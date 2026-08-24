from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess  # nosec B404
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT_EARLY = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "numba"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from models.baseline_common import (
    CLOSE_COLUMN,
    DATE_COLUMN,
    PROJECT_ROOT,
    TARGET_COLUMN,
    binary_direction_metrics,
    discover_folds,
    package_versions,
    regression_metrics,
    sequence_history_features,
)
from models.convolutional_neural_network import make_sequences, make_test_sequences
from models.integrated_multimodal import (
    ARMS,
    FitRequest,
    build_arm_feature_sets,
    build_fit_requests,
    prepare_integrated_fold,
    subset_aligned_regimes,
    validate_cell_integrity,
    validate_regime_training_capacity,
    verify_freeze_manifest,
)
from models.integrated_multimodal_runner import (
    MINIMUM_REGIME_TRAINING_SEQUENCES,
    fold_metrics_from_seed_averaged_predictions,
)
from models.neural_network_folds import inverse_scaled_target
from models.pit_cmm_extension import (
    EVIDENCE_STATUS,
    FINAL_ARM,
    build_compact_six_model_comparison,
    build_six_model_tables,
    evaluate_promotion_gates,
)
from models.pit_cmm_lstm import (
    BATCH_SIZE,
    CONFIG,
    EPOCHS,
    MODEL_KEY,
    MODEL_LABEL,
    PROTOCOL_ID,
    SEQUENCE_WINDOW,
    build_pit_cmm_lstm_model,
)
from models.track_a_analysis import exact_sign_flip_pvalue
from models.track_a_final import FINAL_SEEDS, TRACK_A_MODELS
from models.track_b_forward_news import DAILY_NEWS_FILE
from models.track_b_fusion import _scaled_fold
from models.track_c_inference import average_seed_predictions
from models.track_c_outer import (
    REGIMES,
    route_regime_predictions,
    selected_feature_lookup,
)
from models.track_c_topk_validation import endpoint_regime_mask
from models.track_c_topk_validation_runner import SELECTED_FEATURES_FILE
from models.vmd_feature_pool import FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pit_cmm_lstm_extension_v1"
CELL_ROOT_NAME = "cells"
FREEZE_FILE = PROJECT_ROOT / "test" / "pit_cmm_lstm_freeze_v1.json"
INTEGRATED_FREEZE_MANIFEST = (
    PROJECT_ROOT / "test" / "integrated_multimodal_freeze_v1.json"
)
LOCKED_WINDOWS_FILE = (
    PROJECT_ROOT / "outputs" / "track_a_final_point_in_time_v2" / "locked_windows.csv"
)
REGIME_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_c" / "daily_regime_point_in_time_v2"
FROZEN_RESULTS_DIR = PROJECT_ROOT / "outputs" / "integrated_multimodal_posthoc_v1"
FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4")


@dataclass(frozen=True)
class FitResult:
    prediction: np.ndarray
    training_sequences: int
    fit_seconds: float
    inference_seconds: float
    trainable_parameters: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_extension_freeze() -> dict[str, object]:
    payload = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("PIT-CMM freeze protocol_id is incorrect")
    if payload.get("result_access_at_freeze") is not False:
        raise ValueError("PIT-CMM freeze does not predate result access")
    expected = payload.get("input_sha256")
    if not isinstance(expected, dict):
        raise TypeError("PIT-CMM freeze is missing input hashes")
    paths = {
        "integrated_freeze_manifest": INTEGRATED_FREEZE_MANIFEST,
        "locked_windows": LOCKED_WINDOWS_FILE,
        "daily_news_features": DAILY_NEWS_FILE,
        "regime_shap_selections": SELECTED_FEATURES_FILE,
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Frozen PIT-CMM input is missing: {path}")
        if _sha256(path) != str(expected.get(key)):
            raise ValueError(f"Frozen PIT-CMM input hash changed: {path}")
    return {
        "passed": True,
        "protocol_id": PROTOCOL_ID,
        "verified_inputs": len(paths),
        "freeze_file_sha256": _sha256(FREEZE_FILE),
    }


def cell_directory(output_dir: Path, fold: str, seed: int) -> Path:
    return output_dir / CELL_ROOT_NAME / MODEL_KEY / fold / f"seed_{int(seed)}"


def _cell_paths(directory: Path) -> tuple[Path, ...]:
    return (
        directory / "metrics.csv",
        directory / "fit_registry.csv",
        directory / "run_metadata.json",
        directory / "integrity_audit.json",
        *(directory / f"predictions_{arm}.csv" for arm in ARMS),
    )


def cell_complete(output_dir: Path, fold: str, seed: int) -> bool:
    directory = cell_directory(output_dir, fold, seed)
    try:
        if not all(path.is_file() for path in _cell_paths(directory)):
            return False
        metadata = json.loads(
            (directory / "run_metadata.json").read_text(encoding="utf-8")
        )
        audit = json.loads(
            (directory / "integrity_audit.json").read_text(encoding="utf-8")
        )
        if metadata.get("protocol_id") != PROTOCOL_ID or audit.get("passed") is not True:
            return False
        metrics = pd.read_csv(directory / "metrics.csv")
        registry = pd.read_csv(directory / "fit_registry.csv")
        predictions = {
            arm: pd.read_csv(directory / f"predictions_{arm}.csv") for arm in ARMS
        }
        validate_cell_integrity(
            metrics,
            registry,
            predictions,
            minimum_training_sequences=MINIMUM_REGIME_TRAINING_SEQUENCES,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def _validate_folds(folds: Iterable[str]) -> tuple[str, ...]:
    result = tuple(str(fold) for fold in folds)
    if not result or len(set(result)) != len(result):
        raise ValueError("folds must be non-empty and unique")
    unknown = sorted(set(result).difference(FOLDS))
    if unknown:
        raise ValueError(f"Unknown PIT-CMM folds: {unknown}")
    return result


def _validate_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if not values or len(set(values)) != len(values):
        raise ValueError("seeds must be non-empty and unique")
    return values


def build_cell_commands(
    *,
    python_executable: Path,
    output_dir: Path,
    folds: Iterable[str] = FOLDS,
    seeds: Iterable[int] = FINAL_SEEDS,
    force: bool = False,
) -> list[list[str]]:
    commands: list[list[str]] = []
    for fold in _validate_folds(folds):
        for seed in _validate_seeds(seeds):
            command = [
                str(python_executable),
                "-m",
                "models.pit_cmm_extension_runner",
                "cell",
                "--fold",
                fold,
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
            ]
            if force:
                command.append("--force")
            commands.append(command)
    return commands


def _load_regimes(fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = REGIME_OUTPUT_DIR / fold
    return (
        pd.read_csv(directory / "train_regimes.csv"),
        pd.read_csv(directory / "test_regimes.csv"),
    )


def _selected_shap_features() -> dict[str, tuple[str, ...]]:
    lookup = selected_feature_lookup(pd.read_csv(SELECTED_FEATURES_FILE))
    return {regime: lookup[("shap", regime)] for regime in REGIMES}


def _feature_hash(features: Sequence[str]) -> str:
    return hashlib.sha256("|".join(features).encode("utf-8")).hexdigest()


def _fit_id(fold: str, arm: str, request: FitRequest) -> str:
    material = "|".join(
        [
            PROTOCOL_ID,
            MODEL_KEY,
            fold,
            arm,
            request.scope,
            request.regime,
            str(request.seed),
            *request.features,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def _fit_request(
    request: FitRequest,
    *,
    scaled_fold,
    train_regimes: np.ndarray,
    scaler: Mapping[str, object],
) -> FitResult:
    import tensorflow as tf

    train_features = scaled_fold.train.loc[
        :,
        list(request.features),
    ].to_numpy(dtype=float)
    train_target = scaled_fold.train[TARGET_COLUMN].to_numpy(dtype=float)
    x_train, y_train = make_sequences(
        train_features,
        train_target,
        SEQUENCE_WINDOW,
    )
    if request.scope == "regime":
        mask = endpoint_regime_mask(
            train_regimes,
            regime=request.regime,
            window=SEQUENCE_WINDOW,
        )
        x_fit = x_train[mask]
        y_fit = y_train[mask]
    elif request.scope == "global":
        x_fit = x_train
        y_fit = y_train
    else:
        raise ValueError(f"Unknown PIT-CMM fit scope: {request.scope}")
    if len(x_fit) < MINIMUM_REGIME_TRAINING_SEQUENCES:
        raise ValueError(
            f"{request.scope}/{request.regime} has only {len(x_fit)} sequences"
        )

    feature_indices = [
        scaled_fold.feature_columns.index(name) for name in request.features
    ]
    test_sequences = make_test_sequences(
        sequence_history_features(scaled_fold)[:, feature_indices],
        scaled_fold.test.loc[:, list(request.features)].to_numpy(dtype=float),
        SEQUENCE_WINDOW,
    )
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(request.seed)
    model = build_pit_cmm_lstm_model(
        (SEQUENCE_WINDOW, len(request.features))
    )
    fit_started = time.perf_counter()
    model.fit(
        x_fit,
        y_fit,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    fit_seconds = float(time.perf_counter() - fit_started)
    inference_started = time.perf_counter()
    scaled_prediction = model.predict(test_sequences, verbose=0).reshape(-1)
    inference_seconds = float(time.perf_counter() - inference_started)
    prediction = inverse_scaled_target(scaled_prediction, dict(scaler))
    if prediction.shape != (len(scaled_fold.test),):
        raise ValueError("PIT-CMM prediction shape is invalid")
    if not np.isfinite(prediction).all():
        raise ValueError("PIT-CMM prediction contains non-finite values")
    return FitResult(
        prediction=prediction,
        training_sequences=len(x_fit),
        fit_seconds=fit_seconds,
        inference_seconds=inference_seconds,
        trainable_parameters=int(model.count_params()),
    )


def _prediction_frame(fold, prediction: np.ndarray, regimes: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            DATE_COLUMN: fold.test[DATE_COLUMN],
            "routing_regime": regimes,
            "Close_D": fold.test[CLOSE_COLUMN],
            "y_true": fold.test[TARGET_COLUMN],
            "y_pred": prediction,
        }
    )
    frame["true_direction"] = np.sign(frame["y_true"] - frame["Close_D"])
    frame["pred_direction"] = np.sign(frame["y_pred"] - frame["Close_D"])
    return frame


def _prediction_metrics(fold, prediction: np.ndarray) -> dict[str, float | int]:
    y_true = fold.test[TARGET_COLUMN].to_numpy(dtype=float)
    close = fold.test[CLOSE_COLUMN].to_numpy(dtype=float)
    return {
        **regression_metrics(y_true, prediction),
        **binary_direction_metrics(y_true, prediction, close),
    }


def run_cell(
    *,
    fold: str,
    seed: int,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict[str, object]:
    fold_name = _validate_folds((fold,))[0]
    base_seed = _validate_seeds((seed,))[0]
    if cell_complete(output_dir, fold_name, base_seed) and not force:
        return {"status": "skipped_complete", "fold": fold_name, "seed": base_seed}

    freeze_audit = verify_extension_freeze()
    integrated_audit = verify_freeze_manifest(
        PROJECT_ROOT,
        INTEGRATED_FREEZE_MANIFEST,
    )
    specs = {
        spec.fold: spec
        for spec in discover_folds(FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR)
    }
    spec = specs[fold_name]
    prepared = prepare_integrated_fold(spec, pd.read_csv(DAILY_NEWS_FILE))
    original_numeric = prepared["Global-Numeric"]
    original_news = prepared["Global-Numeric-News"]
    scaled_numeric, scaler_numeric = _scaled_fold(original_numeric)
    scaled_news, scaler_news = _scaled_fold(original_news)

    train_regime_frame, test_regime_frame = _load_regimes(fold_name)
    train_regimes = subset_aligned_regimes(
        original_numeric.train,
        train_regime_frame,
        split="train",
    )
    test_regimes = subset_aligned_regimes(
        original_numeric.test,
        test_regime_frame,
        split="test",
    )
    regime_counts = validate_regime_training_capacity(
        train_regimes,
        window=SEQUENCE_WINDOW,
        minimum=MINIMUM_REGIME_TRAINING_SEQUENCES,
    )
    arm_features = build_arm_feature_sets(
        tuple(original_numeric.feature_columns),
        _selected_shap_features(),
    )
    requests_by_arm = build_fit_requests(
        base_seed=base_seed,
        arm_features=arm_features,
    )

    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass

    metric_rows: list[dict[str, object]] = []
    registry_rows: list[dict[str, object]] = []
    predictions: dict[str, pd.DataFrame] = {}
    cell_started = time.perf_counter()
    for arm in ARMS:
        uses_news = arm.endswith("-News")
        scaled_fold = scaled_news if uses_news else scaled_numeric
        original_fold = original_news if uses_news else original_numeric
        scaler = scaler_news if uses_news else scaler_numeric
        requests = requests_by_arm[arm]
        results: list[FitResult] = []
        for request in requests:
            result = _fit_request(
                request,
                scaled_fold=scaled_fold,
                train_regimes=train_regimes,
                scaler=scaler,
            )
            results.append(result)
            registry_rows.append(
                {
                    "fit_id": _fit_id(fold_name, arm, request),
                    "model": MODEL_KEY,
                    "fold": fold_name,
                    "test_year": spec.test_year,
                    "base_seed": base_seed,
                    "arm": arm,
                    "scope": request.scope,
                    "regime": request.regime,
                    "fit_seed": request.seed,
                    "features": len(request.features),
                    "feature_hash": _feature_hash(request.features),
                    "training_sequences": result.training_sequences,
                    "fit_seconds": result.fit_seconds,
                    "inference_seconds": result.inference_seconds,
                    "trainable_parameters": result.trainable_parameters,
                }
            )
        if arm.startswith("Global-"):
            prediction = results[0].prediction.copy()
        else:
            prediction = route_regime_predictions(
                test_regimes,
                {
                    request.regime: result.prediction
                    for request, result in zip(requests, results, strict=True)
                },
            )
        predictions[arm] = _prediction_frame(
            original_fold,
            prediction,
            test_regimes,
        )
        metric_rows.append(
            {
                "model": MODEL_KEY,
                "fold": fold_name,
                "test_year": spec.test_year,
                "base_seed": base_seed,
                "arm": arm,
                "window": SEQUENCE_WINDOW,
                "models_in_arm": len(requests),
                "conceptual_fit_seconds": float(
                    sum(result.fit_seconds for result in results)
                ),
                "conceptual_inference_seconds": float(
                    sum(result.inference_seconds for result in results)
                ),
                "conceptual_trainable_parameters": int(
                    sum(result.trainable_parameters for result in results)
                ),
                "n_test": len(prediction),
                **_prediction_metrics(original_fold, prediction),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    registry = pd.DataFrame(registry_rows)
    audit = validate_cell_integrity(
        metrics,
        registry,
        predictions,
        minimum_training_sequences=MINIMUM_REGIME_TRAINING_SEQUENCES,
    )
    audit.update(
        {
            "protocol_id": PROTOCOL_ID,
            "extension_freeze_passed": bool(freeze_audit["passed"]),
            "integrated_freeze_passed": bool(integrated_audit["passed"]),
            "regime_training_sequence_counts": regime_counts,
        }
    )
    directory = cell_directory(output_dir, fold_name, base_seed)
    directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(directory / "metrics.csv", index=False)
    registry.to_csv(directory / "fit_registry.csv", index=False)
    for arm, frame in predictions.items():
        frame.to_csv(directory / f"predictions_{arm}.csv", index=False)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "evidence_status": EVIDENCE_STATUS,
        "created_at": _utc_now(),
        "model": MODEL_KEY,
        "model_label": MODEL_LABEL,
        "fold": fold_name,
        "test_year": spec.test_year,
        "base_seed": base_seed,
        "window": SEQUENCE_WINDOW,
        "arms": list(ARMS),
        "unique_fits": len(registry),
        "cell_wall_seconds": float(time.perf_counter() - cell_started),
        "runtime_scope": "model build, fit, and test inference",
        "incremental_api_cost_usd": 0,
        "config": CONFIG,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "scipy", "tensorflow"]
        ),
    }
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (directory / "integrity_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    if not cell_complete(output_dir, fold_name, base_seed):
        raise RuntimeError("PIT-CMM cell failed its post-write integrity audit")
    return {**metadata, "status": "completed"}


def _expected_cell_keys() -> set[tuple[str, int]]:
    return {(fold, int(seed)) for fold in FOLDS for seed in FINAL_SEEDS}


def _collect_csv(output_dir: Path, name: str) -> pd.DataFrame:
    paths = sorted(
        (output_dir / CELL_ROOT_NAME / MODEL_KEY).glob(f"*/seed_*/{name}")
    )
    if not paths:
        raise FileNotFoundError(f"No PIT-CMM {name} files were found")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _collect_seed_averaged_predictions(
    output_dir: Path,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (fold, arm), group in metrics.groupby(["fold", "arm"], sort=False):
        frames_by_seed = {
            int(seed): pd.read_csv(
                cell_directory(output_dir, str(fold), int(seed))
                / f"predictions_{arm}.csv"
            )
            for seed in sorted(group["base_seed"].astype(int).unique())
        }
        averaged = average_seed_predictions(frames_by_seed)
        averaged.insert(0, "model", MODEL_KEY)
        averaged.insert(1, "fold", fold)
        averaged.insert(2, "test_year", int(group["test_year"].iloc[0]))
        averaged.insert(3, "arm", arm)
        rows.append(averaged)
    return pd.concat(rows, ignore_index=True)


def build_lstm_fold_contrast(
    frozen_fold_metrics: pd.DataFrame,
    pit_cmm_fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    frozen = frozen_fold_metrics.loc[
        frozen_fold_metrics["model"].eq("lstm")
        & frozen_fold_metrics["arm"].eq(FINAL_ARM),
        ["fold", "balanced_accuracy"],
    ].copy()
    ours = pit_cmm_fold_metrics.loc[
        pit_cmm_fold_metrics["model"].eq(MODEL_KEY)
        & pit_cmm_fold_metrics["arm"].eq(FINAL_ARM),
        ["fold", "balanced_accuracy"],
    ].copy()
    paired = frozen.merge(
        ours,
        on="fold",
        how="inner",
        validate="one_to_one",
        suffixes=("_lstm", "_pit_cmm"),
    ).sort_values("fold")
    if len(paired) != 4:
        raise ValueError("LSTM contrast requires exactly four final-arm folds")
    paired["balanced_accuracy_delta_pp"] = (
        paired["balanced_accuracy_pit_cmm"]
        - paired["balanced_accuracy_lstm"]
    ) * 100.0
    return paired.reset_index(drop=True)


def _parameter_deltas() -> tuple[pd.DataFrame, list[float]]:
    from models.full_non_ta_experiments import build_lstm_model

    rows: list[dict[str, object]] = []
    deltas: list[float] = []
    for features in (38, 88, 130):
        ours = build_pit_cmm_lstm_model((SEQUENCE_WINDOW, features)).count_params()
        baseline = build_lstm_model((SEQUENCE_WINDOW, features)).count_params()
        delta = (ours - baseline) / float(baseline)
        deltas.append(delta)
        rows.append(
            {
                "features": features,
                "pit_cmm_parameters": ours,
                "lstm_parameters": baseline,
                "parameter_delta_fraction": delta,
            }
        )
    return pd.DataFrame(rows), deltas


def aggregate_experiment(*, output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    freeze_audit = verify_extension_freeze()
    incomplete = [
        key
        for key in sorted(_expected_cell_keys())
        if not cell_complete(output_dir, *key)
    ]
    if incomplete:
        raise ValueError(f"PIT-CMM run has incomplete cells: {incomplete[:5]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _collect_csv(output_dir, "metrics.csv")
    registry = _collect_csv(output_dir, "fit_registry.csv")
    if len(metrics) != len(_expected_cell_keys()) * len(ARMS):
        raise ValueError("PIT-CMM metric row count is incorrect")
    if len(registry) != len(_expected_cell_keys()) * 8:
        raise ValueError("PIT-CMM fit registry row count is incorrect")
    if registry["fit_id"].duplicated().any():
        raise ValueError("PIT-CMM fit registry contains duplicate fit ids")
    metrics.to_csv(output_dir / "metrics_by_seed_fold.csv", index=False)
    registry.to_csv(output_dir / "fit_registry.csv", index=False)

    predictions = _collect_seed_averaged_predictions(output_dir, metrics)
    predictions.to_csv(output_dir / "predictions_seed_averaged.csv", index=False)
    fold_metrics = fold_metrics_from_seed_averaged_predictions(predictions)
    fold_metrics.to_csv(output_dir / "fold_metrics_seed_averaged.csv", index=False)
    arm_summary = (
        fold_metrics.groupby(["model", "arm"], sort=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            direction_accuracy_mean=("direction_accuracy", "mean"),
            direction_accuracy_std=("direction_accuracy", "std"),
            mcc_mean=("mcc", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            temporal_folds=("fold", "nunique"),
        )
        .reset_index()
    )
    arm_summary.to_csv(output_dir / "arm_summary.csv", index=False)

    frozen_summary = pd.read_csv(FROZEN_RESULTS_DIR / "arm_summary.csv")
    six_all, six_final = build_six_model_tables(frozen_summary, arm_summary)
    six_all.to_csv(output_dir / "six_model_all_arms_comparison.csv", index=False)
    six_final.to_csv(output_dir / "six_model_final_arm_comparison.csv", index=False)
    compact = build_compact_six_model_comparison(six_all)
    compact.to_csv(
        output_dir / "six_model_compact_comparison.csv",
        index=False,
    )

    prediction_diagnostics = predictions.copy()
    prediction_diagnostics["predicted_change"] = (
        prediction_diagnostics["y_pred"] - prediction_diagnostics["Close_D"]
    )
    distribution = (
        prediction_diagnostics.groupby(["model", "fold", "arm"], sort=False)
        .agg(
            observations=("y_pred", "size"),
            prediction_std=("y_pred", "std"),
            predicted_change_std=("predicted_change", "std"),
            predicted_up_share=(
                "predicted_change",
                lambda values: float((values > 0.0).mean()),
            ),
            predicted_down_share=(
                "predicted_change",
                lambda values: float((values < 0.0).mean()),
            ),
            predicted_abstention_share=(
                "predicted_change",
                lambda values: float((values == 0.0).mean()),
            ),
        )
        .reset_index()
    )
    distribution["single_class_prediction"] = (
        distribution["predicted_up_share"].eq(0.0)
        | distribution["predicted_down_share"].eq(0.0)
    )
    distribution.to_csv(
        output_dir / "prediction_distribution_diagnostics.csv",
        index=False,
    )

    frozen_fold = pd.read_csv(
        FROZEN_RESULTS_DIR / "fold_metrics_seed_averaged.csv"
    )
    contrast = build_lstm_fold_contrast(frozen_fold, fold_metrics)
    contrast.to_csv(output_dir / "pit_cmm_vs_lstm_fold_contrast.csv", index=False)
    parameter_table, parameter_deltas = _parameter_deltas()
    parameter_table.to_csv(output_dir / "parameter_budget.csv", index=False)
    complete_finite = bool(
        len(metrics) == 80
        and np.isfinite(
            predictions[["Close_D", "y_true", "y_pred"]].to_numpy(dtype=float)
        ).all()
    )
    promotion = evaluate_promotion_gates(
        contrast[["fold", "balanced_accuracy_delta_pp"]],
        parameter_deltas=parameter_deltas,
        complete_finite_predictions=complete_finite,
    )
    promotion["exact_sign_flip_pvalue"] = exact_sign_flip_pvalue(
        contrast["balanced_accuracy_delta_pp"].to_numpy(dtype=float)
    )
    (output_dir / "promotion_decision.json").write_text(
        json.dumps(promotion, indent=2),
        encoding="utf-8",
    )

    runtime = (
        registry.groupby(["model", "arm"], sort=False)
        .agg(
            executed_fits=("fit_id", "size"),
            fit_seconds_total=("fit_seconds", "sum"),
            fit_seconds_mean=("fit_seconds", "mean"),
            inference_seconds_total=("inference_seconds", "sum"),
            trainable_parameters_mean=("trainable_parameters", "mean"),
            training_sequences_min=("training_sequences", "min"),
            training_sequences_max=("training_sequences", "max"),
        )
        .reset_index()
    )
    runtime.to_csv(output_dir / "runtime_summary.csv", index=False)
    frozen_runtime = pd.read_csv(FROZEN_RESULTS_DIR / "runtime_summary.csv")
    six_model_runtime = pd.concat(
        [
            frozen_runtime.loc[frozen_runtime["arm"].eq(FINAL_ARM)].assign(
                evidence_status="frozen_existing_result"
            ),
            runtime.loc[runtime["arm"].eq(FINAL_ARM)].assign(
                evidence_status=EVIDENCE_STATUS
            ),
        ],
        ignore_index=True,
    )
    expected_runtime_models = [*TRACK_A_MODELS, MODEL_KEY]
    runtime_order = {
        model: index for index, model in enumerate(expected_runtime_models)
    }
    six_model_runtime["_order"] = six_model_runtime["model"].map(
        runtime_order
    )
    if six_model_runtime["_order"].isna().any():
        raise ValueError("Unexpected model in six-model runtime comparison")
    six_model_runtime = (
        six_model_runtime.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    if six_model_runtime["model"].tolist() != expected_runtime_models:
        raise ValueError("Six-model runtime comparison is incomplete")
    six_model_runtime.to_csv(
        output_dir / "six_model_runtime_comparison.csv",
        index=False,
    )
    cell_runtime_rows = []
    for fold, seed in sorted(_expected_cell_keys()):
        payload = json.loads(
            (
                cell_directory(output_dir, fold, seed) / "run_metadata.json"
            ).read_text(encoding="utf-8")
        )
        cell_runtime_rows.append(
            {
                "model": MODEL_KEY,
                "fold": fold,
                "base_seed": seed,
                "cell_wall_seconds": float(payload["cell_wall_seconds"]),
            }
        )
    pd.DataFrame(cell_runtime_rows).to_csv(
        output_dir / "runtime_by_cell.csv",
        index=False,
    )

    integrity = {
        "passed": True,
        "protocol_id": PROTOCOL_ID,
        "freeze_audit": freeze_audit,
        "completed_cells": len(_expected_cell_keys()),
        "metric_rows": len(metrics),
        "fit_rows": len(registry),
        "seed_averaged_fold_arm_rows": len(fold_metrics),
        "all_predictions_finite": complete_finite,
        "minimum_training_sequences": int(registry["training_sequences"].min()),
        "promotion_passed": bool(promotion["passed"]),
    }
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "completed_at": _utc_now(),
        "evidence_status": EVIDENCE_STATUS,
        "models_in_comparison": [*TRACK_A_MODELS, MODEL_KEY],
        "new_model_only_executed": True,
        "folds": list(FOLDS),
        "seeds": list(FINAL_SEEDS),
        "arms": list(ARMS),
        "window": SEQUENCE_WINDOW,
        "primary_metric": "balanced_accuracy",
        "primary_arm": FINAL_ARM,
        "primary_comparator": "lstm",
        "incremental_api_cost_usd": 0,
        "freeze_file_sha256": _sha256(FREEZE_FILE),
        "source_hashes": {
            "pit_cmm_lstm.py": _sha256(PROJECT_ROOT / "models" / "pit_cmm_lstm.py"),
            "pit_cmm_extension.py": _sha256(
                PROJECT_ROOT / "models" / "pit_cmm_extension.py"
            ),
            "pit_cmm_extension_runner.py": _sha256(Path(__file__)),
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "scipy", "tensorflow"]
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return {**metadata, "promotion": promotion, "integrity": integrity}


def run_cells_isolated(
    *,
    python_executable: Path = Path(sys.executable),
    output_dir: Path = OUTPUT_DIR,
    folds: Iterable[str] = FOLDS,
    seeds: Iterable[int] = FINAL_SEEDS,
    force: bool = False,
) -> dict[str, int]:
    commands = build_cell_commands(
        python_executable=python_executable,
        output_dir=output_dir,
        folds=folds,
        seeds=seeds,
        force=force,
    )
    completed = skipped = 0
    total = len(commands)
    for index, command in enumerate(commands, start=1):
        fold = command[command.index("--fold") + 1]
        seed = int(command[command.index("--seed") + 1])
        if cell_complete(output_dir, fold, seed) and not force:
            skipped += 1
            print(f"[{index}/{total}] skip {fold}/seed_{seed}", flush=True)
            continue
        print(f"[{index}/{total}] run {fold}/seed_{seed}", flush=True)
        # The argv is built entirely from frozen constants.
        subprocess.run(  # nosec B603
            command,
            cwd=PROJECT_ROOT,
            check=True,
        )
        completed += 1
    return {"total": total, "completed": completed, "skipped": skipped}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen PIT-CMM-LSTM exploratory extension."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    cell = subparsers.add_parser("cell")
    cell.add_argument("--fold", choices=FOLDS, required=True)
    cell.add_argument("--seed", type=int, required=True)
    cell.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    cell.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--fold", action="append", choices=FOLDS)
    run.add_argument("--seed", action="append", type=int)
    run.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    run.add_argument("--force", action="store_true")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if args.command == "cell":
        result = run_cell(
            fold=args.fold,
            seed=args.seed,
            output_dir=args.output_dir,
            force=args.force,
        )
    elif args.command == "run":
        result = run_cells_isolated(
            output_dir=args.output_dir,
            folds=FOLDS if args.fold is None else args.fold,
            seeds=FINAL_SEEDS if args.seed is None else args.seed,
            force=args.force,
        )
    else:
        result = aggregate_experiment(output_dir=args.output_dir)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
