from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

# Keep all mutable runtime caches on the user's D: workspace.
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
)
from models.integrated_multimodal import (
    ARMS,
    CONTRASTS,
    DIRECT_CONTRASTS,
    PROTOCOL_ID,
    FitRequest,
    apply_integrated_holm,
    build_arm_feature_sets,
    build_fit_requests,
    build_integrated_fold_contrasts,
    integrated_fold_inference,
    prepare_integrated_fold,
    subset_aligned_regimes,
    validate_cell_integrity,
    validate_regime_training_capacity,
    verify_freeze_manifest,
)
from models.track_a_final import (
    FINAL_SEEDS,
    TRACK_A_MODELS,
    load_locked_windows,
)
from models.track_b_forward_news import DAILY_NEWS_FILE
from models.track_b_fusion import _scaled_fold
from models.track_c_inference import (
    apply_holm_by_family,
    average_seed_predictions,
    moving_block_bootstrap,
)
from models.track_c_outer import (
    REGIMES,
    route_regime_predictions,
    selected_feature_lookup,
)
from models.track_c_outer_runner import FitResult, _fit_request
from models.track_c_topk_validation_runner import SELECTED_FEATURES_FILE
from models.vmd_feature_pool import FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "integrated_multimodal_posthoc_v1"
CELL_ROOT_NAME = "cells"
FREEZE_MANIFEST = PROJECT_ROOT / "test" / "integrated_multimodal_freeze_v1.json"
REGIME_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_c" / "daily_regime_point_in_time_v2"
MINIMUM_REGIME_TRAINING_SEQUENCES = 200
BOOTSTRAP_BLOCK_LENGTH = 10
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_801
FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_hash(features: Sequence[str]) -> str:
    return hashlib.sha256("|".join(features).encode("utf-8")).hexdigest()


def cell_directory(
    output_dir: Path,
    model: str,
    fold: str,
    seed: int,
) -> Path:
    return output_dir / CELL_ROOT_NAME / model / fold / f"seed_{int(seed)}"


def _cell_paths(directory: Path) -> tuple[Path, ...]:
    return (
        directory / "metrics.csv",
        directory / "fit_registry.csv",
        directory / "run_metadata.json",
        directory / "integrity_audit.json",
        *(directory / f"predictions_{arm}.csv" for arm in ARMS),
    )


def cell_complete(
    output_dir: Path,
    model: str,
    fold: str,
    seed: int,
) -> bool:
    directory = cell_directory(output_dir, model, fold, seed)
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


def fit_execution_id(model: str, fold: str, request: FitRequest) -> str:
    material = "|".join(
        [
            PROTOCOL_ID,
            model,
            fold,
            request.scope,
            request.regime,
            str(request.seed),
            *request.features,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def _validate_models(models: Iterable[str]) -> tuple[str, ...]:
    result = tuple(str(model) for model in models)
    if not result or len(set(result)) != len(result):
        raise ValueError("models must be non-empty and unique")
    unknown = sorted(set(result).difference(TRACK_A_MODELS))
    if unknown:
        raise ValueError(f"Unknown integrated models: {unknown}")
    return result


def _validate_folds(folds: Iterable[str]) -> tuple[str, ...]:
    result = tuple(str(fold) for fold in folds)
    if not result or len(set(result)) != len(result):
        raise ValueError("folds must be non-empty and unique")
    unknown = sorted(set(result).difference(FOLDS))
    if unknown:
        raise ValueError(f"Unknown integrated folds: {unknown}")
    return result


def _validate_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(seed) for seed in seeds)
    if not result or len(set(result)) != len(result):
        raise ValueError("seeds must be non-empty and unique")
    return result


def build_cell_commands(
    *,
    python_executable: Path,
    output_dir: Path,
    models: Iterable[str] = TRACK_A_MODELS,
    folds: Iterable[str] = FOLDS,
    seeds: Iterable[int] = FINAL_SEEDS,
    force: bool = False,
) -> list[list[str]]:
    keys = _validate_models(models)
    fold_names = _validate_folds(folds)
    seed_values = _validate_seeds(seeds)
    commands: list[list[str]] = []
    for model in keys:
        for fold in fold_names:
            for seed in seed_values:
                command = [
                    str(python_executable),
                    "-m",
                    "models.integrated_multimodal_runner",
                    "cell",
                    "--model",
                    model,
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


def _window_map() -> dict[str, int]:
    return {
        str(row.model): int(row.selected_sequence_window)
        for row in load_locked_windows().itertuples(index=False)
    }


def _load_regimes(fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = REGIME_OUTPUT_DIR / fold
    train = pd.read_csv(directory / "train_regimes.csv")
    test = pd.read_csv(directory / "test_regimes.csv")
    return train, test


def _selected_shap_features() -> dict[str, tuple[str, ...]]:
    lookup = selected_feature_lookup(pd.read_csv(SELECTED_FEATURES_FILE))
    return {regime: lookup[("shap", regime)] for regime in REGIMES}


def _prediction_metrics(fold, prediction: np.ndarray) -> dict[str, float | int]:
    y_true = fold.test[TARGET_COLUMN].to_numpy(dtype=float)
    close = fold.test[CLOSE_COLUMN].to_numpy(dtype=float)
    return {
        **regression_metrics(y_true, prediction),
        **binary_direction_metrics(y_true, prediction, close),
    }


def _prediction_frame(
    fold,
    prediction: np.ndarray,
    test_regimes: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            DATE_COLUMN: fold.test[DATE_COLUMN],
            "routing_regime": test_regimes,
            "Close_D": fold.test[CLOSE_COLUMN],
            "y_true": fold.test[TARGET_COLUMN],
            "y_pred": prediction,
        }
    )
    frame["true_direction"] = np.sign(frame["y_true"] - frame["Close_D"])
    frame["pred_direction"] = np.sign(frame["y_pred"] - frame["Close_D"])
    return frame


def _validate_registered_feature_counts(
    arm_features: Mapping[str, Mapping[str, Sequence[str]]],
) -> None:
    expected = {
        ("Global-Numeric", "global"): 122,
        ("Global-Numeric-News", "global"): 130,
        ("Regime-SHAP-Numeric", "bull"): 30,
        ("Regime-SHAP-Numeric", "sideway"): 122,
        ("Regime-SHAP-Numeric", "bear"): 80,
        ("Regime-SHAP-Numeric-News", "bull"): 38,
        ("Regime-SHAP-Numeric-News", "sideway"): 130,
        ("Regime-SHAP-Numeric-News", "bear"): 88,
    }
    actual = {
        (arm, regime): len(features)
        for arm, groups in arm_features.items()
        for regime, features in groups.items()
    }
    if actual != expected:
        raise ValueError(f"Registered feature counts changed: {actual}")


def run_cell(
    *,
    model: str,
    fold: str,
    seed: int,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict[str, object]:
    _validate_models((model,))
    _validate_folds((fold,))
    base_seed = _validate_seeds((seed,))[0]
    if cell_complete(output_dir, model, fold, base_seed) and not force:
        return {
            "status": "skipped_complete",
            "model": model,
            "fold": fold,
            "seed": base_seed,
        }

    manifest_audit = verify_freeze_manifest(PROJECT_ROOT, FREEZE_MANIFEST)
    specs = {
        spec.fold: spec
        for spec in discover_folds(FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR)
    }
    spec = specs[fold]
    daily_news = pd.read_csv(DAILY_NEWS_FILE)
    prepared = prepare_integrated_fold(spec, daily_news)
    original_numeric = prepared["Global-Numeric"]
    original_news = prepared["Global-Numeric-News"]
    scaled_numeric, scaler_numeric = _scaled_fold(original_numeric)
    scaled_news, scaler_news = _scaled_fold(original_news)

    train_regime_frame, test_regime_frame = _load_regimes(fold)
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
    window = _window_map()[model]
    regime_counts = validate_regime_training_capacity(
        train_regimes,
        window=window,
        minimum=MINIMUM_REGIME_TRAINING_SEQUENCES,
    )
    arm_features = build_arm_feature_sets(
        tuple(original_numeric.feature_columns),
        _selected_shap_features(),
    )
    _validate_registered_feature_counts(arm_features)
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

    metrics_rows: list[dict[str, object]] = []
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
                model,
                request,
                scaled_fold=scaled_fold,
                train_regimes=train_regimes,
                window=window,
                scaler=scaler,
            )
            results.append(result)
            registry_rows.append(
                {
                    "fit_id": fit_execution_id(model, fold, request),
                    "model": model,
                    "fold": fold,
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
        metrics_rows.append(
            {
                "model": model,
                "fold": fold,
                "test_year": spec.test_year,
                "base_seed": base_seed,
                "arm": arm,
                "window": window,
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

    metrics = pd.DataFrame(metrics_rows)
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
            "input_manifest_passed": bool(manifest_audit["passed"]),
            "feature_counts_passed": True,
            "regime_training_sequence_counts": regime_counts,
        }
    )
    directory = cell_directory(output_dir, model, fold, base_seed)
    directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(directory / "metrics.csv", index=False)
    registry.to_csv(directory / "fit_registry.csv", index=False)
    for arm, frame in predictions.items():
        frame.to_csv(directory / f"predictions_{arm}.csv", index=False)
    wall_seconds = float(time.perf_counter() - cell_started)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "evidence_status": "post_hoc_integrated_extension",
        "news_feature_source": "frozen_expanding_local_nlp",
        "incremental_api_cost_usd": 0,
        "created_at": _utc_now(),
        "model": model,
        "fold": fold,
        "test_year": spec.test_year,
        "base_seed": base_seed,
        "window": window,
        "arms": list(ARMS),
        "unique_fits": len(registry),
        "cell_wall_seconds": wall_seconds,
        "train_period": [
            original_numeric.train[DATE_COLUMN].min().date().isoformat(),
            original_numeric.train[DATE_COLUMN].max().date().isoformat(),
        ],
        "test_period": [
            original_numeric.test[DATE_COLUMN].min().date().isoformat(),
            original_numeric.test[DATE_COLUMN].max().date().isoformat(),
        ],
        "scaler_fit_scope": "current common-cohort fold train only",
        "daily_news_assignment": "strictly next trading day",
        "runtime_scope": "model build, fit, and test inference",
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "scipy", "tensorflow"]
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (directory / "integrity_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    if not cell_complete(output_dir, model, fold, base_seed):
        raise RuntimeError("Integrated cell failed its post-write integrity audit")
    result = dict(metadata)
    result["status"] = "completed"
    return result


def fold_metrics_from_seed_averaged_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "model",
        "fold",
        "test_year",
        "arm",
        "Close_D",
        "y_true",
        "y_pred",
        "seeds_averaged",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Seed-averaged predictions are missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for (model, fold, test_year, arm), group in predictions.groupby(
        ["model", "fold", "test_year", "arm"], sort=False
    ):
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        close = group["Close_D"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "fold": fold,
                "test_year": int(test_year),
                "arm": arm,
                "seeds_averaged": int(group["seeds_averaged"].iloc[0]),
                "n_test": len(group),
                **regression_metrics(y_true, y_pred),
                **binary_direction_metrics(y_true, y_pred, close),
            }
        )
    return pd.DataFrame(rows)


def _daily_bacc_contribution(
    true_direction: np.ndarray,
    treatment_direction: np.ndarray,
    control_direction: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    true_values = true_direction[eligible]
    treatment = treatment_direction[eligible]
    control = control_direction[eligible]
    if set(true_values) != {-1, 1}:
        raise ValueError("Daily contrast requires both direction classes")
    result = np.empty(len(true_values), dtype=float)
    total = len(true_values)
    for direction in (-1, 1):
        rows = true_values == direction
        class_count = int(np.count_nonzero(rows))
        weight = 0.5 / class_count * total * 100.0
        result[rows] = weight * (
            (treatment[rows] == direction).astype(float)
            - (control[rows] == direction).astype(float)
        )
    return result


def daily_contrast_effects(
    aligned_wide_predictions: pd.DataFrame,
    contrast: str,
) -> dict[str, np.ndarray]:
    required = {"Close_D", "y_true", *ARMS}
    missing = sorted(required.difference(aligned_wide_predictions.columns))
    if missing:
        raise ValueError(f"Wide predictions are missing columns: {missing}")
    if contrast not in CONTRASTS:
        raise ValueError(f"Unknown integrated contrast: {contrast}")
    frame = aligned_wide_predictions.reset_index(drop=True)
    numeric = frame[["Close_D", "y_true", *ARMS]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Wide predictions contain non-finite values")
    y_true = frame["y_true"].to_numpy(dtype=float)
    close = frame["Close_D"].to_numpy(dtype=float)
    true_direction = np.sign(y_true - close).astype(int)
    predictions = {
        arm: frame[arm].to_numpy(dtype=float) for arm in ARMS
    }
    predicted_direction = {
        arm: np.sign(values - close).astype(int)
        for arm, values in predictions.items()
    }

    def pair_effects(
        treatment: str,
        control: str,
        *,
        common_eligible: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        squared_error = np.square(predictions[treatment] - y_true) - np.square(
            predictions[control] - y_true
        )
        eligible = (
            (true_direction != 0)
            & (predicted_direction[treatment] != 0)
            & (predicted_direction[control] != 0)
        )
        if common_eligible is not None:
            eligible &= common_eligible
        bacc = _daily_bacc_contribution(
            true_direction,
            predicted_direction[treatment],
            predicted_direction[control],
            eligible,
        )
        return squared_error, bacc

    if contrast in DIRECT_CONTRASTS:
        treatment, control = DIRECT_CONTRASTS[contrast]
        squared_error, bacc = pair_effects(treatment, control)
    else:
        common = true_direction != 0
        for arm in ARMS:
            common &= predicted_direction[arm] != 0
        regime_squared, regime_bacc = pair_effects(
            "Regime-SHAP-Numeric-News",
            "Regime-SHAP-Numeric",
            common_eligible=common,
        )
        global_squared, global_bacc = pair_effects(
            "Global-Numeric-News",
            "Global-Numeric",
            common_eligible=common,
        )
        squared_error = regime_squared - global_squared
        bacc = regime_bacc - global_bacc
    return {
        "squared_error_loss_delta": squared_error,
        "balanced_accuracy_delta_pp": bacc,
    }


def _expected_cell_keys() -> set[tuple[str, str, int]]:
    return {
        (model, fold, int(seed))
        for model in TRACK_A_MODELS
        for fold in FOLDS
        for seed in FINAL_SEEDS
    }


def _collect_metrics(output_dir: Path) -> pd.DataFrame:
    paths = sorted((output_dir / CELL_ROOT_NAME).glob("*/*/seed_*/metrics.csv"))
    if not paths:
        raise FileNotFoundError("No integrated cell metrics were found")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _collect_registry(output_dir: Path) -> pd.DataFrame:
    paths = sorted(
        (output_dir / CELL_ROOT_NAME).glob("*/*/seed_*/fit_registry.csv")
    )
    if not paths:
        raise FileNotFoundError("No integrated fit registries were found")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _collect_seed_averaged_predictions(
    output_dir: Path,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (model, fold, arm), group in metrics.groupby(
        ["model", "fold", "arm"], sort=False
    ):
        frames_by_seed = {
            int(seed): pd.read_csv(
                cell_directory(output_dir, str(model), str(fold), int(seed))
                / f"predictions_{arm}.csv"
            )
            for seed in sorted(group["base_seed"].astype(int).unique())
        }
        averaged = average_seed_predictions(frames_by_seed)
        averaged.insert(0, "model", model)
        averaged.insert(1, "fold", fold)
        averaged.insert(2, "test_year", int(group["test_year"].iloc[0]))
        averaged.insert(3, "arm", arm)
        rows.append(averaged)
    return pd.concat(rows, ignore_index=True)


def _wide_fold_predictions(group: pd.DataFrame) -> pd.DataFrame:
    reference = group.loc[group["arm"].eq(ARMS[0])].sort_values(DATE_COLUMN)
    if reference.empty:
        raise ValueError("Reference arm predictions are missing")
    result = reference[[DATE_COLUMN, "Close_D", "y_true"]].reset_index(drop=True)
    for arm in ARMS:
        arm_frame = group.loc[group["arm"].eq(arm)].sort_values(DATE_COLUMN)
        if len(arm_frame) != len(result):
            raise ValueError(f"{arm} has an unexpected number of prediction rows")
        if not pd.to_datetime(arm_frame[DATE_COLUMN]).reset_index(drop=True).equals(
            pd.to_datetime(result[DATE_COLUMN]).reset_index(drop=True)
        ):
            raise ValueError("Arm prediction dates do not align")
        for column in ("Close_D", "y_true"):
            if not np.allclose(
                arm_frame[column].to_numpy(dtype=float),
                result[column].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("Arm prediction targets do not align")
        result[arm] = arm_frame["y_pred"].to_numpy(dtype=float)
    return result


def _daily_block_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in TRACK_A_MODELS:
        model_frame = predictions.loc[predictions["model"].eq(model)]
        for contrast in CONTRASTS:
            effects_by_metric: dict[str, list[np.ndarray]] = {
                "squared_error_loss_delta": [],
                "balanced_accuracy_delta_pp": [],
            }
            for fold in FOLDS:
                fold_frame = model_frame.loc[model_frame["fold"].eq(fold)]
                effects = daily_contrast_effects(
                    _wide_fold_predictions(fold_frame),
                    contrast,
                )
                for metric, values in effects.items():
                    effects_by_metric[metric].append(values)
            for metric, fold_effects in effects_by_metric.items():
                result = moving_block_bootstrap(
                    fold_effects,
                    block_length=BOOTSTRAP_BLOCK_LENGTH,
                    replicates=BOOTSTRAP_REPLICATES,
                    seed=BOOTSTRAP_SEED,
                )
                rows.append(
                    {
                        "model": model,
                        "contrast": contrast,
                        "metric": metric,
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def _runtime_summary(
    registry: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    summary = (
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
    cell_rows = []
    for model, fold, seed in sorted(_expected_cell_keys()):
        payload = json.loads(
            (
                cell_directory(output_dir, model, fold, seed)
                / "run_metadata.json"
            ).read_text(encoding="utf-8")
        )
        cell_rows.append(
            {
                "model": model,
                "fold": fold,
                "base_seed": seed,
                "cell_wall_seconds": float(payload["cell_wall_seconds"]),
            }
        )
    pd.DataFrame(cell_rows).to_csv(output_dir / "runtime_by_cell.csv", index=False)
    return summary


def aggregate_experiment(
    *,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    manifest_audit = verify_freeze_manifest(PROJECT_ROOT, FREEZE_MANIFEST)
    incomplete = [
        key
        for key in sorted(_expected_cell_keys())
        if not cell_complete(output_dir, *key)
    ]
    if incomplete:
        raise ValueError(f"Integrated run has incomplete cells: {incomplete[:5]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _collect_metrics(output_dir)
    registry = _collect_registry(output_dir)
    expected_metric_rows = len(_expected_cell_keys()) * len(ARMS)
    expected_fit_rows = len(_expected_cell_keys()) * 8
    if len(metrics) != expected_metric_rows:
        raise ValueError("Integrated metric row count is incorrect")
    if len(registry) != expected_fit_rows or registry["fit_id"].duplicated().any():
        raise ValueError("Integrated fit registry cardinality is incorrect")
    cell_groups = metrics.groupby(["model", "fold", "base_seed"])["arm"]
    if not (
        cell_groups.size().eq(len(ARMS)).all()
        and cell_groups.nunique().eq(len(ARMS)).all()
    ):
        raise ValueError("Integrated metric cells do not contain four unique arms")

    metrics.to_csv(output_dir / "metrics_by_seed_fold.csv", index=False)
    registry.to_csv(output_dir / "fit_registry.csv", index=False)
    predictions = _collect_seed_averaged_predictions(output_dir, metrics)
    predictions.to_csv(output_dir / "predictions_seed_averaged.csv", index=False)
    fold_metrics = fold_metrics_from_seed_averaged_predictions(predictions)
    fold_metrics.to_csv(
        output_dir / "fold_metrics_seed_averaged.csv", index=False
    )
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

    paired = build_integrated_fold_contrasts(fold_metrics)
    paired.to_csv(output_dir / "paired_fold_contrasts.csv", index=False)
    fold_inference = apply_integrated_holm(integrated_fold_inference(paired))
    fold_inference["minimum_attainable_nonzero_pvalue"] = 0.125
    fold_inference.to_csv(output_dir / "fold_inference_holm.csv", index=False)
    bootstrap = apply_holm_by_family(
        _daily_block_bootstrap(predictions),
        pvalue_column="two_sided_pvalue",
    )
    bootstrap.to_csv(output_dir / "daily_block_bootstrap_holm.csv", index=False)

    runtime = _runtime_summary(registry, output_dir)
    runtime.to_csv(output_dir / "runtime_summary.csv", index=False)
    paper_table = arm_summary.copy()
    paper_table["balanced_accuracy_mean_pct"] = (
        paper_table["balanced_accuracy_mean"] * 100.0
    )
    paper_table["direction_accuracy_mean_pct"] = (
        paper_table["direction_accuracy_mean"] * 100.0
    )
    paper_table.to_csv(output_dir / "paper_integrated_table.csv", index=False)

    feature_counts = (
        registry.groupby(["arm", "regime"])["features"]
        .unique()
        .map(lambda values: sorted(int(value) for value in values))
        .to_dict()
    )
    integrity = {
        "passed": True,
        "protocol_id": PROTOCOL_ID,
        "input_manifest": manifest_audit,
        "expected_cells": len(_expected_cell_keys()),
        "completed_cells": len(_expected_cell_keys()),
        "metric_rows": len(metrics),
        "fit_rows": len(registry),
        "seed_averaged_fold_arm_rows": len(fold_metrics),
        "paired_fold_rows": len(paired),
        "fold_inference_rows": len(fold_inference),
        "bootstrap_rows": len(bootstrap),
        "minimum_training_sequences": int(registry["training_sequences"].min()),
        "all_predictions_finite": bool(
            np.isfinite(predictions[["Close_D", "y_true", "y_pred"]]).all().all()
        ),
        "feature_counts": {
            f"{arm}/{regime}": values
            for (arm, regime), values in feature_counts.items()
        },
    }
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "completed_at": _utc_now(),
        "evidence_status": "post_hoc_integrated_extension",
        "news_feature_source": "frozen_expanding_local_nlp",
        "incremental_api_cost_usd": 0,
        "models": list(TRACK_A_MODELS),
        "folds": list(FOLDS),
        "seeds": list(FINAL_SEEDS),
        "arms": list(ARMS),
        "primary_metric": "balanced_accuracy",
        "input_freeze_sha256": _sha256(FREEZE_MANIFEST),
        "source_hashes": {
            "integrated_multimodal.py": _sha256(
                PROJECT_ROOT / "models" / "integrated_multimodal.py"
            ),
            "integrated_multimodal_runner.py": _sha256(Path(__file__)),
        },
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "scipy", "tensorflow"]
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "artifacts": [
            "metrics_by_seed_fold.csv",
            "fit_registry.csv",
            "predictions_seed_averaged.csv",
            "fold_metrics_seed_averaged.csv",
            "arm_summary.csv",
            "paired_fold_contrasts.csv",
            "fold_inference_holm.csv",
            "daily_block_bootstrap_holm.csv",
            "runtime_by_cell.csv",
            "runtime_summary.csv",
            "paper_integrated_table.csv",
            "integrity_audit.json",
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def run_cells_isolated(
    *,
    python_executable: Path = Path(sys.executable),
    output_dir: Path = OUTPUT_DIR,
    models: Iterable[str] = TRACK_A_MODELS,
    folds: Iterable[str] = FOLDS,
    seeds: Iterable[int] = FINAL_SEEDS,
    force: bool = False,
) -> dict[str, int]:
    commands = build_cell_commands(
        python_executable=python_executable,
        output_dir=output_dir,
        models=models,
        folds=folds,
        seeds=seeds,
        force=force,
    )
    completed = skipped = 0
    total = len(commands)
    for index, command in enumerate(commands, start=1):
        model = command[command.index("--model") + 1]
        fold = command[command.index("--fold") + 1]
        seed = int(command[command.index("--seed") + 1])
        if cell_complete(output_dir, model, fold, seed) and not force:
            skipped += 1
            print(
                f"[{index}/{total}] skip complete {model}/{fold}/seed_{seed}",
                flush=True,
            )
            continue
        print(
            f"[{index}/{total}] run {model}/{fold}/seed_{seed}",
            flush=True,
        )
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        completed += 1
    return {"total": total, "completed": completed, "skipped": skipped}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen zero-API-cost integrated multimodal extension."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    cell = subparsers.add_parser("cell")
    cell.add_argument("--model", choices=tuple(TRACK_A_MODELS), required=True)
    cell.add_argument("--fold", choices=FOLDS, required=True)
    cell.add_argument("--seed", type=int, required=True)
    cell.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    cell.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--model", action="append", choices=tuple(TRACK_A_MODELS))
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
            model=args.model,
            fold=args.fold,
            seed=args.seed,
            output_dir=args.output_dir,
            force=args.force,
        )
    elif args.command == "run":
        result = run_cells_isolated(
            output_dir=args.output_dir,
            models=TRACK_A_MODELS if args.model is None else args.model,
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
