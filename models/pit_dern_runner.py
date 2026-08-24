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
    FoldData,
    binary_direction_metrics,
    discover_folds,
    package_versions,
    regression_metrics,
    sequence_history_features,
)
from models.convolutional_neural_network import make_sequences, make_test_sequences
from models.integrated_multimodal import (
    ARMS,
    prepare_integrated_fold,
    verify_freeze_manifest,
)
from models.integrated_multimodal_runner import (
    fold_metrics_from_seed_averaged_predictions,
)
from models.neural_network_folds import inverse_scaled_target
from models.pit_dern import (
    BATCH_SIZE,
    CONFIG,
    EPOCHS,
    MODEL_KEY,
    MODEL_LABEL,
    PROTOCOL_ID,
    SEQUENCE_WINDOW,
    blend_transferable_evidence,
    build_pit_dern_model,
    build_regime_conditioned_features,
    dual_evidence_retrieval,
    permute_memory_outcomes,
    probability_signed_scaled_target,
    standard_retrieval,
    validate_point_in_time_memory,
)
from models.pit_dern_extension import (
    EVIDENCE_STATUS,
    FINAL_ARM,
    build_compact_six_model_comparison,
    build_six_model_tables,
    evaluate_promotion_gates,
)
from models.point_in_time_data import LABEL_DATE_COLUMN
from models.track_a_analysis import exact_sign_flip_pvalue
from models.track_a_final import FINAL_SEEDS, TRACK_A_MODELS
from models.track_b_forward_news import DAILY_NEWS_FILE
from models.track_b_fusion import _scaled_fold
from models.track_c_inference import average_seed_predictions
from models.track_c_outer import REGIMES, selected_feature_lookup
from models.track_c_topk_validation_runner import SELECTED_FEATURES_FILE
from models.vmd_feature_pool import FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pit_dern_extension_v1"
CELL_ROOT_NAME = "cells"
FREEZE_FILE = PROJECT_ROOT / "test" / "pit_dern_freeze_v1.json"
INTEGRATED_FREEZE_MANIFEST = PROJECT_ROOT / "test" / "integrated_multimodal_freeze_v1.json"
LOCKED_WINDOWS_FILE = PROJECT_ROOT / "outputs" / "track_a_final_point_in_time_v2" / "locked_windows.csv"
REGIME_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_c" / "daily_regime_point_in_time_v2"
FROZEN_RESULTS_DIR = PROJECT_ROOT / "outputs" / "integrated_multimodal_posthoc_v1"
FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4")
ABLATIONS = (
    "encoder_only",
    "standard_retrieval",
    "dual_evidence",
    "pit_dern",
    "shuffled_retrieval_control",
)
PREDICTION_COLUMNS = {
    "encoder_only": "y_pred_encoder_only",
    "standard_retrieval": "y_pred_standard_retrieval",
    "dual_evidence": "y_pred_dual_evidence",
    "pit_dern": "y_pred",
    "shuffled_retrieval_control": "y_pred_shuffled_retrieval_control",
}


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
        raise ValueError("PIT-DERN freeze protocol_id is incorrect")
    if payload.get("result_access_at_freeze") is not False:
        raise ValueError("PIT-DERN freeze does not predate result access")
    expected = payload.get("input_sha256")
    if not isinstance(expected, dict):
        raise TypeError("PIT-DERN freeze is missing input hashes")
    paths: dict[str, Path] = {
        "integrated_freeze_manifest": INTEGRATED_FREEZE_MANIFEST,
        "locked_windows": LOCKED_WINDOWS_FILE,
        "daily_news_features": DAILY_NEWS_FILE,
        "regime_shap_selections": SELECTED_FEATURES_FILE,
    }
    for fold in FOLDS:
        paths[f"regime_{fold}_train"] = REGIME_OUTPUT_DIR / fold / "train_regimes.csv"
        paths[f"regime_{fold}_test"] = REGIME_OUTPUT_DIR / fold / "test_regimes.csv"
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Frozen PIT-DERN input is missing: {path}")
        if _sha256(path) != str(expected.get(key)):
            raise ValueError(f"Frozen PIT-DERN input hash changed: {path}")
    if set(expected) != set(paths):
        raise ValueError("PIT-DERN freeze contains an unexpected input set")
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
        directory / "ablation_metrics.csv",
        directory / "fit_registry.csv",
        directory / "retrieval_evidence.csv",
        directory / "run_metadata.json",
        directory / "integrity_audit.json",
        *(directory / f"predictions_{arm}.csv" for arm in ARMS),
    )


def _validate_cell_frames(
    metrics: pd.DataFrame,
    ablations: pd.DataFrame,
    registry: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
) -> dict[str, object]:
    if len(metrics) != len(ARMS) or set(metrics["arm"]) != set(ARMS):
        raise ValueError("PIT-DERN cell must contain four final metric rows")
    if len(ablations) != len(ARMS) * len(ABLATIONS):
        raise ValueError("PIT-DERN cell must contain twenty ablation rows")
    if set(ablations["ablation"]) != set(ABLATIONS):
        raise ValueError("PIT-DERN cell ablations do not match the freeze")
    if len(registry) != len(ARMS) or registry["fit_id"].duplicated().any():
        raise ValueError("PIT-DERN cell must contain four unique fits")
    if set(predictions) != set(ARMS):
        raise ValueError("PIT-DERN cell predictions are incomplete")
    prediction_rows = 0
    for arm, frame in predictions.items():
        required = {
            "Date", "routing_regime", "Close_D", "y_true", *PREDICTION_COLUMNS.values()
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{arm} predictions are missing columns: {missing}")
        numeric = frame.loc[:, ["Close_D", "y_true", *PREDICTION_COLUMNS.values()]]
        if frame.empty or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{arm} predictions are empty or non-finite")
        prediction_rows += len(frame)
    if int(registry["training_sequences"].min()) < 500:
        raise ValueError("PIT-DERN training capacity is unexpectedly small")
    return {
        "passed": True,
        "metric_rows": len(metrics),
        "ablation_metric_rows": len(ablations),
        "fit_rows": len(registry),
        "prediction_rows": prediction_rows,
    }


def cell_complete(output_dir: Path, fold: str, seed: int) -> bool:
    directory = cell_directory(output_dir, fold, seed)
    try:
        if not all(path.is_file() for path in _cell_paths(directory)):
            return False
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        audit = json.loads((directory / "integrity_audit.json").read_text(encoding="utf-8"))
        if metadata.get("protocol_id") != PROTOCOL_ID or audit.get("passed") is not True:
            return False
        _validate_cell_frames(
            pd.read_csv(directory / "metrics.csv"),
            pd.read_csv(directory / "ablation_metrics.csv"),
            pd.read_csv(directory / "fit_registry.csv"),
            {arm: pd.read_csv(directory / f"predictions_{arm}.csv") for arm in ARMS},
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
        raise ValueError(f"Unknown PIT-DERN folds: {unknown}")
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
    folds: Iterable[str] = FOLDS,
    seeds: Iterable[int] = FINAL_SEEDS,
    force: bool = False,
) -> list[list[str]]:
    commands: list[list[str]] = []
    for fold in _validate_folds(folds):
        for seed in _validate_seeds(seeds):
            command = [
                str(python_executable), "-m", "models.pit_dern_runner", "cell",
                "--fold", fold, "--seed", str(seed), "--output-dir", str(output_dir),
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
    return {regime: tuple(lookup[("shap", regime)]) for regime in REGIMES}


def _aligned_regime_context(
    market: pd.DataFrame,
    regime_frame: pd.DataFrame,
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    required = {
        DATE_COLUMN,
        "routing_regime",
        "prob_bull",
        "prob_sideway",
        "prob_bear",
    }
    missing = sorted(required.difference(regime_frame.columns))
    if missing:
        raise ValueError(f"{split} regime frame is missing columns: {missing}")
    left = pd.DataFrame({DATE_COLUMN: pd.to_datetime(market[DATE_COLUMN]).dt.normalize()})
    right = regime_frame.loc[:, list(required)].copy()
    right[DATE_COLUMN] = pd.to_datetime(right[DATE_COLUMN]).dt.normalize()
    if right[DATE_COLUMN].duplicated().any():
        raise ValueError(f"{split} regime dates contain duplicates")
    aligned = left.merge(right, on=DATE_COLUMN, how="left", validate="one_to_one")
    if aligned.isna().any().any():
        raise ValueError(f"{split} regime context does not cover every market row")
    labels = aligned["routing_regime"].astype(str).to_numpy(dtype=object)
    probabilities = aligned[
        ["prob_bull", "prob_sideway", "prob_bear"]
    ].to_numpy(dtype=float)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{split} regime probabilities do not sum to one")
    return labels, probabilities


def _ordered_regime_union(
    numeric_features: Sequence[str],
    selected_by_regime: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    selected = set().union(*(set(selected_by_regime[regime]) for regime in REGIMES))
    missing = sorted(selected.difference(numeric_features))
    if missing:
        raise ValueError(f"Frozen SHAP features are absent from the fold: {missing}")
    result = tuple(feature for feature in numeric_features if feature in selected)
    if not result:
        raise ValueError("Frozen regime feature union is empty")
    return result


def _conditioned_arrays(
    fold: FoldData,
    *,
    full_numeric_features: tuple[str, ...],
    numeric_features: tuple[str, ...],
    selected_by_regime: dict[str, tuple[str, ...]],
    train_regimes: np.ndarray,
    train_probabilities: np.ndarray,
    test_regimes: np.ndarray,
    test_probabilities: np.ndarray,
    include_news: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    news_features = tuple(
        feature
        for feature in fold.feature_columns
        if feature not in full_numeric_features
    )
    if not include_news and news_features:
        raise ValueError("Numeric regime arm unexpectedly contains news features")
    train_numeric, names = build_regime_conditioned_features(
        fold.train.loc[:, numeric_features].to_numpy(dtype=float),
        feature_names=numeric_features,
        regimes=train_regimes,
        regime_probabilities=train_probabilities,
        selected_by_regime=selected_by_regime,
    )
    test_numeric, test_names = build_regime_conditioned_features(
        fold.test.loc[:, numeric_features].to_numpy(dtype=float),
        feature_names=numeric_features,
        regimes=test_regimes,
        regime_probabilities=test_probabilities,
        selected_by_regime=selected_by_regime,
    )
    if names != test_names:
        raise ValueError("Train/test regime-conditioned features differ")
    history_parts = [train_numeric]
    if fold.context is not None:
        context_count = len(fold.context)
        context_numeric, context_names = build_regime_conditioned_features(
            fold.context.loc[:, numeric_features].to_numpy(dtype=float),
            feature_names=numeric_features,
            regimes=np.repeat(train_regimes[-1], context_count),
            regime_probabilities=np.repeat(
                train_probabilities[-1:, :], context_count, axis=0
            ),
            selected_by_regime=selected_by_regime,
        )
        if names != context_names:
            raise ValueError("Context regime-conditioned features differ")
        history_parts.append(context_numeric)
    train_values = train_numeric
    test_values = test_numeric
    history_values = np.vstack(history_parts)
    output_names = names
    if include_news:
        if not news_features:
            raise ValueError("News regime arm is missing its news block")
        train_news = fold.train.loc[:, news_features].to_numpy(dtype=np.float32)
        test_news = fold.test.loc[:, news_features].to_numpy(dtype=np.float32)
        history_news_parts = [train_news]
        if fold.context is not None:
            history_news_parts.append(
                fold.context.loc[:, news_features].to_numpy(dtype=np.float32)
            )
        train_values = np.column_stack([train_values, train_news])
        test_values = np.column_stack([test_values, test_news])
        history_values = np.column_stack(
            [history_values, np.vstack(history_news_parts)]
        )
        output_names = (*names, *news_features)
    return train_values, test_values, history_values, output_names


def _target_scaled_close(close: np.ndarray, scaler: Mapping[str, object]) -> np.ndarray:
    columns = list(scaler["columns"])
    index = columns.index(TARGET_COLUMN)
    scale = float(list(scaler["scale"])[index])
    offset = float(list(scaler["min"])[index])
    return np.asarray(close, dtype=float).reshape(-1) * scale + offset


def _set_tensorflow_runtime(seed: int) -> None:
    import tensorflow as tf

    tf.keras.backend.clear_session()
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    tf.keras.utils.set_random_seed(int(seed))


def _stable_arm_seed(base_seed: int, arm: str) -> int:
    material = f"{PROTOCOL_ID}|{base_seed}|{arm}|shuffled-control"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)


def _metrics_row(
    *,
    model: str,
    fold: str,
    test_year: int,
    arm: str,
    base_seed: int,
    prediction: np.ndarray,
    original_fold: FoldData,
) -> dict[str, object]:
    y_true = original_fold.test[TARGET_COLUMN].to_numpy(dtype=float)
    close = original_fold.test[CLOSE_COLUMN].to_numpy(dtype=float)
    return {
        "model": model,
        "fold": fold,
        "test_year": int(test_year),
        "arm": arm,
        "base_seed": int(base_seed),
        "n_test": len(prediction),
        **regression_metrics(y_true, prediction),
        **binary_direction_metrics(y_true, prediction, close),
    }


def _fit_id(fold: str, arm: str, seed: int, features: Sequence[str]) -> str:
    material = "|".join(
        [PROTOCOL_ID, MODEL_KEY, fold, arm, str(seed), *features]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def _flatten_dates(values: np.ndarray) -> list[str]:
    return [
        "|".join(pd.to_datetime(row).strftime("%Y-%m-%d").tolist())
        for row in values
    ]


def _predict_variants(
    *,
    model: object,
    x_train: np.ndarray,
    x_test: np.ndarray,
    memory_labels: np.ndarray,
    memory_deltas: np.ndarray,
    memory_dates: np.ndarray,
    current_scaled_close: np.ndarray,
    scaler: Mapping[str, object],
    shuffled_seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, float]]:
    started = time.perf_counter()
    train_output = model.predict(x_train, batch_size=BATCH_SIZE, verbose=0)
    test_output = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    inference_seconds = time.perf_counter() - started
    memory_embeddings = np.asarray(train_output["embedding"], dtype=float)
    query_embeddings = np.asarray(test_output["embedding"], dtype=float)
    encoder_probability = np.asarray(test_output["direction"], dtype=float).reshape(-1)
    encoder_delta = np.asarray(test_output["scaled_delta"], dtype=float).reshape(-1)

    retrieval_started = time.perf_counter()
    standard = standard_retrieval(
        query_embeddings, memory_embeddings, memory_labels, memory_deltas
    )
    dual = dual_evidence_retrieval(
        query_embeddings,
        memory_embeddings,
        memory_labels,
        memory_deltas,
        memory_dates=memory_dates,
    )
    blended = blend_transferable_evidence(
        encoder_probability=encoder_probability,
        encoder_delta=encoder_delta,
        retrieval_probability=dual.probability,
        retrieval_delta=dual.scaled_delta,
        best_similarity=dual.best_similarity,
    )
    shuffled_labels, shuffled_deltas = permute_memory_outcomes(
        memory_labels, memory_deltas, seed=shuffled_seed
    )
    shuffled = dual_evidence_retrieval(
        query_embeddings,
        memory_embeddings,
        shuffled_labels,
        shuffled_deltas,
        memory_dates=memory_dates,
    )
    shuffled_blended = blend_transferable_evidence(
        encoder_probability=encoder_probability,
        encoder_delta=encoder_delta,
        retrieval_probability=shuffled.probability,
        retrieval_delta=shuffled.scaled_delta,
        best_similarity=shuffled.best_similarity,
    )
    retrieval_seconds = time.perf_counter() - retrieval_started
    probability_delta = {
        "encoder_only": (encoder_probability, encoder_delta),
        "standard_retrieval": (standard.probability, standard.scaled_delta),
        "dual_evidence": (dual.probability, dual.scaled_delta),
        "pit_dern": (blended.probability, blended.scaled_delta),
        "shuffled_retrieval_control": (
            shuffled_blended.probability,
            shuffled_blended.scaled_delta,
        ),
    }
    predictions: dict[str, np.ndarray] = {}
    for ablation, (probability, delta) in probability_delta.items():
        scaled_target = probability_signed_scaled_target(
            current_scaled_close, probability, delta
        )
        predictions[ablation] = inverse_scaled_target(scaled_target, dict(scaler))
    evidence = pd.DataFrame(
        {
            "encoder_probability": encoder_probability,
            "dual_probability": dual.probability,
            "blended_probability": blended.probability,
            "retrieval_gate": blended.gate,
            "best_similarity": dual.best_similarity,
            "up_best_similarity": dual.up_similarity[:, 0],
            "down_best_similarity": dual.down_similarity[:, 0],
            "up_neighbor_dates": _flatten_dates(dual.up_dates),
            "down_neighbor_dates": _flatten_dates(dual.down_dates),
        }
    )
    return predictions, evidence, {
        "inference_seconds": float(inference_seconds),
        "retrieval_seconds": float(retrieval_seconds),
    }


def run_cell(
    *,
    fold: str,
    seed: int,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict[str, object]:
    fold_name = _validate_folds([fold])[0]
    base_seed = _validate_seeds([seed])[0]
    directory = cell_directory(output_dir, fold_name, base_seed)
    if cell_complete(output_dir, fold_name, base_seed) and not force:
        return {"status": "skipped", "fold": fold_name, "base_seed": base_seed}
    freeze_audit = verify_extension_freeze()
    integrated_audit = verify_freeze_manifest(
        PROJECT_ROOT, INTEGRATED_FREEZE_MANIFEST
    )
    specs = {
        spec.fold: spec
        for spec in discover_folds(FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR)
    }
    if fold_name not in specs:
        raise ValueError(f"Missing registered market fold: {fold_name}")
    spec = specs[fold_name]
    daily_news = pd.read_csv(DAILY_NEWS_FILE)
    original_by_arm = prepare_integrated_fold(spec, daily_news)
    scaled_by_arm = {
        arm: _scaled_fold(original_by_arm[arm]) for arm in ARMS
    }
    original_numeric = original_by_arm["Global-Numeric"]
    full_numeric_features = tuple(original_numeric.feature_columns)
    selected_by_regime = _selected_shap_features()
    numeric_union = _ordered_regime_union(
        full_numeric_features, selected_by_regime
    )
    train_regime_frame, test_regime_frame = _load_regimes(fold_name)
    train_regimes, train_probabilities = _aligned_regime_context(
        original_numeric.train, train_regime_frame, split=f"{fold_name}/train"
    )
    test_regimes, test_probabilities = _aligned_regime_context(
        original_numeric.test, test_regime_frame, split=f"{fold_name}/test"
    )

    metric_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    registry_rows: list[dict[str, object]] = []
    evidence_frames: list[pd.DataFrame] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    cell_started = time.perf_counter()

    for arm in ARMS:
        original_fold = original_by_arm[arm]
        scaled_fold, scaler = scaled_by_arm[arm]
        if arm.startswith("Global-"):
            features = tuple(scaled_fold.feature_columns)
            train_values = scaled_fold.train.loc[:, features].to_numpy(dtype=float)
            test_values = scaled_fold.test.loc[:, features].to_numpy(dtype=float)
            history_values = sequence_history_features(scaled_fold)
        else:
            train_values, test_values, history_values, features = _conditioned_arrays(
                scaled_fold,
                full_numeric_features=full_numeric_features,
                numeric_features=numeric_union,
                selected_by_regime=selected_by_regime,
                train_regimes=train_regimes,
                train_probabilities=train_probabilities,
                test_regimes=test_regimes,
                test_probabilities=test_probabilities,
                include_news=arm.endswith("News"),
            )
        scaled_target = scaled_fold.train[TARGET_COLUMN].to_numpy(dtype=float)
        x_train, _ = make_sequences(
            train_values, scaled_target, SEQUENCE_WINDOW
        )
        x_test = make_test_sequences(
            history_values, test_values, SEQUENCE_WINDOW
        )
        aligned_target = scaled_target[SEQUENCE_WINDOW - 1 :]
        train_close_target_scale = _target_scaled_close(
            original_fold.train[CLOSE_COLUMN].to_numpy(dtype=float), scaler
        )[SEQUENCE_WINDOW - 1 :]
        memory_deltas = aligned_target - train_close_target_scale
        memory_labels = (memory_deltas > 0.0).astype(np.int8)
        memory_dates = pd.to_datetime(
            original_fold.train[DATE_COLUMN].iloc[SEQUENCE_WINDOW - 1 :]
        ).to_numpy()
        memory_label_dates = pd.to_datetime(
            original_fold.train[LABEL_DATE_COLUMN].iloc[SEQUENCE_WINDOW - 1 :]
        ).to_numpy()
        query_dates = pd.to_datetime(original_fold.test[DATE_COLUMN]).to_numpy()
        validate_point_in_time_memory(memory_label_dates, query_dates)
        if set(memory_labels.tolist()) != {0, 1}:
            raise ValueError(f"{fold_name}/{arm} training labels lack one class")

        _set_tensorflow_runtime(base_seed)
        model = build_pit_dern_model((SEQUENCE_WINDOW, len(features)))
        fit_started = time.perf_counter()
        history = model.fit(
            x_train,
            {
                "direction": memory_labels.astype(np.float32),
                "scaled_delta": memory_deltas.astype(np.float32),
                "embedding": memory_labels.astype(np.float32),
            },
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,
            verbose=0,
        )
        fit_seconds = time.perf_counter() - fit_started
        test_close_target_scale = _target_scaled_close(
            original_fold.test[CLOSE_COLUMN].to_numpy(dtype=float), scaler
        )
        variants, evidence, runtime = _predict_variants(
            model=model,
            x_train=x_train,
            x_test=x_test,
            memory_labels=memory_labels,
            memory_deltas=memory_deltas,
            memory_dates=memory_dates,
            current_scaled_close=test_close_target_scale,
            scaler=scaler,
            shuffled_seed=_stable_arm_seed(base_seed, arm),
        )
        test_dates = pd.to_datetime(original_fold.test[DATE_COLUMN])
        prediction_frame = pd.DataFrame(
            {
                "Date": test_dates,
                "routing_regime": test_regimes,
                "Close_D": original_fold.test[CLOSE_COLUMN].to_numpy(dtype=float),
                "y_true": original_fold.test[TARGET_COLUMN].to_numpy(dtype=float),
                **{
                    PREDICTION_COLUMNS[name]: values
                    for name, values in variants.items()
                },
                "encoder_probability": evidence["encoder_probability"].to_numpy(),
                "dual_probability": evidence["dual_probability"].to_numpy(),
                "blended_probability": evidence["blended_probability"].to_numpy(),
                "retrieval_gate": evidence["retrieval_gate"].to_numpy(),
                "best_similarity": evidence["best_similarity"].to_numpy(),
            }
        )
        prediction_frames[arm] = prediction_frame
        metric_rows.append(
            _metrics_row(
                model=MODEL_KEY,
                fold=fold_name,
                test_year=spec.test_year,
                arm=arm,
                base_seed=base_seed,
                prediction=variants["pit_dern"],
                original_fold=original_fold,
            )
        )
        for ablation in ABLATIONS:
            ablation_rows.append(
                {
                    **_metrics_row(
                        model=MODEL_KEY,
                        fold=fold_name,
                        test_year=spec.test_year,
                        arm=arm,
                        base_seed=base_seed,
                        prediction=variants[ablation],
                        original_fold=original_fold,
                    ),
                    "ablation": ablation,
                }
            )
        registry_rows.append(
            {
                "fit_id": _fit_id(fold_name, arm, base_seed, features),
                "protocol_id": PROTOCOL_ID,
                "model": MODEL_KEY,
                "fold": fold_name,
                "test_year": spec.test_year,
                "arm": arm,
                "base_seed": base_seed,
                "window": SEQUENCE_WINDOW,
                "feature_count": len(features),
                "feature_hash": hashlib.sha256(
                    "|".join(features).encode("utf-8")
                ).hexdigest(),
                "training_sequences": len(x_train),
                "memory_up": int(memory_labels.sum()),
                "memory_down": int((memory_labels == 0).sum()),
                "fit_seconds": float(fit_seconds),
                "inference_seconds": runtime["inference_seconds"],
                "retrieval_seconds": runtime["retrieval_seconds"],
                "trainable_parameters": int(model.count_params()),
                "final_training_loss": float(history.history["loss"][-1]),
            }
        )
        evidence.insert(0, "Date", test_dates.to_numpy())
        evidence.insert(0, "base_seed", base_seed)
        evidence.insert(0, "fold", fold_name)
        evidence.insert(0, "arm", arm)
        evidence_frames.append(evidence)

    metrics = pd.DataFrame(metric_rows)
    ablation_metrics = pd.DataFrame(ablation_rows)
    registry = pd.DataFrame(registry_rows)
    evidence_all = pd.concat(evidence_frames, ignore_index=True)
    audit = _validate_cell_frames(
        metrics, ablation_metrics, registry, prediction_frames
    )
    audit.update(
        {
            "protocol_id": PROTOCOL_ID,
            "extension_freeze_passed": bool(freeze_audit["passed"]),
            "integrated_freeze_passed": bool(integrated_audit["passed"]),
            "memory_label_date_max": str(
                max(pd.to_datetime(frame[LABEL_DATE_COLUMN]).max() for frame in [original_numeric.train]).date()
            ),
            "query_date_min": str(pd.to_datetime(original_numeric.test[DATE_COLUMN]).min().date()),
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(directory / "metrics.csv", index=False)
    ablation_metrics.to_csv(directory / "ablation_metrics.csv", index=False)
    registry.to_csv(directory / "fit_registry.csv", index=False)
    evidence_all.to_csv(directory / "retrieval_evidence.csv", index=False)
    for arm, frame in prediction_frames.items():
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
        "ablations": list(ABLATIONS),
        "unique_fits": len(registry),
        "cell_wall_seconds": float(time.perf_counter() - cell_started),
        "runtime_scope": "build, fit, embedding inference, and retrieval",
        "incremental_api_cost_usd": 0,
        "config": CONFIG,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "scipy", "tensorflow"]
        ),
    }
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (directory / "integrity_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    if not cell_complete(output_dir, fold_name, base_seed):
        raise RuntimeError("PIT-DERN cell failed its post-write integrity audit")
    return {**metadata, "status": "completed"}


def _expected_cell_keys() -> set[tuple[str, int]]:
    return {(fold, int(seed)) for fold in FOLDS for seed in FINAL_SEEDS}


def _collect_csv(output_dir: Path, name: str) -> pd.DataFrame:
    paths = sorted(
        (output_dir / CELL_ROOT_NAME / MODEL_KEY).glob(f"*/seed_*/{name}")
    )
    if not paths:
        raise FileNotFoundError(f"No PIT-DERN {name} files were found")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _average_prediction_variant(
    frames_by_seed: Mapping[int, pd.DataFrame],
    *,
    prediction_column: str,
) -> pd.DataFrame:
    prepared = {
        seed: frame.assign(y_pred=frame[prediction_column].to_numpy(dtype=float))
        for seed, frame in frames_by_seed.items()
    }
    return average_seed_predictions(prepared)


def _collect_seed_averaged_predictions(
    output_dir: Path,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_rows: list[pd.DataFrame] = []
    ablation_rows: list[pd.DataFrame] = []
    for (fold, arm), group in metrics.groupby(["fold", "arm"], sort=False):
        seeds = sorted(group["base_seed"].astype(int).unique())
        frames = {
            seed: pd.read_csv(
                cell_directory(output_dir, str(fold), seed)
                / f"predictions_{arm}.csv"
            )
            for seed in seeds
        }
        for ablation, column in PREDICTION_COLUMNS.items():
            averaged = _average_prediction_variant(
                frames, prediction_column=column
            )
            averaged.insert(0, "model", MODEL_KEY)
            averaged.insert(1, "fold", fold)
            averaged.insert(2, "test_year", int(group["test_year"].iloc[0]))
            averaged.insert(3, "arm", arm)
            averaged.insert(4, "ablation", ablation)
            ablation_rows.append(averaged)
            if ablation == "pit_dern":
                final_rows.append(averaged.drop(columns="ablation"))
    return (
        pd.concat(final_rows, ignore_index=True),
        pd.concat(ablation_rows, ignore_index=True),
    )


def _summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
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


def _ablation_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for ablation, group in predictions.groupby("ablation", sort=False):
        metrics = fold_metrics_from_seed_averaged_predictions(
            group.drop(columns="ablation")
        )
        metrics.insert(4, "ablation", ablation)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def _build_primary_contrast(
    frozen_fold_metrics: pd.DataFrame,
    ours_fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    comparator = frozen_fold_metrics.loc[
        frozen_fold_metrics["model"].eq("lstm_cnn_attention")
        & frozen_fold_metrics["arm"].eq(FINAL_ARM),
        ["fold", "balanced_accuracy", "direction_accuracy", "mcc"],
    ]
    ours = ours_fold_metrics.loc[
        ours_fold_metrics["model"].eq(MODEL_KEY)
        & ours_fold_metrics["arm"].eq(FINAL_ARM),
        ["fold", "balanced_accuracy", "direction_accuracy", "mcc"],
    ]
    paired = comparator.merge(
        ours,
        on="fold",
        how="inner",
        validate="one_to_one",
        suffixes=("_lstm_cnn_attention", "_pit_dern"),
    ).sort_values("fold")
    if len(paired) != 4:
        raise ValueError("Primary contrast requires four temporal folds")
    for metric in ("balanced_accuracy", "direction_accuracy"):
        paired[f"{metric}_delta_pp"] = (
            paired[f"{metric}_pit_dern"]
            - paired[f"{metric}_lstm_cnn_attention"]
        ) * 100.0
    paired["mcc_delta"] = paired["mcc_pit_dern"] - paired["mcc_lstm_cnn_attention"]
    return paired.reset_index(drop=True)


def aggregate_experiment(*, output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    freeze_audit = verify_extension_freeze()
    incomplete = [
        key
        for key in sorted(_expected_cell_keys())
        if not cell_complete(output_dir, *key)
    ]
    if incomplete:
        raise ValueError(f"PIT-DERN run has incomplete cells: {incomplete[:5]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _collect_csv(output_dir, "metrics.csv")
    ablation_metrics_seed = _collect_csv(output_dir, "ablation_metrics.csv")
    registry = _collect_csv(output_dir, "fit_registry.csv")
    evidence = _collect_csv(output_dir, "retrieval_evidence.csv")
    expected_cells = len(_expected_cell_keys())
    if len(metrics) != expected_cells * len(ARMS):
        raise ValueError("PIT-DERN metric row count is incorrect")
    if len(ablation_metrics_seed) != expected_cells * len(ARMS) * len(ABLATIONS):
        raise ValueError("PIT-DERN ablation metric row count is incorrect")
    if len(registry) != expected_cells * len(ARMS):
        raise ValueError("PIT-DERN fit registry row count is incorrect")
    if registry["fit_id"].duplicated().any():
        raise ValueError("PIT-DERN fit registry contains duplicate fit ids")
    metrics.to_csv(output_dir / "metrics_by_seed_fold.csv", index=False)
    ablation_metrics_seed.to_csv(
        output_dir / "ablation_metrics_by_seed_fold.csv", index=False
    )
    registry.to_csv(output_dir / "fit_registry.csv", index=False)
    evidence.to_csv(output_dir / "retrieval_evidence.csv", index=False)

    predictions, ablation_predictions = _collect_seed_averaged_predictions(
        output_dir, metrics
    )
    predictions.to_csv(output_dir / "predictions_seed_averaged.csv", index=False)
    ablation_predictions.to_csv(
        output_dir / "ablation_predictions_seed_averaged.csv", index=False
    )
    fold_metrics = fold_metrics_from_seed_averaged_predictions(predictions)
    fold_metrics.to_csv(output_dir / "fold_metrics_seed_averaged.csv", index=False)
    arm_summary = _summarize_fold_metrics(fold_metrics)
    arm_summary.to_csv(output_dir / "arm_summary.csv", index=False)
    ablation_fold = _ablation_fold_metrics(ablation_predictions)
    ablation_fold.to_csv(
        output_dir / "ablation_fold_metrics_seed_averaged.csv", index=False
    )
    ablation_summary = (
        ablation_fold.groupby(["model", "arm", "ablation"], sort=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            direction_accuracy_mean=("direction_accuracy", "mean"),
            mcc_mean=("mcc", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            temporal_folds=("fold", "nunique"),
        )
        .reset_index()
    )
    ablation_summary.to_csv(output_dir / "ablation_summary.csv", index=False)

    frozen_summary = pd.read_csv(FROZEN_RESULTS_DIR / "arm_summary.csv")
    six_all, six_final = build_six_model_tables(frozen_summary, arm_summary)
    six_all.to_csv(output_dir / "six_model_all_arms_comparison.csv", index=False)
    six_final.to_csv(output_dir / "six_model_final_arm_comparison.csv", index=False)
    compact = build_compact_six_model_comparison(six_all)
    compact.to_csv(output_dir / "six_model_compact_comparison.csv", index=False)

    frozen_fold = pd.read_csv(FROZEN_RESULTS_DIR / "fold_metrics_seed_averaged.csv")
    contrast = _build_primary_contrast(frozen_fold, fold_metrics)
    contrast.to_csv(
        output_dir / "pit_dern_vs_lstm_cnn_attention_fold_contrast.csv", index=False
    )
    runtime = (
        registry.groupby(["model", "arm"], sort=False)
        .agg(
            executed_fits=("fit_id", "size"),
            fit_seconds_total=("fit_seconds", "sum"),
            fit_seconds_mean=("fit_seconds", "mean"),
            inference_seconds_total=("inference_seconds", "sum"),
            retrieval_seconds_total=("retrieval_seconds", "sum"),
            trainable_parameters_mean=("trainable_parameters", "mean"),
            training_sequences_min=("training_sequences", "min"),
            training_sequences_max=("training_sequences", "max"),
        )
        .reset_index()
    )
    runtime.to_csv(output_dir / "runtime_summary.csv", index=False)
    frozen_runtime = pd.read_csv(FROZEN_RESULTS_DIR / "runtime_summary.csv")
    six_runtime = pd.concat(
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
    order = {model: index for index, model in enumerate([*TRACK_A_MODELS, MODEL_KEY])}
    six_runtime["_order"] = six_runtime["model"].map(order)
    six_runtime = six_runtime.sort_values("_order").drop(columns="_order")
    six_runtime.to_csv(output_dir / "six_model_runtime_comparison.csv", index=False)

    final_ablation = ablation_summary.loc[ablation_summary["arm"].eq(FINAL_ARM)]
    bacc_lookup = final_ablation.set_index("ablation")["balanced_accuracy_mean"]
    ours_parameter = float(
        runtime.loc[runtime["arm"].eq(FINAL_ARM), "trainable_parameters_mean"].iloc[0]
    )
    frozen_parameter = float(
        frozen_runtime.loc[
            frozen_runtime["model"].eq("lstm_cnn_attention")
            & frozen_runtime["arm"].eq(FINAL_ARM),
            "trainable_parameters_mean",
        ].iloc[0]
    )
    parameter_delta = (ours_parameter - frozen_parameter) / frozen_parameter
    complete_finite = bool(
        np.isfinite(
            ablation_predictions[["Close_D", "y_true", "y_pred"]].to_numpy(dtype=float)
        ).all()
    )
    promotion = evaluate_promotion_gates(
        contrast[["fold", "balanced_accuracy_delta_pp"]],
        ours_bacc=float(bacc_lookup["pit_dern"]),
        encoder_bacc=float(bacc_lookup["encoder_only"]),
        shuffled_control_bacc=float(bacc_lookup["shuffled_retrieval_control"]),
        parameter_delta_fraction=float(parameter_delta),
        complete_finite_predictions=complete_finite,
    )
    promotion["exact_sign_flip_pvalue"] = exact_sign_flip_pvalue(
        contrast["balanced_accuracy_delta_pp"].to_numpy(dtype=float)
    )
    promotion["primary_comparator"] = "lstm_cnn_attention"
    (output_dir / "promotion_decision.json").write_text(
        json.dumps(promotion, indent=2), encoding="utf-8"
    )

    retrieval_diagnostics = (
        evidence.groupby(["fold", "arm"], sort=False)
        .agg(
            observations=("best_similarity", "size"),
            best_similarity_mean=("best_similarity", "mean"),
            best_similarity_min=("best_similarity", "min"),
            retrieval_gate_mean=("retrieval_gate", "mean"),
            retrieval_gate_max=("retrieval_gate", "max"),
        )
        .reset_index()
    )
    retrieval_diagnostics.to_csv(
        output_dir / "retrieval_diagnostics.csv", index=False
    )
    cell_runtime = []
    for fold, seed in sorted(_expected_cell_keys()):
        payload = json.loads(
            (cell_directory(output_dir, fold, seed) / "run_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        cell_runtime.append(
            {
                "model": MODEL_KEY,
                "fold": fold,
                "base_seed": seed,
                "cell_wall_seconds": payload["cell_wall_seconds"],
            }
        )
    pd.DataFrame(cell_runtime).to_csv(output_dir / "runtime_by_cell.csv", index=False)

    integrity = {
        "passed": True,
        "protocol_id": PROTOCOL_ID,
        "freeze_audit": freeze_audit,
        "completed_cells": expected_cells,
        "metric_rows": len(metrics),
        "ablation_metric_rows": len(ablation_metrics_seed),
        "fit_rows": len(registry),
        "seed_averaged_fold_arm_rows": len(fold_metrics),
        "all_predictions_finite": complete_finite,
        "minimum_training_sequences": int(registry["training_sequences"].min()),
        "promotion_passed": bool(promotion["passed"]),
    }
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
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
        "ablations": list(ABLATIONS),
        "window": SEQUENCE_WINDOW,
        "primary_metric": "balanced_accuracy",
        "primary_arm": FINAL_ARM,
        "primary_comparator": "lstm_cnn_attention",
        "incremental_api_cost_usd": 0,
        "source_hashes": {
            "pit_dern.py": _sha256(PROJECT_ROOT / "models" / "pit_dern.py"),
            "pit_dern_extension.py": _sha256(PROJECT_ROOT / "models" / "pit_dern_extension.py"),
            "pit_dern_runner.py": _sha256(Path(__file__)),
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "scipy", "tensorflow"]
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
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
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)  # nosec B603
        completed += 1
    return {"total": total, "completed": completed, "skipped": skipped}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen PIT-DERN exploratory extension."
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
