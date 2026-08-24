from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT_EARLY = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "mpl"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

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
from models.fcta_lstm import (
    PROTOCOL_ID,
    REGISTERED_VARIANTS,
    FCTAConfig,
    build_fcta_model,
    counterfactual_distribution,
    verify_freeze_manifest,
)
from models.integrated_multimodal import prepare_integrated_fold
from models.neural_network_folds import inverse_scaled_target
from models.pit_cdr_lstm import direction_metrics
from models.pit_cdr_lstm_runner import prepare_candidate_fold
from models.track_a_final import FINAL_SEEDS
from models.track_b_forward_news import DAILY_NEWS_FILE
from models.track_b_fusion import _scaled_fold
from models.vmd_feature_pool import FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fcta_lstm_2024_2025_v1"
FREEZE_FILE = PROJECT_ROOT / "test" / "fcta_lstm_freeze_v1.json"
FROZEN_PREDICTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "final_five_model_prediction_visuals_v1"
    / "final_arm_prediction_series.csv"
)
TEST_YEARS = (2024, 2025)
SEEDS = tuple(int(seed) for seed in FINAL_SEEDS)
EXPECTED_TEST_ROWS = {2024: 244, 2025: 234}
WINDOW = 5
EPOCHS = 20
BATCH_SIZE = 32
FROZEN_MODEL_KEYS = (
    "lstm",
    "cnn",
    "lstm_cnn",
    "lstm_attention",
    "lstm_cnn_attention",
)
MODEL_LABELS = {
    "lstm": "LSTM",
    "cnn": "CNN",
    "lstm_cnn": "LSTM-CNN",
    "lstm_attention": "LSTM-Attention",
    "lstm_cnn_attention": "LSTM-CNN-Attention",
}


@dataclass(frozen=True)
class FCTAPreparedFold:
    fold: str
    test_year: int
    feature_columns: tuple[str, ...]
    train_sequence: np.ndarray
    train_target_scaled: np.ndarray
    train_current_scaled: np.ndarray
    train_dates: np.ndarray
    test_sequence: np.ndarray
    test_target_scaled: np.ndarray
    test_current_scaled: np.ndarray
    test_regimes: np.ndarray
    test_dates: np.ndarray
    test_close: np.ndarray
    test_next_close: np.ndarray
    scaler_metadata: Mapping[str, object]
    mask_counts: Mapping[str, int]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_tensorflow(seed: int):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    return tf


def _fold_spec(test_year: int):
    specs = {
        spec.test_year: spec
        for spec in discover_folds(FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR)
    }
    if test_year not in TEST_YEARS or test_year not in specs:
        raise ValueError(f"No registered FCTA-LSTM fold for {test_year}")
    return specs[test_year]


def prepare_fcta_fold(test_year: int) -> FCTAPreparedFold:
    base = prepare_candidate_fold(test_year)
    spec = _fold_spec(test_year)
    daily_news = pd.read_csv(DAILY_NEWS_FILE)
    original = prepare_integrated_fold(spec, daily_news)["Global-Numeric-News"]
    scaled, metadata = _scaled_fold(original)
    positions = np.asarray(base.train_endpoint_positions, dtype=np.int64)
    train_target = scaled.train[TARGET_COLUMN].to_numpy(dtype=np.float64)[positions]
    train_current = scaled.train[CLOSE_COLUMN].to_numpy(dtype=np.float64)[positions]
    train_dates = original.train[DATE_COLUMN].to_numpy()[positions]
    test_target = scaled.test[TARGET_COLUMN].to_numpy(dtype=np.float64)
    test_current = scaled.test[CLOSE_COLUMN].to_numpy(dtype=np.float64)

    if len(train_target) != len(base.train_sequence):
        raise ValueError("FCTA train target and sequence counts differ")
    if not np.array_equal(train_dates, base.train_dates):
        raise ValueError("FCTA train target dates do not align with the sequence endpoints")
    if len(test_target) != EXPECTED_TEST_ROWS[test_year]:
        raise ValueError("FCTA test-row count changed")
    restored_target = inverse_scaled_target(test_target, metadata)
    if not np.allclose(restored_target, base.test_next_close, rtol=0.0, atol=1e-8):
        raise ValueError("FCTA inverse-scaled test target does not match the frozen cohort")
    if not np.isfinite(
        np.concatenate([train_target, train_current, test_target, test_current])
    ).all():
        raise ValueError("FCTA scaled targets contain non-finite values")

    return FCTAPreparedFold(
        fold=base.fold,
        test_year=base.test_year,
        feature_columns=base.feature_columns,
        train_sequence=base.train_sequence,
        train_target_scaled=train_target.astype(np.float32),
        train_current_scaled=train_current.astype(np.float32),
        train_dates=base.train_dates,
        test_sequence=base.test_sequence,
        test_target_scaled=test_target.astype(np.float32),
        test_current_scaled=test_current.astype(np.float32),
        test_regimes=base.test_regimes,
        test_dates=base.test_dates,
        test_close=base.test_close,
        test_next_close=base.test_next_close,
        scaler_metadata=metadata,
        mask_counts=base.mask_counts,
    )


def probability_from_scaled_prediction(
    prediction: np.ndarray,
    current: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    logits = (
        np.asarray(prediction, dtype=np.float64)
        - np.asarray(current, dtype=np.float64)
    ) / float(temperature)
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("Temporal distributions must be finite non-negative matrices")
    total = array.sum(axis=1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("Temporal distributions must have positive row sums")
    return array / total


def attention_fidelity_metrics(
    attention: np.ndarray,
    counterfactual_importance: np.ndarray,
) -> dict[str, float]:
    left = _normalize_rows(attention)
    right = _normalize_rows(counterfactual_importance)
    if left.shape != right.shape:
        raise ValueError("Attention and counterfactual importance shapes differ")
    midpoint = np.clip(0.5 * (left + right), 1e-12, 1.0)
    left_safe = np.clip(left, 1e-12, 1.0)
    right_safe = np.clip(right, 1e-12, 1.0)
    jsd = 0.5 * np.sum(left_safe * np.log(left_safe / midpoint), axis=1)
    jsd += 0.5 * np.sum(right_safe * np.log(right_safe / midpoint), axis=1)
    top_attention = np.argmax(left, axis=1)
    top_importance = np.argmax(right, axis=1)
    selected = right[np.arange(len(right)), top_attention]
    uniform_share = 1.0 / right.shape[1]
    correlations: list[float] = []
    for attention_row, importance_row in zip(left, right, strict=True):
        if np.std(attention_row) <= 1e-12 or np.std(importance_row) <= 1e-12:
            continue
        correlations.append(float(np.corrcoef(attention_row, importance_row)[0, 1]))
    return {
        "attention_counterfactual_jsd": float(np.mean(jsd)),
        "top1_deletion_agreement": float(np.mean(top_attention == top_importance)),
        "top_attention_influence_lift": float(np.mean(selected) / uniform_share),
        "attention_influence_correlation": (
            float(np.mean(correlations)) if correlations else 0.0
        ),
    }


def _direction_temperature(prepared: FCTAPreparedFold) -> float:
    absolute_move = np.abs(
        prepared.train_target_scaled - prepared.train_current_scaled
    )
    nonzero = absolute_move[absolute_move > 0.0]
    if len(nonzero) == 0:
        raise ValueError("FCTA train fold has no non-zero price moves")
    return float(max(np.median(nonzero), 1e-4))


def fit_variant(
    prepared: FCTAPreparedFold,
    *,
    variant: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Any]:
    if variant not in REGISTERED_VARIANTS or seed not in SEEDS:
        raise ValueError("Unknown FCTA-LSTM variant or seed")
    tf = _configure_tensorflow(seed)
    temperature = _direction_temperature(prepared)
    config = FCTAConfig(
        window=WINDOW,
        feature_count=len(prepared.feature_columns),
        direction_temperature=temperature,
        variant=variant,
    )
    model = build_fcta_model(config)
    model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate))
    packed_target = np.column_stack(
        [prepared.train_target_scaled, prepared.train_current_scaled]
    ).astype(np.float32)
    fit_started = time.perf_counter()
    history = model.fit(
        prepared.train_sequence,
        packed_target,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    fit_seconds = float(time.perf_counter() - fit_started)
    inference_started = time.perf_counter()
    full = model(
        prepared.test_sequence,
        training=False,
        return_attention=True,
    )
    scaled_prediction = np.asarray(full["prediction"], dtype=np.float64).reshape(-1)
    attention = np.asarray(full["attention"], dtype=np.float64)
    deleted = np.asarray(
        model.counterfactual_outputs(prepared.test_sequence, training=False),
        dtype=np.float64,
    )
    importance = np.asarray(
        counterfactual_distribution(
            tf.convert_to_tensor(scaled_prediction[:, None], dtype=tf.float32),
            tf.convert_to_tensor(deleted, dtype=tf.float32),
        ),
        dtype=np.float64,
    )
    inference_seconds = float(time.perf_counter() - inference_started)
    if scaled_prediction.shape != (len(prepared.test_sequence),):
        raise ValueError("FCTA-LSTM inference shape is invalid")
    if attention.shape != (len(prepared.test_sequence), WINDOW):
        raise ValueError("FCTA-LSTM attention shape is invalid")
    if not np.isfinite(scaled_prediction).all():
        raise ValueError("FCTA-LSTM inference contains non-finite predictions")
    diagnostics: dict[str, Any] = {
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "inference_parameters": int(model.count_params()),
        "direction_temperature": temperature,
        "final_training_loss": float(history.history["loss"][-1]),
        **attention_fidelity_metrics(attention, importance),
    }
    return scaled_prediction, attention, importance, diagnostics, model


def _cell_directory(output_dir: Path, year: int, seed: int, variant: str) -> Path:
    return output_dir / "cells" / str(year) / f"seed_{seed}" / variant


def _cell_complete(directory: Path) -> bool:
    required = (
        directory / "predictions.csv",
        directory / "temporal_explanations.csv",
        directory / "metrics.json",
        directory / "run_metadata.json",
        directory / "inference.weights.h5",
    )
    return all(path.is_file() for path in required)


def run_cell(
    prepared: FCTAPreparedFold,
    *,
    seed: int,
    variant: str,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    directory = _cell_directory(output_dir, prepared.test_year, seed, variant)
    if _cell_complete(directory) and not force:
        return {
            "status": "skipped_complete",
            "year": prepared.test_year,
            "seed": seed,
            "variant": variant,
        }
    started = time.perf_counter()
    scaled_prediction, attention, importance, diagnostics, model = fit_variant(
        prepared,
        variant=variant,
        seed=seed,
    )
    prediction = inverse_scaled_target(scaled_prediction, dict(prepared.scaler_metadata))
    probability = probability_from_scaled_prediction(
        scaled_prediction,
        prepared.test_current_scaled,
        temperature=float(diagnostics["direction_temperature"]),
    )
    direction = direction_metrics(
        current_close=prepared.test_close,
        next_close=prepared.test_next_close,
        probability=probability,
    )
    level = regression_metrics(prepared.test_next_close, prediction)
    labels = (prepared.test_next_close > prepared.test_close).astype(np.int8)
    metrics = {
        **direction,
        **level,
        "brier_score": float(brier_score_loss(labels, probability)),
        "binary_crossentropy": float(
            log_loss(labels, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1])
        ),
        **diagnostics,
    }
    predictions = pd.DataFrame(
        {
            "protocol_id": PROTOCOL_ID,
            "variant": variant,
            "seed": seed,
            "fold": prepared.fold,
            "test_year": prepared.test_year,
            DATE_COLUMN: prepared.test_dates,
            "routing_regime": prepared.test_regimes,
            CLOSE_COLUMN: prepared.test_close,
            "y_true": prepared.test_next_close,
            "y_pred": prediction,
            "scaled_prediction": scaled_prediction,
            "probability": probability,
            "pred_direction": (probability > 0.5).astype(np.int8),
        }
    )
    explanation_rows: list[dict[str, Any]] = []
    for row, date in enumerate(prepared.test_dates):
        for day in range(WINDOW):
            explanation_rows.append(
                {
                    "protocol_id": PROTOCOL_ID,
                    "variant": variant,
                    "seed": seed,
                    "test_year": prepared.test_year,
                    DATE_COLUMN: date,
                    "lag_index": day - WINDOW + 1,
                    "attention_weight": attention[row, day],
                    "counterfactual_importance": importance[row, day],
                }
            )
    directory.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(explanation_rows).to_csv(
        directory / "temporal_explanations.csv", index=False
    )
    (directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    model.save_weights(directory / "inference.weights.h5")
    config = FCTAConfig(
        window=WINDOW,
        feature_count=len(prepared.feature_columns),
        direction_temperature=float(diagnostics["direction_temperature"]),
        variant=variant,
    )
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "evidence_status": "frozen_retrospective_architecture_screen",
        "created_at": _utc_now(),
        "variant": variant,
        "seed": seed,
        "fold": prepared.fold,
        "test_year": prepared.test_year,
        "train_period": [
            pd.Timestamp(prepared.train_dates.min()).date().isoformat(),
            pd.Timestamp(prepared.train_dates.max()).date().isoformat(),
        ],
        "test_period": [
            pd.Timestamp(prepared.test_dates.min()).date().isoformat(),
            pd.Timestamp(prepared.test_dates.max()).date().isoformat(),
        ],
        "train_rows": len(prepared.train_sequence),
        "test_rows": len(prepared.test_sequence),
        "feature_count": len(prepared.feature_columns),
        "mask_counts": dict(prepared.mask_counts),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "config": asdict(config),
        "inference_parameters": int(model.count_params()),
        "fit_seconds": diagnostics["fit_seconds"],
        "inference_seconds": diagnostics["inference_seconds"],
        "wall_seconds": float(time.perf_counter() - started),
        "python": platform.python_version(),
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "tensorflow"]
        ),
        "incremental_api_cost_usd": 0,
    }
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    del model
    gc.collect()
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except RuntimeError:
        pass
    return {
        "status": "completed",
        "year": prepared.test_year,
        "seed": seed,
        "variant": variant,
    }


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized[DATE_COLUMN] = pd.to_datetime(
        normalized[DATE_COLUMN], errors="raise"
    ).dt.normalize()
    normalized["test_year"] = normalized["test_year"].astype(int)
    return normalized


def validate_candidate_cohort(
    candidate: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    candidate_required = {
        "variant",
        "seed",
        "test_year",
        DATE_COLUMN,
        CLOSE_COLUMN,
        "y_true",
        "y_pred",
        "probability",
    }
    frozen_required = {
        "model",
        "test_year",
        DATE_COLUMN,
        CLOSE_COLUMN,
        "y_true",
        "y_pred",
    }
    if not candidate_required.issubset(candidate.columns):
        raise ValueError("FCTA candidate predictions are missing cohort columns")
    if not frozen_required.issubset(frozen.columns):
        raise ValueError("Frozen predictions are missing cohort columns")
    candidate_values = _normalize_keys(candidate)
    frozen_values = _normalize_keys(frozen)
    candidate_values = candidate_values.loc[
        candidate_values["test_year"].isin(TEST_YEARS)
    ]
    frozen_values = frozen_values.loc[frozen_values["test_year"].isin(TEST_YEARS)]
    if set(candidate_values["variant"]) != set(REGISTERED_VARIANTS):
        raise ValueError("FCTA candidate does not contain all registered variants")
    if set(candidate_values["seed"].astype(int)) != {int(seed) for seed in seeds}:
        raise ValueError("FCTA candidate does not contain all registered seeds")
    if set(frozen_values["model"]) != set(FROZEN_MODEL_KEYS):
        raise ValueError("Frozen predictions do not contain the five registered models")
    reference = frozen_values.loc[frozen_values["model"].eq("lstm")].sort_values(
        ["test_year", DATE_COLUMN]
    )
    keys = reference[["test_year", DATE_COLUMN]].reset_index(drop=True)
    for (variant, seed), group in candidate_values.groupby(
        ["variant", "seed"], sort=False
    ):
        ordered = group.sort_values(["test_year", DATE_COLUMN]).reset_index(drop=True)
        if not ordered[["test_year", DATE_COLUMN]].equals(keys):
            raise ValueError(f"FCTA dates do not match frozen cohort: {variant}/{seed}")
        if not np.allclose(ordered[CLOSE_COLUMN], reference[CLOSE_COLUMN], atol=1e-12):
            raise ValueError("FCTA current-close values do not match frozen cohort")
        if not np.allclose(ordered["y_true"], reference["y_true"], atol=1e-12):
            raise ValueError("FCTA actual values do not match frozen cohort")
    return {
        "passed": True,
        "rows_per_variant_seed": len(reference),
        "candidate_cells": int(candidate_values.groupby(["variant", "seed"]).ngroups),
        "frozen_models": len(FROZEN_MODEL_KEYS),
    }


def _candidate_seed_average(
    candidate: pd.DataFrame,
    *,
    seeds: Sequence[int],
) -> pd.DataFrame:
    values = _normalize_keys(candidate)
    keys = ["variant", "test_year", DATE_COLUMN]
    for column in (CLOSE_COLUMN, "y_true"):
        if values.groupby(keys)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"FCTA seeds disagree on {column}")
    averaged = values.groupby(keys, as_index=False, sort=False).agg(
        Close_D=(CLOSE_COLUMN, "first"),
        y_true=("y_true", "first"),
        y_pred=("y_pred", "mean"),
        probability=("probability", "mean"),
        seeds_averaged=("seed", "nunique"),
    )
    if not averaged["seeds_averaged"].eq(len(seeds)).all():
        raise ValueError("FCTA seed averaging is incomplete")
    return averaged


def build_ablation_fold_metrics(
    candidate: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    averaged = _candidate_seed_average(candidate, seeds=seeds)
    rows: list[dict[str, Any]] = []
    for (variant, year), group in averaged.groupby(
        ["variant", "test_year"], sort=False
    ):
        direction = direction_metrics(
            current_close=group[CLOSE_COLUMN],
            next_close=group["y_true"],
            probability=group["probability"],
        )
        level = regression_metrics(group["y_true"], group["y_pred"])
        labels = (group["y_true"].to_numpy() > group[CLOSE_COLUMN].to_numpy()).astype(int)
        probability = group["probability"].to_numpy(dtype=float)
        rows.append(
            {
                "variant": str(variant),
                "test_year": int(year),
                **direction,
                **level,
                "brier_score": float(brier_score_loss(labels, probability)),
                "binary_crossentropy": float(
                    log_loss(
                        labels,
                        np.clip(probability, 1e-7, 1 - 1e-7),
                        labels=[0, 1],
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "test_year"]).reset_index(
        drop=True
    )


def _diagnostic_summary(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paths = sorted((output_dir / "cells").glob("*/seed_*/*/metrics.json"))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "variant": path.parent.name,
                "attention_counterfactual_jsd": payload[
                    "attention_counterfactual_jsd"
                ],
                "top1_deletion_agreement": payload["top1_deletion_agreement"],
                "top_attention_influence_lift": payload[
                    "top_attention_influence_lift"
                ],
                "attention_influence_correlation": payload[
                    "attention_influence_correlation"
                ],
            }
        )
    return pd.DataFrame(rows).groupby("variant", as_index=False).mean(numeric_only=True)


def build_ablation_table(
    candidate: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    folds = build_ablation_fold_metrics(candidate, seeds=seeds)
    summary = folds.groupby("variant", as_index=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        mcc_mean=("mcc", "mean"),
        rmse_mean=("rmse", "mean"),
        mae_mean=("mae", "mean"),
        brier_score_mean=("brier_score", "mean"),
        binary_crossentropy_mean=("binary_crossentropy", "mean"),
        predicted_up_share_mean=("predicted_up_share", "mean"),
        years=("test_year", "nunique"),
    )
    renamed = diagnostics.rename(
        columns={
            column: f"{column}_mean"
            for column in diagnostics.columns
            if column != "variant"
        }
    )
    return summary.merge(renamed, on="variant", validate="one_to_one").sort_values(
        "balanced_accuracy_mean", ascending=False
    ).reset_index(drop=True)


def _frozen_fold_metrics(frozen: pd.DataFrame) -> pd.DataFrame:
    values = _normalize_keys(frozen)
    values = values.loc[values["test_year"].isin(TEST_YEARS)]
    rows: list[dict[str, Any]] = []
    for (model, year), group in values.groupby(["model", "test_year"], sort=False):
        direction = binary_direction_metrics(
            group["y_true"].to_numpy(dtype=float),
            group["y_pred"].to_numpy(dtype=float),
            group[CLOSE_COLUMN].to_numpy(dtype=float),
        )
        level = regression_metrics(group["y_true"], group["y_pred"])
        rows.append(
            {
                "model_key": str(model),
                "model": MODEL_LABELS[str(model)],
                "test_year": int(year),
                "balanced_accuracy": direction["balanced_accuracy"],
                "direction_accuracy": direction["direction_accuracy"],
                "mcc": direction["mcc"],
                "rmse": level["rmse"],
                "mae": level["mae"],
            }
        )
    return pd.DataFrame(rows)


def build_six_model_fold_metrics(
    candidate: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    validate_candidate_cohort(candidate, frozen, seeds=seeds)
    averaged = _candidate_seed_average(candidate, seeds=seeds)
    ours = averaged.loc[averaged["variant"].eq("fcta_lstm")]
    rows = _frozen_fold_metrics(frozen).to_dict(orient="records")
    for year, group in ours.groupby("test_year", sort=False):
        direction = direction_metrics(
            current_close=group[CLOSE_COLUMN],
            next_close=group["y_true"],
            probability=group["probability"],
        )
        level = regression_metrics(group["y_true"], group["y_pred"])
        rows.append(
            {
                "model_key": "fcta_lstm",
                "model": "FCTA-LSTM",
                "test_year": int(year),
                "balanced_accuracy": direction["balanced_accuracy"],
                "direction_accuracy": direction["direction_accuracy"],
                "mcc": direction["mcc"],
                "rmse": level["rmse"],
                "mae": level["mae"],
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "test_year"]).reset_index(
        drop=True
    )


def build_six_model_table(folds: pd.DataFrame) -> pd.DataFrame:
    table = folds.groupby(["model_key", "model"], as_index=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        direction_accuracy_std=("direction_accuracy", "std"),
        mcc_mean=("mcc", "mean"),
        rmse_mean=("rmse", "mean"),
        mae_mean=("mae", "mean"),
        test_years=("test_year", "nunique"),
    )
    table = table.sort_values("balanced_accuracy_mean", ascending=False).reset_index(
        drop=True
    )
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def promotion_decision(
    ablation: pd.DataFrame,
    annual: pd.DataFrame,
    frozen_models: pd.DataFrame,
    *,
    test_years: Sequence[int] = TEST_YEARS,
) -> dict[str, Any]:
    scores = ablation.set_index("variant")["balanced_accuracy_mean"]
    if set(scores.index) != set(REGISTERED_VARIANTS):
        raise ValueError("FCTA ablation does not contain every registered variant")
    full_score = float(scores["fcta_lstm"])
    controls = tuple(value for value in REGISTERED_VARIANTS if value != "fcta_lstm")
    conditions = {
        f"beats_{control}_mean": full_score > float(scores[control])
        for control in controls
    }
    annual_scores = annual.set_index(["variant", "test_year"])["balanced_accuracy"]
    for year in test_years:
        conditions[f"beats_attention_control_in_{year}"] = float(
            annual_scores.loc[("fcta_lstm", int(year))]
        ) > float(annual_scores.loc[("attention_control", int(year))])
    best_frozen = float(frozen_models["balanced_accuracy_mean"].max())
    conditions["beats_best_frozen_model_mean"] = full_score > best_frozen
    fidelity = ablation.set_index("variant")
    conditions["lower_jsd_than_attention_control"] = float(
        fidelity.loc["fcta_lstm", "attention_counterfactual_jsd_mean"]
    ) < float(
        fidelity.loc["attention_control", "attention_counterfactual_jsd_mean"]
    )
    if "top1_deletion_agreement_mean" in fidelity.columns:
        conditions["higher_top1_fidelity_than_attention_control"] = float(
            fidelity.loc["fcta_lstm", "top1_deletion_agreement_mean"]
        ) > float(
            fidelity.loc["attention_control", "top1_deletion_agreement_mean"]
        )
    return {
        "promoted": bool(all(conditions.values())),
        "conditions": {key: bool(value) for key, value in conditions.items()},
        "fcta_balanced_accuracy_mean": full_score,
        "best_frozen_balanced_accuracy_mean": best_frozen,
        "failure_action": "close_without_tuning_or_second_run",
    }


def _collect_predictions(output_dir: Path) -> pd.DataFrame:
    paths = sorted((output_dir / "cells").glob("*/seed_*/*/predictions.csv"))
    expected = len(TEST_YEARS) * len(SEEDS) * len(REGISTERED_VARIANTS)
    if len(paths) != expected:
        raise ValueError(f"FCTA cells are incomplete: {len(paths)} of {expected}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _runtime_summary(output_dir: Path) -> pd.DataFrame:
    paths = sorted((output_dir / "cells").glob("*/seed_*/*/run_metadata.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    frame = pd.DataFrame(rows)
    return frame.groupby("variant", as_index=False).agg(
        cells=("wall_seconds", "size"),
        wall_seconds_total=("wall_seconds", "sum"),
        wall_seconds_mean=("wall_seconds", "mean"),
        wall_seconds_std=("wall_seconds", "std"),
        fit_seconds_total=("fit_seconds", "sum"),
        inference_seconds_total=("inference_seconds", "sum"),
        inference_parameters=("inference_parameters", "first"),
    )


def aggregate_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    candidate = _collect_predictions(output_dir)
    frozen = pd.read_csv(FROZEN_PREDICTIONS)
    cohort = validate_candidate_cohort(candidate, frozen, seeds=SEEDS)
    reference = _normalize_keys(frozen)
    reference = reference.loc[
        reference["model"].eq("lstm") & reference["test_year"].isin(TEST_YEARS)
    ]
    rows_by_year = {
        int(year): len(group) for year, group in reference.groupby("test_year")
    }
    if rows_by_year != EXPECTED_TEST_ROWS:
        raise ValueError(f"Frozen cohort row counts changed: {rows_by_year}")
    diagnostics = _diagnostic_summary(output_dir)
    ablation_by_year = build_ablation_fold_metrics(candidate, seeds=SEEDS)
    ablation = build_ablation_table(candidate, diagnostics, seeds=SEEDS)
    six_by_year = build_six_model_fold_metrics(candidate, frozen, seeds=SEEDS)
    six_model = build_six_model_table(six_by_year)
    frozen_summary = six_model.loc[six_model["model_key"].isin(FROZEN_MODEL_KEYS)]
    decision = promotion_decision(ablation, ablation_by_year, frozen_summary)
    runtime = _runtime_summary(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output_dir / "all_seed_predictions.csv", index=False)
    _candidate_seed_average(candidate, seeds=SEEDS).to_csv(
        output_dir / "predictions_seed_averaged.csv", index=False
    )
    diagnostics.to_csv(output_dir / "fidelity_diagnostics.csv", index=False)
    ablation_by_year.to_csv(output_dir / "ablation_by_year_2024_2025.csv", index=False)
    ablation.to_csv(output_dir / "ablation_summary_2024_2025.csv", index=False)
    six_by_year.to_csv(output_dir / "six_model_by_year_2024_2025.csv", index=False)
    six_model.to_csv(output_dir / "six_model_comparison_2024_2025.csv", index=False)
    runtime.to_csv(output_dir / "runtime_summary.csv", index=False)
    (output_dir / "promotion_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    integrity = {
        "passed": True,
        "protocol_id": PROTOCOL_ID,
        "evidence_status": "retrospective_development_evaluation",
        "cohort": cohort,
        "expected_test_rows": EXPECTED_TEST_ROWS,
        "candidate_cells": len(TEST_YEARS) * len(SEEDS) * len(REGISTERED_VARIANTS),
        "variants": list(REGISTERED_VARIANTS),
        "seeds": list(SEEDS),
        "test_years": list(TEST_YEARS),
        "promoted": decision["promoted"],
    }
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    return {
        "integrity": integrity,
        "decision": decision,
        "ablation_by_year": ablation_by_year,
        "ablation": ablation,
        "six_model_by_year": six_by_year,
        "six_model": six_model,
        "runtime": runtime,
    }


def run_experiment(
    *,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
    variants: Iterable[str] = REGISTERED_VARIANTS,
) -> dict[str, Any]:
    requested = tuple(str(value) for value in variants)
    if set(requested) != set(REGISTERED_VARIANTS):
        raise ValueError("A complete registered FCTA run requires every variant")
    freeze_audit = verify_freeze_manifest(PROJECT_ROOT, FREEZE_FILE)
    started = time.perf_counter()
    for year in TEST_YEARS:
        prepared = prepare_fcta_fold(year)
        for seed in SEEDS:
            for variant in REGISTERED_VARIANTS:
                print(
                    f"FCTA-LSTM: year={year}, seed={seed}, variant={variant}",
                    flush=True,
                )
                run_cell(
                    prepared,
                    seed=seed,
                    variant=variant,
                    output_dir=output_dir,
                    force=force,
                )
    result = aggregate_experiment(output_dir)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "created_at": _utc_now(),
        "freeze_audit": freeze_audit,
        "code_sha256": {
            "models/fcta_lstm.py": _sha256(PROJECT_ROOT / "models" / "fcta_lstm.py"),
            "models/fcta_lstm_runner.py": _sha256(
                PROJECT_ROOT / "models" / "fcta_lstm_runner.py"
            ),
        },
        "total_wall_seconds": float(time.perf_counter() - started),
        "tensorflow_devices": [],
        "incremental_api_cost_usd": 0,
    }
    try:
        import tensorflow as tf

        metadata["tensorflow_devices"] = [
            device.name for device in tf.config.list_physical_devices()
        ]
    except (ImportError, RuntimeError):
        pass
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen FCTA-LSTM retrospective architecture screen"
    )
    parser.add_argument("command", choices=("run", "aggregate"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run_experiment(output_dir=args.output_dir, force=args.force)
    else:
        result = aggregate_experiment(args.output_dir)
    print(result["six_model"].to_string(index=False), flush=True)
    print(json.dumps(result["decision"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
