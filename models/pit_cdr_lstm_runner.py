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
    sequence_history_features,
)
from models.convolutional_neural_network import make_sequences, make_test_sequences
from models.integrated_multimodal import (
    NEWS_FEATURES,
    prepare_integrated_fold,
    subset_aligned_regimes,
)
from models.integrated_multimodal_runner import REGIME_OUTPUT_DIR
from models.pit_cdr_lstm import (
    PROTOCOL_ID,
    CDRConfig,
    apply_endpoint_regime_masks,
    build_pit_cdr_models,
    build_relation_pairs,
    compile_pit_cdr_model,
    direction_metrics,
    regime_feature_masks,
    verify_freeze_manifest,
)
from models.track_a_final import FINAL_SEEDS
from models.track_b_forward_news import DAILY_NEWS_FILE
from models.track_b_fusion import _scaled_fold
from models.track_c_outer import REGIMES, selected_feature_lookup
from models.track_c_topk_validation_runner import SELECTED_FEATURES_FILE
from models.vmd_feature_pool import FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pit_cdr_lstm_direct_2024_2025_v1"
FREEZE_FILE = PROJECT_ROOT / "test" / "pit_cdr_lstm_direct_freeze_v1.json"
FROZEN_PREDICTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "final_five_model_prediction_visuals_v1"
    / "final_arm_prediction_series.csv"
)
TEST_YEARS = (2024, 2025)
SEEDS = tuple(int(seed) for seed in FINAL_SEEDS)
VARIANTS = (
    "direct_lstm",
    "random_relations",
    "counter_direction_only",
    "cross_state_only",
    "pit_cdr_lstm",
    "permuted_regime_cdr",
)
STATE_COLUMNS = (
    "composite_trend_score",
    "directional_strength",
    "prob_bull",
    "prob_sideway",
    "prob_bear",
    "routing_confidence",
    "routing_entropy",
)
TRAIN_START_YEAR = 2019
WINDOW = 5
EPOCHS = 20
BATCH_SIZE = 32
MINIMUM_SEPARATION = 20
EXPECTED_TEST_ROWS = {2024: 244, 2025: 234}
REGISTERED_EXPECTED_DIRECTION_ROWS = 477
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
class PreparedFold:
    fold: str
    test_year: int
    feature_columns: tuple[str, ...]
    train_sequence: np.ndarray
    train_labels: np.ndarray
    train_regimes: np.ndarray
    train_state: np.ndarray
    train_endpoint_positions: np.ndarray
    train_dates: np.ndarray
    test_sequence: np.ndarray
    test_regimes: np.ndarray
    test_dates: np.ndarray
    test_close: np.ndarray
    test_next_close: np.ndarray
    mask_counts: Mapping[str, int]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _derived_seed(seed: int, *parts: object) -> int:
    material = "|".join([PROTOCOL_ID, str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") % (
        2**31
    )


def _configure_tensorflow(seed: int):
    import tensorflow as tf

    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    return tf


def _selected_by_regime() -> dict[str, tuple[str, ...]]:
    lookup = selected_feature_lookup(pd.read_csv(SELECTED_FEATURES_FILE))
    return {regime: lookup[("shap", regime)] for regime in REGIMES}


def _load_regime_frames(fold: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = REGIME_OUTPUT_DIR / fold
    return (
        pd.read_csv(directory / "train_regimes.csv"),
        pd.read_csv(directory / "test_regimes.csv"),
    )


def _aligned_state(market: pd.DataFrame, regime: pd.DataFrame) -> np.ndarray:
    required = {DATE_COLUMN, *STATE_COLUMNS}
    missing = sorted(required.difference(regime.columns))
    if missing:
        raise ValueError(f"Regime state is missing columns: {missing}")
    dates = pd.to_datetime(market[DATE_COLUMN], errors="raise").dt.normalize()
    state = regime.loc[:, [DATE_COLUMN, *STATE_COLUMNS]].copy()
    state[DATE_COLUMN] = pd.to_datetime(state[DATE_COLUMN], errors="raise").dt.normalize()
    if state[DATE_COLUMN].duplicated().any():
        raise ValueError("Regime state contains duplicate dates")
    aligned = state.set_index(DATE_COLUMN).reindex(dates)
    if aligned.isna().any().any():
        raise ValueError("Regime state does not cover the market cohort")
    values = aligned.loc[:, STATE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Aligned regime state contains non-finite values")
    return values


def _fold_spec(test_year: int):
    specs = {
        spec.test_year: spec
        for spec in discover_folds(FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR)
    }
    if test_year not in TEST_YEARS or test_year not in specs:
        raise ValueError(f"No registered PIT-CDR fold for {test_year}")
    return specs[test_year]


def prepare_candidate_fold(test_year: int) -> PreparedFold:
    spec = _fold_spec(test_year)
    daily_news = pd.read_csv(DAILY_NEWS_FILE)
    original = prepare_integrated_fold(spec, daily_news)["Global-Numeric-News"]
    scaled, _ = _scaled_fold(original)
    if int(original.train[DATE_COLUMN].dt.year.min()) != TRAIN_START_YEAR:
        raise ValueError("PIT-CDR training cohort does not begin in 2019")
    feature_columns = tuple(original.feature_columns)
    if len(feature_columns) != 130 or tuple(feature_columns[-len(NEWS_FEATURES) :]) != tuple(
        NEWS_FEATURES
    ):
        raise ValueError("PIT-CDR final-arm feature pool changed")

    train_regime_frame, test_regime_frame = _load_regime_frames(spec.fold)
    train_regimes_all = subset_aligned_regimes(
        original.train, train_regime_frame, split="train"
    )
    test_regimes = subset_aligned_regimes(original.test, test_regime_frame, split="test")
    state_all = _aligned_state(original.train, train_regime_frame)
    masks = regime_feature_masks(
        feature_columns,
        _selected_by_regime(),
        news_features=NEWS_FEATURES,
    )
    mask_counts = {regime: int(mask.sum()) for regime, mask in masks.items()}
    if mask_counts != {"bull": 38, "sideway": 130, "bear": 88}:
        raise ValueError(f"PIT-CDR feature-mask counts changed: {mask_counts}")

    train_features = scaled.train.loc[:, feature_columns].to_numpy(dtype=np.float64)
    train_target = original.train[TARGET_COLUMN].to_numpy(dtype=np.float64)
    train_close = original.train[CLOSE_COLUMN].to_numpy(dtype=np.float64)
    train_sequence_all, _ = make_sequences(train_features, train_target, WINDOW)
    endpoint_slice = slice(WINDOW - 1, None)
    delta = train_target[endpoint_slice] - train_close[endpoint_slice]
    nonzero = delta != 0.0
    train_labels = (delta[nonzero] > 0.0).astype(np.int8)
    train_regimes = train_regimes_all[endpoint_slice][nonzero]
    train_sequence = apply_endpoint_regime_masks(
        train_sequence_all[nonzero], train_regimes, masks
    )
    train_state = state_all[endpoint_slice][nonzero]
    train_positions = np.arange(WINDOW - 1, len(original.train), dtype=np.int64)[nonzero]
    train_dates = original.train[DATE_COLUMN].to_numpy()[endpoint_slice][nonzero]

    test_features = scaled.test.loc[:, feature_columns].to_numpy(dtype=np.float64)
    history_features = sequence_history_features(scaled)
    history_indices = [scaled.feature_columns.index(column) for column in feature_columns]
    test_sequence_all = make_test_sequences(
        history_features[:, history_indices],
        test_features,
        WINDOW,
    )
    test_sequence = apply_endpoint_regime_masks(test_sequence_all, test_regimes, masks)
    if len(test_sequence) != EXPECTED_TEST_ROWS[test_year]:
        raise ValueError(f"PIT-CDR {test_year} test-row count changed")
    if pd.Timestamp(train_dates.max()) >= pd.Timestamp(original.test[DATE_COLUMN].min()):
        raise ValueError("PIT-CDR train dates do not precede the test fold")
    if len(np.unique(train_labels)) != 2:
        raise ValueError("PIT-CDR training data does not contain both directions")

    return PreparedFold(
        fold=spec.fold,
        test_year=test_year,
        feature_columns=feature_columns,
        train_sequence=train_sequence,
        train_labels=train_labels,
        train_regimes=train_regimes,
        train_state=train_state,
        train_endpoint_positions=train_positions,
        train_dates=train_dates,
        test_sequence=test_sequence,
        test_regimes=test_regimes,
        test_dates=original.test[DATE_COLUMN].to_numpy(),
        test_close=original.test[CLOSE_COLUMN].to_numpy(dtype=np.float64),
        test_next_close=original.test[TARGET_COLUMN].to_numpy(dtype=np.float64),
        mask_counts=mask_counts,
    )


def _relation_plan(prepared: PreparedFold, *, variant: str, seed: int):
    pairing_regimes = prepared.train_regimes
    strategy = "matched"
    relation_seed = _derived_seed(seed, variant, prepared.test_year, "pairs")
    if variant == "random_relations":
        strategy = "random"
    elif variant == "permuted_regime_cdr":
        pairing_regimes = np.random.default_rng(relation_seed).permutation(pairing_regimes)
    return build_relation_pairs(
        prepared.train_labels,
        pairing_regimes,
        prepared.train_state,
        endpoint_positions=prepared.train_endpoint_positions,
        minimum_separation=MINIMUM_SEPARATION,
        seed=relation_seed,
        strategy=strategy,
    )


def _relation_weights(variant: str, relation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown PIT-CDR variant: {variant}")
    use_counter = variant in {
        "random_relations",
        "counter_direction_only",
        "pit_cdr_lstm",
        "permuted_regime_cdr",
    }
    use_transport = variant in {
        "random_relations",
        "cross_state_only",
        "pit_cdr_lstm",
        "permuted_regime_cdr",
    }
    relation_values = np.asarray(relation, dtype=object)
    counter = (relation_values == "counter").astype(np.float32) if use_counter else np.zeros(
        len(relation_values), dtype=np.float32
    )
    transport = (
        (relation_values == "transport").astype(np.float32)
        if use_transport
        else np.zeros(len(relation_values), dtype=np.float32)
    )
    return counter, transport


def fit_variant(
    prepared: PreparedFold,
    *,
    variant: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], Any]:
    if variant not in VARIANTS or seed not in SEEDS:
        raise ValueError("Unknown PIT-CDR variant or seed")
    plan = _relation_plan(prepared, variant=variant, seed=seed)
    counter_weight, transport_weight = _relation_weights(variant, plan.relation)
    labels = prepared.train_labels.astype(np.float32)
    inputs = {
        "left_sequence": prepared.train_sequence[plan.left],
        "right_sequence": prepared.train_sequence[plan.right],
    }
    targets = {
        "left_probability": labels[plan.left],
        "right_probability": labels[plan.right],
        "counter_rank": np.ones(len(plan.left), dtype=np.float32),
        "transport_distance": np.zeros(len(plan.left), dtype=np.float32),
    }
    sample_weights = {
        "left_probability": np.ones(len(plan.left), dtype=np.float32),
        "right_probability": np.ones(len(plan.left), dtype=np.float32),
        "counter_rank": counter_weight,
        "transport_distance": transport_weight,
    }
    tf = _configure_tensorflow(seed)
    tf.keras.backend.clear_session()
    config = CDRConfig(window=WINDOW, feature_count=len(prepared.feature_columns))
    training_model, inference_model = build_pit_cdr_models(config)
    compile_pit_cdr_model(training_model, config)
    fit_started = time.perf_counter()
    training_model.fit(
        inputs,
        targets,
        sample_weight=sample_weights,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    fit_seconds = float(time.perf_counter() - fit_started)
    inference_started = time.perf_counter()
    probability = inference_model.predict(prepared.test_sequence, verbose=0).reshape(-1)
    inference_seconds = float(time.perf_counter() - inference_started)
    if probability.shape != (len(prepared.test_sequence),) or not np.isfinite(probability).all():
        raise ValueError("PIT-CDR inference probabilities are invalid")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("PIT-CDR probabilities are outside [0, 1]")
    diagnostics = {
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "training_pairs": len(plan.left),
        "counter_pairs": int(np.count_nonzero(plan.relation == "counter")),
        "transport_pairs": int(np.count_nonzero(plan.relation == "transport")),
        "active_counter_pairs": int(np.count_nonzero(counter_weight)),
        "active_transport_pairs": int(np.count_nonzero(transport_weight)),
        "inference_parameters": int(inference_model.count_params()),
        "training_parameters": int(training_model.count_params()),
    }
    return probability, diagnostics, inference_model


def _normalized_prediction_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], errors="raise").dt.normalize()
    result["test_year"] = result["test_year"].astype(int)
    return result


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
        raise ValueError("Candidate predictions are missing cohort columns")
    if not frozen_required.issubset(frozen.columns):
        raise ValueError("Frozen predictions are missing cohort columns")
    candidate_values = _normalized_prediction_keys(candidate)
    frozen_values = _normalized_prediction_keys(frozen)
    candidate_values = candidate_values.loc[candidate_values["test_year"].isin(TEST_YEARS)]
    frozen_values = frozen_values.loc[frozen_values["test_year"].isin(TEST_YEARS)]
    if set(candidate_values["variant"]) != set(VARIANTS):
        raise ValueError("Candidate predictions do not contain all frozen variants")
    if set(candidate_values["seed"].astype(int)) != {int(seed) for seed in seeds}:
        raise ValueError("Candidate predictions do not contain all requested seeds")
    if set(frozen_values["model"]) != set(FROZEN_MODEL_KEYS):
        raise ValueError("Frozen predictions do not contain the five registered models")
    reference = frozen_values.loc[frozen_values["model"].eq("lstm")].sort_values(
        ["test_year", DATE_COLUMN]
    )
    reference_keys = reference[["test_year", DATE_COLUMN]].reset_index(drop=True)
    rows_per_cell = len(reference)
    for (variant, seed), group in candidate_values.groupby(["variant", "seed"], sort=False):
        ordered = group.sort_values(["test_year", DATE_COLUMN]).reset_index(drop=True)
        if not ordered[["test_year", DATE_COLUMN]].equals(reference_keys):
            raise ValueError(f"Candidate dates do not match frozen cohort: {variant}/{seed}")
        if not np.allclose(ordered[CLOSE_COLUMN], reference[CLOSE_COLUMN], rtol=0.0, atol=1e-12):
            raise ValueError("Candidate current-close values do not match frozen cohort")
        if not np.allclose(ordered["y_true"], reference["y_true"], rtol=0.0, atol=1e-12):
            raise ValueError("Candidate actual values do not match frozen cohort")
    return {
        "passed": True,
        "rows_per_variant_seed": rows_per_cell,
        "candidate_cells": int(candidate_values.groupby(["variant", "seed"]).ngroups),
        "frozen_models": len(FROZEN_MODEL_KEYS),
    }


def _candidate_seed_average(
    candidate: pd.DataFrame,
    *,
    seeds: Sequence[int],
) -> pd.DataFrame:
    values = _normalized_prediction_keys(candidate)
    keys = ["variant", "test_year", DATE_COLUMN]
    for column in (CLOSE_COLUMN, "y_true"):
        if values.groupby(keys)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Candidate seeds disagree on {column}")
    averaged = values.groupby(keys, as_index=False, sort=False).agg(
        Close_D=(CLOSE_COLUMN, "first"),
        y_true=("y_true", "first"),
        probability=("probability", "mean"),
        seeds_averaged=("seed", "nunique"),
    )
    if not averaged["seeds_averaged"].eq(len(seeds)).all():
        raise ValueError("Candidate seed averaging is incomplete")
    return averaged


def build_ablation_fold_metrics(
    candidate: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    averaged = _candidate_seed_average(candidate, seeds=seeds)
    rows: list[dict[str, Any]] = []
    for (variant, year), group in averaged.groupby(["variant", "test_year"], sort=False):
        metrics = direction_metrics(
            current_close=group[CLOSE_COLUMN],
            next_close=group["y_true"],
            probability=group["probability"],
        )
        eligible = np.sign(group["y_true"].to_numpy() - group[CLOSE_COLUMN].to_numpy()) != 0
        labels = (group.loc[eligible, "y_true"].to_numpy() > group.loc[eligible, CLOSE_COLUMN].to_numpy()).astype(int)
        probabilities = group.loc[eligible, "probability"].to_numpy(dtype=float)
        rows.append(
            {
                "variant": str(variant),
                "test_year": int(year),
                **metrics,
                "brier_score": float(brier_score_loss(labels, probabilities)),
                "binary_crossentropy": float(
                    log_loss(labels, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "test_year"]).reset_index(
        drop=True
    )


def build_ablation_table(
    candidate: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    fold_metrics = build_ablation_fold_metrics(candidate, seeds=seeds)
    summary = fold_metrics.groupby("variant", as_index=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        mcc_mean=("mcc", "mean"),
        brier_score_mean=("brier_score", "mean"),
        binary_crossentropy_mean=("binary_crossentropy", "mean"),
        predicted_up_share_mean=("predicted_up_share", "mean"),
        years=("test_year", "nunique"),
    )
    return summary.sort_values("balanced_accuracy_mean", ascending=False).reset_index(drop=True)


def build_six_model_fold_metrics(
    candidate: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    validate_candidate_cohort(candidate, frozen, seeds=seeds)
    candidate_average = _candidate_seed_average(candidate, seeds=seeds)
    ours = candidate_average.loc[candidate_average["variant"].eq("pit_cdr_lstm")]
    frozen_values = _normalized_prediction_keys(frozen)
    frozen_values = frozen_values.loc[frozen_values["test_year"].isin(TEST_YEARS)]
    fold_rows: list[dict[str, Any]] = []
    for (model, year), group in frozen_values.groupby(["model", "test_year"], sort=False):
        direction = binary_direction_metrics(
            group["y_true"].to_numpy(dtype=float),
            group["y_pred"].to_numpy(dtype=float),
            group[CLOSE_COLUMN].to_numpy(dtype=float),
        )
        level = regression_metrics(
            group["y_true"].to_numpy(dtype=float),
            group["y_pred"].to_numpy(dtype=float),
        )
        fold_rows.append(
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
    for year, group in ours.groupby("test_year", sort=False):
        direction = direction_metrics(
            current_close=group[CLOSE_COLUMN],
            next_close=group["y_true"],
            probability=group["probability"],
        )
        fold_rows.append(
            {
                "model_key": "pit_cdr_lstm",
                "model": "PIT-CDR-LSTM",
                "test_year": int(year),
                "balanced_accuracy": direction["balanced_accuracy"],
                "direction_accuracy": direction["direction_accuracy"],
                "mcc": direction["mcc"],
                "rmse": np.nan,
                "mae": np.nan,
            }
        )
    return pd.DataFrame(fold_rows).sort_values(["model", "test_year"]).reset_index(
        drop=True
    )


def build_six_model_table(
    candidate: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    folds = build_six_model_fold_metrics(candidate, frozen, seeds=seeds)
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
    table = table.sort_values("balanced_accuracy_mean", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def _cell_directory(output_dir: Path, year: int, seed: int, variant: str) -> Path:
    return output_dir / "cells" / str(year) / f"seed_{seed}" / variant


def _cell_complete(directory: Path) -> bool:
    required = (
        directory / "predictions.csv",
        directory / "metrics.json",
        directory / "run_metadata.json",
        directory / "inference.weights.h5",
    )
    return all(path.is_file() for path in required)


def run_cell(
    prepared: PreparedFold,
    *,
    seed: int,
    variant: str,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    directory = _cell_directory(output_dir, prepared.test_year, seed, variant)
    if _cell_complete(directory) and not force:
        return {"status": "skipped_complete", "year": prepared.test_year, "seed": seed, "variant": variant}
    started = time.perf_counter()
    probability, diagnostics, inference_model = fit_variant(
        prepared,
        variant=variant,
        seed=seed,
    )
    metrics = direction_metrics(
        current_close=prepared.test_close,
        next_close=prepared.test_next_close,
        probability=probability,
    )
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
            "probability": probability,
            "pred_direction": (probability > 0.5).astype(np.int8),
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(directory / "predictions.csv", index=False)
    (directory / "metrics.json").write_text(
        json.dumps({**metrics, **diagnostics}, indent=2), encoding="utf-8"
    )
    inference_model.save_weights(directory / "inference.weights.h5")
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "evidence_status": "frozen_retrospective_extension",
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
        "configuration": asdict(CDRConfig(window=WINDOW, feature_count=len(prepared.feature_columns))),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "minimum_separation": MINIMUM_SEPARATION,
        "feature_count": len(prepared.feature_columns),
        "mask_counts": dict(prepared.mask_counts),
        "training_rows": len(prepared.train_sequence),
        "test_rows": len(prepared.test_sequence),
        "runtime_scope": "pair construction, model build, fit, inference, and artifact write",
        "wall_seconds": float(time.perf_counter() - started),
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "tensorflow"]
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if not _cell_complete(directory):
        raise RuntimeError("PIT-CDR cell failed post-write completeness check")
    del inference_model
    gc.collect()
    return {"status": "completed", **metadata}


def _collect_predictions(output_dir: Path) -> pd.DataFrame:
    paths = sorted((output_dir / "cells").glob("*/seed_*/*/predictions.csv"))
    expected = len(TEST_YEARS) * len(SEEDS) * len(VARIANTS)
    if len(paths) != expected:
        raise ValueError(f"PIT-CDR cells are incomplete: {len(paths)} of {expected}")
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
    )


def _mechanism_decision(ablation: pd.DataFrame) -> dict[str, Any]:
    scores = ablation.set_index("variant")["balanced_accuracy_mean"]
    full = float(scores["pit_cdr_lstm"])
    controls = (
        "direct_lstm",
        "random_relations",
        "counter_direction_only",
        "cross_state_only",
        "permuted_regime_cdr",
    )
    conditions = {f"beats_{name}": bool(full > float(scores[name])) for name in controls}
    return {
        "supported": bool(all(conditions.values())),
        "conditions": conditions,
        "pit_cdr_lstm_balanced_accuracy_mean": full,
        "deltas_pp": {
            name: (full - float(scores[name])) * 100.0 for name in controls
        },
        "reporting_rule": (
            "mechanism supported only if every registered control is beaten; "
            "otherwise report as unsupported performance change"
        ),
    }


def aggregate_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    candidate = _collect_predictions(output_dir)
    frozen = pd.read_csv(FROZEN_PREDICTIONS)
    cohort = validate_candidate_cohort(candidate, frozen, seeds=SEEDS)
    frozen_reference = _normalized_prediction_keys(frozen)
    frozen_reference = frozen_reference.loc[
        frozen_reference["model"].eq("lstm")
        & frozen_reference["test_year"].isin(TEST_YEARS)
    ]
    observed_rows_by_year = {
        int(year): len(group)
        for year, group in frozen_reference.groupby("test_year")
    }
    if observed_rows_by_year != EXPECTED_TEST_ROWS:
        raise ValueError(f"Frozen cohort row counts changed: {observed_rows_by_year}")
    observed_direction_rows = int(
        np.count_nonzero(
            frozen_reference["y_true"].to_numpy(dtype=float)
            - frozen_reference[CLOSE_COLUMN].to_numpy(dtype=float)
        )
    )
    if observed_direction_rows != sum(EXPECTED_TEST_ROWS.values()):
        raise ValueError("Frozen cohort unexpectedly contains actual direction ties")
    protocol_deviation = {
        "declared": True,
        "type": "pre_execution_bookkeeping_count_error",
        "registered_direction_evaluable_rows": REGISTERED_EXPECTED_DIRECTION_ROWS,
        "observed_direction_evaluable_rows": observed_direction_rows,
        "explanation": (
            "The freeze correctly registered 244 plus 234 test rows but incorrectly "
            "stated that one row was an actual zero-return tie. Recalculation from "
            "the frozen targets found no actual ties; all 478 rows are evaluable for "
            "every model. No row, prediction, threshold, or hyperparameter changed."
        ),
    }
    ablation_by_year = build_ablation_fold_metrics(candidate, seeds=SEEDS)
    ablation = build_ablation_table(candidate, seeds=SEEDS)
    six_model_by_year = build_six_model_fold_metrics(candidate, frozen, seeds=SEEDS)
    six_model = build_six_model_table(candidate, frozen, seeds=SEEDS)
    mechanism = _mechanism_decision(ablation)
    runtime = _runtime_summary(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output_dir / "all_seed_predictions.csv", index=False)
    _candidate_seed_average(candidate, seeds=SEEDS).to_csv(
        output_dir / "predictions_seed_averaged.csv", index=False
    )
    ablation_by_year.to_csv(output_dir / "ablation_by_year_2024_2025.csv", index=False)
    ablation.to_csv(output_dir / "ablation_summary_2024_2025.csv", index=False)
    six_model_by_year.to_csv(
        output_dir / "six_model_by_year_2024_2025.csv", index=False
    )
    six_model.to_csv(output_dir / "six_model_comparison_2024_2025.csv", index=False)
    runtime.to_csv(output_dir / "runtime_summary.csv", index=False)
    (output_dir / "mechanism_decision.json").write_text(
        json.dumps(mechanism, indent=2), encoding="utf-8"
    )
    (output_dir / "protocol_deviation.json").write_text(
        json.dumps(protocol_deviation, indent=2), encoding="utf-8"
    )
    integrity = {
        "passed": True,
        "status": "passed_with_declared_bookkeeping_deviation",
        "protocol_id": PROTOCOL_ID,
        "cohort": cohort,
        "expected_test_rows": EXPECTED_TEST_ROWS,
        "registered_direction_evaluable_rows": REGISTERED_EXPECTED_DIRECTION_ROWS,
        "observed_direction_evaluable_rows": observed_direction_rows,
        "protocol_deviation": protocol_deviation,
        "candidate_cells": len(TEST_YEARS) * len(SEEDS) * len(VARIANTS),
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "test_years": list(TEST_YEARS),
        "mechanism_supported": mechanism["supported"],
    }
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    return {
        "integrity": integrity,
        "mechanism": mechanism,
        "ablation_by_year": ablation_by_year,
        "ablation": ablation,
        "six_model_by_year": six_model_by_year,
        "six_model": six_model,
        "runtime": runtime,
    }


def run_experiment(
    *,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
    variants: Iterable[str] = VARIANTS,
) -> dict[str, Any]:
    requested = tuple(str(value) for value in variants)
    if set(requested) != set(VARIANTS):
        raise ValueError("A complete registered run requires all PIT-CDR variants")
    freeze_audit = verify_freeze_manifest(PROJECT_ROOT, FREEZE_FILE)
    started = time.perf_counter()
    for year in TEST_YEARS:
        prepared = prepare_candidate_fold(year)
        for seed in SEEDS:
            for variant in VARIANTS:
                print(
                    f"PIT-CDR direct: year={year}, seed={seed}, variant={variant}",
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
            "models/pit_cdr_lstm.py": _sha256(PROJECT_ROOT / "models" / "pit_cdr_lstm.py"),
            "models/pit_cdr_lstm_runner.py": _sha256(
                PROJECT_ROOT / "models" / "pit_cdr_lstm_runner.py"
            ),
        },
        "total_wall_seconds": float(time.perf_counter() - started),
        "tensorflow_devices": [],
        "incremental_api_cost_usd": 0,
    }
    try:
        import tensorflow as tf

        metadata["tensorflow_devices"] = [device.name for device in tf.config.list_physical_devices()]
    except (ImportError, RuntimeError):
        pass
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen direct PIT-CDR-LSTM evaluation")
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
    print(json.dumps(result["mechanism"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
