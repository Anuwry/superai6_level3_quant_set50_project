from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT_EARLY = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "numba"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
)

from models.baseline_common import (
    PROJECT_ROOT,
    discover_folds,
    get_feature_columns,
    package_versions,
    read_frame,
)
from models.pit_fcg_development import (
    CONTEXT_FEATURES,
    NEWS_FEATURES,
    PreparedInnerFold,
    prepare_inner_fold,
    verify_frozen_inputs,
)
from models.track_b_forward_news import DAILY_NEWS_FILE
from models.track_b_fusion import join_news_features
from models.vmd_feature_pool import FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR

PROTOCOL_ID = "pit-fcg-lstm-inner-development-v1"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pit_fcg_lstm_inner_development_v1"
FREEZE_FILE = PROJECT_ROOT / "test" / "pit_fcg_lstm_freeze_v1.json"
REGIME_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "track_c"
    / "daily_regime_point_in_time_v2"
    / "fold_1"
    / "train_regimes.csv"
)
INNER_YEARS = (2020, 2021)
SEEDS = (42, 123, 456, 789, 2025)
EPOCHS = 20
BATCH_SIZE = 32
VARIANTS = (
    "direct_numeric_lstm",
    "concat_lstm",
    "bounded_residual",
    "random_control_fcg",
    "matched_control_fcg",
)
FCG_VARIANTS = ("random_control_fcg", "matched_control_fcg")
CELL_ROOT = "cells"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    y_true = np.asarray(labels, dtype=np.int8).reshape(-1)
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(y_true) < 1 or y_true.shape != probability.shape:
        raise ValueError("Labels and probabilities must be non-empty and aligned")
    if set(np.unique(y_true)).difference({0, 1}):
        raise ValueError("Direction labels must be binary")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    prediction = (probability > 0.5).astype(np.int8)
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return {
        "observations": len(y_true),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "direction_accuracy": float(accuracy_score(y_true, prediction)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "binary_crossentropy": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probability)),
    }


def evaluate_inner_promotion(
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    parameter_increase_fraction: float,
    integrity_passed: bool,
) -> dict[str, Any]:
    required_columns = {"variant", "balanced_accuracy_mean"}
    if not required_columns.issubset(summary.columns):
        raise ValueError("Inner summary is missing promotion columns")
    if set(summary["variant"].astype(str)) != set(VARIANTS):
        raise ValueError("Inner summary does not contain exactly the frozen variants")
    scores = summary.set_index("variant")["balanced_accuracy_mean"]
    ours = float(scores["matched_control_fcg"])
    baselines = [
        float(scores[name])
        for name in (
            "direct_numeric_lstm",
            "concat_lstm",
            "bounded_residual",
        )
    ]
    matched = diagnostics.loc[
        diagnostics["variant"].eq("matched_control_fcg")
    ]
    required_diagnostics = {
        "inner_fold",
        "aligned_balanced_accuracy",
        "placebo_balanced_accuracy",
        "aligned_gate_median",
    }
    if not required_diagnostics.issubset(matched.columns) or len(matched) != 2:
        raise ValueError("Matched-control diagnostics require exactly two inner folds")
    aligned_beats_placebo = bool(
        (
            matched["aligned_balanced_accuracy"]
            > matched["placebo_balanced_accuracy"]
        ).all()
    )
    gate_nontrivial = bool(
        matched["aligned_gate_median"].between(0.01, 0.99, inclusive="neither").all()
    )
    conditions = {
        "beats_required_inner_baselines": bool(all(ours > value for value in baselines)),
        "aligned_beats_placebo_each_fold": aligned_beats_placebo,
        "gate_is_nontrivial_each_fold": gate_nontrivial,
        "parameter_budget_within_15_percent": bool(
            np.isfinite(parameter_increase_fraction)
            and parameter_increase_fraction <= 0.15
        ),
        "integrity_passed": bool(integrity_passed),
    }
    return {
        "passed": bool(all(conditions.values())),
        "conditions": conditions,
        "matched_control_mean_balanced_accuracy": ours,
        "parameter_increase_fraction": float(parameter_increase_fraction),
    }


def load_development_frame() -> tuple[pd.DataFrame, tuple[str, ...], dict[str, Any]]:
    audit = verify_frozen_inputs(PROJECT_ROOT, FREEZE_FILE)
    specs = {
        spec.fold: spec
        for spec in discover_folds(FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR)
    }
    spec = specs["fold_1"]
    market = read_frame(spec.train_path)
    market = market.loc[market["Date"].dt.year.ge(2019)].reset_index(drop=True)
    numeric_features = tuple(get_feature_columns(market))
    if len(numeric_features) != 122:
        raise ValueError("PIT-FCG numerical feature count changed")
    fused = join_news_features(market, pd.read_csv(DAILY_NEWS_FILE))
    if tuple(get_feature_columns(fused)[-len(NEWS_FEATURES) :]) != NEWS_FEATURES:
        raise ValueError("PIT-FCG news feature order changed")
    regime = pd.read_csv(REGIME_FILE)
    regime["Date"] = pd.to_datetime(regime["Date"], errors="raise")
    gate_columns = ["Date", *CONTEXT_FEATURES, "routing_regime"]
    regime = regime.loc[:, gate_columns]
    frame = fused.merge(
        regime,
        on="Date",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not frame["_merge"].eq("both").all():
        raise ValueError("Regime context does not cover the development frame")
    frame = frame.drop(columns="_merge").sort_values("Date").reset_index(drop=True)
    if len(frame) != 727 or frame["Date"].dt.year.unique().tolist() != [2019, 2020, 2021]:
        raise ValueError("Development cohort changed from the frozen 2019-2021 rows")
    return frame, numeric_features, audit


def cell_directory(output_dir: Path, inner_year: int, seed: int) -> Path:
    return output_dir / CELL_ROOT / f"inner_{int(inner_year)}" / f"seed_{int(seed)}"


def _cell_paths(directory: Path) -> tuple[Path, ...]:
    return (
        directory / "metrics.csv",
        directory / "predictions.csv",
        directory / "diagnostics.csv",
        directory / "control_pair_audit.csv",
        directory / "run_metadata.json",
        directory / "integrity_audit.json",
    )


def cell_complete(output_dir: Path, inner_year: int, seed: int) -> bool:
    directory = cell_directory(output_dir, inner_year, seed)
    try:
        if not all(path.is_file() for path in _cell_paths(directory)):
            return False
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        integrity = json.loads(
            (directory / "integrity_audit.json").read_text(encoding="utf-8")
        )
        metrics = pd.read_csv(directory / "metrics.csv")
        predictions = pd.read_csv(directory / "predictions.csv")
        return bool(
            metadata.get("protocol_id") == PROTOCOL_ID
            and integrity.get("passed") is True
            and set(metrics["variant"].astype(str)) == set(VARIANTS)
            and len(metrics) == len(VARIANTS)
            and predictions.groupby("variant").size().nunique() == 1
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _configure_tensorflow(seed: int) -> Any:
    import tensorflow as tf

    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    return tf


def _custom_inputs(
    prepared: PreparedInnerFold,
    *,
    controls: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    anchors = prepared.matched_train_controls.anchor_indices
    train_inputs = {
        "numeric": prepared.train.numeric[anchors],
        "news": prepared.train.news[anchors],
        "context": prepared.train.context[anchors],
    }
    validation_inputs = {
        "numeric": prepared.validation.numeric,
        "news": prepared.validation.news,
        "context": prepared.validation.context,
    }
    if controls == "matched":
        train_inputs["placebo_news"] = prepared.train.news[
            prepared.matched_train_controls.source_indices
        ]
    elif controls == "random":
        train_inputs["placebo_news"] = prepared.train.news[
            prepared.random_train_controls.source_indices
        ]
    elif controls is not None:
        raise ValueError("Unknown custom control type")
    return train_inputs, validation_inputs


def _fit_variant(
    variant: str,
    prepared: PreparedInnerFold,
    *,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any] | None]:
    from models.pit_fcg_lstm import (
        build_concat_lstm,
        build_direct_numeric_lstm,
        build_pit_fcg_lstm_model,
    )

    tf = _configure_tensorflow(seed)
    tf.keras.backend.clear_session()
    _configure_tensorflow(seed)
    anchors = prepared.matched_train_controls.anchor_indices
    y_train = prepared.train.labels[anchors]
    history: Any
    gate = placebo_probability = placebo_gate = None
    if variant == "direct_numeric_lstm":
        model = build_direct_numeric_lstm(
            (prepared.train.numeric.shape[1], prepared.train.numeric.shape[2])
        )
        x_train = prepared.train.numeric[anchors]
        x_validation = prepared.validation.numeric
    elif variant == "concat_lstm":
        x_train = np.concatenate(
            [prepared.train.numeric[anchors], prepared.train.news[anchors]],
            axis=2,
        )
        x_validation = np.concatenate(
            [prepared.validation.numeric, prepared.validation.news],
            axis=2,
        )
        model = build_concat_lstm((x_train.shape[1], x_train.shape[2]))
    else:
        control = {
            "bounded_residual": None,
            "random_control_fcg": "random",
            "matched_control_fcg": "matched",
        }.get(variant)
        if variant not in VARIANTS:
            raise ValueError(f"Unknown PIT-FCG variant: {variant}")
        x_train, x_validation = _custom_inputs(prepared, controls=control)
        model = build_pit_fcg_lstm_model(
            numeric_shape=(x_train["numeric"].shape[1], x_train["numeric"].shape[2]),
            news_shape=(x_train["news"].shape[1], x_train["news"].shape[2]),
            context_features=x_train["context"].shape[1],
            use_fcg_loss=control is not None,
        )

    parameters = int(model.count_params())
    fit_started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    fit_seconds = time.perf_counter() - fit_started
    inference_started = time.perf_counter()
    if variant in ("direct_numeric_lstm", "concat_lstm"):
        probability = model.predict(x_validation, verbose=0).reshape(-1)
    else:
        aligned_outputs = model(x_validation, training=False)
        probability = aligned_outputs["probability"].numpy().reshape(-1)
        gate = aligned_outputs["gate"].numpy().reshape(-1)
        if variant in FCG_VARIANTS:
            controls = (
                prepared.validation_controls
                if variant == "matched_control_fcg"
                else prepared.random_validation_controls
            )
            placebo_inputs = {
                **x_validation,
                "news": prepared.train.news[controls.source_indices],
            }
            placebo_outputs = model(placebo_inputs, training=False)
            placebo_probability = (
                placebo_outputs["probability"].numpy().reshape(-1)
            )
            placebo_gate = placebo_outputs["gate"].numpy().reshape(-1)
    inference_seconds = time.perf_counter() - inference_started
    metrics = classification_metrics(prepared.validation.labels, probability)
    metric_row: dict[str, Any] = {
        "inner_fold": prepared.name,
        "validation_year": prepared.validation_year,
        "seed": int(seed),
        "variant": variant,
        "training_sequences": len(y_train),
        "trainable_parameters": parameters,
        "fit_seconds": float(fit_seconds),
        "inference_seconds": float(inference_seconds),
        "final_training_loss": float(history.history["loss"][-1]),
        **metrics,
    }
    prediction = pd.DataFrame(
        {
            "inner_fold": prepared.name,
            "validation_year": prepared.validation_year,
            "seed": int(seed),
            "variant": variant,
            "Date": prepared.validation.endpoint_dates,
            "y_true": prepared.validation.labels.astype(np.int8),
            "probability": probability,
            "gate": np.nan if gate is None else gate,
            "placebo_probability": (
                np.nan if placebo_probability is None else placebo_probability
            ),
            "placebo_gate": np.nan if placebo_gate is None else placebo_gate,
        }
    )
    diagnostic = None
    if variant in FCG_VARIANTS:
        if placebo_probability is None or placebo_gate is None or gate is None:
            raise RuntimeError("FCG diagnostic predictions are missing")
        diagnostic = {
            "inner_fold": prepared.name,
            "validation_year": prepared.validation_year,
            "seed": int(seed),
            "variant": variant,
            "aligned_balanced_accuracy": metrics["balanced_accuracy"],
            "placebo_balanced_accuracy": classification_metrics(
                prepared.validation.labels,
                placebo_probability,
            )["balanced_accuracy"],
            "aligned_gate_mean": float(np.mean(gate)),
            "aligned_gate_median": float(np.median(gate)),
            "placebo_gate_mean": float(np.mean(placebo_gate)),
            "placebo_gate_median": float(np.median(placebo_gate)),
            "final_epoch_gate_target_rate": float(
                history.history["gate_target_rate"][-1]
            ),
        }
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return metric_row, prediction, diagnostic


def _control_rows(
    prepared: PreparedInnerFold,
    *,
    seed: int,
) -> pd.DataFrame:
    specifications = (
        ("train", "matched", prepared.matched_train_controls, prepared.train),
        ("train", "random", prepared.random_train_controls, prepared.train),
        ("validation", "matched", prepared.validation_controls, prepared.validation),
        (
            "validation",
            "random",
            prepared.random_validation_controls,
            prepared.validation,
        ),
    )
    rows: list[dict[str, Any]] = []
    for split, strategy, matches, anchors in specifications:
        for position, (anchor, source) in enumerate(
            zip(matches.anchor_indices, matches.source_indices, strict=True)
        ):
            source_date = prepared.train.endpoint_dates[source]
            anchor_date = anchors.endpoint_dates[anchor]
            rows.append(
                {
                    "inner_fold": prepared.name,
                    "seed": int(seed),
                    "anchor_split": split,
                    "strategy": strategy,
                    "anchor_index": int(anchor),
                    "source_index": int(source),
                    "anchor_date": anchor_date,
                    "source_date": source_date,
                    "source_partition": "inner_train",
                    "source_before_anchor": bool(source_date < anchor_date),
                    "endpoint_gap_rows": (
                        int(anchor - source) if split == "train" else np.nan
                    ),
                    "regime_agreement": bool(
                        anchors.regimes[anchor] == prepared.train.regimes[source]
                    ),
                    "coverage_distance": float(matches.coverage_distance[position]),
                    "same_year": bool(matches.same_year[position]),
                }
            )
    return pd.DataFrame(rows)


def run_cell(
    *,
    inner_year: int,
    seed: int,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    if inner_year not in INNER_YEARS or seed not in SEEDS:
        raise ValueError("Cell is outside the frozen inner year/seed grid")
    if cell_complete(output_dir, inner_year, seed) and not force:
        return {"status": "skipped_complete", "inner_year": inner_year, "seed": seed}
    frame, numeric_features, freeze_audit = load_development_frame()
    prepared = prepare_inner_fold(
        frame,
        numeric_features=numeric_features,
        validation_year=inner_year,
        seed=seed,
    )
    started = time.perf_counter()
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for variant in VARIANTS:
        metric, prediction, diagnostic = _fit_variant(
            variant,
            prepared,
            seed=seed,
        )
        metrics.append(metric)
        predictions.append(prediction)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    diagnostic_frame = pd.DataFrame(diagnostics)
    control_frame = _control_rows(prepared, seed=seed)
    integrity = {
        "passed": bool(
            len(metric_frame) == len(VARIANTS)
            and set(metric_frame["variant"]) == set(VARIANTS)
            and prediction_frame.groupby("variant").size().nunique() == 1
            and np.isfinite(prediction_frame["probability"]).all()
            and control_frame["source_before_anchor"].all()
            and control_frame.loc[
                control_frame["anchor_split"].eq("train"), "endpoint_gap_rows"
            ].ge(5).all()
            and control_frame.loc[
                control_frame["strategy"].eq("matched"), "regime_agreement"
            ].all()
        ),
        "protocol_id": PROTOCOL_ID,
        "freeze_audit": freeze_audit,
        "metric_rows": len(metric_frame),
        "prediction_rows": len(prediction_frame),
        "diagnostic_rows": len(diagnostic_frame),
        "control_pair_rows": len(control_frame),
        "common_training_sequences": len(
            prepared.matched_train_controls.anchor_indices
        ),
        "validation_sequences": len(prepared.validation.labels),
    }
    if not integrity["passed"]:
        raise RuntimeError("PIT-FCG cell failed its integrity audit")
    directory = cell_directory(output_dir, inner_year, seed)
    directory.mkdir(parents=True, exist_ok=True)
    metric_frame.to_csv(directory / "metrics.csv", index=False)
    prediction_frame.to_csv(directory / "predictions.csv", index=False)
    diagnostic_frame.to_csv(directory / "diagnostics.csv", index=False)
    control_frame.to_csv(directory / "control_pair_audit.csv", index=False)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "created_at": _utc_now(),
        "inner_year": int(inner_year),
        "seed": int(seed),
        "variants": list(VARIANTS),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "cell_wall_seconds": float(time.perf_counter() - started),
        "outer_years_accessed": [],
        "incremental_api_cost_usd": 0,
    }
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (directory / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2),
        encoding="utf-8",
    )
    if not cell_complete(output_dir, inner_year, seed):
        raise RuntimeError("PIT-FCG cell was not complete after writing artifacts")
    return {**metadata, "status": "completed"}


def _collect_cells(output_dir: Path, name: str) -> pd.DataFrame:
    paths = sorted((output_dir / CELL_ROOT).glob(f"inner_*/seed_*/{name}"))
    if not paths:
        raise FileNotFoundError(f"No PIT-FCG {name} cell files were found")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _seed_averaged_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["inner_fold", "validation_year", "variant", "Date", "y_true"]
    result = (
        predictions.groupby(keys, sort=False, as_index=False)
        .agg(
            probability=("probability", "mean"),
            gate=("gate", "mean"),
            placebo_probability=("placebo_probability", "mean"),
            placebo_gate=("placebo_gate", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    if not result["seeds"].eq(len(SEEDS)).all():
        raise ValueError("Seed-averaged predictions do not contain all five seeds")
    return result


def _fold_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for (inner_fold, year, variant), group in predictions.groupby(
        ["inner_fold", "validation_year", "variant"],
        sort=False,
    ):
        metric_rows.append(
            {
                "inner_fold": inner_fold,
                "validation_year": int(year),
                "variant": variant,
                **classification_metrics(group["y_true"], group["probability"]),
            }
        )
        if variant in FCG_VARIANTS:
            if group["placebo_probability"].isna().any():
                raise ValueError("FCG seed-averaged placebo predictions are missing")
            diagnostic_rows.append(
                {
                    "inner_fold": inner_fold,
                    "validation_year": int(year),
                    "variant": variant,
                    "aligned_balanced_accuracy": classification_metrics(
                        group["y_true"], group["probability"]
                    )["balanced_accuracy"],
                    "placebo_balanced_accuracy": classification_metrics(
                        group["y_true"], group["placebo_probability"]
                    )["balanced_accuracy"],
                    "aligned_gate_mean": float(group["gate"].mean()),
                    "aligned_gate_median": float(group["gate"].median()),
                    "placebo_gate_mean": float(group["placebo_gate"].mean()),
                    "placebo_gate_median": float(group["placebo_gate"].median()),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(diagnostic_rows)


def aggregate_experiment(*, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    incomplete = [
        (year, seed)
        for year in INNER_YEARS
        for seed in SEEDS
        if not cell_complete(output_dir, year, seed)
    ]
    if incomplete:
        raise ValueError(f"PIT-FCG inner run has incomplete cells: {incomplete}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _collect_cells(output_dir, "metrics.csv")
    predictions = _collect_cells(output_dir, "predictions.csv")
    diagnostics_by_seed = _collect_cells(output_dir, "diagnostics.csv")
    controls = _collect_cells(output_dir, "control_pair_audit.csv")
    metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    predictions.to_csv(output_dir / "predictions_by_seed.csv", index=False)
    diagnostics_by_seed.to_csv(
        output_dir / "diagnostics_by_seed.csv",
        index=False,
    )
    controls.to_csv(output_dir / "control_pair_audit.csv", index=False)

    averaged = _seed_averaged_predictions(predictions)
    averaged.to_csv(output_dir / "predictions_seed_averaged.csv", index=False)
    fold_metrics, fold_diagnostics = _fold_tables(averaged)
    fold_metrics.to_csv(output_dir / "inner_fold_metrics.csv", index=False)
    fold_diagnostics.to_csv(
        output_dir / "inner_fold_falsification_diagnostics.csv",
        index=False,
    )
    summary = (
        fold_metrics.groupby("variant", sort=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            direction_accuracy_mean=("direction_accuracy", "mean"),
            mcc_mean=("mcc", "mean"),
            binary_crossentropy_mean=("binary_crossentropy", "mean"),
            brier_score_mean=("brier_score", "mean"),
            inner_folds=("inner_fold", "nunique"),
        )
        .reindex(VARIANTS)
        .reset_index()
    )
    summary.to_csv(output_dir / "inner_summary.csv", index=False)
    runtime = (
        metrics.groupby("variant", sort=False)
        .agg(
            fits=("seed", "size"),
            fit_seconds_total=("fit_seconds", "sum"),
            fit_seconds_mean=("fit_seconds", "mean"),
            inference_seconds_total=("inference_seconds", "sum"),
            trainable_parameters=("trainable_parameters", "first"),
            training_sequences_min=("training_sequences", "min"),
            training_sequences_max=("training_sequences", "max"),
        )
        .reindex(VARIANTS)
        .reset_index()
    )
    runtime.to_csv(output_dir / "runtime_summary.csv", index=False)
    parameter_lookup = runtime.set_index("variant")["trainable_parameters"]
    parameter_increase = (
        float(parameter_lookup["matched_control_fcg"])
        - float(parameter_lookup["direct_numeric_lstm"])
    ) / float(parameter_lookup["direct_numeric_lstm"])
    integrity_passed = bool(
        len(metrics) == len(INNER_YEARS) * len(SEEDS) * len(VARIANTS)
        and set(metrics["variant"]) == set(VARIANTS)
        and controls["source_before_anchor"].all()
        and controls.loc[
            controls["strategy"].eq("matched"), "regime_agreement"
        ].all()
        and np.isfinite(averaged["probability"]).all()
    )
    promotion = evaluate_inner_promotion(
        summary,
        fold_diagnostics,
        parameter_increase_fraction=parameter_increase,
        integrity_passed=integrity_passed,
    )
    (output_dir / "promotion_decision.json").write_text(
        json.dumps(promotion, indent=2),
        encoding="utf-8",
    )
    integrity = {
        "passed": integrity_passed,
        "protocol_id": PROTOCOL_ID,
        "completed_cells": len(INNER_YEARS) * len(SEEDS),
        "metric_rows": len(metrics),
        "prediction_rows": len(predictions),
        "seed_averaged_prediction_rows": len(averaged),
        "control_pair_rows": len(controls),
        "outer_years_accessed": [],
        "parameter_increase_fraction": parameter_increase,
        "promotion_passed": bool(promotion["passed"]),
    }
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "completed_at": _utc_now(),
        "evidence_status": "post_freeze_exploratory_inner_development",
        "validation_years": list(INNER_YEARS),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "outer_years_accessed": [],
        "incremental_api_cost_usd": 0,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "tensorflow"]
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return {**metadata, "promotion": promotion, "integrity": integrity}


def run_all_cells(
    *,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict[str, int]:
    completed = skipped = 0
    total = len(INNER_YEARS) * len(SEEDS)
    index = 0
    for year in INNER_YEARS:
        for seed in SEEDS:
            index += 1
            if cell_complete(output_dir, year, seed) and not force:
                skipped += 1
                print(f"[{index}/{total}] skip inner_{year}/seed_{seed}", flush=True)
                continue
            print(f"[{index}/{total}] run inner_{year}/seed_{seed}", flush=True)
            run_cell(
                inner_year=year,
                seed=seed,
                output_dir=output_dir,
                force=force,
            )
            completed += 1
    return {"total": total, "completed": completed, "skipped": skipped}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen pre-2022 PIT-FCG-LSTM development.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    cell = subparsers.add_parser("cell")
    cell.add_argument("--inner-year", type=int, choices=INNER_YEARS, required=True)
    cell.add_argument("--seed", type=int, choices=SEEDS, required=True)
    cell.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    cell.add_argument("--force", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    run.add_argument("--force", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    if args.command == "cell":
        result = run_cell(
            inner_year=args.inner_year,
            seed=args.seed,
            output_dir=args.output_dir,
            force=args.force,
        )
    elif args.command == "run":
        result = run_all_cells(output_dir=args.output_dir, force=args.force)
    else:
        result = aggregate_experiment(output_dir=args.output_dir)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
