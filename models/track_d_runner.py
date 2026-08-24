from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

from models.baseline_common import (
    CLOSE_COLUMN,
    DATE_COLUMN,
    PROJECT_ROOT,
    TARGET_COLUMN,
    discover_folds,
    load_fold,
    package_versions,
    sequence_history_features,
)
from models.convolutional_neural_network import make_test_sequences
from models.track_a_data import (
    FULL_TA_VMD_POINT_IN_TIME_SELECTION_DIR,
    FULL_TA_VMD_POINT_IN_TIME_SELECTION_NN_DIR,
)
from models.track_a_final import TRACK_A_MODELS
from models.track_d_economics import (
    annualized_sharpe,
    backtest_positions,
    deflated_sharpe_ratio,
    positions_from_probabilities,
)
from models.track_d_models import (
    build_track_d_model,
    make_direction_sequences,
    standardize_return_targets,
)
from models.track_d_protocol import (
    TrackDConfig,
    direction_labels,
    verify_freeze_manifest,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_d_q2"
CELL_DIR = OUTPUT_DIR / "cells"
FORWARD_ORIGINAL_DIR = PROJECT_ROOT / "data-track-d" / "forward_2026"
FORWARD_SCALED_DIR = PROJECT_ROOT / "data-track-d" / "forward_2026_nn"
SELECTION_SEED = 42


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def binary_probability_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | int]:
    target = np.asarray(y_true, dtype=float).reshape(-1)
    score = np.asarray(probability, dtype=float).reshape(-1)
    if target.shape != score.shape or len(target) < 2:
        raise ValueError("Binary metric inputs have invalid shapes")
    if not np.isfinite(target).all() or not np.isfinite(score).all():
        raise ValueError("Binary metric inputs must be finite")
    if not set(np.unique(target)).issubset({0.0, 1.0}):
        raise ValueError("Binary targets must contain only zero and one")
    if np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("Probabilities must be in [0, 1]")
    predicted = (score >= 0.5).astype(int)
    auc = (
        float(roc_auc_score(target, score))
        if len(np.unique(target)) == 2
        else float("nan")
    )
    clipped = np.clip(score, 1e-7, 1.0 - 1e-7)
    return {
        "rows": len(target),
        "positive_share": float(target.mean()),
        "direction_accuracy": float(np.mean(predicted == target)),
        "balanced_accuracy": float(
            balanced_accuracy_score(target, predicted)
        ),
        "mcc": float(matthews_corrcoef(target, predicted)),
        "auc": auc,
        "brier": float(brier_score_loss(target, score)),
        "log_loss": float(log_loss(target, clipped, labels=[0, 1])),
        "ece_10": expected_calibration_error(target, score, bins=10),
    }


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int,
) -> float:
    target = np.asarray(y_true, dtype=float).reshape(-1)
    score = np.asarray(probability, dtype=float).reshape(-1)
    if bins < 2 or target.shape != score.shape:
        raise ValueError("Calibration inputs are invalid")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.digitize(score, edges[1:-1]), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(score[mask].mean()) - float(target[mask].mean())
            )
    return float(error)


def next_session_implementation_returns(frame: pd.DataFrame) -> pd.DataFrame:
    required = {DATE_COLUMN, "Open_D", CLOSE_COLUMN, TARGET_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Implementation-return frame lacks columns: {missing}")
    result = pd.DataFrame({DATE_COLUMN: frame[DATE_COLUMN].astype(str)})
    result["next_open"] = frame["Open_D"].shift(-1).to_numpy(dtype=float)
    target = frame[TARGET_COLUMN].to_numpy(dtype=float)
    current = frame[CLOSE_COLUMN].to_numpy(dtype=float)
    next_open = result["next_open"].to_numpy(dtype=float)
    eligible = np.isfinite(next_open) & (next_open > 0.0)
    result["eligible"] = eligible
    result["implementation_return"] = np.where(
        eligible,
        target / next_open - 1.0,
        np.nan,
    )
    result["idealized_close_return"] = target / current - 1.0
    return result


def seed_average_predictions(
    frames_by_seed: Mapping[int, pd.DataFrame],
) -> pd.DataFrame:
    if not frames_by_seed:
        raise ValueError("At least one seed prediction frame is required")
    seeds = sorted(frames_by_seed)
    reference = frames_by_seed[seeds[0]].reset_index(drop=True).copy()
    key_columns = [DATE_COLUMN, "y_true"]
    for seed in seeds[1:]:
        candidate = frames_by_seed[seed].reset_index(drop=True)
        if len(candidate) != len(reference) or not candidate[
            key_columns
        ].equals(reference[key_columns]):
            raise ValueError("Seed prediction keys do not align")
    averaged = reference.drop(columns=["probability"]).copy()
    averaged["probability"] = np.mean(
        [frames_by_seed[seed]["probability"].to_numpy(dtype=float) for seed in seeds],
        axis=0,
    )
    if all("predicted_return" in frames_by_seed[seed] for seed in seeds):
        averaged["predicted_return"] = np.mean(
            [
                frames_by_seed[seed]["predicted_return"].to_numpy(dtype=float)
                for seed in seeds
            ],
            axis=0,
        )
    averaged["seeds_averaged"] = len(seeds)
    return averaged


def _validate_keys(keys: Iterable[str], allowed: Iterable[str]) -> tuple[str, ...]:
    values = tuple(keys)
    unknown = sorted(set(values).difference(allowed))
    if not values or unknown:
        raise ValueError(f"Invalid registered keys: {unknown}")
    return values


def _fit_predict_fold(
    *,
    model_key: str,
    objective: str,
    scaled_fold,
    original_fold,
    window: int,
    seed: int,
    return_loss_weight: float,
    weights_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()

    features = tuple(scaled_fold.feature_columns)
    if features != tuple(original_fold.feature_columns):
        raise ValueError("Scaled and original feature pools differ")
    train_features = scaled_fold.train.loc[:, list(features)].to_numpy(
        dtype=float
    )
    train_labels, train_eligible = direction_labels(
        original_fold.train[TARGET_COLUMN].to_numpy(dtype=float),
        original_fold.train[CLOSE_COLUMN].to_numpy(dtype=float),
    )
    train_returns = np.log(
        original_fold.train[TARGET_COLUMN].to_numpy(dtype=float)
        / original_fold.train[CLOSE_COLUMN].to_numpy(dtype=float)
    )
    x_train, y_direction = make_direction_sequences(
        train_features,
        train_labels,
        window=window,
    )
    endpoint_eligible = train_eligible[window - 1 :]
    endpoint_returns = train_returns[window - 1 :]
    x_fit = x_train[endpoint_eligible]
    y_direction_fit = y_direction[endpoint_eligible]
    return_scaled, _, return_scaler = standardize_return_targets(
        endpoint_returns[endpoint_eligible],
        np.array([0.0]),
    )
    test_sequences = make_test_sequences(
        sequence_history_features(scaled_fold),
        scaled_fold.test.loc[:, list(features)].to_numpy(dtype=float),
        window,
    )
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_track_d_model(
        model_key,
        input_shape=(window, len(features)),
        objective=objective,
        return_loss_weight=return_loss_weight,
    )
    parameters = TRACK_A_MODELS[model_key].parameters
    fit_started = time.perf_counter()
    fit_target: np.ndarray | dict[str, np.ndarray]
    if objective == "direct":
        fit_target = y_direction_fit
    else:
        fit_target = {
            "direction": y_direction_fit,
            "return": return_scaled,
        }
    model.fit(
        x_fit,
        fit_target,
        epochs=int(parameters["epochs"]),
        batch_size=int(parameters["batch_size"]),
        shuffle=False,
        verbose=0,
    )
    fit_seconds = float(time.perf_counter() - fit_started)
    inference_started = time.perf_counter()
    raw_prediction = model.predict(test_sequences, verbose=0)
    inference_seconds = float(time.perf_counter() - inference_started)
    if objective == "direct":
        probability = np.asarray(raw_prediction, dtype=float).reshape(-1)
        predicted_return = np.full(len(probability), np.nan)
    else:
        probability = np.asarray(
            raw_prediction["direction"], dtype=float
        ).reshape(-1)
        scaled_return = np.asarray(raw_prediction["return"], dtype=float).reshape(-1)
        predicted_return = (
            scaled_return * return_scaler["std"] + return_scaler["mean"]
        )
    test_labels, test_eligible = direction_labels(
        original_fold.test[TARGET_COLUMN].to_numpy(dtype=float),
        original_fold.test[CLOSE_COLUMN].to_numpy(dtype=float),
    )
    implementation = next_session_implementation_returns(original_fold.test)
    prediction = pd.DataFrame(
        {
            DATE_COLUMN: original_fold.test[DATE_COLUMN].astype(str),
            "Label_Date": original_fold.test["Label_Date"].astype(str),
            CLOSE_COLUMN: original_fold.test[CLOSE_COLUMN].to_numpy(dtype=float),
            "Open_D": original_fold.test["Open_D"].to_numpy(dtype=float),
            TARGET_COLUMN: original_fold.test[TARGET_COLUMN].to_numpy(dtype=float),
            "y_true": test_labels,
            "direction_eligible": test_eligible,
            "probability": probability,
            "predicted_return": predicted_return,
            "next_open": implementation["next_open"],
            "economic_eligible": implementation["eligible"],
            "implementation_return": implementation["implementation_return"],
            "idealized_close_return": implementation["idealized_close_return"],
        }
    )
    if not np.isfinite(probability).all() or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("Model produced invalid direction probabilities")
    if weights_path is not None:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(weights_path)
    metrics = {
        "model": model_key,
        "objective": objective,
        "seed": seed,
        "window": window,
        "features": len(features),
        "training_sequences": len(x_fit),
        "trainable_parameters": int(model.count_params()),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "return_target_mean": return_scaler["mean"],
        "return_target_std": return_scaler["std"],
        **binary_probability_metrics(
            test_labels[test_eligible],
            probability[test_eligible],
        ),
    }
    return prediction, metrics


def _selection_specs(years: tuple[int, ...]):
    scaled = {
        spec.test_year: spec
        for spec in discover_folds(FULL_TA_VMD_POINT_IN_TIME_SELECTION_NN_DIR)
        if spec.test_year in years
    }
    original = {
        spec.test_year: spec
        for spec in discover_folds(FULL_TA_VMD_POINT_IN_TIME_SELECTION_DIR)
        if spec.test_year in years
    }
    if set(scaled) != set(years) or set(original) != set(years):
        raise ValueError("Track D selection folds are incomplete")
    return scaled, original


def run_validation(
    *,
    model_keys: Iterable[str] | None = None,
    objectives: Iterable[str] | None = None,
    force: bool = False,
) -> dict[str, object]:
    config = TrackDConfig()
    models = _validate_keys(
        config.models if model_keys is None else model_keys,
        config.models,
    )
    objective_keys = _validate_keys(
        config.objectives if objectives is None else objectives,
        config.objectives,
    )
    scaled_specs, original_specs = _selection_specs(config.selection_years)
    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict[str, object]] = []
    cached_cells = 0
    computed_cells = 0
    started = time.perf_counter()
    for model_key in models:
        for objective in objective_keys:
            for year in config.selection_years:
                cell_dir = CELL_DIR / "validation" / model_key / objective / str(year)
                prediction_path = cell_dir / "predictions.csv"
                metric_path = cell_dir / "metrics.json"
                if prediction_path.is_file() and metric_path.is_file() and not force:
                    prediction = pd.read_csv(prediction_path)
                    metric = json.loads(metric_path.read_text(encoding="utf-8"))
                    cached_cells += 1
                else:
                    prediction, metric = _fit_predict_fold(
                        model_key=model_key,
                        objective=objective,
                        scaled_fold=load_fold(scaled_specs[year]),
                        original_fold=load_fold(original_specs[year]),
                        window=int(config.windows[model_key]),
                        seed=SELECTION_SEED,
                        return_loss_weight=config.return_loss_weight,
                    )
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    prediction.to_csv(prediction_path, index=False)
                    metric_path.write_text(
                        json.dumps({**metric, "validation_year": year}, indent=2),
                        encoding="utf-8",
                    )
                    computed_cells += 1
                prediction.insert(0, "model", model_key)
                prediction.insert(1, "objective", objective)
                prediction.insert(2, "validation_year", year)
                all_predictions.append(prediction)
                all_metrics.append({**metric, "validation_year": year})
    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.DataFrame(all_metrics)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    gate_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for (model, objective), group in predictions.groupby(
        ["model", "objective"], sort=False
    ):
        selection_candidates: list[dict[str, object]] = []
        for threshold in config.confidence_thresholds:
            yearly_net = []
            yearly_coverage = []
            for year, year_frame in group.groupby("validation_year", sort=False):
                eligible = year_frame["economic_eligible"].astype(bool)
                frame = year_frame.loc[eligible].reset_index(drop=True)
                positions = positions_from_probabilities(
                    frame["probability"].to_numpy(dtype=float),
                    threshold=threshold,
                    strategy="long_short",
                )
                backtest = backtest_positions(
                    positions,
                    frame["implementation_return"].to_numpy(dtype=float),
                    cost_bps=config.primary_cost_bps,
                )
                yearly_net.append(float(backtest["net_return"].sum()))
                yearly_coverage.append(float(np.mean(positions != 0.0)))
                gate_rows.append(
                    {
                        "model": model,
                        "objective": objective,
                        "validation_year": year,
                        "threshold": threshold,
                        "coverage": yearly_coverage[-1],
                        "net_return": yearly_net[-1],
                        "net_sharpe": annualized_sharpe(
                            backtest["net_return"].to_numpy(dtype=float)
                        ),
                    }
                )
            selection_candidates.append(
                {
                    "threshold": threshold,
                    "mean_net_return": float(np.mean(yearly_net)),
                    "coverage": float(np.mean(yearly_coverage)),
                    "positive_years": int(np.sum(np.asarray(yearly_net) > 0.0)),
                }
            )
        from models.track_d_protocol import select_confidence_threshold

        selected = select_confidence_threshold(
            selection_candidates,
            minimum_coverage=config.minimum_coverage,
            minimum_positive_years=config.minimum_positive_years,
        )
        candidate = next(
            row for row in selection_candidates if row["threshold"] == selected
        )
        selected_rows.append(
            {
                "model": model,
                "objective": objective,
                "selected_threshold": selected,
                **{f"selected_{key}": value for key, value in candidate.items() if key != "threshold"},
                "fallback_to_0_50": not any(
                    row["coverage"] >= config.minimum_coverage
                    and row["positive_years"] >= config.minimum_positive_years
                    for row in selection_candidates
                ),
            }
        )
    pd.DataFrame(gate_rows).to_csv(
        OUTPUT_DIR / "threshold_selection.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(
        OUTPUT_DIR / "selected_thresholds.csv", index=False
    )
    metadata = {
        "protocol_version": config.protocol_version,
        "completed_at": _utc_now(),
        "models": list(models),
        "objectives": list(objective_keys),
        "selection_years": list(config.selection_years),
        "seed": SELECTION_SEED,
        "runtime_seconds": float(time.perf_counter() - started),
        "computed_cells": computed_cells,
        "cached_cells": cached_cells,
        "cell_fit_inference_seconds_sum": float(
            metrics["fit_seconds"].sum() + metrics["inference_seconds"].sum()
        ),
        "forward_data_accessed": False,
        "deterministic_environment": {
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            "TF_ENABLE_ONEDNN_OPTS": os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
        },
    }
    (OUTPUT_DIR / "validation_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def _forward_specs():
    scaled = discover_folds(FORWARD_SCALED_DIR)
    original = discover_folds(FORWARD_ORIGINAL_DIR)
    if len(scaled) != 1 or len(original) != 1:
        raise ValueError("Expected exactly one Track D forward fold")
    if scaled[0].test_year != 2026 or original[0].test_year != 2026:
        raise ValueError("Track D forward fold must test 2026")
    return scaled[0], original[0]


def _summarize_backtest(frame: pd.DataFrame) -> dict[str, float | int]:
    net = frame["net_return"].to_numpy(dtype=float)
    gross = frame["gross_return"].to_numpy(dtype=float)
    position = frame["position"].to_numpy(dtype=float)
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    active = position != 0.0
    round_trip_units = float(frame["round_trip_units"].sum())
    return {
        "rows": len(frame),
        "coverage": float(active.mean()),
        "round_trip_units": round_trip_units,
        "position_change_units": float(frame["position_change"].sum()),
        "gross_cumulative_return": float(np.prod(1.0 + gross) - 1.0),
        "net_cumulative_return": float(equity[-1] - 1.0),
        "net_mean_daily_return": float(net.mean()),
        "net_annualized_sharpe": annualized_sharpe(net),
        "maximum_drawdown": float(drawdown.min()),
        "active_win_rate": float(np.mean(net[active] > 0.0)) if active.any() else float("nan"),
        "trade_entries": int(active.sum()),
        "break_even_cost_bps": (
            float(gross.sum() * 10_000.0 / round_trip_units)
            if round_trip_units > 0.0
            else float("nan")
        ),
    }


def run_forward(
    *,
    model_keys: Iterable[str] | None = None,
    objectives: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    force: bool = False,
) -> dict[str, object]:
    config = TrackDConfig()
    models = _validate_keys(config.models if model_keys is None else model_keys, config.models)
    objective_keys = _validate_keys(
        config.objectives if objectives is None else objectives, config.objectives
    )
    seed_values = tuple(config.seeds if seeds is None else seeds)
    if not seed_values or not set(seed_values).issubset(config.seeds):
        raise ValueError("Forward seeds must be registered")
    freeze_path = OUTPUT_DIR / "freeze_manifest.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("Track D freeze manifest is required")
    verify_freeze_manifest(freeze_path)
    selected_thresholds_path = OUTPUT_DIR / "selected_thresholds.csv"
    if not selected_thresholds_path.is_file():
        raise FileNotFoundError("Validation-frozen thresholds are required")
    thresholds = pd.read_csv(selected_thresholds_path).set_index(
        ["model", "objective"]
    )
    scaled_spec, original_spec = _forward_specs()
    scaled_fold = load_fold(scaled_spec)
    original_fold = load_fold(original_spec)
    all_seed_predictions: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for model_key in models:
        for objective in objective_keys:
            for seed in seed_values:
                cell_dir = CELL_DIR / "forward" / model_key / objective / f"seed_{seed}"
                prediction_path = cell_dir / "predictions.csv"
                metric_path = cell_dir / "metrics.json"
                weights_path = cell_dir / "model.weights.h5"
                if prediction_path.is_file() and metric_path.is_file() and not force:
                    prediction = pd.read_csv(prediction_path)
                    metric = json.loads(metric_path.read_text(encoding="utf-8"))
                else:
                    prediction, metric = _fit_predict_fold(
                        model_key=model_key,
                        objective=objective,
                        scaled_fold=scaled_fold,
                        original_fold=original_fold,
                        window=int(config.windows[model_key]),
                        seed=int(seed),
                        return_loss_weight=config.return_loss_weight,
                        weights_path=(
                            weights_path
                            if objective == "direct" and seed == SELECTION_SEED
                            else None
                        ),
                    )
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    prediction.to_csv(prediction_path, index=False)
                    metric_path.write_text(
                        json.dumps({**metric, "stage": "forward_2026"}, indent=2),
                        encoding="utf-8",
                    )
                prediction.insert(0, "model", model_key)
                prediction.insert(1, "objective", objective)
                prediction.insert(2, "seed", seed)
                all_seed_predictions.append(prediction)
                runtime_rows.append({**metric, "stage": "forward_2026"})
    seed_predictions = pd.concat(all_seed_predictions, ignore_index=True)
    seed_predictions.to_csv(
        OUTPUT_DIR / "forward_predictions_by_seed.csv", index=False
    )
    averaged_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for (model, objective), group in seed_predictions.groupby(
        ["model", "objective"], sort=False
    ):
        frames = {
            int(seed): seed_frame.drop(columns=["model", "objective", "seed"])
            for seed, seed_frame in group.groupby("seed", sort=False)
        }
        averaged = seed_average_predictions(frames)
        averaged.insert(0, "model", model)
        averaged.insert(1, "objective", objective)
        averaged_frames.append(averaged)
        eligible = averaged["direction_eligible"].astype(bool)
        metric_rows.append(
            {
                "model": model,
                "objective": objective,
                "seeds": len(frames),
                **binary_probability_metrics(
                    averaged.loc[eligible, "y_true"].to_numpy(dtype=float),
                    averaged.loc[eligible, "probability"].to_numpy(dtype=float),
                ),
            }
        )
    averaged = pd.concat(averaged_frames, ignore_index=True)
    averaged.to_csv(
        OUTPUT_DIR / "forward_predictions_seed_averaged.csv", index=False
    )
    pd.DataFrame(metric_rows).to_csv(
        OUTPUT_DIR / "forward_metrics.csv", index=False
    )
    daily_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    selective_rows: list[dict[str, object]] = []
    for (model, objective), group in averaged.groupby(
        ["model", "objective"], sort=False
    ):
        selected_threshold = float(
            thresholds.loc[(model, objective), "selected_threshold"]
        )
        classification_eligible = group["direction_eligible"].astype(bool)
        economic_eligible = group["economic_eligible"].astype(bool)
        frame = group.loc[economic_eligible].reset_index(drop=True)
        for threshold in config.confidence_thresholds:
            threshold_role = (
                "validation_selected_primary"
                if threshold == selected_threshold
                else "registered_sensitivity"
            )
            probability = group["probability"].to_numpy(dtype=float)
            confident = (probability >= threshold) | (
                probability < 1.0 - threshold
            )
            selective = classification_eligible.to_numpy() & confident
            selective_metric = (
                binary_probability_metrics(
                    group.loc[selective, "y_true"].to_numpy(dtype=float),
                    group.loc[selective, "probability"].to_numpy(dtype=float),
                )
                if selective.sum() >= 2
                else {
                    key: float("nan")
                    for key in (
                        "positive_share",
                        "direction_accuracy",
                        "balanced_accuracy",
                        "mcc",
                        "auc",
                        "brier",
                        "log_loss",
                        "ece_10",
                    )
                }
            )
            selective_rows.append(
                {
                    "model": model,
                    "objective": objective,
                    "threshold": threshold,
                    "threshold_role": threshold_role,
                    "eligible_rows": int(classification_eligible.sum()),
                    "selected_rows": int(selective.sum()),
                    "coverage": float(
                        selective.sum() / classification_eligible.sum()
                    ),
                    **selective_metric,
                }
            )
            for strategy in ("long_flat", "long_short"):
                positions = positions_from_probabilities(
                    frame["probability"].to_numpy(dtype=float),
                    threshold=threshold,
                    strategy=strategy,
                )
                for cost_bps in config.cost_bps_grid:
                    backtest = backtest_positions(
                        positions,
                        frame["implementation_return"].to_numpy(dtype=float),
                        cost_bps=cost_bps,
                    )
                    daily = pd.concat(
                        [
                            frame[[DATE_COLUMN, "probability"]].reset_index(
                                drop=True
                            ),
                            backtest,
                        ],
                        axis=1,
                    )
                    daily.insert(0, "model", model)
                    daily.insert(1, "objective", objective)
                    daily.insert(2, "threshold", threshold)
                    daily.insert(3, "threshold_role", threshold_role)
                    daily.insert(4, "strategy", strategy)
                    daily.insert(5, "cost_bps", cost_bps)
                    daily_rows.append(daily)
                    summary_rows.append(
                        {
                            "model": model,
                            "objective": objective,
                            "threshold": threshold,
                            "threshold_role": threshold_role,
                            "strategy": strategy,
                            "cost_bps": cost_bps,
                            **_summarize_backtest(backtest),
                        }
                    )
    pd.DataFrame(selective_rows).to_csv(
        OUTPUT_DIR / "selective_prediction_metrics.csv", index=False
    )
    economic_daily = pd.concat(daily_rows, ignore_index=True)
    economic_summary = pd.DataFrame(summary_rows)
    sharpe_std = float(economic_summary["net_annualized_sharpe"].std(ddof=1))
    trials = len(economic_summary)
    dsr_values = []
    for row in economic_summary.itertuples(index=False):
        net = economic_daily.loc[
            economic_daily["model"].eq(row.model)
            & economic_daily["objective"].eq(row.objective)
            & economic_daily["threshold"].eq(row.threshold)
            & economic_daily["strategy"].eq(row.strategy)
            & economic_daily["cost_bps"].eq(row.cost_bps),
            "net_return",
        ].to_numpy(dtype=float)
        if len(net) < 3 or float(np.std(net, ddof=1)) <= np.finfo(float).eps:
            dsr_values.append(float("nan"))
        else:
            dsr_values.append(
                deflated_sharpe_ratio(
                    observed_sharpe=float(row.net_annualized_sharpe)
                    / np.sqrt(252.0),
                    return_count=len(net),
                    return_skewness=float(skew(net, bias=False)),
                    return_kurtosis=float(
                        kurtosis(net, fisher=False, bias=False)
                    ),
                    trials=trials,
                    sharpe_std_across_trials=sharpe_std / np.sqrt(252.0),
                )
            )
    economic_summary["deflated_sharpe_probability"] = dsr_values
    economic_daily.to_csv(OUTPUT_DIR / "economic_daily.csv", index=False)
    economic_summary.to_csv(OUTPUT_DIR / "economic_summary.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(
        OUTPUT_DIR / "runtime_summary.csv", index=False
    )
    metadata = {
        "protocol_version": config.protocol_version,
        "completed_at": _utc_now(),
        "models": list(models),
        "objectives": list(objective_keys),
        "seeds": list(seed_values),
        "forward_year": config.forward_year,
        "forward_rows": len(original_fold.test),
        "runtime_seconds": float(time.perf_counter() - started),
        "packages": package_versions(
            ["numpy", "pandas", "scipy", "scikit-learn", "tensorflow"]
        ),
        "platform": platform.platform(),
        "deterministic_environment": {
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            "TF_ENABLE_ONEDNN_OPTS": os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
        },
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen Track D experiments")
    parser.add_argument("stage", choices=("validation", "forward"))
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--objective", action="append", dest="objectives")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if args.stage == "validation":
        return run_validation(
            model_keys=args.models,
            objectives=args.objectives,
            force=args.force,
        )
    return run_forward(
        model_keys=args.models,
        objectives=args.objectives,
        seeds=args.seeds,
        force=args.force,
    )


if __name__ == "__main__":
    main()
