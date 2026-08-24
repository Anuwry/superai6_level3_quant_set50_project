from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_PROJECT_ROOT_EARLY = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "numba"))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

from models.pit_fcg_development import CONTEXT_FEATURES, NEWS_FEATURES
from models.pit_fcg_runner import load_development_frame
from models.tcrc_lstm import (
    PROTOCOL_ID,
    TCRCConfig,
    VARIANTS,
    balanced_positive_weight,
    build_tcrc_lstm,
    compute_tcrc_loss,
    turning_point_targets,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tcrc_lstm_inner_development_v1"
FREEZE_FILE = PROJECT_ROOT / "test" / "tcrc_lstm_freeze_v1.json"
CELL_ROOT = "cells"
INNER_YEARS = (2020, 2021)
SEEDS = (42, 123, 456, 789, 2025)
WINDOW = 20
LSTM_WINDOW = 5
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GRADIENT_CLIP_NORM = 1.0
PROMOTION_MARGIN = 0.005


@dataclass(frozen=True)
class InnerFoldArrays:
    train_x: np.ndarray
    validation_x: np.ndarray
    train_standardized_return: np.ndarray
    validation_standardized_return: np.ndarray
    train_direction: np.ndarray
    validation_direction: np.ndarray
    train_turn: np.ndarray
    validation_turn: np.ndarray
    train_turn_valid: np.ndarray
    validation_turn_valid: np.ndarray
    train_dates: pd.DatetimeIndex
    validation_dates: pd.DatetimeIndex
    validation_current_close: np.ndarray
    validation_next_close: np.ndarray
    return_mean: float
    return_std: float
    feature_count: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze_manifest(
    project_root: Path,
    manifest_path: Path,
) -> dict[str, object]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("TCRC freeze protocol id is incorrect")
    if payload.get("result_access_at_freeze") is not False:
        raise ValueError("TCRC freeze must precede candidate result access")
    entries = payload.get("inputs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("TCRC freeze contains no registered inputs")
    root = project_root.resolve()
    for entry in entries:
        path = (root / str(entry["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("Frozen input path escapes the project root") from error
        if not path.is_file():
            raise FileNotFoundError(f"Frozen input is missing: {path}")
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"Frozen input size changed: {path}")
        if _sha256(path) != str(entry["sha256"]):
            raise ValueError(f"Frozen input hash changed: {path}")
    return {"passed": True, "files_checked": len(entries)}


def _validate_frame(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    required = {
        "Date",
        "Close_D",
        "Target_Next_Close",
        *feature_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Development frame is missing columns: {missing}")
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns must be non-empty and unique")
    result = frame.copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="raise")
    if "Label_Date" in result:
        result["Label_Date"] = pd.to_datetime(result["Label_Date"], errors="raise")
        if not (result["Label_Date"] > result["Date"]).all():
            raise ValueError("Every label date must follow its endpoint date")
    if result["Date"].duplicated().any() or not result["Date"].is_monotonic_increasing:
        raise ValueError("Development dates must be unique and increasing")
    numeric = result.loc[
        :, [*feature_columns, "Close_D", "Target_Next_Close"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Development data contain non-finite values")
    return result.reset_index(drop=True)


def _windowed(values: np.ndarray, *, window: int) -> np.ndarray:
    if values.ndim != 2 or len(values) < window:
        raise ValueError("Insufficient two-dimensional features for windowing")
    return np.stack(
        [values[index - window + 1 : index + 1] for index in range(window - 1, len(values))]
    ).astype(np.float32)


def _return_percent(current: np.ndarray, future: np.ndarray) -> np.ndarray:
    if np.any(current == 0.0):
        raise ValueError("Current close must be non-zero")
    return ((future - current) / current * 100.0).astype(np.float64)


def build_inner_fold_arrays(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    validation_year: int,
    window: int = WINDOW,
) -> InnerFoldArrays:
    features = tuple(str(column) for column in feature_columns)
    validated = _validate_frame(frame, features)
    if isinstance(validation_year, bool) or validation_year < 2:
        raise ValueError("validation_year must be an integer year")
    if isinstance(window, bool) or window < 2:
        raise ValueError("window must be at least two")
    years = validated["Date"].dt.year
    pre_validation = validated.loc[years < validation_year].reset_index(drop=True)
    validation = validated.loc[years == validation_year].reset_index(drop=True)
    if len(pre_validation) < window or validation.empty:
        raise ValueError("Inner fold has insufficient train or validation rows")
    if pre_validation["Date"].max() >= validation["Date"].min():
        raise ValueError("Inner train and validation periods overlap")
    train = pre_validation
    if "Label_Date" in pre_validation:
        train = pre_validation.loc[
            pre_validation["Label_Date"] < validation["Date"].min()
        ].reset_index(drop=True)
    if len(train) < window:
        raise ValueError("Label-date purge leaves insufficient training rows")

    scaler = StandardScaler().fit(train.loc[:, features])
    train_features = scaler.transform(train.loc[:, features])
    boundary_features = scaler.transform(pre_validation.loc[:, features])
    validation_features = scaler.transform(validation.loc[:, features])
    train_x = _windowed(train_features, window=window)
    history = np.vstack([boundary_features[-(window - 1) :], validation_features])
    validation_x = _windowed(history, window=window)
    endpoint = np.arange(window - 1, len(train), dtype=int)

    train_close_all = train["Close_D"].to_numpy(dtype=np.float64)
    train_next_all = train["Target_Next_Close"].to_numpy(dtype=np.float64)
    validation_close = validation["Close_D"].to_numpy(dtype=np.float64)
    validation_next = validation["Target_Next_Close"].to_numpy(dtype=np.float64)
    train_return_all = _return_percent(train_close_all, train_next_all)
    fitted_returns = train_return_all[endpoint]
    return_mean = float(np.mean(fitted_returns))
    return_std = float(np.std(fitted_returns, ddof=0))
    if not np.isfinite(return_std) or return_std <= 0.0:
        raise ValueError("Training return scale is degenerate")
    validation_return = _return_percent(validation_close, validation_next)

    train_turn_all, train_turn_valid_all = turning_point_targets(
        train_close_all,
        train_next_all,
        previous_close=float(train_close_all[0]),
    )
    validation_turn, validation_turn_valid = turning_point_targets(
        validation_close,
        validation_next,
        previous_close=float(pre_validation["Close_D"].iloc[-1]),
    )
    return InnerFoldArrays(
        train_x=train_x,
        validation_x=validation_x,
        train_standardized_return=((fitted_returns - return_mean) / return_std).astype(np.float32),
        validation_standardized_return=((validation_return - return_mean) / return_std).astype(np.float32),
        train_direction=(fitted_returns > 0.0).astype(np.float32),
        validation_direction=(validation_return > 0.0).astype(np.float32),
        train_turn=train_turn_all[endpoint].astype(np.float32),
        validation_turn=validation_turn.astype(np.float32),
        train_turn_valid=train_turn_valid_all[endpoint],
        validation_turn_valid=validation_turn_valid,
        train_dates=pd.DatetimeIndex(train.loc[endpoint, "Date"]),
        validation_dates=pd.DatetimeIndex(validation["Date"]),
        validation_current_close=validation_close,
        validation_next_close=validation_next,
        return_mean=return_mean,
        return_std=return_std,
        feature_count=len(features),
    )


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _tensor_targets(arrays: InnerFoldArrays, indices: np.ndarray, device: Any) -> dict[str, Any]:
    import torch

    return {
        "standardized_return": torch.as_tensor(
            arrays.train_standardized_return[indices], dtype=torch.float32, device=device
        ),
        "direction": torch.as_tensor(
            arrays.train_direction[indices], dtype=torch.float32, device=device
        ),
        "turn": torch.as_tensor(
            arrays.train_turn[indices], dtype=torch.float32, device=device
        ),
        "turn_valid": torch.as_tensor(
            arrays.train_turn_valid[indices], dtype=torch.bool, device=device
        ),
    }


def _fit_variant(
    arrays: InnerFoldArrays,
    *,
    variant: str,
    seed: int,
    config: TCRCConfig,
) -> tuple[dict[str, np.ndarray], dict[str, float | int | str]]:
    import torch

    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tcrc_lstm(
        input_features=arrays.feature_count,
        variant=variant,
        return_mean=arrays.return_mean,
        return_std=arrays.return_std,
        config=config,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    direction_weight = balanced_positive_weight(arrays.train_direction)
    valid_turn = arrays.train_turn[arrays.train_turn_valid]
    turn_weight = balanced_positive_weight(valid_turn)
    training_started = time.perf_counter()
    final_terms: Mapping[str, Any] | None = None
    model.train()
    for _ in range(EPOCHS):
        for start in range(0, len(arrays.train_x), BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, len(arrays.train_x)))
            x_batch = torch.as_tensor(
                arrays.train_x[indices], dtype=torch.float32, device=device
            )
            targets = _tensor_targets(arrays, indices, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x_batch)
            final_terms = compute_tcrc_loss(
                outputs,
                targets,
                variant=variant,
                direction_positive_weight=direction_weight,
                turn_positive_weight=turn_weight,
                config=config,
            )
            final_terms["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
    fit_seconds = float(time.perf_counter() - training_started)
    if final_terms is None:
        raise RuntimeError("TCRC training produced no loss terms")

    model.eval()
    inference_started = time.perf_counter()
    with torch.no_grad():
        validation_x = torch.as_tensor(
            arrays.validation_x, dtype=torch.float32, device=device
        )
        output = model(validation_x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = float(time.perf_counter() - inference_started)
    values = {
        key: tensor.detach().cpu().numpy()
        for key, tensor in output.items()
        if key != "attention_weights"
    }
    values["attention_entropy"] = (
        -(
            output["attention_weights"]
            * torch.log(torch.clamp(output["attention_weights"], min=1e-8))
        ).sum(dim=1)
    ).detach().cpu().numpy()
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("TCRC inference produced non-finite outputs")
    diagnostics: dict[str, float | int | str] = {
        "device": str(device),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "final_training_loss": float(final_terms["loss"].detach().cpu()),
        "training_sequences": len(arrays.train_x),
        "validation_sequences": len(arrays.validation_x),
    }
    return values, diagnostics


def _metrics(
    arrays: InnerFoldArrays,
    outputs: Mapping[str, np.ndarray],
) -> dict[str, float | int]:
    probability = np.asarray(outputs["direction_probability"], dtype=np.float64)
    raw_return = np.asarray(outputs["raw_return_percent"], dtype=np.float64)
    prediction = (probability > 0.5).astype(np.int8)
    labels = arrays.validation_direction.astype(np.int8)
    predicted_close = arrays.validation_current_close * (1.0 + raw_return / 100.0)
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    valid_turn = arrays.validation_turn_valid
    turn_prediction = np.asarray(outputs["turn_probability"] > 0.5, dtype=np.int8)
    return {
        "observations": len(labels),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "direction_accuracy": float(accuracy_score(labels, prediction)),
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "binary_crossentropy": float(log_loss(labels, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probability)),
        "rmse": float(mean_squared_error(arrays.validation_next_close, predicted_close) ** 0.5),
        "mae": float(mean_absolute_error(arrays.validation_next_close, predicted_close)),
        "predicted_up_share": float(np.mean(prediction)),
        "turning_observations": int(np.count_nonzero(valid_turn)),
        "turning_accuracy": float(
            accuracy_score(arrays.validation_turn[valid_turn], turn_prediction[valid_turn])
        ),
    }


def cell_directory(output_dir: Path, validation_year: int, seed: int) -> Path:
    return output_dir / CELL_ROOT / f"inner_{validation_year}" / f"seed_{seed}"


def run_cell(
    *,
    validation_year: int,
    seed: int,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict[str, object]:
    if validation_year not in INNER_YEARS or seed not in SEEDS:
        raise ValueError("Unknown TCRC development cell")
    directory = cell_directory(output_dir, validation_year, seed)
    integrity_path = directory / "integrity_audit.json"
    if integrity_path.is_file() and not force:
        cached = json.loads(integrity_path.read_text(encoding="utf-8"))
        if cached.get("passed") is True:
            return {"status": "skipped_complete", "year": validation_year, "seed": seed}

    freeze_audit = verify_freeze_manifest(PROJECT_ROOT, FREEZE_FILE)
    frame, numeric_features, _ = load_development_frame()
    feature_columns = tuple(dict.fromkeys((*numeric_features, *NEWS_FEATURES, *CONTEXT_FEATURES)))
    arrays = build_inner_fold_arrays(
        frame,
        feature_columns=feature_columns,
        validation_year=validation_year,
        window=WINDOW,
    )
    config = TCRCConfig(window=WINDOW, lstm_window=LSTM_WINDOW)
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    started = time.perf_counter()
    for variant in VARIANTS:
        outputs, diagnostics = _fit_variant(
            arrays,
            variant=variant,
            seed=seed,
            config=config,
        )
        predicted_close = arrays.validation_current_close * (
            1.0 + outputs["raw_return_percent"] / 100.0
        )
        metrics_rows.append(
            {
                "protocol_id": PROTOCOL_ID,
                "variant": variant,
                "validation_year": validation_year,
                "seed": seed,
                **diagnostics,
                **_metrics(arrays, outputs),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "validation_year": validation_year,
                    "seed": seed,
                    "Date": arrays.validation_dates,
                    "Close_D": arrays.validation_current_close,
                    "y_true": arrays.validation_next_close,
                    "y_pred": predicted_close,
                    "true_direction": arrays.validation_direction.astype(np.int8),
                    "probability": outputs["direction_probability"],
                    "predicted_return_percent": outputs["raw_return_percent"],
                    "true_turn": arrays.validation_turn.astype(np.int8),
                    "turn_valid": arrays.validation_turn_valid,
                    "turn_probability": outputs["turn_probability"],
                    "gate": outputs["gate"],
                    "correction": outputs["correction"],
                    "attention_entropy": outputs["attention_entropy"],
                }
            )
        )

    metrics = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    passed = bool(
        len(metrics) == len(VARIANTS)
        and set(metrics["variant"]) == set(VARIANTS)
        and predictions.groupby("variant").size().nunique() == 1
        and np.isfinite(
            predictions[["y_pred", "probability", "turn_probability", "gate"]]
        ).all().all()
        and arrays.train_dates.max() < arrays.validation_dates.min()
        and freeze_audit["passed"]
    )
    directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(directory / "metrics.csv", index=False)
    predictions.to_csv(directory / "predictions.csv", index=False)
    audit = {
        "passed": passed,
        "protocol_id": PROTOCOL_ID,
        "freeze_verified": bool(freeze_audit["passed"]),
        "train_precedes_validation": bool(arrays.train_dates.max() < arrays.validation_dates.min()),
        "finite_predictions": True,
        "variants": len(VARIANTS),
        "feature_count": arrays.feature_count,
    }
    integrity_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "created_at": _utc_now(),
        "validation_year": validation_year,
        "seed": seed,
        "config": asdict(config),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "shuffle": False,
        "wall_seconds": float(time.perf_counter() - started),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if not passed:
        raise RuntimeError("TCRC cell integrity failed")
    return {"status": "completed", "year": validation_year, "seed": seed, "wall_seconds": metadata["wall_seconds"]}


def _seed_averaged_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["variant", "validation_year", "Date"]
    actual = ["Close_D", "y_true", "true_direction", "true_turn", "turn_valid"]
    for column in actual:
        if predictions.groupby(keys)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Seed predictions disagree on {column}")
    averaged = predictions.groupby(keys, as_index=False, sort=False).agg(
        Close_D=("Close_D", "first"),
        y_true=("y_true", "first"),
        true_direction=("true_direction", "first"),
        true_turn=("true_turn", "first"),
        turn_valid=("turn_valid", "first"),
        y_pred=("y_pred", "mean"),
        probability=("probability", "mean"),
        predicted_return_percent=("predicted_return_percent", "mean"),
        turn_probability=("turn_probability", "mean"),
        gate=("gate", "mean"),
        correction=("correction", "mean"),
        attention_entropy=("attention_entropy", "mean"),
        seeds=("seed", "nunique"),
    )
    if not averaged["seeds"].eq(len(SEEDS)).all():
        raise ValueError("Seed averaging is incomplete")
    return averaged


def _fold_metrics_from_averaged(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, year), group in predictions.groupby(
        ["variant", "validation_year"], sort=False
    ):
        labels = group["true_direction"].to_numpy(dtype=np.int8)
        prediction = (
            group["y_pred"].to_numpy(dtype=np.float64)
            > group["Close_D"].to_numpy(dtype=np.float64)
        ).astype(np.int8)
        valid_turn = group["turn_valid"].astype(bool).to_numpy()
        turn_prediction = (group["turn_probability"].to_numpy() > 0.5).astype(np.int8)
        rows.append(
            {
                "variant": variant,
                "validation_year": int(year),
                "observations": len(group),
                "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
                "direction_accuracy": float(accuracy_score(labels, prediction)),
                "mcc": float(matthews_corrcoef(labels, prediction)),
                "rmse": float(mean_squared_error(group["y_true"], group["y_pred"]) ** 0.5),
                "mae": float(mean_absolute_error(group["y_true"], group["y_pred"])),
                "turning_accuracy": float(
                    accuracy_score(group.loc[valid_turn, "true_turn"], turn_prediction[valid_turn])
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_promotion(
    fold_metrics: pd.DataFrame,
    *,
    integrity_passed: bool,
) -> dict[str, object]:
    required = {"variant", "validation_year", "balanced_accuracy"}
    if not required.issubset(fold_metrics.columns):
        raise ValueError("Fold metrics are missing promotion columns")
    if set(fold_metrics["variant"].astype(str)) != set(VARIANTS):
        raise ValueError("Promotion metrics do not contain all TCRC variants")
    pivot = fold_metrics.pivot(
        index="validation_year", columns="variant", values="balanced_accuracy"
    )
    if set(pivot.index.astype(int)) != set(INNER_YEARS):
        raise ValueError("Promotion requires both inner years")
    full = pivot["tcrc_full"]
    ablations = pivot.loc[:, [variant for variant in VARIANTS if variant != "tcrc_full"]]
    full_beats_each_year = bool((ablations.lt(full, axis=0)).all().all())
    margin = float(full.mean() - pivot["lstm_anchor"].mean())
    conditions = {
        "full_beats_every_ablation_each_inner_year": full_beats_each_year,
        "mean_margin_over_lstm_anchor_at_least_0_5pp": margin >= PROMOTION_MARGIN,
        "integrity_passed": bool(integrity_passed),
    }
    return {
        "passed": bool(all(conditions.values())),
        "conditions": conditions,
        "mean_balanced_accuracy": {column: float(pivot[column].mean()) for column in pivot},
        "full_minus_anchor_pp": margin * 100.0,
    }


def aggregate_experiment(*, output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    metric_paths = sorted((output_dir / CELL_ROOT).glob("inner_*/seed_*/metrics.csv"))
    prediction_paths = sorted((output_dir / CELL_ROOT).glob("inner_*/seed_*/predictions.csv"))
    expected_cells = len(INNER_YEARS) * len(SEEDS)
    if len(metric_paths) != expected_cells or len(prediction_paths) != expected_cells:
        raise ValueError("TCRC development cells are incomplete")
    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    predictions = pd.concat([pd.read_csv(path) for path in prediction_paths], ignore_index=True)
    integrity_paths = sorted((output_dir / CELL_ROOT).glob("inner_*/seed_*/integrity_audit.json"))
    integrity_passed = bool(
        len(integrity_paths) == expected_cells
        and all(json.loads(path.read_text(encoding="utf-8"))["passed"] for path in integrity_paths)
    )
    averaged = _seed_averaged_predictions(predictions)
    fold_metrics = _fold_metrics_from_averaged(averaged)
    summary = fold_metrics.groupby("variant", sort=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        mcc_mean=("mcc", "mean"),
        rmse_mean=("rmse", "mean"),
        mae_mean=("mae", "mean"),
        turning_accuracy_mean=("turning_accuracy", "mean"),
    ).reset_index()
    promotion = evaluate_promotion(fold_metrics, integrity_passed=integrity_passed)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    predictions.to_csv(output_dir / "predictions_by_seed.csv", index=False)
    averaged.to_csv(output_dir / "predictions_seed_averaged.csv", index=False)
    fold_metrics.to_csv(output_dir / "inner_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "inner_summary.csv", index=False)
    (output_dir / "promotion_decision.json").write_text(
        json.dumps(promotion, indent=2), encoding="utf-8"
    )
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "created_at": _utc_now(),
        "completed_cells": expected_cells,
        "metric_rows": len(metrics),
        "prediction_rows": len(predictions),
        "integrity_passed": integrity_passed,
        "promotion_passed": promotion["passed"],
        "seeds": list(SEEDS),
        "inner_years": list(INNER_YEARS),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {**metadata, "promotion": promotion}


def run_all_cells(
    *,
    years: Iterable[int] = INNER_YEARS,
    seeds: Iterable[int] = SEEDS,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict[str, object]:
    for year in tuple(years):
        for seed in tuple(seeds):
            result = run_cell(
                validation_year=int(year),
                seed=int(seed),
                output_dir=output_dir,
                force=force,
            )
            print(json.dumps(result), flush=True)
    return aggregate_experiment(output_dir=output_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen TCRC-LSTM development")
    parser.add_argument("command", choices=("cell", "run", "aggregate"))
    parser.add_argument("--year", type=int, choices=INNER_YEARS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if args.command == "cell":
        if args.year is None or args.seed is None:
            raise ValueError("cell requires --year and --seed")
        return run_cell(validation_year=args.year, seed=args.seed, force=args.force)
    if args.command == "aggregate":
        return aggregate_experiment()
    return run_all_cells(force=args.force)


if __name__ == "__main__":
    main()
