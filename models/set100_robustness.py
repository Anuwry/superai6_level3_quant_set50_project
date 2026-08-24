from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import roc_auc_score

from models.baseline_common import (
    binary_direction_metrics,
    regression_metrics,
)
from models.track_a_analysis import exact_sign_flip_pvalue
from models.track_c_inference import holm_adjust

PROTOCOL_ID = "set100-same-exchange-robustness-v1"
MODELS = (
    "lstm",
    "cnn",
    "lstm_cnn",
    "lstm_attention",
    "lstm_cnn_attention",
)
WINDOWS = {
    "lstm": 5,
    "cnn": 20,
    "lstm_cnn": 20,
    "lstm_attention": 10,
    "lstm_cnn_attention": 20,
}
SEEDS = (42, 123, 456, 789, 2025)
FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4")
TEST_YEARS = (2022, 2023, 2024, 2025)
FEATURE_COUNT = 122

DELTA_SPECS: dict[str, tuple[str, float, str]] = {
    "balanced_accuracy": (
        "balanced_accuracy_delta_pp",
        100.0,
        "positive_favours_set100",
    ),
    "direction_accuracy": (
        "direction_accuracy_delta_pp",
        100.0,
        "positive_favours_set100",
    ),
    "mcc": ("mcc_delta", 1.0, "positive_favours_set100"),
    "roc_auc": ("roc_auc_delta", 1.0, "positive_favours_set100"),
    "direction_coverage": (
        "direction_coverage_delta_pp",
        100.0,
        "positive_favours_set100",
    ),
    "mape": ("mape_delta_pp", 1.0, "negative_favours_set100"),
    "nrmse_percent": (
        "nrmse_percent_delta_pp",
        1.0,
        "negative_favours_set100",
    ),
    "nmae_percent": (
        "nmae_percent_delta_pp",
        1.0,
        "negative_favours_set100",
    ),
}
COMPARABLE_DELTA_METRICS = tuple(
    specification[0] for specification in DELTA_SPECS.values()
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze_manifest(
    project_root: Path,
    manifest_path: Path,
) -> dict[str, object]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_values = {
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_any_set100_model_fit",
        "set100_model_results_seen_before_freeze": False,
        "feature_count": FEATURE_COUNT,
        "models": WINDOWS,
        "seeds": list(SEEDS),
        "outer_test_years": list(TEST_YEARS),
    }
    for key, expected in required_values.items():
        if payload.get(key) != expected:
            raise ValueError(f"Frozen protocol field changed: {key}")
    checked: list[dict[str, object]] = []
    for item in payload.get("frozen_inputs", []):
        relative_path = Path(str(item["path"]))
        path = project_root / relative_path
        actual = {
            "path": relative_path.as_posix(),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        expected_bytes = int(item["bytes"])
        expected_sha = str(item["sha256"])
        matches = (
            actual["exists"]
            and actual["bytes"] == expected_bytes
            and actual["sha256"] == expected_sha
        )
        checked.append({**actual, "matches": bool(matches)})
    mismatches = [item["path"] for item in checked if not item["matches"]]
    if mismatches:
        raise ValueError(f"Frozen input mismatch: {mismatches[:5]}")
    return {
        "protocol_id": PROTOCOL_ID,
        "all_inputs_match": True,
        "checked_inputs": checked,
    }


def evaluate_robustness_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_close: np.ndarray,
) -> dict[str, float | int]:
    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)
    close_values = np.asarray(current_close, dtype=float)
    regression = regression_metrics(true_values, pred_values)
    direction = binary_direction_metrics(
        true_values,
        pred_values,
        close_values,
    )
    true_direction = np.sign(true_values - close_values)
    actual_binary = true_direction != 0
    binary_labels = (true_direction[actual_binary] > 0).astype(int)
    direction_scores = (pred_values - close_values)[actual_binary]
    if len(np.unique(binary_labels)) == 2:
        auc = float(roc_auc_score(binary_labels, direction_scores))
    else:
        auc = float("nan")
    denominator = float(np.mean(np.abs(true_values)))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Target scale is invalid for normalized errors")
    return {
        **regression,
        **direction,
        "roc_auc": auc,
        "nrmse_percent": float(regression["rmse"] / denominator * 100.0),
        "nmae_percent": float(regression["mae"] / denominator * 100.0),
    }


def average_seed_predictions(
    frames_by_seed: Mapping[int, pd.DataFrame],
) -> pd.DataFrame:
    if not frames_by_seed:
        raise ValueError("At least one seed prediction frame is required")
    required = {"Date", "Close_D", "y_true", "y_pred"}
    ordered = sorted(frames_by_seed.items())
    reference = ordered[0][1].reset_index(drop=True).copy()
    missing = sorted(required.difference(reference.columns))
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {missing}")
    reference["Date"] = pd.to_datetime(reference["Date"], errors="raise")
    predictions = [reference["y_pred"].to_numpy(dtype=float)]
    for _, raw_frame in ordered[1:]:
        frame = raw_frame.reset_index(drop=True).copy()
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Prediction frame is missing columns: {missing}")
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
        if not frame["Date"].equals(reference["Date"]):
            raise ValueError("Seed prediction dates differ")
        for column in ("Close_D", "y_true"):
            if not np.allclose(
                frame[column].to_numpy(dtype=float),
                reference[column].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("Seed prediction targets differ")
        predictions.append(frame["y_pred"].to_numpy(dtype=float))
    stacked = np.stack(predictions, axis=0)
    if not np.isfinite(stacked).all():
        raise ValueError("Seed predictions contain non-finite values")
    result = reference[["Date", "Close_D", "y_true"]].copy()
    result["y_pred"] = stacked.mean(axis=0)
    result["seeds_averaged"] = len(predictions)
    return result


def build_market_fold_deltas(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "fold", "test_year"]
    required = {*keys, "market", *DELTA_SPECS}
    missing = sorted(required.difference(fold_metrics.columns))
    if missing:
        raise ValueError(f"Fold metrics are missing columns: {missing}")
    if set(fold_metrics["market"]) != {"SET50", "SET100"}:
        raise ValueError("Fold metrics must contain SET50 and SET100")
    set50 = fold_metrics.loc[
        fold_metrics["market"].eq("SET50"),
        [*keys, *DELTA_SPECS],
    ]
    set100 = fold_metrics.loc[
        fold_metrics["market"].eq("SET100"),
        [*keys, *DELTA_SPECS],
    ]
    paired = set100.merge(
        set50,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_set100", "_set50"),
    )
    if len(paired) != len(set50) or len(paired) != len(set100):
        raise ValueError("SET50 and SET100 fold metrics do not pair completely")
    for source, (target, scale, _) in DELTA_SPECS.items():
        paired[target] = (
            paired[f"{source}_set100"] - paired[f"{source}_set50"]
        ) * scale
    return paired


def _confidence_interval(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    margin = float(t.ppf(0.975, df=len(values) - 1) * standard_error)
    return mean - margin, mean + margin


def fold_level_market_inference(deltas: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "fold", *COMPARABLE_DELTA_METRICS}
    missing = sorted(required.difference(deltas.columns))
    if missing:
        raise ValueError(f"Market deltas are missing columns: {missing}")
    rows: list[dict[str, object]] = []
    favour_lookup = {
        target: favour
        for target, _, favour in DELTA_SPECS.values()
    }
    for model, group in deltas.groupby("model", sort=False):
        if len(group) != 4 or group["fold"].nunique() != 4:
            raise ValueError(f"{model} does not contain four unique folds")
        for metric in COMPARABLE_DELTA_METRICS:
            values = group[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite market delta: {model}/{metric}")
            lower, upper = _confidence_interval(values)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "outer_folds": len(values),
                    "mean_delta": float(values.mean()),
                    "std_delta": float(values.std(ddof=1)),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "positive_folds": int(np.sum(values > 0.0)),
                    "negative_folds": int(np.sum(values < 0.0)),
                    "zero_folds": int(np.sum(values == 0.0)),
                    "exact_sign_flip_pvalue": exact_sign_flip_pvalue(values),
                    "favourable_direction": favour_lookup[metric],
                }
            )
    return pd.DataFrame(rows)


def apply_market_holm(inference: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "metric", "exact_sign_flip_pvalue"}
    missing = sorted(required.difference(inference.columns))
    if missing:
        raise ValueError(f"Inference is missing columns: {missing}")
    adjusted_groups: list[pd.DataFrame] = []
    for _, group in inference.groupby("metric", sort=False):
        values = group.copy()
        values["holm_adjusted_pvalue"] = holm_adjust(
            values["exact_sign_flip_pvalue"].to_numpy(dtype=float)
        )
        values["models_in_family"] = len(values)
        adjusted_groups.append(values)
    return pd.concat(adjusted_groups, ignore_index=True)


def validate_registered_design(cell_metrics: pd.DataFrame) -> dict[str, int]:
    required = {"model", "fold", "seed"}
    missing = sorted(required.difference(cell_metrics.columns))
    if missing:
        raise ValueError(f"Cell metrics are missing columns: {missing}")
    keys = list(zip(
        cell_metrics["model"].astype(str),
        cell_metrics["fold"].astype(str),
        cell_metrics["seed"].astype(int),
        strict=True,
    ))
    if len(keys) != len(set(keys)):
        raise ValueError("Registered design contains duplicate cells")
    expected = {
        (model, fold, seed)
        for model in MODELS
        for fold in FOLDS
        for seed in SEEDS
    }
    observed = set(keys)
    if observed != expected:
        missing_cells = sorted(expected.difference(observed))
        extra_cells = sorted(observed.difference(expected))
        raise ValueError(
            "Registered design is incomplete or contains extras: "
            f"missing={missing_cells[:3]}, extra={extra_cells[:3]}"
        )
    return {
        "expected_cells": len(expected),
        "observed_cells": len(observed),
    }
