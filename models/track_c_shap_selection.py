from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from models.baseline_common import DATE_COLUMN
from models.point_in_time_data import LABEL_DATE_COLUMN

SHAP_SELECTION_PROTOCOL_VERSION = "track-c-shap-point-in-time-v2"


def aggregate_shap_importance(values: np.ndarray) -> np.ndarray:
    attributions = np.asarray(values, dtype=np.float64)
    if attributions.ndim != 3:
        raise ValueError("SHAP values must have sample, lag, and feature axes")
    if not np.isfinite(attributions).all():
        raise ValueError("SHAP values must be finite")
    if attributions.shape[0] < 1 or attributions.shape[1] < 1:
        raise ValueError("SHAP values must contain samples and lags")
    return np.mean(np.abs(attributions), axis=(0, 1))


def normalized_descending_ranks(importance: np.ndarray) -> np.ndarray:
    values = np.asarray(importance, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("importance must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("importance must be finite")
    if len(values) == 1:
        return np.zeros(1, dtype=np.float64)
    ranks = rankdata(-values, method="average")
    return (ranks - 1.0) / (len(values) - 1.0)


def build_consensus_ranking(records: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "selection_fold",
        "regime",
        "feature",
        "importance",
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"SHAP rank records missing columns: {missing}")
    if records.empty:
        raise ValueError("SHAP rank records are empty")
    values = records.loc[:, list(records.columns)].copy()
    if not np.isfinite(values["importance"].to_numpy(dtype=float)).all():
        raise ValueError("SHAP importance contains non-finite values")
    cell_columns = ["model", "selection_fold", "regime"]
    if values.duplicated([*cell_columns, "feature"]).any():
        raise ValueError("SHAP rank records contain duplicate cell features")

    ranked_frames: list[pd.DataFrame] = []
    expected_features: tuple[str, ...] | None = None
    for _, cell in values.groupby(cell_columns, sort=False):
        feature_set = tuple(sorted(cell["feature"].astype(str)))
        if expected_features is None:
            expected_features = feature_set
        elif feature_set != expected_features:
            raise ValueError("SHAP cells do not contain the same features")
        ranked = cell.copy()
        ranked["normalized_rank"] = normalized_descending_ranks(
            ranked["importance"].to_numpy(dtype=float)
        )
        ranked_frames.append(ranked)
    ranked_cells = pd.concat(ranked_frames, ignore_index=True)
    consensus = (
        ranked_cells.groupby(["regime", "feature"], sort=False)
        .agg(
            consensus_normalized_rank=("normalized_rank", "mean"),
            normalized_rank_std=("normalized_rank", "std"),
            contributing_cells=("normalized_rank", "size"),
        )
        .reset_index()
    )
    result_frames: list[pd.DataFrame] = []
    for _, regime_frame in consensus.groupby("regime", sort=False):
        ordered = regime_frame.sort_values(
            ["consensus_normalized_rank", "feature"],
            ascending=[True, True],
        ).reset_index(drop=True)
        ordered["consensus_rank"] = np.arange(1, len(ordered) + 1)
        result_frames.append(ordered)
    return pd.concat(result_frames, ignore_index=True)


def derive_protocol_seed(base_seed: int, *cell_parts: object) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    material = "|".join([str(base_seed), *(str(part) for part in cell_parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def purge_ranking_endpoints(
    ranking_frame: pd.DataFrame,
    *,
    first_validation_date: pd.Timestamp,
) -> np.ndarray:
    required = {DATE_COLUMN, LABEL_DATE_COLUMN}
    missing = sorted(required.difference(ranking_frame.columns))
    if missing:
        raise ValueError(f"Ranking frame missing columns: {missing}")
    if pd.isna(first_validation_date):
        raise ValueError("first_validation_date must be valid")
    label_dates = pd.to_datetime(
        ranking_frame[LABEL_DATE_COLUMN],
        errors="coerce",
    )
    if label_dates.isna().any():
        raise ValueError("Ranking Label_Date contains invalid values")
    dates = pd.to_datetime(ranking_frame[DATE_COLUMN], errors="coerce")
    if dates.isna().any():
        raise ValueError("Ranking Date contains invalid values")
    if (label_dates < dates).any():
        raise ValueError("Ranking Label_Date precedes feature Date")
    return (label_dates < pd.Timestamp(first_validation_date)).to_numpy(
        dtype=bool
    )


def select_registered_top_k(
    validation_metrics: pd.DataFrame,
    stability_by_k: pd.Series,
) -> tuple[int, pd.DataFrame]:
    required = {
        "model",
        "selection_fold",
        "top_k",
        "balanced_accuracy",
        "rmse",
    }
    missing = sorted(required.difference(validation_metrics.columns))
    if missing:
        raise ValueError(f"Validation metrics missing columns: {missing}")
    if validation_metrics.empty:
        raise ValueError("Validation metrics are empty")
    metrics = validation_metrics.loc[:, list(validation_metrics.columns)].copy()
    numeric = metrics[["top_k", "balanced_accuracy", "rmse"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all():
        raise ValueError("Validation metrics contain non-finite values")
    if metrics.duplicated(["model", "selection_fold", "top_k"]).any():
        raise ValueError("Validation metrics contain duplicate model-fold-k cells")
    if 122 not in set(metrics["top_k"].astype(int)):
        raise ValueError("All-features k=122 baseline is required")
    if metrics["model"].nunique() < 5:
        raise ValueError("The registered rule requires all five models")

    cell_counts = metrics.groupby("top_k").size()
    if cell_counts.nunique() != 1:
        raise ValueError("Every top-k candidate must contain the same cells")
    baseline = metrics.loc[
        metrics["top_k"].eq(122),
        ["model", "selection_fold", "balanced_accuracy", "rmse"],
    ].rename(
        columns={
            "balanced_accuracy": "baseline_balanced_accuracy",
            "rmse": "baseline_rmse",
        }
    )
    merged = metrics.merge(
        baseline,
        on=["model", "selection_fold"],
        how="left",
        validate="many_to_one",
    )
    if merged[
        ["baseline_balanced_accuracy", "baseline_rmse"]
    ].isna().any().any():
        raise ValueError("A validation cell is missing its k=122 baseline")
    merged["paired_ba_delta"] = (
        merged["balanced_accuracy"]
        - merged["baseline_balanced_accuracy"]
    )
    merged["paired_rmse_increase_fraction"] = (
        merged["rmse"] / merged["baseline_rmse"] - 1.0
    )

    summary = (
        merged.groupby("top_k", sort=True)
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            validation_cells=("balanced_accuracy", "size"),
            median_paired_ba_delta=("paired_ba_delta", "median"),
        )
        .reset_index()
    )
    best_index = summary["mean_balanced_accuracy"].idxmax()
    best_mean = float(summary.loc[best_index, "mean_balanced_accuracy"])
    best_std = float(summary.loc[best_index, "std_balanced_accuracy"])
    if not np.isfinite(best_std):
        best_std = 0.0
    best_cells = int(summary.loc[best_index, "validation_cells"])
    one_se_threshold = best_mean - best_std / np.sqrt(best_cells)

    audit_rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        top_k = int(row.top_k)
        candidate = merged.loc[merged["top_k"].eq(top_k)]
        by_model = (
            candidate.groupby("model", sort=False)
            .agg(
                mean_ba_delta=("paired_ba_delta", "mean"),
                mean_rmse_increase=(
                    "paired_rmse_increase_fraction",
                    "mean",
                ),
            )
            .reset_index()
        )
        stability = float(stability_by_k.get(top_k, np.nan))
        within_one_se = bool(
            float(row.mean_balanced_accuracy) >= one_se_threshold
        )
        median_nonnegative = bool(row.median_paired_ba_delta >= 0.0)
        noninferior_models = int(by_model["mean_ba_delta"].ge(0.0).sum())
        three_models_noninferior = noninferior_models >= 3
        worst_model_ba_delta = float(by_model["mean_ba_delta"].min())
        no_model_ba_loss_gt_1pp = worst_model_ba_delta >= -0.01
        worst_model_rmse_increase = float(
            by_model["mean_rmse_increase"].max()
        )
        no_model_rmse_increase_gt_5pct = (
            worst_model_rmse_increase <= 0.05
        )
        stable = bool(np.isfinite(stability) and stability >= 0.50)
        gates = (
            within_one_se,
            median_nonnegative,
            three_models_noninferior,
            no_model_ba_loss_gt_1pp,
            no_model_rmse_increase_gt_5pct,
            stable,
        )
        audit_rows.append(
            {
                "top_k": top_k,
                "mean_balanced_accuracy": float(
                    row.mean_balanced_accuracy
                ),
                "std_balanced_accuracy": float(
                    row.std_balanced_accuracy
                ),
                "validation_cells": int(row.validation_cells),
                "best_mean_balanced_accuracy": best_mean,
                "one_se_threshold": one_se_threshold,
                "within_one_se": within_one_se,
                "median_paired_ba_delta": float(
                    row.median_paired_ba_delta
                ),
                "median_delta_nonnegative": median_nonnegative,
                "noninferior_models": noninferior_models,
                "at_least_three_models_noninferior": (
                    three_models_noninferior
                ),
                "worst_model_mean_ba_delta": worst_model_ba_delta,
                "no_model_ba_loss_gt_1pp": no_model_ba_loss_gt_1pp,
                "worst_model_mean_rmse_increase_fraction": (
                    worst_model_rmse_increase
                ),
                "no_model_rmse_increase_gt_5pct": (
                    no_model_rmse_increase_gt_5pct
                ),
                "median_temporal_jaccard": stability,
                "jaccard_at_least_0_50": stable,
                "all_gates_pass": bool(all(gates)),
            }
        )
    audit = pd.DataFrame(audit_rows).sort_values("top_k").reset_index(
        drop=True
    )
    passing = audit.loc[audit["all_gates_pass"], "top_k"]
    selected = 122 if passing.empty else int(passing.min())
    if selected != 122 and not bool(
        audit.loc[audit["top_k"].eq(selected), "all_gates_pass"].item()
    ):
        raise RuntimeError("Selected top-k does not pass registered gates")
    return selected, audit
