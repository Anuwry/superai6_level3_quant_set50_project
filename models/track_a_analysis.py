from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

METRIC_COLUMNS = (
    "rmse",
    "mae",
    "mape",
    "r2",
    "direction_accuracy",
    "balanced_accuracy",
    "mcc",
    "direction_coverage",
)


def build_paired_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model",
        "seed",
        "fold",
        "test_year",
        "sequence_window",
    ]
    baseline = metrics.loc[
        metrics["feature_set"] == "full_ta",
        [*keys, *METRIC_COLUMNS, "runtime_seconds"],
    ].copy()
    vmd = metrics.loc[
        metrics["feature_set"] == "full_ta_vmd",
        [*keys, *METRIC_COLUMNS, "runtime_seconds"],
    ].copy()
    paired = baseline.merge(
        vmd,
        on=keys,
        suffixes=("_full_ta", "_full_ta_vmd"),
        how="inner",
        validate="one_to_one",
    )
    if len(paired) * 2 != len(metrics):
        raise ValueError("Final metrics are not fully paired by feature set")
    for metric in ("rmse", "mae", "mape", "r2", "runtime_seconds"):
        paired[f"{metric}_delta_vmd_minus_full_ta"] = (
            paired[f"{metric}_full_ta_vmd"] - paired[f"{metric}_full_ta"]
        )
    paired["direction_accuracy_delta_pp"] = (
        paired["direction_accuracy_full_ta_vmd"] - paired["direction_accuracy_full_ta"]
    ) * 100.0
    paired["balanced_accuracy_delta_pp"] = (
        paired["balanced_accuracy_full_ta_vmd"]
        - paired["balanced_accuracy_full_ta"]
    ) * 100.0
    paired["mcc_delta_vmd_minus_full_ta"] = (
        paired["mcc_full_ta_vmd"] - paired["mcc_full_ta"]
    )
    paired["direction_coverage_delta_pp"] = (
        paired["direction_coverage_full_ta_vmd"]
        - paired["direction_coverage_full_ta"]
    ) * 100.0
    return paired


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Differences must be a non-empty vector")
    if not np.isfinite(values).all():
        raise ValueError("Differences contain non-finite values")
    observed = abs(float(values.mean()))
    permutations = [
        abs(float(np.mean(values * np.asarray(signs, dtype=float))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(permutations) >= observed - np.finfo(float).eps))


def _confidence_interval(values: np.ndarray) -> tuple[float, float]:
    from scipy.stats import t

    data = np.asarray(values, dtype=float)
    mean = float(data.mean())
    if len(data) < 2:
        return np.nan, np.nan
    standard_error = float(data.std(ddof=1) / np.sqrt(len(data)))
    margin = float(t.ppf(0.975, df=len(data) - 1) * standard_error)
    return mean - margin, mean + margin


def summarize_final_results(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_means = (
        metrics.groupby(
            ["model", "feature_set", "sequence_window", "fold"],
            sort=False,
        )[[*METRIC_COLUMNS]]
        .mean()
        .reset_index()
    )
    performance = fold_means.groupby(
        ["model", "feature_set", "sequence_window"],
        sort=False,
    )[[*METRIC_COLUMNS]].agg(["mean", "std"])
    performance.columns = [
        f"{metric}_{statistic}" for metric, statistic in performance.columns
    ]
    performance = performance.reset_index()
    runtime = (
        metrics.groupby(
            ["model", "feature_set", "sequence_window"],
            sort=False,
        )["runtime_seconds"]
        .agg(["mean", "std", "sum"])
        .rename(
            columns={
                "mean": "runtime_seconds_mean",
                "std": "runtime_seconds_std",
                "sum": "runtime_seconds_total",
            }
        )
        .reset_index()
    )
    performance = performance.merge(
        runtime,
        on=["model", "feature_set", "sequence_window"],
        validate="one_to_one",
    )

    paired = build_paired_deltas(metrics)
    delta_columns = [
        "rmse_delta_vmd_minus_full_ta",
        "direction_accuracy_delta_pp",
        "balanced_accuracy_delta_pp",
        "mcc_delta_vmd_minus_full_ta",
        "runtime_seconds_delta_vmd_minus_full_ta",
    ]
    fold_deltas = (
        paired.groupby(["model", "fold"], sort=False)[delta_columns]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for model_key, model_deltas in fold_deltas.groupby("model", sort=False):
        rmse_values = model_deltas["rmse_delta_vmd_minus_full_ta"].to_numpy(dtype=float)
        direction_values = model_deltas["direction_accuracy_delta_pp"].to_numpy(
            dtype=float
        )
        balanced_values = model_deltas[
            "balanced_accuracy_delta_pp"
        ].to_numpy(dtype=float)
        mcc_values = model_deltas[
            "mcc_delta_vmd_minus_full_ta"
        ].to_numpy(dtype=float)
        runtime_values = model_deltas[
            "runtime_seconds_delta_vmd_minus_full_ta"
        ].to_numpy(dtype=float)
        rmse_lower, rmse_upper = _confidence_interval(rmse_values)
        direction_lower, direction_upper = _confidence_interval(direction_values)
        balanced_lower, balanced_upper = _confidence_interval(balanced_values)
        rows.append(
            {
                "model": model_key,
                "paired_outer_folds": len(model_deltas),
                "seeds_per_fold": int(
                    paired.loc[paired["model"] == model_key, "seed"].nunique()
                ),
                "rmse_delta_mean": float(rmse_values.mean()),
                "rmse_delta_std": float(rmse_values.std(ddof=1)),
                "rmse_delta_ci95_lower": rmse_lower,
                "rmse_delta_ci95_upper": rmse_upper,
                "rmse_exact_sign_flip_pvalue": exact_sign_flip_pvalue(rmse_values),
                "direction_accuracy_delta_pp_mean": float(direction_values.mean()),
                "direction_accuracy_delta_pp_std": float(direction_values.std(ddof=1)),
                "direction_accuracy_delta_pp_ci95_lower": direction_lower,
                "direction_accuracy_delta_pp_ci95_upper": direction_upper,
                "direction_exact_sign_flip_pvalue": exact_sign_flip_pvalue(
                    direction_values
                ),
                "balanced_accuracy_delta_pp_mean": float(
                    balanced_values.mean()
                ),
                "balanced_accuracy_delta_pp_std": float(
                    balanced_values.std(ddof=1)
                ),
                "balanced_accuracy_delta_pp_ci95_lower": balanced_lower,
                "balanced_accuracy_delta_pp_ci95_upper": balanced_upper,
                "balanced_accuracy_exact_sign_flip_pvalue": (
                    exact_sign_flip_pvalue(balanced_values)
                ),
                "mcc_delta_mean": float(mcc_values.mean()),
                "mcc_delta_std": float(mcc_values.std(ddof=1)),
                "runtime_seconds_delta_mean": float(runtime_values.mean()),
            }
        )
    return performance, pd.DataFrame(rows)


def build_paper_table(
    performance: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["model", "sequence_window"]
    baseline = performance.loc[performance["feature_set"] == "full_ta"].drop(
        columns=["feature_set"]
    )
    vmd = performance.loc[performance["feature_set"] == "full_ta_vmd"].drop(
        columns=["feature_set"]
    )
    wide = baseline.merge(
        vmd,
        on=keys,
        suffixes=("_full_ta", "_full_ta_vmd"),
        validate="one_to_one",
    )
    return wide.merge(
        paired_summary,
        on="model",
        validate="one_to_one",
    )


def build_compact_paper_table(
    performance: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> pd.DataFrame:
    wide = build_paper_table(performance, paired_summary)
    return wide.loc[
        :,
        [
            "model",
            "sequence_window",
            "rmse_mean_full_ta",
            "rmse_std_full_ta",
            "rmse_mean_full_ta_vmd",
            "rmse_std_full_ta_vmd",
            "rmse_delta_mean",
            "rmse_delta_ci95_lower",
            "rmse_delta_ci95_upper",
            "rmse_exact_sign_flip_pvalue",
            "direction_accuracy_mean_full_ta",
            "direction_accuracy_std_full_ta",
            "direction_accuracy_mean_full_ta_vmd",
            "direction_accuracy_std_full_ta_vmd",
            "direction_accuracy_delta_pp_mean",
            "direction_accuracy_delta_pp_ci95_lower",
            "direction_accuracy_delta_pp_ci95_upper",
            "direction_exact_sign_flip_pvalue",
            "balanced_accuracy_mean_full_ta",
            "balanced_accuracy_std_full_ta",
            "balanced_accuracy_mean_full_ta_vmd",
            "balanced_accuracy_std_full_ta_vmd",
            "balanced_accuracy_delta_pp_mean",
            "balanced_accuracy_delta_pp_ci95_lower",
            "balanced_accuracy_delta_pp_ci95_upper",
            "balanced_accuracy_exact_sign_flip_pvalue",
            "mcc_mean_full_ta",
            "mcc_std_full_ta",
            "mcc_mean_full_ta_vmd",
            "mcc_std_full_ta_vmd",
            "mcc_delta_mean",
            "direction_coverage_mean_full_ta",
            "direction_coverage_mean_full_ta_vmd",
            "runtime_seconds_mean_full_ta",
            "runtime_seconds_mean_full_ta_vmd",
            "paired_outer_folds",
            "seeds_per_fold",
        ],
    ].rename(
        columns={
            "sequence_window": "selected_sequence_window",
            "rmse_mean_full_ta": "full_ta_rmse_mean",
            "rmse_std_full_ta": "full_ta_rmse_std",
            "rmse_mean_full_ta_vmd": "vmd_rmse_mean",
            "rmse_std_full_ta_vmd": "vmd_rmse_std",
            "rmse_delta_mean": "rmse_delta_vmd_minus_full_ta",
            "direction_accuracy_mean_full_ta": ("full_ta_direction_accuracy_mean"),
            "direction_accuracy_std_full_ta": ("full_ta_direction_accuracy_std"),
            "direction_accuracy_mean_full_ta_vmd": ("vmd_direction_accuracy_mean"),
            "direction_accuracy_std_full_ta_vmd": ("vmd_direction_accuracy_std"),
            "direction_accuracy_delta_pp_mean": ("direction_accuracy_delta_pp"),
            "balanced_accuracy_mean_full_ta": (
                "full_ta_balanced_accuracy_mean"
            ),
            "balanced_accuracy_std_full_ta": (
                "full_ta_balanced_accuracy_std"
            ),
            "balanced_accuracy_mean_full_ta_vmd": (
                "vmd_balanced_accuracy_mean"
            ),
            "balanced_accuracy_std_full_ta_vmd": (
                "vmd_balanced_accuracy_std"
            ),
            "balanced_accuracy_delta_pp_mean": (
                "balanced_accuracy_delta_pp"
            ),
            "mcc_mean_full_ta": "full_ta_mcc_mean",
            "mcc_std_full_ta": "full_ta_mcc_std",
            "mcc_mean_full_ta_vmd": "vmd_mcc_mean",
            "mcc_std_full_ta_vmd": "vmd_mcc_std",
            "mcc_delta_mean": "mcc_delta_vmd_minus_full_ta",
            "direction_coverage_mean_full_ta": (
                "full_ta_direction_coverage_mean"
            ),
            "direction_coverage_mean_full_ta_vmd": (
                "vmd_direction_coverage_mean"
            ),
            "runtime_seconds_mean_full_ta": ("full_ta_runtime_seconds_mean"),
            "runtime_seconds_mean_full_ta_vmd": ("vmd_runtime_seconds_mean"),
        }
    )


def build_runtime_summary(
    selection_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat(
        [selection_metrics, final_metrics],
        ignore_index=True,
    )
    required = {
        "stage",
        "model",
        "feature_set",
        "sequence_window",
        "runtime_seconds",
    }
    missing = sorted(required.difference(combined.columns))
    if missing:
        raise ValueError(f"Runtime metrics are missing columns: {missing}")
    if (
        not np.isfinite(combined["runtime_seconds"]).all()
        or (combined["runtime_seconds"] <= 0.0).any()
    ):
        raise ValueError("Runtime metrics must be finite and positive")
    return (
        combined.groupby(
            ["stage", "model", "feature_set", "sequence_window"],
            sort=False,
        )["runtime_seconds"]
        .agg(
            completed_fits="size",
            runtime_seconds_mean="mean",
            runtime_seconds_std="std",
            runtime_seconds_total="sum",
        )
        .reset_index()
    )
