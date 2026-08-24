from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t

from models.track_a_analysis import exact_sign_flip_pvalue

CONTRASTS: dict[str, tuple[str, str]] = {
    "global_shap_reduction": ("Global-SHAP", "Global-All"),
    "global_shap_specificity": ("Global-SHAP", "Global-Spearman"),
    "regime_shap_reduction": ("Regime-SHAP", "Regime-All"),
    "regime_shap_specificity": ("Regime-SHAP", "Regime-Spearman"),
    "regime_routing": ("Regime-All", "Global3-All"),
}
FOLD_DELTA_METRICS = (
    "balanced_accuracy_delta_pp",
    "direction_accuracy_delta_pp",
    "mcc_delta",
    "rmse_delta",
    "mae_delta",
)


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("pvalues must be a non-empty vector")
    if not np.isfinite(values).all() or np.any(
        (values < 0.0) | (values > 1.0)
    ):
        raise ValueError("pvalues must be finite and in [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    adjusted_sorted = np.empty(len(values), dtype=float)
    running_maximum = 0.0
    for rank, value in enumerate(sorted_values):
        candidate = min(1.0, (len(values) - rank) * float(value))
        running_maximum = max(running_maximum, candidate)
        adjusted_sorted[rank] = running_maximum
    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def average_seed_predictions(
    frames_by_seed: Mapping[int, pd.DataFrame],
) -> pd.DataFrame:
    if not frames_by_seed:
        raise ValueError("At least one seed prediction frame is required")
    required = {
        "Date",
        "Close_D",
        "y_true",
        "routing_regime",
        "y_pred",
    }
    ordered = sorted(frames_by_seed.items())
    reference = ordered[0][1].reset_index(drop=True).copy()
    missing = sorted(required.difference(reference.columns))
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {missing}")
    reference["Date"] = pd.to_datetime(reference["Date"], errors="raise")
    prediction_columns = [
        reference["y_pred"].to_numpy(dtype=float)
    ]
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
        if not frame["routing_regime"].astype(str).equals(
            reference["routing_regime"].astype(str)
        ):
            raise ValueError("Seed prediction regimes differ")
        prediction_columns.append(frame["y_pred"].to_numpy(dtype=float))
    predictions = np.stack(prediction_columns, axis=0)
    if not np.isfinite(predictions).all():
        raise ValueError("Seed predictions contain non-finite values")
    result = reference[
        ["Date", "routing_regime", "Close_D", "y_true"]
    ].copy()
    result["y_pred"] = predictions.mean(axis=0)
    result["seeds_averaged"] = len(prediction_columns)
    return result


def build_paired_fold_contrasts(
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "model",
        "fold",
        "test_year",
        "arm",
        "balanced_accuracy",
        "direction_accuracy",
        "mcc",
        "rmse",
        "mae",
    }
    missing = sorted(required.difference(fold_metrics.columns))
    if missing:
        raise ValueError(f"Fold metrics are missing columns: {missing}")
    keys = ["model", "fold", "test_year"]
    rows: list[pd.DataFrame] = []
    available_arms = set(fold_metrics["arm"])
    for contrast, (treatment_arm, control_arm) in CONTRASTS.items():
        if not {treatment_arm, control_arm}.issubset(available_arms):
            continue
        treatment = fold_metrics.loc[
            fold_metrics["arm"].eq(treatment_arm),
            [
                *keys,
                "balanced_accuracy",
                "direction_accuracy",
                "mcc",
                "rmse",
                "mae",
            ],
        ]
        control = fold_metrics.loc[
            fold_metrics["arm"].eq(control_arm),
            [
                *keys,
                "balanced_accuracy",
                "direction_accuracy",
                "mcc",
                "rmse",
                "mae",
            ],
        ]
        paired = treatment.merge(
            control,
            on=keys,
            how="inner",
            validate="one_to_one",
            suffixes=("_treatment", "_control"),
        )
        if len(paired) != len(treatment) or len(paired) != len(control):
            raise ValueError(f"Incomplete fold pairing for {contrast}")
        paired.insert(3, "contrast", contrast)
        paired.insert(4, "treatment_arm", treatment_arm)
        paired.insert(5, "control_arm", control_arm)
        paired["balanced_accuracy_delta_pp"] = (
            paired["balanced_accuracy_treatment"]
            - paired["balanced_accuracy_control"]
        ) * 100.0
        paired["direction_accuracy_delta_pp"] = (
            paired["direction_accuracy_treatment"]
            - paired["direction_accuracy_control"]
        ) * 100.0
        for metric in ("mcc", "rmse", "mae"):
            paired[f"{metric}_delta"] = (
                paired[f"{metric}_treatment"]
                - paired[f"{metric}_control"]
            )
        rows.append(paired)
    if not rows:
        raise ValueError("No registered arm contrast can be paired")
    return pd.concat(rows, ignore_index=True)


def _confidence_interval(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    if len(values) < 2:
        return float("nan"), float("nan")
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    margin = float(t.ppf(0.975, df=len(values) - 1) * standard_error)
    return mean - margin, mean + margin


def fold_level_inference(paired: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "contrast", "fold", *FOLD_DELTA_METRICS}
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ValueError(f"Paired contrasts are missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for (model, contrast), group in paired.groupby(
        ["model", "contrast"],
        sort=False,
    ):
        if group["fold"].nunique() != 4 or len(group) != 4:
            raise ValueError(
                f"{model}/{contrast} does not contain four unique folds"
            )
        for metric in FOLD_DELTA_METRICS:
            values = group[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Fold delta contains non-finite values")
            lower, upper = _confidence_interval(values)
            rows.append(
                {
                    "model": model,
                    "contrast": contrast,
                    "metric": metric,
                    "outer_folds": len(values),
                    "mean_delta": float(values.mean()),
                    "std_delta": float(values.std(ddof=1)),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "positive_folds": int(np.sum(values > 0.0)),
                    "negative_folds": int(np.sum(values < 0.0)),
                    "zero_folds": int(np.sum(values == 0.0)),
                    "exact_sign_flip_pvalue": exact_sign_flip_pvalue(
                        values
                    ),
                }
            )
    return pd.DataFrame(rows)


def circular_moving_block_indices(
    total: int,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if isinstance(total, bool) or total < 1:
        raise ValueError("total must be a positive integer")
    if isinstance(block_length, bool) or block_length < 1:
        raise ValueError("block_length must be a positive integer")
    blocks = math.ceil(total / block_length)
    starts = rng.integers(0, total, size=blocks)
    offsets = np.arange(block_length, dtype=int)
    indices = (
        starts[:, np.newaxis] + offsets[np.newaxis, :]
    ) % total
    return indices.reshape(-1)[:total]


def moving_block_bootstrap(
    fold_effects: Sequence[np.ndarray],
    *,
    block_length: int,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    arrays = [np.asarray(values, dtype=float) for values in fold_effects]
    if not arrays or any(values.ndim != 1 or len(values) < 1 for values in arrays):
        raise ValueError("fold_effects must contain non-empty vectors")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("fold_effects contain non-finite values")
    if isinstance(replicates, bool) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    rng = np.random.default_rng(seed)
    distribution = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        fold_means = []
        for values in arrays:
            indices = circular_moving_block_indices(
                len(values),
                block_length=block_length,
                rng=rng,
            )
            fold_means.append(float(values[indices].mean()))
        distribution[replicate] = float(np.mean(fold_means))
    probability_nonpositive = float(np.mean(distribution <= 0.0))
    probability_nonnegative = float(np.mean(distribution >= 0.0))
    return {
        "point_estimate": float(
            np.mean([values.mean() for values in arrays])
        ),
        "ci95_lower": float(np.quantile(distribution, 0.025)),
        "ci95_upper": float(np.quantile(distribution, 0.975)),
        "two_sided_pvalue": min(
            1.0,
            2.0 * min(
                probability_nonpositive,
                probability_nonnegative,
            ),
        ),
        "replicates": replicates,
        "block_length": block_length,
        "folds": len(arrays),
        "daily_rows": int(sum(len(values) for values in arrays)),
    }


def paired_daily_effects(
    treatment: pd.DataFrame,
    control: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    required = {"Date", "Close_D", "y_true", "y_pred"}
    for frame in (treatment, control):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Daily prediction is missing columns: {missing}")
    left = treatment.reset_index(drop=True).copy()
    right = control.reset_index(drop=True).copy()
    left["Date"] = pd.to_datetime(left["Date"], errors="raise")
    right["Date"] = pd.to_datetime(right["Date"], errors="raise")
    if not left["Date"].equals(right["Date"]):
        raise ValueError("Treatment and control dates differ")
    for column in ("Close_D", "y_true"):
        if not np.allclose(
            left[column].to_numpy(dtype=float),
            right[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("Treatment and control targets differ")
    y_true = left["y_true"].to_numpy(dtype=float)
    close = left["Close_D"].to_numpy(dtype=float)
    treatment_prediction = left["y_pred"].to_numpy(dtype=float)
    control_prediction = right["y_pred"].to_numpy(dtype=float)
    squared_error_delta = np.square(treatment_prediction - y_true) - np.square(
        control_prediction - y_true
    )

    true_direction = np.sign(y_true - close).astype(int)
    treatment_direction = np.sign(treatment_prediction - close).astype(int)
    control_direction = np.sign(control_prediction - close).astype(int)
    eligible = (
        (true_direction != 0)
        & (treatment_direction != 0)
        & (control_direction != 0)
    )
    eligible_true = true_direction[eligible]
    if set(eligible_true) != {-1, 1}:
        raise ValueError("Paired daily direction rows require both classes")
    n_eligible = len(eligible_true)
    balanced_accuracy_delta = np.empty(n_eligible, dtype=float)
    eligible_treatment = treatment_direction[eligible]
    eligible_control = control_direction[eligible]
    for direction in (-1, 1):
        class_count = int(np.sum(eligible_true == direction))
        weight = 0.5 / class_count * n_eligible * 100.0
        class_rows = eligible_true == direction
        treatment_correct = eligible_treatment[class_rows] == direction
        control_correct = eligible_control[class_rows] == direction
        balanced_accuracy_delta[class_rows] = (
            weight
            * (
                treatment_correct.astype(float)
                - control_correct.astype(float)
            )
        )
    if not (
        np.isfinite(squared_error_delta).all()
        and np.isfinite(balanced_accuracy_delta).all()
    ):
        raise ValueError("Daily paired effects contain non-finite values")
    return squared_error_delta, balanced_accuracy_delta


def apply_holm_by_family(
    frame: pd.DataFrame,
    *,
    pvalue_column: str,
) -> pd.DataFrame:
    required = {"model", "contrast", "metric", pvalue_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Inference frame is missing columns: {missing}")
    result = frame.copy()
    adjusted = pd.Series(index=result.index, dtype=float)
    for indices in result.groupby(
        ["contrast", "metric"],
        sort=False,
    ).groups.values():
        family = result.loc[indices]
        if family["model"].nunique() != 5 or len(family) != 5:
            raise ValueError("Each Holm family must contain five models")
        adjusted.loc[indices] = holm_adjust(
            family[pvalue_column].to_numpy(dtype=float)
        )
    result[f"{pvalue_column}_holm"] = adjusted
    return result
