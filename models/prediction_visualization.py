"""Utilities for plotting saved out-of-sample model predictions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "Date",
    "Close_D",
    "y_true",
    "y_pred",
    "true_direction",
    "pred_direction",
}
PREDICTION_PATTERN = re.compile(r"predictions_fold_(\d+)\.csv$")


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest parent containing the project's outputs directory."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "outputs").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find an 'outputs' directory. Run the notebook from inside "
        "the SET50_direction_prediction_paper project."
    )


def discover_experiments(
    project_root: Path, model_aliases: Sequence[str]
) -> list[Path]:
    """Find output directories with predictions matching a model family."""
    aliases = tuple(alias.lower() for alias in model_aliases)
    output_root = project_root / "outputs"
    if not output_root.is_dir():
        raise FileNotFoundError(f"Outputs directory not found: {output_root}")

    experiments = [
        path
        for path in output_root.glob("*/*")
        if path.is_dir()
        and any(alias in path.name.lower() for alias in aliases)
        and any(path.glob("predictions_fold_*.csv"))
    ]
    return sorted(experiments, key=lambda path: str(path).lower())


def load_predictions(experiment_dir: Path) -> pd.DataFrame:
    """Load every available test fold and return chronologically sorted rows."""
    frames: list[pd.DataFrame] = []
    prediction_files = sorted(experiment_dir.glob("predictions_fold_*.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction CSV files found in {experiment_dir}")

    for path in prediction_files:
        match = PREDICTION_PATTERN.search(path.name)
        if match is None:
            continue
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        selected = frame.loc[:, sorted(REQUIRED_COLUMNS)].copy()
        selected["Date"] = pd.to_datetime(selected["Date"], errors="raise")
        selected["fold"] = int(match.group(1))
        frames.append(selected)

    predictions = pd.concat(frames, ignore_index=True)
    numeric_columns = list(REQUIRED_COLUMNS.difference({"Date"}))
    predictions[numeric_columns] = predictions[numeric_columns].apply(
        pd.to_numeric, errors="raise"
    )
    if predictions[numeric_columns].isna().any().any():
        raise ValueError(f"NaN found in prediction values from {experiment_dir}")

    predictions = predictions.sort_values(["Date", "fold"]).reset_index(drop=True)
    return predictions.assign(
        residual=predictions["y_pred"] - predictions["y_true"],
        direction_correct=(
            predictions["true_direction"] == predictions["pred_direction"]
        ),
    )


def regression_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Calculate regression and direction metrics from saved predictions."""
    actual = predictions["y_true"].to_numpy(dtype=float)
    predicted = predictions["y_pred"].to_numpy(dtype=float)
    residual = predicted - actual
    nonzero = actual != 0
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "mape": float(np.mean(np.abs(residual[nonzero] / actual[nonzero])) * 100),
        "r2": float(1 - np.sum(residual**2) / denominator) if denominator else np.nan,
        "direction_accuracy": float(
            np.mean(
                predictions["true_direction"].to_numpy()
                == predictions["pred_direction"].to_numpy()
            )
        ),
    }


def metrics_by_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one recalculated metric row per out-of-sample fold."""
    rows = [
        {"fold": int(fold), **regression_metrics(frame)}
        for fold, frame in predictions.groupby("fold", sort=True)
    ]
    return pd.DataFrame(rows).set_index("fold")


def plot_predictions(
    predictions: pd.DataFrame, title: str
) -> tuple[plt.Figure, np.ndarray]:
    """Plot all OOS predictions, final fold, scatter, and fold stability."""
    final_fold = int(predictions["fold"].max())
    final = predictions.loc[predictions["fold"] == final_fold]
    fold_metrics = metrics_by_fold(predictions)

    figure, axes = plt.subplots(2, 2, figsize=(17, 11), constrained_layout=True)
    figure.suptitle(title, fontsize=16, fontweight="bold")

    axes[0, 0].plot(predictions["Date"], predictions["y_true"], label="Actual", lw=1.7)
    axes[0, 0].plot(
        predictions["Date"], predictions["y_pred"], label="Predicted", lw=1.25, alpha=0.85
    )
    axes[0, 0].set(title="All out-of-sample folds", ylabel="SET50 value")
    axes[0, 0].legend()

    axes[0, 1].plot(final["Date"], final["y_true"], label="Actual", lw=1.8)
    axes[0, 1].plot(final["Date"], final["y_pred"], label="Predicted", lw=1.4)
    axes[0, 1].set(title=f"Final test fold (fold {final_fold})", ylabel="SET50 value")
    axes[0, 1].legend()

    scatter = axes[1, 0].scatter(
        predictions["y_true"], predictions["y_pred"],
        c=predictions["fold"], cmap="viridis", alpha=0.65, s=18,
    )
    bounds = [
        min(predictions["y_true"].min(), predictions["y_pred"].min()),
        max(predictions["y_true"].max(), predictions["y_pred"].max()),
    ]
    axes[1, 0].plot(bounds, bounds, "k--", lw=1, label="Perfect prediction")
    axes[1, 0].set(
        title="Actual vs predicted", xlabel="Actual", ylabel="Predicted",
        xlim=bounds, ylim=bounds,
    )
    axes[1, 0].legend()
    figure.colorbar(scatter, ax=axes[1, 0], label="Fold")

    accuracy = fold_metrics["direction_accuracy"] * 100
    axes[1, 1].bar(accuracy.index.astype(str), accuracy, color="#4c78a8")
    axes[1, 1].axhline(50, color="black", ls="--", lw=1, label="50% reference")
    axes[1, 1].set(
        title="Direction accuracy by fold", xlabel="Fold", ylabel="Accuracy (%)",
        ylim=(0, 100),
    )
    axes[1, 1].legend()
    for index, value in enumerate(accuracy):
        axes[1, 1].text(index, value + 2, f"{value:.1f}%", ha="center")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    return figure, axes
