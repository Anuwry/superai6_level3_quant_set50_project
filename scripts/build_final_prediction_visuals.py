"""Build publication-ready actual-vs-predicted figures for the five frozen models.

The figures use only seed-averaged, out-of-sample predictions from the final
integrated arm (Regime-SHAP-Numeric-News).  No model fitting occurs here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "outputs" / "integrated_multimodal_posthoc_v1"
PREDICTIONS_PATH = SOURCE_DIR / "predictions_seed_averaged.csv"
SUMMARY_PATH = SOURCE_DIR / "arm_summary.csv"
WINDOWS_PATH = ROOT / "outputs" / "track_a_final_point_in_time_v2" / "locked_windows.csv"
OUTPUT_DIR = ROOT / "outputs" / "final_five_model_prediction_visuals_v1"

FINAL_ARM = "Regime-SHAP-Numeric-News"
MODEL_ORDER = [
    "lstm",
    "cnn",
    "lstm_cnn",
    "lstm_attention",
    "lstm_cnn_attention",
]
MODEL_LABELS = {
    "lstm": "LSTM",
    "cnn": "CNN",
    "lstm_cnn": "LSTM-CNN",
    "lstm_attention": "LSTM-Attention",
    "lstm_cnn_attention": "LSTM-CNN-Attention",
}
MODEL_COLORS = {
    "lstm": "#2F6690",
    "cnn": "#D95D39",
    "lstm_cnn": "#2A9D8F",
    "lstm_attention": "#6A4C93",
    "lstm_cnn_attention": "#E9A23B",
}


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary balanced accuracy without introducing a sklearn dependency."""

    recalls: list[float] = []
    for cls in (0, 1):
        mask = y_true == cls
        recalls.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(recalls))


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9E2EC", linewidth=0.7, alpha=0.75)
    ax.tick_params(colors="#334E68", labelsize=8)


def make_line_figure(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    windows: dict[str, int],
    *,
    start_date: str | None,
    stem: str,
    title_suffix: str,
    paper_clean: bool = False,
) -> None:
    plot_data = predictions if start_date is None else predictions[predictions["Date"] >= start_date]
    values = np.concatenate([plot_data["y_true"].to_numpy(), plot_data["y_pred"].to_numpy()])
    padding = max((float(np.nanmax(values)) - float(np.nanmin(values))) * 0.04, 5.0)
    y_limits = (float(np.nanmin(values)) - padding, float(np.nanmax(values)) + padding)

    fig, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(14.2, 13.2),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.985 if paper_clean else 0.925,
        bottom=0.075 if paper_clean else 0.095,
        hspace=0.22,
    )
    if not paper_clean:
        fig.suptitle(
            f"Out-of-sample next-day SET50 forecasts: actual vs predicted {title_suffix}",
            fontsize=17,
            fontweight="bold",
            color="#102A43",
            y=0.975,
        )
        fig.text(
            0.5,
            0.945,
            "Final arm: Regime-SHAP-Numeric-News | seed-averaged predictions (5 seeds) | walk-forward test folds",
            ha="center",
            fontsize=9.5,
            color="#486581",
        )

    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        frame = plot_data[plot_data["model"] == model].sort_values("Date")
        row = metrics.loc[model]
        if start_date is None:
            displayed_bacc = float(row["balanced_accuracy_mean"])
            displayed_da = float(row["direction_accuracy_mean"])
            displayed_rmse = float(row["rmse_mean"])
            metric_scope = "4-fold mean"
        else:
            actual_delta = frame["y_true"].to_numpy() - frame["Close_D"].to_numpy()
            predicted_delta = frame["y_pred"].to_numpy() - frame["Close_D"].to_numpy()
            valid = actual_delta != 0
            actual_binary = (actual_delta[valid] > 0).astype(int)
            predicted_binary = (predicted_delta[valid] > 0).astype(int)
            displayed_bacc = balanced_accuracy(actual_binary, predicted_binary)
            displayed_da = float(np.mean(actual_binary == predicted_binary))
            displayed_rmse = float(np.sqrt(np.mean(np.square(frame["y_pred"] - frame["y_true"]))))
            metric_scope = "visible 2025 fold"
        ax.plot(frame["Date"], frame["y_true"], color="#1F2933", linewidth=1.55, label="Actual next-day close", zorder=3)
        ax.plot(
            frame["Date"],
            frame["y_pred"],
            color=MODEL_COLORS[model],
            linewidth=1.05,
            alpha=0.92,
            label="Predicted next-day close",
            zorder=2,
        )
        ax.set_ylim(*y_limits)
        ax.set_ylabel("SET50", fontsize=8.5, color="#334E68")
        style_axis(ax)
        ax.text(
            0.008,
            0.89,
            MODEL_LABELS[model],
            transform=ax.transAxes,
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#102A43",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BCCCDC", "alpha": 0.9},
        )
        ax.text(
            0.992,
            0.89,
            (
                f"W={windows[model]}  |  BAcc={displayed_bacc * 100:.2f}%  |  "
                f"DA={displayed_da * 100:.2f}%  |  RMSE={displayed_rmse:.2f}  [{metric_scope}]"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.4,
            color="#334E68",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )

    handles = [
        Line2D([0], [0], color="#1F2933", linewidth=1.8, label="Actual next-day close"),
        Line2D([0], [0], color="#7B8794", linewidth=1.4, label="Model prediction"),
    ]
    axes[0].legend(handles=handles, loc="lower right", frameon=True, framealpha=0.92, fontsize=8.5)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2 if start_date else 6))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].set_xlabel("Out-of-sample prediction date", fontsize=9.5, color="#334E68")
    for label in axes[-1].get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    if not paper_clean:
        fig.text(
            0.075,
            0.012,
            "Interpretation: closeness of the two level lines reflects next-day index-level error; directional correctness is shown separately.",
            fontsize=8.2,
            color="#627D98",
        )
    save_figure(fig, stem)


def make_direction_figure(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    reference = predictions[predictions["model"] == MODEL_ORDER[0]].sort_values("Date").copy()
    dates = reference["Date"].reset_index(drop=True)
    actual_delta = reference["y_true"].to_numpy() - reference["Close_D"].to_numpy()
    actual_direction = np.sign(actual_delta)
    tie_mask = actual_direction == 0

    actual_code = np.where(actual_direction > 0, 1, np.where(actual_direction < 0, 0, 2)).reshape(1, -1)
    correctness_rows: list[np.ndarray] = []
    labels: list[str] = []
    for model in MODEL_ORDER:
        frame = predictions[predictions["model"] == model].sort_values("Date")
        predicted_direction = np.sign(frame["y_pred"].to_numpy() - frame["Close_D"].to_numpy())
        code = np.where(tie_mask | (predicted_direction == 0), 2, (predicted_direction == actual_direction).astype(int))
        correctness_rows.append(code)
        row = metrics.loc[model]
        labels.append(
            f"{MODEL_LABELS[model]}   BAcc {row['balanced_accuracy_mean'] * 100:.2f}%   DA {row['direction_accuracy_mean'] * 100:.2f}%"
        )
    correctness = np.vstack(correctness_rows)

    fig = plt.figure(figsize=(15.5, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4.4], hspace=0.18, left=0.205, right=0.985, top=0.83, bottom=0.18)
    actual_ax = fig.add_subplot(gs[0])
    correct_ax = fig.add_subplot(gs[1], sharex=actual_ax)
    fig.suptitle(
        "Daily direction alignment with the realized SET50 move (out-of-sample, 2022–2025)",
        fontsize=16,
        fontweight="bold",
        color="#102A43",
        y=0.965,
    )
    fig.text(
        0.5,
        0.91,
        "Each column is one next-day forecast; model cells show whether the predicted direction matched the realized direction",
        ha="center",
        fontsize=9.3,
        color="#486581",
    )

    actual_ax.imshow(actual_code, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#D95F02", "#1B9E77", "#9AA5B1"]), vmin=0, vmax=2)
    actual_ax.set_yticks([0], ["Actual direction"], fontsize=9.5, fontweight="bold", color="#102A43")
    actual_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in actual_ax.spines.values():
        spine.set_visible(False)

    correct_ax.imshow(correctness, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#D95D39", "#2A9D8F", "#9AA5B1"]), vmin=0, vmax=2)
    correct_ax.set_yticks(np.arange(len(labels)), labels, fontsize=8.8, color="#102A43")
    for spine in correct_ax.spines.values():
        spine.set_visible(False)

    years = dates.dt.year.to_numpy()
    tick_positions: list[int] = []
    tick_labels: list[str] = []
    for year in sorted(np.unique(years)):
        indices = np.where(years == year)[0]
        tick_positions.append(int(np.mean(indices)))
        tick_labels.append(str(year))
        for ax in (actual_ax, correct_ax):
            ax.axvline(indices[0] - 0.5, color="white", linewidth=1.5)
    correct_ax.set_xticks(tick_positions, tick_labels, fontsize=9.5, color="#334E68")
    correct_ax.set_xlabel("Walk-forward test year", fontsize=9.5, color="#334E68", labelpad=8)

    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#1B9E77", markeredgecolor="none", markersize=9, label="Actual up"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#D95F02", markeredgecolor="none", markersize=9, label="Actual down"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#2A9D8F", markeredgecolor="none", markersize=9, label="Correct model direction"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#D95D39", markeredgecolor="none", markersize=9, label="Wrong model direction"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#9AA5B1", markeredgecolor="none", markersize=9, label="Realized tie / abstention"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.59, 0.035))
    save_figure(fig, "direction_alignment_oos_2022_2025")


def make_scatter_figure(predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    values = np.concatenate([predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy()])
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    padding = max((upper - lower) * 0.035, 5.0)
    limits = (lower - padding, upper + padding)

    fig = plt.figure(figsize=(13.8, 7.7), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        6,
        left=0.07,
        right=0.985,
        top=0.985,
        bottom=0.085,
        wspace=0.95,
        hspace=0.32,
    )
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]

    for ax, model in zip(axes, MODEL_ORDER, strict=True):
        frame = predictions[predictions["model"] == model]
        row = metrics.loc[model]
        ax.scatter(
            frame["y_true"],
            frame["y_pred"],
            s=10,
            alpha=0.28,
            color=MODEL_COLORS[model],
            edgecolors="none",
            rasterized=True,
        )
        ax.plot(limits, limits, color="#1F2933", linestyle="--", linewidth=1.1, label="45° identity")
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")
        style_axis(ax)
        ax.grid(axis="both", color="#D9E2EC", linewidth=0.65, alpha=0.7)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold", color="#102A43", pad=7)
        ax.text(
            0.03,
            0.96,
            f"RMSE {row['rmse_mean']:.2f}  |  MAE {row['mae_mean']:.2f}\nBAcc {row['balanced_accuracy_mean'] * 100:.2f}%",
            transform=ax.transAxes,
            va="top",
            fontsize=8.2,
            color="#334E68",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        ax.set_xlabel("Observed next-day SET50", fontsize=8.5, color="#334E68")
        ax.set_ylabel("Predicted next-day SET50", fontsize=8.5, color="#334E68")

    save_figure(fig, "observed_vs_predicted_scatter_oos_2022_2025")


def make_summary_table(metrics: pd.DataFrame, windows: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        row = metrics.loc[model]
        rows.append(
            {
                "Model": MODEL_LABELS[model],
                "Window": windows[model],
                "Balanced Accuracy Mean (%)": row["balanced_accuracy_mean"] * 100,
                "Balanced Accuracy SD (pp)": row["balanced_accuracy_std"] * 100,
                "Direction Accuracy Mean (%)": row["direction_accuracy_mean"] * 100,
                "Direction Accuracy SD (pp)": row["direction_accuracy_std"] * 100,
                "MCC": row["mcc_mean"],
                "RMSE": row["rmse_mean"],
                "MAE": row["mae_mean"],
                "Temporal Folds": int(row["temporal_folds"]),
            }
        )
    summary = pd.DataFrame(rows).sort_values("Balanced Accuracy Mean (%)", ascending=False).reset_index(drop=True)
    summary.insert(0, "Rank", np.arange(1, len(summary) + 1))
    summary.to_csv(OUTPUT_DIR / "final_five_model_metrics.csv", index=False, encoding="utf-8-sig")

    display = summary.copy()
    display["Balanced Accuracy (%)"] = display.apply(
        lambda x: f"{x['Balanced Accuracy Mean (%)']:.2f} ± {x['Balanced Accuracy SD (pp)']:.2f}", axis=1
    )
    display["Direction Accuracy (%)"] = display.apply(
        lambda x: f"{x['Direction Accuracy Mean (%)']:.2f} ± {x['Direction Accuracy SD (pp)']:.2f}", axis=1
    )
    display = display.drop(
        columns=[
            "Balanced Accuracy Mean (%)",
            "Balanced Accuracy SD (pp)",
            "Direction Accuracy Mean (%)",
            "Direction Accuracy SD (pp)",
        ]
    )
    display = display[
        [
            "Rank",
            "Model",
            "Window",
            "Balanced Accuracy (%)",
            "Direction Accuracy (%)",
            "MCC",
            "RMSE",
            "MAE",
            "Temporal Folds",
        ]
    ]
    display["MCC"] = display["MCC"].map(lambda x: f"{x:.4f}")
    for col in ("RMSE", "MAE"):
        display[col] = display[col].map(lambda x: f"{x:.2f}")

    fig, ax = plt.subplots(figsize=(14.8, 3.55))
    ax.axis("off")
    fig.suptitle(
        "Final five-model results on the integrated out-of-sample arm",
        fontsize=16,
        fontweight="bold",
        color="#102A43",
        y=0.94,
    )
    fig.text(
        0.5,
        0.84,
        "Regime-SHAP-Numeric-News | mean across four temporal folds (2022–2025); five seeds per fold",
        ha="center",
        fontsize=9.3,
        color="#486581",
    )
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.21, 1.0, 0.50],
        colWidths=[0.055, 0.18, 0.075, 0.15, 0.15, 0.08, 0.09, 0.09, 0.09],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#BCCCDC")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor("#243B53")
            cell.set_text_props(color="white", weight="bold")
        elif r == 1:
            cell.set_facecolor("#E7F5EF")
            if c in (0, 1, 3):
                cell.set_text_props(weight="bold", color="#102A43")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F7FA")
        else:
            cell.set_facecolor("white")
    fig.text(
        0.06,
        0.055,
        "Primary ranking metric: balanced accuracy. Lower RMSE/MAE is better; higher balanced accuracy, direction accuracy, and MCC is better.",
        fontsize=8.1,
        color="#627D98",
    )
    save_figure(fig, "final_five_model_results_table")
    return summary


def write_behavior_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Write level and directional diagnostics used to interpret Figures 8 and 9."""

    rows: list[dict[str, object]] = []
    scopes = {
        "2022-2025": predictions,
        "2025": predictions[predictions["Date"].dt.year == 2025],
    }
    for scope, data in scopes.items():
        for model in MODEL_ORDER:
            frame = data[data["model"] == model].sort_values("Date")
            actual = frame["y_true"].to_numpy()
            predicted = frame["y_pred"].to_numpy()
            current = frame["Close_D"].to_numpy()
            actual_direction = (actual > current).astype(int)
            predicted_direction = (predicted > current).astype(int)
            actual_up = actual_direction == 1
            actual_down = actual_direction == 0
            rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "observations": len(frame),
                    "level_pearson_correlation": float(np.corrcoef(actual, predicted)[0, 1]),
                    "mean_prediction_bias": float(np.mean(predicted - actual)),
                    "prediction_to_observed_sd_ratio": float(np.std(predicted) / np.std(actual)),
                    "actual_up_fraction": float(np.mean(actual_direction)),
                    "predicted_up_fraction": float(np.mean(predicted_direction)),
                    "up_recall": float(np.mean(predicted_direction[actual_up] == 1)),
                    "down_recall": float(np.mean(predicted_direction[actual_down] == 0)),
                    "balanced_accuracy": float(
                        0.5
                        * (
                            np.mean(predicted_direction[actual_up] == 1)
                            + np.mean(predicted_direction[actual_down] == 0)
                        )
                    ),
                    "direction_accuracy": float(np.mean(predicted_direction == actual_direction)),
                    "rmse": float(np.sqrt(np.mean(np.square(predicted - actual)))),
                    "mae": float(np.mean(np.abs(predicted - actual))),
                }
            )
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(OUTPUT_DIR / "model_behavior_diagnostics.csv", index=False, encoding="utf-8-sig")
    return diagnostics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["Date"])
    predictions = predictions[
        (predictions["arm"] == FINAL_ARM) & (predictions["model"].isin(MODEL_ORDER))
    ].copy()
    predictions = predictions.sort_values(["model", "Date"]).reset_index(drop=True)

    summary = pd.read_csv(SUMMARY_PATH)
    summary = summary[(summary["arm"] == FINAL_ARM) & (summary["model"].isin(MODEL_ORDER))].copy()
    summary = summary.set_index("model").loc[MODEL_ORDER]
    locked = pd.read_csv(WINDOWS_PATH).set_index("model")
    windows = {model: int(locked.loc[model, "selected_sequence_window"]) for model in MODEL_ORDER}

    expected_rows = 962
    counts = predictions.groupby("model").size().to_dict()
    if set(counts) != set(MODEL_ORDER) or any(count != expected_rows for count in counts.values()):
        raise ValueError(f"Unexpected prediction coverage: {counts}")
    if predictions["seeds_averaged"].nunique() != 1 or int(predictions["seeds_averaged"].iloc[0]) != 5:
        raise ValueError("Expected exactly five averaged seeds per prediction row.")

    reference = predictions[predictions["model"] == MODEL_ORDER[0]][["Date", "Close_D", "y_true"]].reset_index(drop=True)
    actual_series_identical = True
    metric_checks: dict[str, dict[str, float]] = {}
    for model in MODEL_ORDER:
        frame = predictions[predictions["model"] == model].sort_values("Date").reset_index(drop=True)
        actual_series_identical &= bool(
            frame[["Date", "Close_D", "y_true"]].equals(reference)
        )

        fold_metrics: list[dict[str, float]] = []
        for _, fold in frame.groupby("fold", sort=True):
            actual_delta = fold["y_true"].to_numpy() - fold["Close_D"].to_numpy()
            predicted_delta = fold["y_pred"].to_numpy() - fold["Close_D"].to_numpy()
            valid = actual_delta != 0
            actual_binary = (actual_delta[valid] > 0).astype(int)
            predicted_binary = (predicted_delta[valid] > 0).astype(int)
            fold_metrics.append(
                {
                    "balanced_accuracy": balanced_accuracy(actual_binary, predicted_binary),
                    "direction_accuracy": float(np.mean(actual_binary == predicted_binary)),
                }
            )
        recomputed_bacc = float(np.mean([x["balanced_accuracy"] for x in fold_metrics]))
        recomputed_da = float(np.mean([x["direction_accuracy"] for x in fold_metrics]))
        metric_checks[model] = {
            "published_balanced_accuracy": float(summary.loc[model, "balanced_accuracy_mean"]),
            "recomputed_balanced_accuracy": recomputed_bacc,
            "balanced_accuracy_abs_diff": abs(recomputed_bacc - float(summary.loc[model, "balanced_accuracy_mean"])),
            "published_direction_accuracy": float(summary.loc[model, "direction_accuracy_mean"]),
            "recomputed_direction_accuracy": recomputed_da,
            "direction_accuracy_abs_diff": abs(recomputed_da - float(summary.loc[model, "direction_accuracy_mean"])),
        }

    if not actual_series_identical:
        raise ValueError("Actual target series is not identical across models.")
    if any(
        values[metric] > 1e-12
        for values in metric_checks.values()
        for metric in ("balanced_accuracy_abs_diff", "direction_accuracy_abs_diff")
    ):
        raise ValueError(f"Recomputed metrics do not match frozen summary: {metric_checks}")

    make_line_figure(
        predictions,
        summary,
        windows,
        start_date=None,
        stem="actual_vs_predicted_oos_2022_2025",
        title_suffix="(2022–2025)",
    )
    make_line_figure(
        predictions,
        summary,
        windows,
        start_date="2025-01-01",
        stem="actual_vs_predicted_oos_2025_zoom",
        title_suffix="(2025 detail)",
        paper_clean=True,
    )
    make_direction_figure(predictions, summary)
    make_scatter_figure(predictions, summary)
    summary_table = make_summary_table(summary, windows)
    behavior_diagnostics = write_behavior_diagnostics(predictions)

    selected_columns = [
        "model",
        "fold",
        "test_year",
        "Date",
        "routing_regime",
        "Close_D",
        "y_true",
        "y_pred",
        "seeds_averaged",
    ]
    predictions[selected_columns].to_csv(
        OUTPUT_DIR / "final_arm_prediction_series.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "source_predictions": str(PREDICTIONS_PATH.relative_to(ROOT)),
        "source_summary": str(SUMMARY_PATH.relative_to(ROOT)),
        "source_locked_windows": str(WINDOWS_PATH.relative_to(ROOT)),
        "final_arm": FINAL_ARM,
        "models": MODEL_ORDER,
        "rows_per_model": counts,
        "date_min": str(reference["Date"].min().date()),
        "date_max": str(reference["Date"].max().date()),
        "test_years": sorted(int(x) for x in predictions["test_year"].unique()),
        "folds": sorted(predictions["fold"].unique().tolist()),
        "seeds_averaged": 5,
        "actual_series_identical_across_models": actual_series_identical,
        "metric_recomputation_checks": metric_checks,
        "ranked_models": summary_table[["Rank", "Model", "Balanced Accuracy Mean (%)"]].to_dict(orient="records"),
        "behavior_diagnostics_file": "outputs/final_five_model_prediction_visuals_v1/model_behavior_diagnostics.csv",
        "behavior_diagnostic_rows": len(behavior_diagnostics),
    }
    (OUTPUT_DIR / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
