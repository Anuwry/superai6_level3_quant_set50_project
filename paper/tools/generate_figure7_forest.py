from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\SET50_direction_prediction_paper")
TABLE_DIR = ROOT / "outputs" / "manuscript_tables_v1"
OUTPUT_DIR = ROOT / "paper" / "figures" / "results_v1"

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


def read_rows(filename: str) -> list[dict[str, str]]:
    with (TABLE_DIR / filename).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def indexed(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["model"]: row for row in rows}


def model_series(
    rows: list[dict[str, str]],
    estimate: str,
    lower: str,
    upper: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = indexed(rows)
    points = np.array([float(table[model][estimate]) for model in MODEL_ORDER])
    lows = np.array([float(table[model][lower]) for model in MODEL_ORDER])
    highs = np.array([float(table[model][upper]) for model in MODEL_ORDER])
    return points, lows, highs


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#777777")
    axis.spines["bottom"].set_color("#777777")
    axis.spines["left"].set_linewidth(0.7)
    axis.spines["bottom"].set_linewidth(0.7)
    axis.tick_params(colors="#333333", width=0.7, length=3)
    axis.grid(axis="x", color="#dddddd", linewidth=0.55, linestyle="-", zorder=0)


def forest_panel(
    axis: plt.Axes,
    title: str,
    points: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    show_labels: bool,
) -> None:
    y = np.arange(len(MODEL_ORDER))[::-1]
    xerr = np.vstack((points - lows, highs - points))
    axis.axvline(0, color="#555555", linewidth=0.9, linestyle="--", zorder=1)
    axis.errorbar(
        points,
        y,
        xerr=xerr,
        fmt="o",
        markersize=4.8,
        markerfacecolor="#52789c",
        markeredgecolor="#35546f",
        markeredgewidth=0.7,
        ecolor="#52789c",
        elinewidth=1.0,
        capsize=2.5,
        capthick=0.8,
        zorder=3,
    )
    axis.set_xlim(-7.0, 5.0)
    axis.set_xticks([-6, -3, 0, 3])
    axis.set_ylim(-0.75, len(MODEL_ORDER) - 0.25)
    axis.set_yticks(y)
    if show_labels:
        axis.set_yticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
    else:
        axis.set_yticklabels([])
        axis.tick_params(axis="y", length=0)
    axis.set_title(title, loc="left", fontsize=9.6, fontweight="bold", pad=7)
    style_axis(axis)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vmd_rows = read_rows("table_2_numerical_ablation.csv")
    vmd = model_series(
        vmd_rows,
        "balanced_accuracy_delta_pp",
        "balanced_accuracy_delta_pp_ci95_lower",
        "balanced_accuracy_delta_pp_ci95_upper",
    )

    news_rows = [
        row
        for row in read_rows("table_3a_multimodal_falsification.csv")
        if row["contrast"] == "observed_news_effect"
        and row["metric"] == "balanced_accuracy_delta_pp"
    ]
    news = model_series(news_rows, "point_estimate", "ci95_lower", "ci95_upper")

    shap_rows = [
        row
        for row in read_rows("table_4_regime_shap.csv")
        if row["inference_type"] == "four_fold_exact_sign_flip"
        and row["contrast"] == "regime_shap_reduction"
    ]
    shap = model_series(shap_rows, "point_estimate", "ci95_lower", "ci95_upper")

    transfer_rows = read_rows("table_5b_set100_transfer.csv")
    transfer = model_series(
        transfer_rows,
        "balanced_accuracy_delta_pp",
        "balanced_accuracy_delta_ci95_lower",
        "balanced_accuracy_delta_ci95_upper",
    )

    llm_rows = read_rows("table_3b_llm_intrinsic_separate.csv")
    llm_order = [
        "leader_minus_self_consistency_3",
        "leader_minus_self_consistency_4",
    ]
    llm_table = {row["comparison_id"]: row for row in llm_rows}
    llm_points = np.array([float(llm_table[key]["accuracy_delta_pp"]) for key in llm_order])
    llm_lows = np.array([float(llm_table[key]["accuracy_delta_pp_ci95_lower"]) for key in llm_order])
    llm_highs = np.array([float(llm_table[key]["accuracy_delta_pp_ci95_upper"]) for key in llm_order])

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.7,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    figure = plt.figure(figsize=(7.35, 7.25), facecolor="white")
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[1.0, 1.0, 0.66],
        hspace=0.58,
        wspace=0.24,
        left=0.18,
        right=0.98,
        top=0.965,
        bottom=0.115,
    )

    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])
    axis_e = figure.add_subplot(grid[2, :])

    forest_panel(axis_a, "A  Causal VMD - Full TA", *vmd, show_labels=True)
    forest_panel(axis_b, "B  Observed News - Market-Only", *news, show_labels=False)
    forest_panel(axis_c, "C  Regime-SHAP - Regime-All", *shap, show_labels=True)
    forest_panel(axis_d, "D  SET100 - matched SET50", *transfer, show_labels=False)

    axis_a.set_xlabel("Balanced-accuracy difference (pp)")
    axis_b.set_xlabel("Balanced-accuracy difference (pp)")
    axis_c.set_xlabel("Balanced-accuracy difference (pp)")
    axis_d.set_xlabel("Balanced-accuracy difference (pp)")

    llm_y = np.array([1, 0])
    llm_xerr = np.vstack((llm_points - llm_lows, llm_highs - llm_points))
    axis_e.axvline(0, color="#555555", linewidth=0.9, linestyle="--", zorder=1)
    axis_e.errorbar(
        llm_points,
        llm_y,
        xerr=llm_xerr,
        fmt="o",
        markersize=4.8,
        markerfacecolor="#66895f",
        markeredgecolor="#3e6338",
        markeredgewidth=0.7,
        ecolor="#66895f",
        elinewidth=1.0,
        capsize=2.5,
        capthick=0.8,
        zorder=3,
    )
    axis_e.set_xlim(0.0, 10.0)
    axis_e.set_xticks([0, 2, 4, 6, 8, 10])
    axis_e.set_ylim(-0.65, 1.65)
    axis_e.set_yticks(llm_y)
    axis_e.set_yticklabels(["vs SC3 (equal calls)", "vs SC4 (near cost)"])
    axis_e.set_xlabel("Intrinsic sentiment-accuracy difference (pp)")
    axis_e.set_title(
        "E  Leader - self-consistency controls (separate endpoint)",
        loc="left",
        fontsize=9.6,
        fontweight="bold",
        pad=7,
    )
    style_axis(axis_e)
    axis_e.text(
        9.9,
        1.47,
        "Both Holm-adjusted p < 0.001",
        ha="right",
        va="top",
        fontsize=7.8,
        color="#3e6338",
    )

    figure.text(
        0.18,
        0.038,
        "Points are paired effects; horizontal bars are 95% intervals. Panels A-D use balanced accuracy;\n"
        "Panel E uses intrinsic sentiment accuracy. No A-D contrast survives Holm adjustment.",
        ha="left",
        va="bottom",
        fontsize=7.7,
        color="#333333",
    )

    stem = OUTPUT_DIR / "figure_7_architecture_effects_forest"
    figure.savefig(stem.with_suffix(".png"), dpi=400, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    figure.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(figure)

    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
