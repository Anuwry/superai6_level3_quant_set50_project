"""Generate classic academic flowcharts for Methods Sections 3.1--3.8.

The figures deliberately follow a conventional journal-diagram aesthetic:
thin grey borders, pale functional fills, straight arrows, short labels, and
no embedded figure titles. Detailed explanations belong in the captions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "methods_v2_simple"

INK = "#333333"
EDGE = "#666666"
ARROW = "#666666"
CREAM = "#F4F0E5"      # data / input
BLUE = "#CCD9E8"       # processing / output
GREEN = "#DDE9D5"      # model / selection
GREY = "#EEEEEE"       # control / neutral
WHITE = "#FFFFFF"


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
)


def canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    fill: str = WHITE,
    fontsize: float = 8.2,
    weight: str = "normal",
    align: str = "center",
    radius: float = 0.010,
    lw: float = 0.85,
    linestyle: str = "-",
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fill,
        edgecolor=EDGE,
        linewidth=lw,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(patch)
    tx = x + w / 2 if align == "center" else x + 0.012
    ax.text(
        tx,
        y + h / 2,
        label,
        ha=align,
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=INK,
        linespacing=1.18,
        zorder=3,
    )
    return patch


def group(ax, x: float, y: float, w: float, h: float, label: str | None = None):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="none",
            edgecolor=EDGE,
            linewidth=0.85,
            zorder=0,
        )
    )
    if label:
        ax.text(x + 0.012, y + h - 0.018, label, ha="left", va="top", fontsize=8.0, color=EDGE)


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    connectionstyle: str = "arc3",
    lw: float = 0.9,
):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            color=ARROW,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=connectionstyle,
            zorder=1,
        )
    )
    if label:
        lx, ly = label_xy if label_xy else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=7.4, color=EDGE, backgroundcolor=WHITE)


def line(ax, start: tuple[float, float], end: tuple[float, float], *, lw: float = 0.8):
    ax.plot([start[0], end[0]], [start[1], end[1]], color=ARROW, lw=lw, zorder=1)


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def figure1_overall_pipeline():
    fig, ax = canvas(7.5, 5.2)

    # Numerical path.
    box(ax, 0.06, 0.855, 0.18, 0.075, "Historical SET50\nmarket data", fill=CREAM)
    arrow(ax, (0.15, 0.855), (0.15, 0.795))
    box(ax, 0.045, 0.705, 0.21, 0.09, "Point-in-time\npreprocessing", fill=BLUE)
    arrow(ax, (0.15, 0.705), (0.15, 0.645))
    box(ax, 0.045, 0.555, 0.21, 0.09, "Technical features\nand causal VMD", fill=BLUE)

    # News path.
    group(ax, 0.705, 0.825, 0.25, 0.145)
    box(ax, 0.725, 0.865, 0.095, 0.07, "Labeled\nnews", fill=CREAM, fontsize=7.8)
    box(ax, 0.840, 0.865, 0.095, 0.07, "Forward\nnews", fill=CREAM, fontsize=7.8)
    arrow(ax, (0.83, 0.825), (0.83, 0.765))
    box(ax, 0.725, 0.675, 0.21, 0.09, "Date, ticker and\nrelevance filtering", fill=BLUE)
    arrow(ax, (0.83, 0.675), (0.83, 0.615))
    box(ax, 0.725, 0.525, 0.21, 0.09, "Out-of-sample\nsentiment", fill=BLUE)

    # Regime/feature selection block.
    group(ax, 0.035, 0.285, 0.31, 0.19, "Feature construction")
    box(ax, 0.055, 0.315, 0.125, 0.095, "Bull\nSideway\nBear", fill=GREEN, fontsize=7.8)
    box(ax, 0.205, 0.315, 0.12, 0.095, "Train-only\nSHAP\nselection", fill=GREEN, fontsize=7.8)
    arrow(ax, (0.18, 0.362), (0.205, 0.362))
    arrow(ax, (0.15, 0.555), (0.12, 0.475))

    # Central five-model panel.
    group(ax, 0.385, 0.295, 0.29, 0.30, "Registered architecture panel")
    model_labels = ["LSTM", "CNN", "LSTM-CNN", "LSTM-Attention", "LSTM-CNN-Attention"]
    ys = [0.495, 0.454, 0.413, 0.372, 0.331]
    for label, y in zip(model_labels, ys):
        box(ax, 0.425, y, 0.21, 0.032, label, fill=GREEN, fontsize=7.7, radius=0.004)

    arrow(ax, (0.345, 0.385), (0.385, 0.385))
    arrow(ax, (0.725, 0.57), (0.675, 0.50))

    # Evaluation and output.
    arrow(ax, (0.53, 0.295), (0.53, 0.25))
    box(ax, 0.405, 0.17, 0.25, 0.08, "Expanding-window\nouter evaluation", fill=BLUE)
    arrow(ax, (0.53, 0.17), (0.53, 0.115))
    box(ax, 0.405, 0.035, 0.25, 0.08, "Next-day direction\nand reliability audit", fill=BLUE)

    # Robustness outcomes.
    line(ax, (0.655, 0.075), (0.82, 0.075))
    line(ax, (0.82, 0.075), (0.82, 0.20))
    arrow(ax, (0.82, 0.20), (0.82, 0.235))
    box(ax, 0.715, 0.235, 0.21, 0.07, "SET100 transfer and\npartial-2026 stress test", fill=CREAM, fontsize=7.8)

    save(fig, "figure1_overall_reliability_audit_pipeline")


def figure2_news_pipeline():
    fig, ax = canvas(7.4, 4.4)

    group(ax, 0.04, 0.765, 0.42, 0.18)
    box(ax, 0.065, 0.815, 0.17, 0.075, "StockTBSA\n2018-2023", fill=CREAM)
    box(ax, 0.265, 0.815, 0.17, 0.075, "Forward news\n2024-2025", fill=CREAM)

    arrow(ax, (0.25, 0.765), (0.25, 0.69))
    box(ax, 0.135, 0.60, 0.23, 0.09, "Timestamp and\nticker normalization", fill=BLUE)
    arrow(ax, (0.25, 0.60), (0.25, 0.525))
    box(ax, 0.135, 0.435, 0.23, 0.09, "SET50 relevance\nand duplicate filtering", fill=BLUE)

    # Training route.
    arrow(ax, (0.365, 0.48), (0.54, 0.69), label="labeled subset", label_xy=(0.45, 0.60))
    box(ax, 0.535, 0.635, 0.19, 0.11, "Train sentiment model\nusing past labels only", fill=GREEN)
    arrow(ax, (0.63, 0.635), (0.63, 0.545))
    box(ax, 0.535, 0.455, 0.19, 0.09, "Out-of-sample\nsentiment prediction", fill=BLUE)

    # Causal aggregation route.
    arrow(ax, (0.725, 0.50), (0.77, 0.40))
    box(ax, 0.77, 0.345, 0.19, 0.11, "Map publication time\nto next tradable\nsession", fill=CREAM, fontsize=7.8)
    arrow(ax, (0.865, 0.345), (0.865, 0.265))
    box(ax, 0.745, 0.175, 0.24, 0.09, "Daily sentiment features\nknown at prediction time", fill=BLUE)

    # Compact audit summary, not a decorative chart.
    group(ax, 0.055, 0.11, 0.59, 0.20, "Locked data audit")
    ax.text(
        0.075,
        0.245,
        "2018-2023: 12,706 valid labeled article-ticker pairs",
        ha="left",
        va="center",
        fontsize=8.0,
        color=INK,
    )
    ax.text(
        0.075,
        0.195,
        "2024: 1,223 selected records / 225 sessions",
        ha="left",
        va="center",
        fontsize=8.0,
        color=INK,
    )
    ax.text(
        0.075,
        0.145,
        "2025: 1,569 selected records / 224 sessions",
        ha="left",
        va="center",
        fontsize=8.0,
        color=INK,
    )

    save(fig, "figure2_news_data_and_oos_sentiment_pipeline")


def figure3_point_in_time():
    fig, ax = canvas(7.4, 4.7)

    # Information contract.
    box(ax, 0.055, 0.80, 0.18, 0.08, "Features available\nby close of day t", fill=CREAM)
    arrow(ax, (0.235, 0.84), (0.34, 0.84))
    box(ax, 0.34, 0.80, 0.18, 0.08, "Predict direction\nof day t+1", fill=BLUE)
    arrow(ax, (0.52, 0.84), (0.625, 0.84))
    box(ax, 0.625, 0.80, 0.18, 0.08, "Observe label only\nafter day t+1", fill=CREAM)
    ax.text(0.50, 0.745, "No future observations cross the purge boundary", ha="center", va="center", fontsize=8.1, color=EDGE)

    # Expanding folds.
    years = list(range(2012, 2026))
    x0, x1 = 0.09, 0.94
    dx = (x1 - x0) / len(years)
    rows = [("Outer 2022", 2022), ("Outer 2023", 2023), ("Outer 2024", 2024), ("Outer 2025", 2025)]
    y_rows = [0.62, 0.50, 0.38, 0.26]
    for (label, test_year), y in zip(rows, y_rows):
        ax.text(0.075, y + 0.03, label, ha="right", va="center", fontsize=7.7, color=INK)
        for i, year in enumerate(years):
            x = x0 + i * dx
            if year < test_year:
                fill = GREEN
            elif year == test_year:
                fill = BLUE
            else:
                fill = WHITE
            ax.add_patch(Rectangle((x, y), dx * 0.92, 0.06, facecolor=fill, edgecolor=EDGE, linewidth=0.55))

    for i, year in enumerate(years):
        if year in (2012, 2015, 2018, 2021, 2022, 2023, 2024, 2025):
            ax.text(x0 + i * dx + dx * 0.46, 0.225, str(year), ha="center", va="top", fontsize=6.5, color=EDGE)

    box(ax, 0.19, 0.035, 0.20, 0.055, "Expanding training data", fill=GREEN, fontsize=7.5)
    box(ax, 0.60, 0.035, 0.20, 0.055, "Held-out test year", fill=BLUE, fontsize=7.5)
    ax.text(0.50, 0.145, "Window selection is completed before the outer years; all transforms are fitted on training data only.", ha="center", va="center", fontsize=7.5, color=EDGE)

    save(fig, "figure3_point_in_time_expanding_window_design")


def figure4_vmd_models():
    fig, ax = canvas(7.5, 4.9)

    box(ax, 0.055, 0.84, 0.18, 0.075, "Historical market data", fill=CREAM)
    arrow(ax, (0.145, 0.84), (0.145, 0.775))
    box(ax, 0.04, 0.685, 0.21, 0.09, "Preprocessing and\ntechnical indicators", fill=BLUE)
    arrow(ax, (0.145, 0.685), (0.145, 0.62))

    group(ax, 0.025, 0.36, 0.31, 0.25, "Numerical inputs")
    box(ax, 0.05, 0.465, 0.12, 0.085, "Full TA\n116 features", fill=CREAM, fontsize=7.8)
    box(ax, 0.19, 0.465, 0.12, 0.085, "Rolling VMD\n60-day window", fill=GREEN, fontsize=7.8)
    arrow(ax, (0.17, 0.507), (0.19, 0.507))
    box(ax, 0.115, 0.385, 0.13, 0.055, "122 features", fill=CREAM, fontsize=7.8)
    arrow(ax, (0.25, 0.465), (0.18, 0.44))

    group(ax, 0.39, 0.34, 0.30, 0.46, "Five fixed architectures")
    labels = ["LSTM", "CNN", "LSTM-CNN", "LSTM-Attention", "LSTM-CNN-Attention"]
    yvals = [0.70, 0.625, 0.55, 0.475, 0.40]
    for label, y in zip(labels, yvals):
        box(ax, 0.425, y, 0.23, 0.05, label, fill=GREEN, fontsize=8.0, radius=0.005)
    arrow(ax, (0.335, 0.49), (0.39, 0.49))

    arrow(ax, (0.69, 0.57), (0.77, 0.57))
    box(ax, 0.77, 0.515, 0.18, 0.11, "Next-day\ndirection\nprobability", fill=BLUE)
    arrow(ax, (0.86, 0.515), (0.86, 0.43))
    box(ax, 0.75, 0.34, 0.22, 0.09, "Paired comparison:\nTA vs TA + VMD", fill=GREY)

    # Minimal architecture key.
    group(ax, 0.13, 0.045, 0.74, 0.18, "Layer sequence")
    keys = [
        "LSTM: LSTM - Dense - Output",
        "CNN: Conv1D - Pooling - Dense - Output",
        "Hybrid: CNN and/or LSTM - Attention where registered - Dense - Output",
    ]
    for i, text in enumerate(keys):
        ax.text(0.16, 0.165 - i * 0.048, text, ha="left", va="center", fontsize=7.5, color=INK)

    save(fig, "figure4_vmd_and_registered_architectures")


def figure5_multimodal():
    fig, ax = canvas(7.6, 5.3)

    # Market branch and two directly comparable forecasts.
    box(ax, 0.05, 0.85, 0.22, 0.075, "Point-in-time\nmarket features", fill=CREAM)
    arrow(ax, (0.16, 0.85), (0.16, 0.69))
    box(ax, 0.05, 0.60, 0.22, 0.09, "Five models\nmarket-only", fill=GREEN)
    arrow(ax, (0.16, 0.60), (0.16, 0.34))
    box(ax, 0.045, 0.25, 0.23, 0.09, "Forecast without\nsentiment", fill=BLUE)

    # News debate branch.
    box(ax, 0.71, 0.85, 0.22, 0.075, "Point-in-time\nfinancial news", fill=CREAM)
    arrow(ax, (0.82, 0.85), (0.82, 0.79))
    box(ax, 0.71, 0.70, 0.22, 0.09, "Worker 1\nSET50 relevance", fill=GREEN)
    arrow(ax, (0.82, 0.70), (0.82, 0.66))
    group(ax, 0.68, 0.50, 0.28, 0.16)
    box(ax, 0.70, 0.545, 0.105, 0.065, "Bull\nworker", fill=GREEN, fontsize=7.6)
    box(ax, 0.835, 0.545, 0.105, 0.065, "Bear\nworker", fill=GREEN, fontsize=7.6)
    line(ax, (0.82, 0.66), (0.82, 0.63))
    line(ax, (0.752, 0.63), (0.888, 0.63))
    arrow(ax, (0.752, 0.63), (0.752, 0.61))
    arrow(ax, (0.888, 0.63), (0.888, 0.61))
    line(ax, (0.752, 0.545), (0.752, 0.515))
    line(ax, (0.888, 0.545), (0.888, 0.515))
    line(ax, (0.752, 0.515), (0.888, 0.515))
    arrow(ax, (0.82, 0.515), (0.82, 0.48))
    box(ax, 0.71, 0.40, 0.22, 0.08, "Leader sentiment score", fill=BLUE)

    # The same market inputs are combined with the dated Leader score.
    line(ax, (0.27, 0.888), (0.52, 0.888))
    arrow(ax, (0.52, 0.888), (0.52, 0.49))
    box(ax, 0.405, 0.40, 0.23, 0.09, "Five models\nmarket + sentiment", fill=GREEN)
    arrow(ax, (0.71, 0.44), (0.635, 0.44))
    arrow(ax, (0.52, 0.40), (0.52, 0.34))
    box(ax, 0.405, 0.25, 0.23, 0.09, "Forecast with\nsentiment", fill=BLUE)

    # Paired comparison.
    line(ax, (0.16, 0.25), (0.16, 0.205))
    line(ax, (0.52, 0.25), (0.52, 0.205))
    line(ax, (0.16, 0.205), (0.52, 0.205))
    arrow(ax, (0.34, 0.205), (0.34, 0.16))
    box(ax, 0.22, 0.075, 0.24, 0.085, "Paired forecast\ncomparison", fill=BLUE)

    # Controls remain visible but outside the principal causal path.
    box(ax, 0.68, 0.235, 0.28, 0.105, "Forecast controls\nshuffled, lagged, news-only,\nand random features", fill=GREY, fontsize=7.3)
    box(ax, 0.68, 0.075, 0.28, 0.10, "Compute-matched controls\nsingle pass and\nself-consistency", fill=GREY, fontsize=7.3)

    save(fig, "figure5_multimodal_debate_and_falsification")


def figure6_regime_shap():
    fig, ax = canvas(7.5, 4.8)

    box(ax, 0.055, 0.83, 0.19, 0.075, "Past market returns\nand volatility", fill=CREAM)
    arrow(ax, (0.15, 0.83), (0.15, 0.755))
    box(ax, 0.045, 0.665, 0.21, 0.09, "Causal daily\nregime labeling", fill=BLUE)

    group(ax, 0.035, 0.35, 0.23, 0.24, "Market regime")
    box(ax, 0.065, 0.485, 0.17, 0.045, "Bull", fill=GREEN, fontsize=7.8)
    box(ax, 0.065, 0.425, 0.17, 0.045, "Sideway", fill=CREAM, fontsize=7.8)
    box(ax, 0.065, 0.365, 0.17, 0.045, "Bear", fill=GREY, fontsize=7.8)
    arrow(ax, (0.15, 0.665), (0.15, 0.59))

    arrow(ax, (0.265, 0.47), (0.36, 0.47))
    group(ax, 0.36, 0.26, 0.27, 0.37, "Training fold only")
    box(ax, 0.395, 0.505, 0.20, 0.055, "Fit reference model", fill=GREEN, fontsize=7.8)
    arrow(ax, (0.495, 0.505), (0.495, 0.46))
    box(ax, 0.395, 0.405, 0.20, 0.055, "Compute SHAP values", fill=BLUE, fontsize=7.8)
    arrow(ax, (0.495, 0.405), (0.495, 0.365))
    box(ax, 0.38, 0.285, 0.23, 0.08, "Select features within\neach regime", fill=GREEN)

    arrow(ax, (0.61, 0.325), (0.70, 0.325))
    box(ax, 0.70, 0.275, 0.23, 0.10, "Apply frozen feature set\nto held-out year", fill=BLUE)
    arrow(ax, (0.815, 0.275), (0.815, 0.205))
    box(ax, 0.70, 0.115, 0.23, 0.09, "Five-model paired\nregime comparison", fill=GREEN)

    # Inference footer.
    group(ax, 0.07, 0.04, 0.54, 0.15, "Inference")
    box(ax, 0.095, 0.06, 0.13, 0.045, "Seed repeats", fill=GREY, fontsize=7.3)
    box(ax, 0.275, 0.06, 0.13, 0.045, "Year effects", fill=GREY, fontsize=7.3)
    box(ax, 0.455, 0.06, 0.13, 0.045, "Holm adjustment", fill=GREY, fontsize=7.3)
    arrow(ax, (0.225, 0.082), (0.275, 0.082))
    arrow(ax, (0.405, 0.082), (0.455, 0.082))

    save(fig, "figure6_regime_shap_and_inference")


def main():
    figure1_overall_pipeline()
    figure2_news_pipeline()
    figure3_point_in_time()
    figure4_vmd_models()
    figure5_multimodal()
    figure6_regime_shap()
    print(f"Generated six simple Methods figures in: {OUT}")


if __name__ == "__main__":
    main()
