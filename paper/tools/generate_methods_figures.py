"""Generate journal-style Methods figures for Sections 3.1--3.8.

The figures intentionally use a restrained, single-accent visual language and
avoid decorative elements.  Outputs are written as editable SVG, vector PDF,
and 400-dpi PNG files suitable for Microsoft Word.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "methods_v1"
STOCKTBSA = (
    ROOT
    / "data-raw"
    / "track_b"
    / "Bilingual_StockTBSA"
    / "Thai_Financial_TBSA_dataset_Updated.json"
)
DOMAIN_SHIFT = ROOT / "outputs" / "track_b" / "forward_news" / "domain_shift_audit.csv"

INK = "#202427"
GRAY = "#6A7075"
LIGHT_GRAY = "#D7DDE0"
PALE = "#F5F7F8"
BLUE = "#3F667A"
BLUE_LIGHT = "#E8EFF2"
BLUE_MID = "#B9CDD6"
NEG = "#A86666"
NEUTRAL = "#B8B8B8"
POS = "#52798C"
WHITE = "#FFFFFF"


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.2,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": WHITE,
        "figure.facecolor": WHITE,
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
    text: str,
    *,
    face: str = WHITE,
    edge: str = INK,
    lw: float = 0.9,
    fontsize: float = 7.6,
    weight: str = "normal",
    align: str = "center",
    pad: float = 0.008,
    linestyle: str = "-",
    rounding: float = 0.008,
    zorder: int = 2,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={rounding}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    tx = x + w / 2 if align == "center" else x + 0.018
    ax.text(
        tx,
        y + h / 2,
        text,
        ha=align,
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=INK,
        linespacing=1.22,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    lw: float = 1.0,
    style: str = "-|>",
    mutation: float = 9,
    linestyle: str = "-",
    connectionstyle: str = "arc3",
    zorder: int = 1,
):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def panel_label(ax, label: str, title: str, x: float, y: float):
    ax.text(x, y, label, fontsize=9.5, fontweight="bold", ha="left", va="top", color=INK, transform=ax.transAxes)
    ax.text(x + 0.055, y, title, fontsize=9.0, fontweight="bold", ha="left", va="top", color=INK, transform=ax.transAxes)


def section_rule(ax, y: float, x0: float = 0.03, x1: float = 0.97):
    ax.plot([x0, x1], [y, y], color=LIGHT_GRAY, lw=0.8, zorder=0)


def save(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def draw_pipeline_figure():
    fig, ax = canvas(7.2, 5.0)
    panel_label(ax, "A", "Point-in-time foundation", 0.03, 0.975)
    box(
        ax,
        0.08,
        0.845,
        0.84,
        0.085,
        "Market-data governance and target definition\n"
        "Provenance · Asia/Bangkok cutoff · purged t+1 labels · train-only fitting",
        face=BLUE_LIGHT,
        edge=BLUE,
        fontsize=8.0,
        weight="bold",
    )
    for x in (0.18, 0.50, 0.82):
        arrow(ax, (0.50, 0.845), (x, 0.775), color=GRAY)

    panel_label(ax, "B", "Three architecture-paired audit branches", 0.03, 0.785)
    branch_specs = [
        (
            0.035,
            "Numerical / denoising",
            "Full TA (116)\nvs\nFull TA + causal VMD (122)",
        ),
        (
            0.355,
            "Multimodal / LLM",
            "Market-only\nvs predicted news\n+ falsification controls",
        ),
        (
            0.675,
            "Regime / explainability",
            "Causal Bull–Sideway–Bear\n+ train-only SHAP\n+ size-matched comparator",
        ),
    ]
    for x, heading, body in branch_specs:
        box(ax, x, 0.54, 0.29, 0.20, "", face=WHITE, edge=BLUE, lw=1.0)
        ax.add_patch(Rectangle((x, 0.685), 0.29, 0.055, facecolor=BLUE, edgecolor=BLUE, lw=0, zorder=3))
        ax.text(x + 0.145, 0.712, heading, ha="center", va="center", color=WHITE, fontsize=7.8, fontweight="bold", zorder=4)
        ax.text(x + 0.145, 0.615, body, ha="center", va="center", color=INK, fontsize=7.5, linespacing=1.25)
        arrow(ax, (x + 0.145, 0.54), (0.50, 0.455), color=GRAY)

    box(
        ax,
        0.08,
        0.355,
        0.84,
        0.095,
        "Common fixed architecture panel\n"
        "LSTM  |  CNN  |  LSTM–CNN  |  LSTM–Attention  |  LSTM–CNN–Attention",
        face=PALE,
        edge=INK,
        fontsize=8.0,
        weight="bold",
    )
    arrow(ax, (0.50, 0.355), (0.50, 0.292), color=GRAY)
    box(
        ax,
        0.20,
        0.205,
        0.60,
        0.083,
        "Common SET50 next-day direction contract\nBalanced accuracy as the primary endpoint",
        face=BLUE_LIGHT,
        edge=BLUE,
        fontsize=8.0,
        weight="bold",
    )

    for x in (0.20, 0.50, 0.80):
        arrow(ax, (0.50, 0.205), (x, 0.145), color=GRAY)
    for x, text in [
        (0.04, "Historical audit\n2022–2025"),
        (0.36, "Source-contingent\npartial-2026"),
        (0.68, "Frozen SET100\nsame-exchange transfer"),
    ]:
        box(ax, x, 0.045, 0.28, 0.095, text, face=WHITE, edge=INK, fontsize=7.5, weight="bold")
    save(fig, "figure1_reliability_audit_pipeline_simple")


def load_news_counts():
    records = json.loads(STOCKTBSA.read_text(encoding="utf-8"))
    counts: dict[int, Counter] = defaultdict(Counter)
    articles = Counter()
    for article in records:
        year = int(article["Year"])
        articles[year] += 1
        for item in article.get("Ticker_sentiments", []):
            counts[year][str(item.get("sentiment", "")).strip().lower()] += 1
    forward = pd.read_csv(DOMAIN_SHIFT)
    return counts, articles, forward


def draw_news_figure():
    counts, articles, forward = load_news_counts()
    fig = plt.figure(figsize=(7.2, 7.0))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.55, 1.05], hspace=0.52, wspace=0.34)

    # A: coverage timeline
    ax = fig.add_subplot(grid[0, :])
    panel_label(ax, "A", "Temporal coverage and model status", 0.0, 1.12)
    ax.set_xlim(2011.7, 2025.85)
    ax.set_ylim(-0.65, 3.75)
    bars = [
        (2.9, 2012, 2025.0, BLUE, "SET50 market data"),
        (1.9, 2018, 2023.0, GRAY, "Labelled StockTBSA"),
        (0.9, 2019, 2023.0, BLUE_MID, "Expanding OOS sentiment"),
        (-0.1, 2024, 2025.0, INK, "Frozen 2018–2023 model → official SET headlines"),
    ]
    for y, start, end, color, label in bars:
        ax.barh(y, end - start + 1, left=start - 0.5, height=0.48, color=color, edgecolor=INK, lw=0.55)
        if start == 2024:
            ax.text(2024.5, y, "Frozen → SET", va="center", ha="center", color=WHITE, fontsize=6.4, fontweight="bold")
        else:
            ax.text(start - 0.37, y, label, va="center", ha="left", color=WHITE if color in (BLUE, GRAY, INK) else INK, fontsize=7.3, fontweight="bold")
    ax.add_patch(Rectangle((2011.5, 1.63), 6.0, 0.54, facecolor=WHITE, edgecolor=LIGHT_GRAY, hatch="////", lw=0.6))
    ax.text(2014.5, 1.90, "No comparable ticker + sentiment labels", ha="center", va="center", fontsize=7.1, color=GRAY)
    ax.set_xticks(range(2012, 2026))
    ax.set_xticklabels(range(2012, 2026), rotation=45, ha="right")
    ax.set_yticks([])
    ax.tick_params(axis="x", length=2.5, color=GRAY)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT_GRAY)

    # B: labelled class distribution
    ax = fig.add_subplot(grid[1, 0])
    panel_label(ax, "B", "Valid labelled article–ticker pairs", 0.0, 1.12)
    years = np.array(sorted(y for y in counts if 2018 <= y <= 2023))
    neg = np.array([counts[int(y)]["negative"] for y in years])
    neu = np.array([counts[int(y)]["neutral"] for y in years])
    pos = np.array([counts[int(y)]["positive"] for y in years])
    ax.bar(years, neg, color=NEG, edgecolor=WHITE, lw=0.35, label="Negative")
    ax.bar(years, neu, bottom=neg, color=NEUTRAL, edgecolor=WHITE, lw=0.35, label="Neutral")
    ax.bar(years, pos, bottom=neg + neu, color=POS, edgecolor=WHITE, lw=0.35, label="Positive")
    totals = neg + neu + pos
    for x, total in zip(years, totals):
        ax.text(x, total + 45, f"{total:,}", ha="center", va="bottom", fontsize=6.8, color=INK)
    ax.set_ylabel("Valid pairs")
    ax.set_xticks(years)
    ax.set_ylim(0, max(totals) * 1.18)
    ax.grid(axis="y", color=LIGHT_GRAY, lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15), columnspacing=0.8, handlelength=1.1)
    ax.text(2023, 46, "92", ha="center", va="center", fontsize=5.8, color=WHITE, fontweight="bold")
    ax.text(2023, 92 + 585 / 2, "585", ha="center", va="center", fontsize=5.8, color=INK, fontweight="bold")
    ax.text(2023, 92 + 585 + 656 / 2, "656", ha="center", va="center", fontsize=5.8, color=WHITE, fontweight="bold")

    # C: forward filtration
    ax = fig.add_subplot(grid[1, 1])
    panel_label(ax, "C", "Official SET forward-news filtration", 0.0, 1.12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fwd = forward[forward["year"].isin([2024, 2025])].set_index("year")
    y_positions = [0.62, 0.23]
    column_specs = [
        (0.14, 0.17, "Official\nheadlines"),
        (0.36, 0.17, "PIT SET50\npairs"),
        (0.58, 0.17, "Relevant\npairs"),
        (0.80, 0.16, "Sessions\nwith news"),
    ]
    for x, width, label in column_specs:
        ax.text(x + width / 2, 0.88, label, ha="center", va="center", fontsize=5.7, color=GRAY, fontweight="bold")
    for year, y in zip((2024, 2025), y_positions):
        row = fwd.loc[year]
        ax.text(0.10, y + 0.07, str(year), ha="right", va="center", fontsize=7.6, fontweight="bold")
        values = [
            int(row["pairs_before_membership"]),
            int(row["pairs_after_membership"]),
            int(row["selected_pairs"]),
            225 if year == 2024 else 224,
        ]
        for i, (value, (x, width, _)) in enumerate(zip(values, column_specs)):
            box(ax, x, y, width, 0.14, f"{value:,}", face=BLUE_LIGHT if i in (1, 2) else WHITE, edge=BLUE if i in (1, 2) else INK, fontsize=7.8, weight="bold")
            if i < 3:
                next_x = column_specs[i + 1][0]
                arrow(ax, (x + width, y + 0.07), (next_x - 0.01, y + 0.07), mutation=6)
    ax.text(0.5, 0.01, "2024–2025 headlines are unlabeled; intrinsic accuracy is not claimed.", ha="center", va="bottom", fontsize=6.9, color=GRAY)

    # D: causal assignment
    ax = fig.add_subplot(grid[2, :])
    panel_label(ax, "D", "Conservative point-in-time news assignment", 0.0, 1.08)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = [
        (0.01, "Headline dated d\nno reliable timestamp"),
        (0.265, "First SET session\nstrictly after d"),
        (0.52, "Eight daily features\nsentiment · ratios\ncounts / availability"),
        (0.775, "Forecast at t\nfor direction t+1"),
    ]
    for i, (x, label) in enumerate(nodes):
        box(ax, x, 0.36, 0.205, 0.30, label, face=BLUE_LIGHT if i in (1, 2) else WHITE, edge=BLUE if i in (1, 2) else INK, fontsize=6.4, weight="bold")
        if i < len(nodes) - 1:
            arrow(ax, (x + 0.205, 0.51), (nodes[i + 1][0] - 0.012, 0.51))
    ax.text(0.50, 0.18, "No same-day assignment when publication time is unknown · No pseudo-label retraining · No Optuna tuning", ha="center", va="center", fontsize=7.1, color=GRAY)
    save(fig, "figure2_news_data_and_oos_sentiment")


def draw_point_in_time_figure():
    fig, ax = canvas(7.2, 5.5)
    panel_label(ax, "A", "Prediction-time contract at an evaluation boundary", 0.03, 0.975)
    y = 0.79
    xs = np.linspace(0.10, 0.90, 7)
    labels = ["t−W+1", "…", "t−2", "t−1", "t", "t+1", "t+2"]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        face = BLUE_LIGHT if i <= 4 else WHITE
        edge = BLUE if i <= 4 else INK
        ax.add_patch(FancyBboxPatch((x - 0.045, y - 0.035), 0.09, 0.07, boxstyle="round,pad=0.004", facecolor=face, edgecolor=edge, lw=0.8))
        ax.text(x, y, lab, ha="center", va="center", fontsize=7.5, fontweight="bold")
    arrow(ax, (xs[0], y - 0.06), (xs[4], y - 0.06), color=BLUE, lw=1.3)
    ax.text((xs[0] + xs[4]) / 2, y - 0.10, "Input sequence: information available through close t", ha="center", va="top", fontsize=7.3, color=BLUE)
    arrow(ax, (xs[4], y + 0.05), (xs[5], y + 0.05), color=INK, lw=1.1)
    ax.text((xs[4] + xs[5]) / 2, y + 0.095, "Predict C(t+1)", ha="center", va="bottom", fontsize=7.3, fontweight="bold")
    ax.plot([xs[5], xs[5]], [0.67, 0.89], color=NEG, lw=1.2, linestyle="--")
    ax.text(xs[5] + 0.012, 0.68, "Label becomes observable\nonly on t+1", ha="left", va="bottom", fontsize=7.0, color=NEG)
    box(ax, 0.06, 0.56, 0.88, 0.07, "Boundary rule: retain a training row only when Label_Date < first evaluation Date", face=PALE, edge=INK, fontsize=7.7, weight="bold")

    section_rule(ax, 0.515)
    panel_label(ax, "B", "Expanding model selection and frozen outer evaluation", 0.03, 0.49)
    years = list(range(2012, 2026))
    x0, x1 = 0.18, 0.96
    colw = (x1 - x0) / len(years)
    for i, yr in enumerate(years):
        ax.text(x0 + (i + 0.5) * colw, 0.445, str(yr)[-2:], ha="center", va="center", fontsize=6.3, color=GRAY)
    rows = [
        ("Selection 1", 2017, 2018, "V"),
        ("Selection 2", 2018, 2019, "V"),
        ("Selection 3", 2019, 2020, "V"),
        ("Selection 4", 2020, 2021, "V"),
        ("Outer 1", 2021, 2022, "T"),
        ("Outer 2", 2022, 2023, "T"),
        ("Outer 3", 2023, 2024, "T"),
        ("Outer 4", 2024, 2025, "T"),
    ]
    y_start, dy = 0.407, 0.034
    for r, (name, train_end, eval_year, marker) in enumerate(rows):
        yy = y_start - r * dy
        ax.text(0.16, yy, name, ha="right", va="center", fontsize=6.6, color=INK)
        for i, yr in enumerate(years):
            x = x0 + i * colw
            if yr <= train_end:
                face, edge = BLUE_LIGHT, BLUE
            elif yr == eval_year:
                face, edge = (PALE, INK) if marker == "V" else (BLUE, BLUE)
            else:
                face, edge = WHITE, LIGHT_GRAY
            ax.add_patch(Rectangle((x, yy - 0.014), colw * 0.92, 0.028, facecolor=face, edgecolor=edge, lw=0.45))
            if yr == eval_year:
                ax.text(x + colw * 0.46, yy, marker, ha="center", va="center", fontsize=5.8, color=WHITE if marker == "T" else INK, fontweight="bold")

    ax.text(0.18, 0.025, "Legend:", ha="left", va="center", fontsize=6.4, fontweight="bold")
    ax.add_patch(Rectangle((0.255, 0.012), 0.03, 0.026, facecolor=BLUE_LIGHT, edgecolor=BLUE, lw=0.5))
    ax.text(0.292, 0.025, "Training", va="center", fontsize=6.3)
    ax.add_patch(Rectangle((0.385, 0.012), 0.03, 0.026, facecolor=PALE, edgecolor=INK, lw=0.5))
    ax.text(0.422, 0.025, "Validation", va="center", fontsize=6.3)
    ax.add_patch(Rectangle((0.545, 0.012), 0.03, 0.026, facecolor=BLUE, edgecolor=BLUE, lw=0.5))
    ax.text(0.582, 0.025, "Outer test", va="center", fontsize=6.3)
    ax.text(0.95, 0.025, "Seeds: 42, 123, 456, 789, 2025", ha="right", va="center", fontsize=6.2, color=GRAY)

    box(
        ax,
        0.62,
        0.075,
        0.34,
        0.065,
        "Frozen windows\nLSTM 5 · CNN 20\nLSTM–CNN 20 · LSTM–Attention 10\nLSTM–CNN–Attention 20",
        face=WHITE,
        edge=INK,
        fontsize=5.1,
        align="left",
    )
    box(
        ax,
        0.18,
        0.075,
        0.40,
        0.065,
        "Training-only fitting\nScaler · feature selection · model weights\nregime thresholds · SHAP ranks",
        face=WHITE,
        edge=INK,
        fontsize=5.3,
        align="left",
    )
    save(fig, "figure3_point_in_time_expanding_evaluation")


def draw_features_architectures_figure():
    fig, ax = canvas(7.2, 7.1)
    panel_label(ax, "A", "Numerical feature construction", 0.03, 0.98)
    feature_groups = [
        "Daily OHLCV\n+ completed weekly /\nmonthly OHLCV",
        "Returns and lags\n1, 3, 5, 10, 20, 60 d",
        "Trend · volatility\nmomentum · ROC",
        "Candlestick\ncross-timeframe\nvolume ratios",
        "RSI · MACD · ADX\nCCI · Williams %R",
    ]
    xs = [0.025, 0.22, 0.415, 0.61, 0.805]
    for x, label in zip(xs, feature_groups):
        box(ax, x, 0.84, 0.17, 0.09, label, face=WHITE, edge=INK, fontsize=5.3, weight="bold")
        arrow(ax, (x + 0.085, 0.84), (0.50, 0.775), mutation=7)
    box(ax, 0.36, 0.71, 0.28, 0.06, "Full-TA control: 116 variables", face=BLUE_LIGHT, edge=BLUE, fontsize=7.6, weight="bold")

    section_rule(ax, 0.675)
    panel_label(ax, "B", "Causal rolling VMD addition", 0.03, 0.65)
    vmd_nodes = [
        (0.035, 0.50, 0.16, "60 closes\nt−59 … t"),
        (0.235, 0.50, 0.18, "VMD at each t\nK=5 · α=1000"),
        (0.455, 0.50, 0.18, "Identify highest\ncentre frequency"),
        (0.675, 0.50, 0.29, "Retain 4 IMFs + denoised close\n+ removed-mode energy ratio\n= 6 causal features"),
    ]
    for i, (x, y, w, label) in enumerate(vmd_nodes):
        box(ax, x, y, w, 0.105, label, face=BLUE_LIGHT if i in (1, 3) else WHITE, edge=BLUE if i in (1, 3) else INK, fontsize=7.0, weight="bold")
        if i < len(vmd_nodes) - 1:
            arrow(ax, (x + w, y + 0.052), (vmd_nodes[i + 1][0] - 0.012, y + 0.052))
    ax.text(0.50, 0.468, "No full-series decomposition · No future observations", ha="center", va="center", fontsize=6.7, color=NEG, fontweight="bold")
    box(ax, 0.34, 0.385, 0.32, 0.06, "Full TA + VMD: 122 variables", face=PALE, edge=INK, fontsize=7.5, weight="bold")

    section_rule(ax, 0.35)
    panel_label(ax, "C", "Five compact forecasting architectures", 0.03, 0.325)
    architecture_rows = [
        ("LSTM  (W=5)", ["LSTM(16)", "Dense(8, ReLU)", "Linear"]),
        ("CNN  (W=20)", ["Causal Conv1D(32, k=3)", "Global average pool", "Dense(8)", "Linear"]),
        ("LSTM–CNN  (W=20)", ["LSTM(16), seq.", "Causal Conv1D", "Global pool", "Dense(8)", "Linear"]),
        ("LSTM–Attention  (W=10)", ["LSTM(16), seq.", "2-head causal attn.", "Global pool", "Dense(8)", "Linear"]),
        ("LSTM–CNN–Attention  (W=20)", ["LSTM(16)", "Causal Conv1D", "2-head attn.", "Global pool", "Dense(8)", "Linear"]),
    ]
    y0, dy = 0.275, 0.052
    for row, (name, nodes) in enumerate(architecture_rows):
        yy = y0 - row * dy
        ax.text(0.23, yy, name, ha="right", va="center", fontsize=5.9, fontweight="bold")
        start_x = 0.25
        usable = 0.705
        gap = 0.008
        node_w = (usable - gap * (len(nodes) - 1)) / len(nodes)
        for i, node in enumerate(nodes):
            x = start_x + i * (node_w + gap)
            ax.add_patch(FancyBboxPatch((x, yy - 0.017), node_w, 0.034, boxstyle="round,pad=0.003", facecolor=BLUE_LIGHT if i == 0 else WHITE, edgecolor=BLUE if i == 0 else INK, lw=0.55))
            ax.text(x + node_w / 2, yy, node, ha="center", va="center", fontsize=4.6)
            if i < len(nodes) - 1:
                arrow(ax, (x + node_w, yy), (x + node_w + gap * 0.85, yy), mutation=5, lw=0.6)
    ax.text(0.50, 0.025, "Common training: Adam · MSE · 20 epochs · batch 32 · shuffle=false · deterministic operations", ha="center", va="bottom", fontsize=6.8, color=GRAY)
    save(fig, "figure4_numerical_vmd_and_architectures")


def draw_multimodal_figure():
    fig, ax = canvas(7.2, 6.4)
    panel_label(ax, "A", "Forecasting falsification arms", 0.03, 0.98)
    inputs = [
        ("Market-only", WHITE, INK),
        ("Observed predicted news", BLUE_LIGHT, BLUE),
        ("Date-shuffled news", WHITE, INK),
        ("News lagged 5 rows", WHITE, INK),
        ("Matched random features", WHITE, INK),
        ("News-only", WHITE, INK),
    ]
    for i, (label, face, edge) in enumerate(inputs):
        row, col = divmod(i, 3)
        x = 0.04 + col * 0.315
        y = 0.82 - row * 0.095
        box(ax, x, y, 0.28, 0.06, label, face=face, edge=edge, fontsize=7.1, weight="bold")
        arrow(ax, (x + 0.14, y), (0.50, 0.64), mutation=6, lw=0.75)
    box(ax, 0.28, 0.56, 0.44, 0.075, "Same dates · frozen windows · five seeds\nfive architectures · identical training budget", face=PALE, edge=INK, fontsize=6.7, weight="bold")
    arrow(ax, (0.50, 0.56), (0.50, 0.49))
    box(ax, 0.32, 0.425, 0.36, 0.06, "Paired balanced-accuracy contrasts", face=BLUE_LIGHT, edge=BLUE, fontsize=7.5, weight="bold")

    section_rule(ax, 0.39)
    panel_label(ax, "B", "Locked intrinsic Bull/Bear/Leader benchmark", 0.03, 0.365)
    box(ax, 0.04, 0.245, 0.18, 0.075, "Locked 2023\n1,333 pairs\n738 articles", face=WHITE, edge=INK, fontsize=5.9, weight="bold")
    arrow(ax, (0.22, 0.282), (0.28, 0.282))
    box(ax, 0.28, 0.255, 0.13, 0.055, "Bull worker", face=BLUE_LIGHT, edge=BLUE, fontsize=6.8, weight="bold")
    box(ax, 0.28, 0.185, 0.13, 0.055, "Bear worker", face=BLUE_LIGHT, edge=BLUE, fontsize=6.8, weight="bold")
    arrow(ax, (0.41, 0.282), (0.48, 0.247), mutation=6)
    arrow(ax, (0.41, 0.212), (0.48, 0.247), mutation=6)
    box(ax, 0.48, 0.215, 0.13, 0.065, "Leader", face=BLUE_LIGHT, edge=BLUE, fontsize=7.2, weight="bold")
    arrow(ax, (0.61, 0.247), (0.68, 0.247))
    box(ax, 0.68, 0.195, 0.28, 0.105, "Intrinsic controls\nSingle pass · SC3 equal calls\nSC4 near-cost (±15%)", face=WHITE, edge=INK, fontsize=6.8, weight="bold")
    ax.text(0.50, 0.158, "Accuracy primary · Macro-F1 supporting · article-cluster uncertainty · Holm-adjusted Leader contrasts", ha="center", va="center", fontsize=6.7, color=GRAY)

    section_rule(ax, 0.125)
    panel_label(ax, "C", "Downstream dated-news routes", 0.03, 0.118)
    box(ax, 0.10, 0.048, 0.25, 0.025, "Expanding Local NLP", face=WHITE, edge=INK, fontsize=5.8, weight="bold")
    box(ax, 0.10, 0.012, 0.25, 0.025, "Dated Leader outputs", face=WHITE, edge=INK, fontsize=5.8, weight="bold")
    arrow(ax, (0.35, 0.061), (0.58, 0.040), mutation=6)
    arrow(ax, (0.35, 0.025), (0.58, 0.040), mutation=6)
    box(ax, 0.58, 0.015, 0.37, 0.050, "Identical-cohort SET50\npaired forecasting comparison", face=BLUE_LIGHT, edge=BLUE, fontsize=5.8, weight="bold")
    save(fig, "figure5_multimodal_falsification_and_leader")


def draw_regime_inference_figure():
    fig, ax = canvas(7.2, 6.6)
    panel_label(ax, "A", "Causal daily regime routing", 0.03, 0.98)
    box(ax, 0.04, 0.82, 0.20, 0.09, "Past-only inputs\nReturns 1–60 d\nVolatility 20/60 · ADX(14)", face=WHITE, edge=INK, fontsize=5.9, weight="bold")
    arrow(ax, (0.24, 0.865), (0.30, 0.865))
    box(ax, 0.30, 0.82, 0.20, 0.09, "Risk-adjusted trend\n× ADX strength\nEWMA(3)", face=BLUE_LIGHT, edge=BLUE, fontsize=6.0, weight="bold")
    arrow(ax, (0.50, 0.865), (0.56, 0.865))
    box(ax, 0.56, 0.82, 0.19, 0.09, "Training-only deadband\nθ = Q₀.₃₅(|Sₜ|)", face=WHITE, edge=INK, fontsize=6.0, weight="bold")
    arrow(ax, (0.75, 0.865), (0.80, 0.865))
    box(ax, 0.80, 0.82, 0.16, 0.09, "Daily hard route\nat close t", face=BLUE_LIGHT, edge=BLUE, fontsize=6.0, weight="bold")
    for x, label, hatch in [(0.28, "Bear\nSₜ < −θ", "////"), (0.46, "Sideway\n|Sₜ| ≤ θ", ""), (0.64, "Bull\nSₜ > θ", "\\\\")]:
        patch = FancyBboxPatch((x, 0.71), 0.14, 0.065, boxstyle="round,pad=0.004", facecolor=WHITE, edgecolor=INK, lw=0.7, hatch=hatch)
        ax.add_patch(patch)
        ax.text(x + 0.07, 0.742, label, ha="center", va="center", fontsize=6.5, fontweight="bold")
        arrow(ax, (0.88, 0.82), (x + 0.07, 0.775), mutation=6, lw=0.7)

    section_rule(ax, 0.665)
    panel_label(ax, "B", "Train-only SHAP selection and capacity controls", 0.03, 0.64)
    box(ax, 0.04, 0.51, 0.18, 0.075, "122-feature pool\nwithin each train split", face=WHITE, edge=INK, fontsize=5.8, weight="bold")
    arrow(ax, (0.22, 0.547), (0.28, 0.547))
    box(ax, 0.28, 0.49, 0.20, 0.115, "Gradient Explainer\n100 backgrounds\n≤128 ranking sequences\n200 samples", face=BLUE_LIGHT, edge=BLUE, fontsize=5.7, weight="bold")
    arrow(ax, (0.48, 0.547), (0.54, 0.547))
    box(ax, 0.54, 0.49, 0.18, 0.115, "Candidate k\n10 · 20 · 30 · 40\n60 · 80 · 100 · 122", face=WHITE, edge=INK, fontsize=5.8, weight="bold")
    arrow(ax, (0.72, 0.547), (0.78, 0.547))
    box(ax, 0.78, 0.49, 0.18, 0.115, "One-SE rule\n+ error / RMSE\n+ Jaccard guardrails", face=BLUE_LIGHT, edge=BLUE, fontsize=5.8, weight="bold")
    ax.text(0.50, 0.445, "Frozen counts: Global 122 · Bull 30 · Sideway 122 · Bear 80", ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK)
    arms = ["Global-All", "Global3-All", "Global-SHAP", "Global-Spearman", "Regime-All", "Regime-SHAP", "Regime-Spearman"]
    ax.text(0.50, 0.405, "Seven outer arms:  " + "  |  ".join(arms), ha="center", va="center", fontsize=5.8, color=GRAY)

    section_rule(ax, 0.37)
    panel_label(ax, "C", "Temporal inference hierarchy", 0.03, 0.345)
    hierarchy = [
        (0.04, "Predictions\nper date"),
        (0.23, "Average seeds\nwithin model–year"),
        (0.44, "Four annual\npaired effects"),
        (0.65, "Exact two-sided\nsign-flip test"),
        (0.84, "Holm familywise\nadjustment"),
    ]
    for i, (x, label) in enumerate(hierarchy):
        box(ax, x, 0.22, 0.13, 0.075, label, face=BLUE_LIGHT if i in (1, 3) else WHITE, edge=BLUE if i in (1, 3) else INK, fontsize=5.5, weight="bold")
        if i < len(hierarchy) - 1:
            arrow(ax, (x + 0.13, 0.257), (hierarchy[i + 1][0] - 0.01, 0.257), mutation=6)
    ax.text(0.50, 0.175, "Seeds are repetitions, not independent market samples", ha="center", va="center", fontsize=7.1, color=NEG, fontweight="bold")
    box(ax, 0.11, 0.055, 0.35, 0.075, "Sensitivity\n10-day circular moving-block\nbootstrap", face=WHITE, edge=INK, fontsize=5.9, weight="bold")
    box(ax, 0.54, 0.055, 0.35, 0.075, "Intrinsic LLM uncertainty\n5,000 article-cluster\nbootstrap replicates", face=WHITE, edge=INK, fontsize=5.9, weight="bold")
    save(fig, "figure6_regime_shap_and_inference")


def main():
    draw_pipeline_figure()
    draw_news_figure()
    draw_point_in_time_figure()
    draw_features_architectures_figure()
    draw_multimodal_figure()
    draw_regime_inference_figure()
    print(f"Generated Methods figures in: {OUT}")


if __name__ == "__main__":
    main()
