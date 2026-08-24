from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(r"D:\SET50_direction_prediction_paper")
DATA_PATH = ROOT / "data-raw" / "SET50_days.csv"
ASSET_DIR = ROOT / "paper" / "assets"
PNG_DIR = ROOT / "output" / "graphical_abstract_v2"
PDF_DIR = ROOT / "output" / "pdf"
PNG_PATH = PNG_DIR / "graphical_abstract_set_reliability_minimal_v2.png"
PDF_PATH = PDF_DIR / "graphical_abstract_set_reliability_minimal_v2.pdf"


BG = "#FCFCFB"
INK = "#26373F"
MUTED = "#687980"
PRIMARY = "#7896A3"
PRIMARY_DARK = "#486775"
TINT = "#EDF3F5"
TINT_LIGHT = "#F5F8F8"
SAGE = "#E4ECE8"
LINE = "#D5DFE2"
WHITE = "#FFFFFF"


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def rounded(ax, x, y, w, h, fill, edge="none", lw=0.7, radius=0.018, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def flow_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.5,
            color=PRIMARY,
            shrinkA=2,
            shrinkB=2,
            zorder=5,
        )
    )


def load_set50():
    dates, closes = [], []
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                dates.append(datetime.strptime(row["วันเดือนปี"], "%m/%d/%Y"))
                closes.append(float(row["ล่าสุด"].replace(",", "")))
            except (KeyError, TypeError, ValueError):
                continue
    order = np.argsort(np.array(dates, dtype="datetime64[ns]"))
    dates = np.asarray(dates, dtype=object)[order]
    closes = np.asarray(closes, dtype=float)[order]
    keep = np.asarray([date.year >= 2012 for date in dates])
    return dates[keep], closes[keep]


def draw_news_icon(ax, x, y):
    for dx, dy in [(0.010, 0.006), (0.005, 0.003), (0, 0)]:
        rounded(ax, x + dx, y + dy, 0.027, 0.040, WHITE, edge=PRIMARY, lw=0.55, radius=0.004, z=4)
    for offset in [0.028, 0.020, 0.012]:
        ax.plot([x + 0.005, x + 0.022], [y + offset, y + offset], color=PRIMARY, lw=0.45, zorder=5)


def draw_input_column(fig, ax, dates, closes):
    x, y, w, h = 0.040, 0.235, 0.235, 0.535
    rounded(ax, x, y, w, h, TINT_LIGHT, radius=0.022)

    chart = fig.add_axes([0.061, 0.565, 0.193, 0.137])
    chart.set_facecolor(TINT_LIGHT)
    chart.axvspan(datetime(2022, 1, 1), datetime(2025, 12, 31), color="#DFE9ED", linewidth=0)
    chart.plot(dates, closes, color=PRIMARY_DARK, linewidth=1.25)
    chart.set_xlim(datetime(2012, 1, 1), datetime(2025, 12, 31))
    chart.set_yticks([])
    chart.set_xticks([datetime(2012, 1, 1), datetime(2022, 1, 1), datetime(2025, 1, 1)])
    chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    chart.tick_params(axis="x", labelsize=6.2, length=0, colors=MUTED, pad=3)
    for spine in chart.spines.values():
        spine.set_visible(False)
    chart.text(0.02, 0.90, "SET50 daily close", transform=chart.transAxes, fontsize=7.1, fontweight="bold", color=INK)
    chart.text(0.73, 0.08, "outer tests", transform=chart.transAxes, fontsize=5.7, color=PRIMARY_DARK)

    ax.plot([x + 0.025, x + w - 0.025], [0.535, 0.535], color=LINE, lw=0.7)
    draw_news_icon(ax, x + 0.027, 0.469)
    ax.text(x + 0.076, 0.505, "Dated financial news", ha="left", va="center", fontsize=8.2, fontweight="bold", color=INK)
    ax.text(x + 0.076, 0.477, "2018-2025  |  out-of-sample sentiment", ha="left", va="center", fontsize=6.7, color=MUTED)

    ax.plot([x + 0.025, x + w - 0.025], [0.445, 0.445], color=LINE, lw=0.7)
    ax.text(x + 0.027, 0.414, "FIVE FROZEN ARCHITECTURES", ha="left", va="center", fontsize=6.5, fontweight="bold", color=PRIMARY_DARK)
    ax.text(x + 0.027, 0.382, "LSTM  |  CNN  |  LSTM-CNN", ha="left", va="center", fontsize=7.2, color=INK)
    ax.text(x + 0.027, 0.353, "LSTM-Attention  |  LSTM-CNN-Attention", ha="left", va="center", fontsize=6.6, color=INK)

    rounded(ax, x + 0.027, 0.280, w - 0.054, 0.045, SAGE, radius=0.012, z=2)
    ax.text(x + w / 2, 0.3025, "Target: next-session Up / Down", ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK)


def draw_method_column(ax):
    x, y, w, h = 0.350, 0.235, 0.285, 0.535
    rounded(ax, x, y, w, h, TINT, radius=0.022)
    spine_x = x + 0.036
    ax.plot([spine_x, spine_x], [0.310, 0.700], color=PRIMARY, lw=1.5, zorder=2)

    rows = [
        ("01", "Point-in-time design", "Expanding windows; train-only transforms"),
        ("02", "Causal VMD ablation", "Full-TA versus Full-TA + rolling VMD"),
        ("03", "Multimodal and LLM", "OOS news; Bull LLM + Bear LLM -> Leader (intrinsic)"),
        ("04", "Regime-aware XAI", "Bull/Sideway/Bear; train-only SHAP; LIME check"),
        ("05", "Forward and transfer", "Partial-2026 stress; frozen SET100"),
    ]
    centers = [0.690, 0.605, 0.520, 0.435, 0.350]
    for idx, ((number, title, detail), cy) in enumerate(zip(rows, centers)):
        ax.add_patch(Circle((spine_x, cy), 0.018, facecolor=WHITE, edgecolor=PRIMARY, linewidth=1.1, zorder=4))
        ax.text(spine_x, cy, number, ha="center", va="center", fontsize=5.7, fontweight="bold", color=PRIMARY_DARK, zorder=5)
        ax.text(x + 0.068, cy + 0.013, title, ha="left", va="center", fontsize=8.0, fontweight="bold", color=INK)
        ax.text(x + 0.068, cy - 0.016, detail, ha="left", va="center", fontsize=6.25, color=MUTED)
        if idx < len(rows) - 1:
            ax.plot([x + 0.066, x + w - 0.026], [cy - 0.043, cy - 0.043], color=LINE, lw=0.65)

    rounded(ax, x + 0.026, 0.260, w - 0.052, 0.047, WHITE, edge=LINE, lw=0.65, radius=0.010, z=2)
    ax.text(
        x + w / 2,
        0.2835,
        "Matched cohorts  |  controls  |  Holm adjustment  |  frozen transfer",
        ha="center",
        va="center",
        fontsize=6.0,
        color=MUTED,
    )


def draw_output_column(ax):
    x, y, w, h = 0.710, 0.235, 0.250, 0.535
    rounded(ax, x, y, w, h, TINT_LIGHT, radius=0.022)

    rows = [
        ("962 held-out sessions", "2022-2025  |  five fixed seeds"),
        ("VMD: -0.60 to +0.35 pp", "No conclusive directional effect"),
        ("News: no robust BAcc gain", "Leader: +5.93 to +6.00 pp (intrinsic only)"),
        ("Regime-SHAP: CNN +1.46 pp", "Mixed elsewhere  |  LIME low fidelity 71.83%"),
        ("Forward shift revealed failure modes", "2026 one-sided collapse  |  SET100 weaker 5/5"),
    ]
    centers = [0.690, 0.605, 0.520, 0.435, 0.350]
    for idx, ((title, detail), cy) in enumerate(zip(rows, centers)):
        ax.add_patch(Circle((x + 0.030, cy), 0.008, facecolor=PRIMARY, edgecolor="none", zorder=4))
        ax.text(x + 0.050, cy + 0.013, title, ha="left", va="center", fontsize=7.5, fontweight="bold", color=INK)
        ax.text(x + 0.050, cy - 0.016, detail, ha="left", va="center", fontsize=6.15, color=MUTED)
        if idx < len(rows) - 1:
            ax.plot([x + 0.028, x + w - 0.025], [cy - 0.043, cy - 0.043], color=LINE, lw=0.65)

    rounded(ax, x + 0.028, 0.260, w - 0.056, 0.047, SAGE, radius=0.010, z=2)
    ax.text(x + w / 2, 0.2835, "No primary forecasting BAcc contrast passed Holm", ha="center", va="center", fontsize=6.2, fontweight="bold", color=PRIMARY_DARK)


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    dates, closes = load_set50()

    fig = plt.figure(figsize=(12, 6), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.948,
        "Reliable Next-Day SET Index Forecasting Requires More Than a Winning Model",
        ha="center",
        va="center",
        fontsize=15.0,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.5,
        0.907,
        "A point-in-time audit of numerical denoising, predicted news, LLM inference, regime-aware XAI and transfer",
        ha="center",
        va="center",
        fontsize=8.4,
        color=MUTED,
    )

    headings = [
        (0.1575, "INPUT", "market, news and frozen models"),
        (0.4925, "METHOD", "five reliability dimensions"),
        (0.8350, "OUTPUT", "paired evidence and failure modes"),
    ]
    for center, title, subtitle in headings:
        ax.text(center, 0.835, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=PRIMARY_DARK)
        ax.text(center, 0.805, subtitle, ha="center", va="center", fontsize=6.9, color=MUTED)

    draw_input_column(fig, ax, dates, closes)
    draw_method_column(ax)
    draw_output_column(ax)

    flow_arrow(ax, (0.287, 0.505), (0.337, 0.505))
    flow_arrow(ax, (0.647, 0.505), (0.697, 0.505))

    # One quiet conclusion band unifies the flow without adding another color family.
    rounded(ax, 0.065, 0.055, 0.870, 0.112, TINT, radius=0.020, z=1)
    ax.add_patch(Rectangle((0.065, 0.055), 0.010, 0.112, facecolor=PRIMARY, edgecolor="none", zorder=3))
    ax.text(0.095, 0.135, "TAKE-HOME MESSAGE", ha="left", va="center", fontsize=6.9, fontweight="bold", color=PRIMARY_DARK)
    ax.text(
        0.500,
        0.102,
        "Effects were architecture-specific; no primary forecasting balanced-accuracy contrast",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.500,
        0.074,
        "survived the registered temporal, falsification, multiplicity and transfer checks.",
        ha="center",
        va="center",
        fontsize=9.1,
        color=INK,
    )

    fig.savefig(PNG_PATH, dpi=300, pad_inches=0)
    fig.savefig(PDF_PATH, pad_inches=0)
    plt.close(fig)

    shutil.copy2(PNG_PATH, ASSET_DIR / PNG_PATH.name)
    shutil.copy2(PDF_PATH, ASSET_DIR / PDF_PATH.name)
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
