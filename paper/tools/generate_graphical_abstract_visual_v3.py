from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge


ROOT = Path(r"D:\SET50_direction_prediction_paper")
DATA_PATH = ROOT / "data-raw" / "SET50_days.csv"
ASSET_DIR = ROOT / "paper" / "assets"
PNG_DIR = ROOT / "output" / "graphical_abstract_v3"
PDF_DIR = ROOT / "output" / "pdf"
PNG_PATH = PNG_DIR / "graphical_abstract_set_reliability_visual_v3.png"
PDF_PATH = PDF_DIR / "graphical_abstract_set_reliability_visual_v3.pdf"


BG = "#FCFCFB"
INK = "#26373F"
MUTED = "#687980"
PRIMARY = "#7896A3"
PRIMARY_DARK = "#476674"
PALE = "#EDF3F5"
PALE_2 = "#F5F8F8"
SAGE = "#E3ECE7"
SAND = "#F1EEE7"
LINE = "#D4DFE2"
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
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=PRIMARY, lw=1.2, scale=11, z=5):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=z,
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


def document_icon(ax, x, y, scale=1.0):
    w, h = 0.026 * scale, 0.038 * scale
    for dx, dy in [(0.008 * scale, 0.005 * scale), (0.004 * scale, 0.0025 * scale), (0, 0)]:
        rounded(ax, x + dx, y + dy, w, h, WHITE, edge=PRIMARY, lw=0.55, radius=0.004, z=4)
    for frac in [0.72, 0.50, 0.28]:
        ax.plot([x + 0.005 * scale, x + 0.021 * scale], [y + h * frac, y + h * frac], color=PRIMARY, lw=0.45, zorder=5)


def calendar_icon(ax, x, y, w=0.048, h=0.048, label="2026"):
    rounded(ax, x, y, w, h, WHITE, edge=PRIMARY, lw=0.8, radius=0.006, z=3)
    ax.add_patch(Rectangle((x, y + h - 0.012), w, 0.012, facecolor=PRIMARY, edgecolor="none", zorder=4))
    ax.text(x + w / 2, y + h * 0.42, label, ha="center", va="center", fontsize=5.7, fontweight="bold", color=PRIMARY_DARK, zorder=5)


def mini_neural_icon(ax, x, y, scale=1.0):
    xs = [x, x + 0.030 * scale, x + 0.060 * scale]
    layers = [[y, y + 0.024 * scale], [y - 0.008 * scale, y + 0.012 * scale, y + 0.032 * scale], [y + 0.012 * scale]]
    for left, right in zip(range(len(layers) - 1), range(1, len(layers))):
        for y1 in layers[left]:
            for y2 in layers[right]:
                ax.plot([xs[left], xs[right]], [y1, y2], color=LINE, lw=0.65, zorder=2)
    for layer_x, layer in zip(xs, layers):
        for node_y in layer:
            ax.add_patch(Circle((layer_x, node_y), 0.0045 * scale, facecolor=WHITE, edgecolor=PRIMARY, linewidth=0.7, zorder=3))


def module_card(ax, x, y, w, h, title, fill=WHITE):
    rounded(ax, x, y, w, h, fill, edge=LINE, lw=0.65, radius=0.015, z=1)
    ax.text(x + 0.016, y + h - 0.022, title, ha="left", va="center", fontsize=7.2, fontweight="bold", color=INK, zorder=4)


def draw_input_panel(fig, ax, dates, closes):
    x, y, w, h = 0.030, 0.045, 0.250, 0.910
    rounded(ax, x, y, w, h, PALE_2, radius=0.024)

    chart = fig.add_axes([0.054, 0.742, 0.202, 0.178])
    chart.set_facecolor(PALE_2)
    chart.axvspan(datetime(2022, 1, 1), datetime(2025, 12, 31), color="#DFE9ED", linewidth=0)
    chart.plot(dates, closes, color=PRIMARY_DARK, linewidth=1.25)
    chart.set_xlim(datetime(2012, 1, 1), datetime(2025, 12, 31))
    chart.set_yticks([])
    chart.set_xticks([datetime(2012, 1, 1), datetime(2022, 1, 1), datetime(2025, 1, 1)])
    chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    chart.tick_params(axis="x", labelsize=6.1, length=0, colors=MUTED, pad=2)
    for spine in chart.spines.values():
        spine.set_visible(False)
    chart.text(0.02, 0.90, "SET50 price-index data", transform=chart.transAxes, fontsize=7.0, fontweight="bold", color=INK)
    chart.text(0.72, 0.08, "2022-2025 tests", transform=chart.transAxes, fontsize=5.5, color=PRIMARY_DARK)

    module_card(ax, x + 0.020, 0.585, w - 0.040, 0.125, "DATED FINANCIAL NEWS", SAND)
    document_icon(ax, x + 0.037, 0.617, 1.25)
    ax.text(x + 0.092, 0.660, "2018-2025", ha="left", va="center", fontsize=7.2, fontweight="bold", color=INK)
    ax.text(x + 0.092, 0.633, "out-of-sample sentiment", ha="left", va="center", fontsize=6.4, color=MUTED)
    for cx, label in [(x + 0.107, "+"), (x + 0.143, "0"), (x + 0.179, "-")]:
        ax.add_patch(Circle((cx, 0.608), 0.012, facecolor=WHITE, edgecolor=PRIMARY, linewidth=0.7))
        ax.text(cx, 0.608, label, ha="center", va="center", fontsize=6.3, fontweight="bold", color=PRIMARY_DARK)

    module_card(ax, x + 0.020, 0.360, w - 0.040, 0.195, "FIVE FROZEN ARCHITECTURES", PALE)
    mini_neural_icon(ax, x + 0.055, 0.476, 1.15)
    rounded(ax, x + 0.140, 0.454, 0.075, 0.055, WHITE, edge=PRIMARY, lw=0.7, radius=0.010, z=3)
    ax.text(x + 0.1775, 0.482, "5 models", ha="center", va="center", fontsize=7.0, fontweight="bold", color=PRIMARY_DARK)
    ax.text(x + 0.045, 0.425, "LSTM  |  CNN  |  LSTM-CNN", ha="left", va="center", fontsize=6.5, color=INK)
    ax.text(x + 0.045, 0.395, "LSTM-Attention  |  LSTM-CNN-Attention", ha="left", va="center", fontsize=5.9, color=INK)

    module_card(ax, x + 0.020, 0.205, w - 0.040, 0.125, "FROZEN TEMPORAL EVALUATION", WHITE)
    block_x = x + 0.047
    for i in range(8):
        fill = PRIMARY if i >= 6 else "#DCE7EA"
        ax.add_patch(Rectangle((block_x + i * 0.019, 0.250), 0.015, 0.028, facecolor=fill, edgecolor=WHITE, linewidth=0.5))
    arrow(ax, (block_x, 0.236), (block_x + 0.150, 0.236), color=PRIMARY_DARK, lw=0.8, scale=7)
    ax.text(block_x, 0.220, "expanding train", ha="left", va="center", fontsize=5.7, color=MUTED)
    ax.text(block_x + 0.115, 0.220, "held-out", ha="left", va="center", fontsize=5.7, color=PRIMARY_DARK)

    rounded(ax, x + 0.045, 0.095, w - 0.090, 0.070, SAGE, radius=0.014, z=2)
    ax.text(x + w / 2, 0.137, "NEXT-SESSION", ha="center", va="center", fontsize=5.8, fontweight="bold", color=PRIMARY_DARK)
    ax.text(x + w / 2, 0.113, "UP  /  DOWN", ha="center", va="center", fontsize=8.4, fontweight="bold", color=INK)


def draw_method_panel(ax):
    x, y, w, h = 0.345, 0.045, 0.315, 0.910
    rounded(ax, x, y, w, h, PALE, radius=0.024)
    card_x, card_w, card_h = x + 0.020, w - 0.040, 0.145
    ys = [0.785, 0.615, 0.445, 0.275, 0.105]

    # Point-in-time module.
    module_card(ax, card_x, ys[0], card_w, card_h, "POINT-IN-TIME DESIGN")
    for i in range(8):
        fill = PRIMARY if i == 7 else "#DDE7EA"
        ax.add_patch(Rectangle((card_x + 0.025 + i * 0.023, ys[0] + 0.050), 0.018, 0.032, facecolor=fill, edgecolor=WHITE, linewidth=0.5))
    arrow(ax, (card_x + 0.030, ys[0] + 0.038), (card_x + 0.205, ys[0] + 0.038), color=PRIMARY_DARK, lw=0.8, scale=7)
    ax.text(card_x + 0.222, ys[0] + 0.066, "t+1", ha="center", va="center", fontsize=6.2, fontweight="bold", color=PRIMARY_DARK)
    ax.text(card_x + 0.025, ys[0] + 0.022, "train-only transforms and expanding folds", ha="left", va="center", fontsize=5.9, color=MUTED)

    # VMD module with component waves.
    module_card(ax, card_x, ys[1], card_w, card_h, "CAUSAL ROLLING VMD")
    t = np.linspace(0, 2 * np.pi, 120)
    base_x = card_x + 0.025
    width = 0.150
    for offset, freq, alpha in [(0.085, 1.0, 1.0), (0.055, 2.0, 0.75), (0.025, 4.0, 0.55)]:
        ax.plot(base_x + width * t / (2 * np.pi), ys[1] + offset + 0.008 * np.sin(freq * t), color=PRIMARY_DARK, lw=0.8, alpha=alpha)
    rounded(ax, card_x + 0.205, ys[1] + 0.040, 0.045, 0.055, SAGE, edge=LINE, lw=0.5, radius=0.008, z=2)
    ax.text(card_x + 0.2275, ys[1] + 0.068, "past\n60 days", ha="center", va="center", fontsize=5.6, color=INK)
    arrow(ax, (card_x + 0.180, ys[1] + 0.068), (card_x + 0.202, ys[1] + 0.068), color=PRIMARY, lw=0.7, scale=6)
    ax.text(card_x + 0.025, ys[1] + 0.013, "Full-TA  vs  Full-TA + six VMD features", ha="left", va="center", fontsize=5.9, color=MUTED)

    # Multimodal and LLM module.
    module_card(ax, card_x, ys[2], card_w, card_h, "MULTIMODAL + LLM")
    rounded(ax, card_x + 0.020, ys[2] + 0.065, 0.052, 0.038, PALE, edge=PRIMARY, lw=0.55, radius=0.007, z=3)
    ax.text(card_x + 0.046, ys[2] + 0.084, "Market", ha="center", va="center", fontsize=5.8, color=INK)
    document_icon(ax, card_x + 0.089, ys[2] + 0.066, 0.85)
    arrow(ax, (card_x + 0.074, ys[2] + 0.084), (card_x + 0.087, ys[2] + 0.084), lw=0.6, scale=5)
    rounded(ax, card_x + 0.132, ys[2] + 0.065, 0.058, 0.038, SAGE, edge=PRIMARY, lw=0.55, radius=0.007, z=3)
    ax.text(card_x + 0.161, ys[2] + 0.084, "Forecast", ha="center", va="center", fontsize=5.6, fontweight="bold", color=INK)
    ax.text(card_x + 0.105, ys[2] + 0.045, "OOS news", ha="center", va="center", fontsize=5.3, color=MUTED)
    rounded(ax, card_x + 0.202, ys[2] + 0.073, 0.035, 0.026, SAND, edge=LINE, lw=0.45, radius=0.005, z=3)
    rounded(ax, card_x + 0.239, ys[2] + 0.073, 0.035, 0.026, SAND, edge=LINE, lw=0.45, radius=0.005, z=3)
    ax.text(card_x + 0.2195, ys[2] + 0.086, "Bull", ha="center", va="center", fontsize=4.7, color=INK)
    ax.text(card_x + 0.2565, ys[2] + 0.086, "Bear", ha="center", va="center", fontsize=4.7, color=INK)
    rounded(ax, card_x + 0.220, ys[2] + 0.035, 0.040, 0.025, WHITE, edge=PRIMARY, lw=0.55, radius=0.005, z=3)
    ax.text(card_x + 0.240, ys[2] + 0.0475, "Leader", ha="center", va="center", fontsize=4.8, fontweight="bold", color=PRIMARY_DARK)
    arrow(ax, (card_x + 0.220, ys[2] + 0.073), (card_x + 0.232, ys[2] + 0.060), lw=0.5, scale=4)
    arrow(ax, (card_x + 0.256, ys[2] + 0.073), (card_x + 0.248, ys[2] + 0.060), lw=0.5, scale=4)
    ax.text(card_x + 0.240, ys[2] + 0.018, "intrinsic only", ha="center", va="center", fontsize=4.8, color=MUTED)

    # Regime and explainability module.
    module_card(ax, card_x, ys[3], card_w, card_h, "REGIME-AWARE XAI")
    labels = ["Bull", "Sideway", "Bear"]
    for i, label in enumerate(labels):
        rounded(ax, card_x + 0.020 + i * 0.055, ys[3] + 0.072, 0.050, 0.030, WHITE, edge=PRIMARY, lw=0.5, radius=0.006, z=3)
        ax.text(card_x + 0.045 + i * 0.055, ys[3] + 0.087, label, ha="center", va="center", fontsize=4.8, color=INK)
    arrow(ax, (card_x + 0.186, ys[3] + 0.087), (card_x + 0.204, ys[3] + 0.087), lw=0.6, scale=5)
    for i, height in enumerate([0.025, 0.040, 0.055, 0.032]):
        ax.add_patch(Rectangle((card_x + 0.210 + i * 0.012, ys[3] + 0.055), 0.008, height, facecolor=PRIMARY, edgecolor="none"))
    ax.text(card_x + 0.232, ys[3] + 0.041, "SHAP", ha="center", va="center", fontsize=5.2, fontweight="bold", color=PRIMARY_DARK)
    ax.add_patch(Circle((card_x + 0.064, ys[3] + 0.035), 0.018, facecolor=WHITE, edgecolor=PRIMARY, linewidth=0.7))
    ax.plot([card_x + 0.077, card_x + 0.090], [ys[3] + 0.022, ys[3] + 0.009], color=PRIMARY, lw=1.0)
    ax.text(card_x + 0.064, ys[3] + 0.035, "LIME", ha="center", va="center", fontsize=4.2, color=INK)
    ax.text(card_x + 0.108, ys[3] + 0.025, "train-only selection + fidelity check", ha="left", va="center", fontsize=5.4, color=MUTED)

    # Forward and transfer module.
    module_card(ax, card_x, ys[4], card_w, card_h, "FORWARD + TRANSFER")
    calendar_icon(ax, card_x + 0.024, ys[4] + 0.052, 0.052, 0.052, "2026")
    rounded(ax, card_x + 0.105, ys[4] + 0.061, 0.050, 0.042, PALE, edge=PRIMARY, lw=0.55, radius=0.006, z=3)
    rounded(ax, card_x + 0.206, ys[4] + 0.052, 0.058, 0.060, SAGE, edge=PRIMARY, lw=0.55, radius=0.006, z=3)
    ax.text(card_x + 0.130, ys[4] + 0.082, "SET50", ha="center", va="center", fontsize=5.5, fontweight="bold", color=INK)
    ax.text(card_x + 0.235, ys[4] + 0.082, "SET100", ha="center", va="center", fontsize=5.5, fontweight="bold", color=INK)
    arrow(ax, (card_x + 0.158, ys[4] + 0.082), (card_x + 0.202, ys[4] + 0.082), lw=0.8, scale=7)
    ax.text(card_x + 0.130, ys[4] + 0.038, "frozen", ha="center", va="center", fontsize=5.0, color=MUTED)
    ax.text(card_x + 0.235, ys[4] + 0.038, "same exchange", ha="center", va="center", fontsize=5.0, color=MUTED)


def draw_output_panel(ax):
    x, y, w, h = 0.720, 0.045, 0.250, 0.910
    rounded(ax, x, y, w, h, PALE_2, radius=0.024)
    card_x, card_w, card_h = x + 0.020, w - 0.040, 0.145
    ys = [0.785, 0.615, 0.445, 0.275, 0.105]

    # Held-out evaluation output.
    module_card(ax, card_x, ys[0], card_w, card_h, "LOCKED OUTER EVALUATION")
    calendar_icon(ax, card_x + 0.020, ys[0] + 0.050, 0.052, 0.052, "4 years")
    ax.text(card_x + 0.095, ys[0] + 0.082, "962 sessions", ha="left", va="center", fontsize=8.1, fontweight="bold", color=INK)
    ax.text(card_x + 0.095, ys[0] + 0.052, "2022-2025  |  five fixed seeds", ha="left", va="center", fontsize=6.1, color=MUTED)
    ax.text(card_x + 0.095, ys[0] + 0.026, "No primary BAcc contrast passed Holm", ha="left", va="center", fontsize=5.3, fontweight="bold", color=PRIMARY_DARK)

    # VMD interval/range output.
    module_card(ax, card_x, ys[1], card_w, card_h, "VMD EFFECT ON BALANCED ACCURACY")
    plot_x0, plot_x1, plot_y = card_x + 0.030, card_x + card_w - 0.030, ys[1] + 0.062
    ax.plot([plot_x0, plot_x1], [plot_y, plot_y], color=LINE, lw=1.2)
    zero_x = plot_x0 + (0.60 / 1.20) * (plot_x1 - plot_x0)
    ax.plot([zero_x, zero_x], [plot_y - 0.022, plot_y + 0.022], color=PRIMARY_DARK, lw=0.8)
    low_x = plot_x0
    high_x = plot_x0 + (0.95 / 1.20) * (plot_x1 - plot_x0)
    ax.plot([low_x, high_x], [plot_y, plot_y], color=PRIMARY, lw=3.0, solid_capstyle="round")
    ax.add_patch(Circle((low_x, plot_y), 0.006, facecolor=PRIMARY_DARK, edgecolor="none"))
    ax.add_patch(Circle((high_x, plot_y), 0.006, facecolor=PRIMARY_DARK, edgecolor="none"))
    ax.text(low_x, plot_y + 0.026, "-0.60", ha="center", va="center", fontsize=5.5, color=INK)
    ax.text(high_x, plot_y + 0.026, "+0.35 pp", ha="center", va="center", fontsize=5.5, color=INK)
    ax.text(zero_x, plot_y - 0.028, "0", ha="center", va="center", fontsize=5.0, color=MUTED)
    ax.text(card_x + card_w / 2, ys[1] + 0.024, "no conclusive directional effect", ha="center", va="center", fontsize=5.8, color=MUTED)

    # News and Leader output.
    module_card(ax, card_x, ys[2], card_w, card_h, "NEWS + LEADER")
    ax.text(card_x + 0.022, ys[2] + 0.088, "News", ha="left", va="center", fontsize=5.8, fontweight="bold", color=INK)
    ax.plot([card_x + 0.065, card_x + 0.125], [ys[2] + 0.088, ys[2] + 0.088], color=LINE, lw=3.0, solid_capstyle="round")
    ax.text(card_x + 0.135, ys[2] + 0.088, "no robust BAcc gain", ha="left", va="center", fontsize=5.5, color=MUTED)
    ax.text(card_x + 0.022, ys[2] + 0.050, "Leader", ha="left", va="center", fontsize=5.8, fontweight="bold", color=INK)
    ax.add_patch(Rectangle((card_x + 0.065, ys[2] + 0.038), 0.095, 0.024, facecolor=PRIMARY, edgecolor="none"))
    ax.text(card_x + 0.166, ys[2] + 0.050, "+5.93 to +6.00 pp", ha="left", va="center", fontsize=5.5, color=INK)
    ax.text(card_x + card_w - 0.020, ys[2] + 0.020, "intrinsic only", ha="right", va="center", fontsize=5.0, color=MUTED)

    # Regime-SHAP and LIME output.
    module_card(ax, card_x, ys[3], card_w, card_h, "REGIME-SHAP + LIME")
    values = [-0.10, 1.46, 0.05, -1.03, -0.74]
    base = ys[3] + 0.064
    bar_x = card_x + 0.020
    for i, value in enumerate(values):
        height = abs(value) * 0.026
        y0 = base if value >= 0 else base - height
        ax.add_patch(Rectangle((bar_x + i * 0.020, y0), 0.012, height, facecolor=PRIMARY if i == 1 else "#B9C9CF", edgecolor="none"))
    ax.plot([bar_x - 0.005, bar_x + 0.097], [base, base], color=LINE, lw=0.7)
    ax.text(bar_x + 0.030, ys[3] + 0.105, "CNN +1.46 pp", ha="center", va="center", fontsize=5.4, fontweight="bold", color=PRIMARY_DARK)
    center = (card_x + 0.165, ys[3] + 0.067)
    ax.add_patch(Wedge(center, 0.040, 90, 90 - 360 * 0.7183, facecolor=PRIMARY, edgecolor="none"))
    ax.add_patch(Wedge(center, 0.040, 90 - 360 * 0.7183, -270, facecolor="#DDE6E9", edgecolor="none"))
    ax.add_patch(Circle(center, 0.025, facecolor=WHITE, edgecolor="none"))
    ax.text(center[0], center[1] + 0.003, "71.83%", ha="center", va="center", fontsize=5.2, fontweight="bold", color=INK)
    ax.text(center[0], center[1] - 0.014, "low fidelity", ha="center", va="center", fontsize=4.4, color=MUTED)
    ax.text(card_x + card_w / 2, ys[3] + 0.019, "mixed elsewhere  |  no Holm significance", ha="center", va="center", fontsize=5.2, color=MUTED)

    # Forward and transfer output.
    module_card(ax, card_x, ys[4], card_w, card_h, "FORWARD + SET100")
    ax.text(card_x + 0.020, ys[4] + 0.092, "2026", ha="left", va="center", fontsize=5.8, fontweight="bold", color=INK)
    for i in range(7):
        fill = PRIMARY if i >= 1 else "#DDE6E9"
        ax.add_patch(Rectangle((card_x + 0.062 + i * 0.016, ys[4] + 0.080), 0.012, 0.024, facecolor=fill, edgecolor=WHITE, linewidth=0.4))
    ax.text(card_x + 0.181, ys[4] + 0.092, "one-sided collapse", ha="left", va="center", fontsize=5.3, color=MUTED)
    ax.text(card_x + 0.020, ys[4] + 0.045, "SET100", ha="left", va="center", fontsize=5.8, fontweight="bold", color=INK)
    for i in range(5):
        start_x = card_x + 0.072 + i * 0.027
        arrow(ax, (start_x, ys[4] + 0.062), (start_x, ys[4] + 0.030), color=PRIMARY, lw=0.8, scale=6)
    ax.text(card_x + 0.205, ys[4] + 0.045, "weaker 5/5", ha="left", va="center", fontsize=5.3, color=MUTED)


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

    draw_input_panel(fig, ax, dates, closes)
    draw_method_panel(ax)
    draw_output_panel(ax)

    # The arrows provide the requested input -> method -> output reading order
    # without adding large section headings.
    arrow(ax, (0.290, 0.500), (0.335, 0.500), color=PRIMARY, lw=1.8, scale=16)
    arrow(ax, (0.670, 0.500), (0.710, 0.500), color=PRIMARY, lw=1.8, scale=16)

    fig.savefig(PNG_PATH, dpi=300, pad_inches=0)
    fig.savefig(PDF_PATH, pad_inches=0)
    plt.close(fig)

    shutil.copy2(PNG_PATH, ASSET_DIR / PNG_PATH.name)
    shutil.copy2(PDF_PATH, ASSET_DIR / PDF_PATH.name)
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
