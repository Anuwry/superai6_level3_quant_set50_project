from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(r"D:\SET50_direction_prediction_paper")
DATA_PATH = ROOT / "data-raw" / "SET50_days.csv"
ASSET_DIR = ROOT / "paper" / "assets"
PNG_DIR = ROOT / "output" / "graphical_abstract_v1"
PDF_DIR = ROOT / "output" / "pdf"
PNG_PATH = PNG_DIR / "graphical_abstract_set_reliability_v1.png"
PDF_PATH = PDF_DIR / "graphical_abstract_set_reliability_v1.pdf"


NAVY = "#18324A"
TEXT = "#1B1F23"
MUTED = "#5D6770"
LINE = "#6B737A"
PANEL = "#FBFBFA"
CREAM = "#F5F1E7"
BLUE = "#DCE9F5"
BLUE_DARK = "#4D789D"
GREEN = "#E2EEDB"
GREEN_DARK = "#5F8A55"
AMBER = "#F7E9C8"
AMBER_DARK = "#A87824"
ROSE = "#F4DEDA"
ROSE_DARK = "#A9544C"
GREY = "#ECEEEF"


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 9,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def rounded(ax, x, y, w, h, facecolor, edgecolor="#70777D", lw=0.9, radius=0.018, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=LINE, lw=1.2, scale=11, z=2):
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
    dates = np.array(dates, dtype=object)[order]
    closes = np.asarray(closes, dtype=float)[order]
    keep = np.array([date.year >= 2012 for date in dates])
    return dates[keep], closes[keep]


def draw_document_icon(ax, x, y, w=0.035, h=0.050):
    for dx, dy in [(0.012, 0.008), (0.006, 0.004), (0, 0)]:
        rect = FancyBboxPatch(
            (x + dx, y + dy),
            w,
            h,
            boxstyle="round,pad=0.002,rounding_size=0.004",
            facecolor="white",
            edgecolor=LINE,
            linewidth=0.7,
            zorder=3,
        )
        ax.add_patch(rect)
    for offset in [0.035, 0.025, 0.015]:
        ax.plot([x + 0.006, x + w - 0.005], [y + offset, y + offset], color="#9AA1A6", lw=0.55, zorder=4)


def draw_model_pills(ax, x, y, w):
    labels = [
        ("LSTM", x, y + 0.076, w * 0.31),
        ("CNN", x + w * 0.35, y + 0.076, w * 0.25),
        ("LSTM-CNN", x + w * 0.64, y + 0.076, w * 0.36),
        ("LSTM-Attention", x, y + 0.025, w * 0.47),
        ("LSTM-CNN-Attention", x + w * 0.51, y + 0.025, w * 0.49),
    ]
    for label, px, py, pw in labels:
        rounded(ax, px, py, pw, 0.035, GREEN, edgecolor="#7C8C75", lw=0.7, radius=0.008, z=3)
        ax.text(px + pw / 2, py + 0.0175, label, ha="center", va="center", fontsize=6.7, color=TEXT, zorder=4)


def audit_card(ax, number, y, title, subtitle, fill, accent):
    x, w, h = 0.305, 0.355, 0.090
    rounded(ax, x, y, w, h, fill, edgecolor="#8A9095", lw=0.8, radius=0.013, z=2)
    ax.add_patch(Rectangle((x, y), 0.008, h, facecolor=accent, edgecolor="none", zorder=3))
    ax.add_patch(Circle((x + 0.030, y + h / 2), 0.017, facecolor="white", edgecolor=accent, linewidth=1.0, zorder=4))
    ax.text(x + 0.030, y + h / 2, str(number), ha="center", va="center", fontsize=8.0, fontweight="bold", color=accent, zorder=5)
    ax.text(x + 0.057, y + 0.059, title, ha="left", va="center", fontsize=8.8, fontweight="bold", color=TEXT, zorder=4)
    ax.text(x + 0.057, y + 0.030, subtitle, ha="left", va="center", fontsize=7.2, color=MUTED, zorder=4)


def evidence_card(ax, y, title, detail, fill, accent, detail2=None):
    x, w, h = 0.720, 0.250, 0.090
    rounded(ax, x, y, w, h, fill, edgecolor="#8A9095", lw=0.8, radius=0.013, z=2)
    ax.add_patch(Rectangle((x, y), 0.007, h, facecolor=accent, edgecolor="none", zorder=3))
    ax.text(x + 0.020, y + 0.061, title, ha="left", va="center", fontsize=8.5, fontweight="bold", color=TEXT, zorder=4)
    if detail2 is None:
        ax.text(x + 0.020, y + 0.029, detail, ha="left", va="center", fontsize=7.4, color=MUTED, zorder=4)
    else:
        ax.text(x + 0.020, y + 0.037, detail, ha="left", va="center", fontsize=7.1, color=MUTED, zorder=4)
        ax.text(x + 0.020, y + 0.017, detail2, ha="left", va="center", fontsize=7.1, color=MUTED, zorder=4)


def draw_llm_inset(ax, y):
    """Show the LLM roles separately from the market-regime labels."""
    bull_x, bear_x, worker_y = 0.535, 0.593, y + 0.056
    worker_w, worker_h = 0.050, 0.025
    leader_x, leader_y, leader_w, leader_h = 0.558, y + 0.012, 0.064, 0.026

    for x, label in [(bull_x, "Bull LLM"), (bear_x, "Bear LLM")]:
        rounded(ax, x, worker_y, worker_w, worker_h, CREAM, edgecolor="#8E8A7D", lw=0.55, radius=0.005, z=5)
        ax.text(x + worker_w / 2, worker_y + worker_h / 2, label, ha="center", va="center", fontsize=4.8, color=TEXT, zorder=6)

    rounded(ax, leader_x, leader_y, leader_w, leader_h, "white", edgecolor=GREEN_DARK, lw=0.65, radius=0.005, z=5)
    ax.text(leader_x + leader_w / 2, leader_y + leader_h / 2, "Leader", ha="center", va="center", fontsize=5.1, fontweight="bold", color=GREEN_DARK, zorder=6)
    arrow(ax, (bull_x + worker_w / 2, worker_y), (leader_x + 0.020, leader_y + leader_h), color=GREEN_DARK, lw=0.85, scale=7, z=4)
    arrow(ax, (bear_x + worker_w / 2, worker_y), (leader_x + leader_w - 0.020, leader_y + leader_h), color=GREEN_DARK, lw=0.85, scale=7, z=4)


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    dates, closes = load_set50()

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.957,
        "Point-in-Time Reliability Audit for Next-Day SET Index Direction Forecasting",
        ha="center",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        0.5,
        0.917,
        "SET50 2012-2025  •  five frozen neural architectures  •  expanding out-of-sample evaluation",
        ha="center",
        va="center",
        fontsize=8.8,
        color=MUTED,
    )

    # Three-column structure.
    rounded(ax, 0.025, 0.205, 0.245, 0.665, PANEL, edgecolor="#A4AAAE", lw=0.9, radius=0.012)
    rounded(ax, 0.290, 0.205, 0.390, 0.665, PANEL, edgecolor="#A4AAAE", lw=0.9, radius=0.012)
    rounded(ax, 0.700, 0.205, 0.275, 0.665, PANEL, edgecolor="#A4AAAE", lw=0.9, radius=0.012)

    ax.text(0.1475, 0.838, "1  POINT-IN-TIME INPUTS", ha="center", va="center", fontsize=10.5, fontweight="bold", color=NAVY)
    ax.text(0.485, 0.838, "2  FIVE AUDIT DIMENSIONS", ha="center", va="center", fontsize=10.5, fontweight="bold", color=NAVY)
    ax.text(0.8375, 0.838, "3  AUDIT EVIDENCE", ha="center", va="center", fontsize=10.5, fontweight="bold", color=NAVY)

    # Actual SET50 mini-series.
    chart = fig.add_axes([0.048, 0.645, 0.198, 0.145])
    chart.plot(dates, closes, color=BLUE_DARK, linewidth=1.2)
    chart.axvspan(datetime(2022, 1, 1), datetime(2025, 12, 31), color="#D7E6F2", alpha=0.8, linewidth=0)
    chart.plot(dates, closes, color=BLUE_DARK, linewidth=1.2)
    chart.set_xlim(datetime(2012, 1, 1), datetime(2025, 12, 31))
    chart.set_yticks([])
    chart.set_xticks([datetime(2012, 1, 1), datetime(2022, 1, 1), datetime(2025, 1, 1)])
    chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    chart.tick_params(axis="x", labelsize=6.3, length=2, colors=MUTED, pad=2)
    chart.grid(axis="y", color="#E5E7E9", linewidth=0.5)
    for spine in ["top", "right", "left"]:
        chart.spines[spine].set_visible(False)
    chart.spines["bottom"].set_color("#A4AAAE")
    chart.text(0.02, 0.88, "SET50 close", transform=chart.transAxes, fontsize=7.2, fontweight="bold", color=NAVY)
    chart.text(0.73, 0.08, "outer tests", transform=chart.transAxes, fontsize=6.0, color=BLUE_DARK)

    # News input.
    rounded(ax, 0.045, 0.525, 0.205, 0.085, CREAM, edgecolor="#9C9789", lw=0.8, radius=0.012, z=2)
    draw_document_icon(ax, 0.061, 0.540)
    ax.text(0.118, 0.578, "Dated financial news", ha="left", va="center", fontsize=8.5, fontweight="bold", color=TEXT)
    ax.text(0.118, 0.550, "2018-2025 • OOS sentiment", ha="left", va="center", fontsize=7.2, color=MUTED)

    # Frozen architecture input.
    rounded(ax, 0.045, 0.320, 0.205, 0.165, BLUE, edgecolor="#8699AA", lw=0.8, radius=0.012, z=2)
    ax.text(0.1475, 0.463, "Five frozen architectures", ha="center", va="center", fontsize=8.6, fontweight="bold", color=TEXT)
    ax.text(0.1475, 0.441, "windows • seeds • training budget", ha="center", va="center", fontsize=6.9, color=MUTED)
    draw_model_pills(ax, 0.061, 0.307, 0.172)

    rounded(ax, 0.075, 0.220, 0.145, 0.052, GREEN, edgecolor="#7C8C75", lw=0.7, radius=0.010, z=2)
    ax.text(0.1475, 0.246, "Target: next-session Up / Down", ha="center", va="center", fontsize=7.2, fontweight="bold", color=TEXT)

    # Audit cards and matched evidence rows.
    ys = [0.715, 0.605, 0.495, 0.385, 0.275]
    audit_card(ax, 1, ys[0], "Temporal contract", "Expanding windows • train-only transforms", BLUE, BLUE_DARK)
    audit_card(ax, 2, ys[1], "Numerical denoising", "Full-TA versus causal rolling VMD", AMBER, AMBER_DARK)
    audit_card(ax, 3, ys[2], "Multimodal and LLM", "Predicted news vs Market-Only", GREEN, GREEN_DARK)
    draw_llm_inset(ax, ys[2])
    audit_card(ax, 4, ys[3], "Regime and explainability", "Bull/Sideway/Bear • train-only SHAP • LIME", BLUE, BLUE_DARK)
    audit_card(ax, 5, ys[4], "Forward and transfer", "Partial-2026 stress • frozen SET100", ROSE, ROSE_DARK)

    evidence_card(ax, ys[0], "962 held-out sessions", "2022-2025 • five fixed seeds", BLUE, BLUE_DARK)
    evidence_card(ax, ys[1], "VMD: -0.60 to +0.35 pp", "No conclusive directional effect", AMBER, AMBER_DARK)
    evidence_card(
        ax,
        ys[2],
        "News: no robust BAcc gain",
        "Leader: +5.93 to +6.00 pp",
        GREEN,
        GREEN_DARK,
        detail2="Intrinsic sentiment endpoint only",
    )
    evidence_card(
        ax,
        ys[3],
        "Regime-SHAP: CNN +1.46 pp",
        "Mixed elsewhere • no Holm significance",
        BLUE,
        BLUE_DARK,
        detail2="LIME low fidelity: 71.83%",
    )
    evidence_card(
        ax,
        ys[4],
        "Forward failure modes detected",
        "2026: one-sided collapse",
        ROSE,
        ROSE_DARK,
        detail2="SET100: weaker for 5/5 models",
    )

    # Direction arrows between the three conceptual columns.
    arrow(ax, (0.271, 0.535), (0.289, 0.535), color=NAVY, lw=1.4, scale=12)
    arrow(ax, (0.681, 0.535), (0.699, 0.535), color=NAVY, lw=1.4, scale=12)

    # Small safeguards band beneath the five audit rows.
    rounded(ax, 0.315, 0.222, 0.330, 0.038, GREY, edgecolor="#A4AAAE", lw=0.7, radius=0.008, z=2)
    ax.text(
        0.480,
        0.241,
        "Matched cohorts  •  falsification controls  •  Holm adjustment  •  frozen transfer",
        ha="center",
        va="center",
        fontsize=6.8,
        color=MUTED,
    )

    # Main conclusion.
    rounded(ax, 0.025, 0.035, 0.950, 0.125, BLUE, edgecolor=BLUE_DARK, lw=1.1, radius=0.016, z=2)
    ax.text(0.055, 0.127, "MAIN CONCLUSION", ha="left", va="center", fontsize=8.0, fontweight="bold", color=BLUE_DARK)
    ax.text(
        0.500,
        0.094,
        "Apparent gains were architecture-dependent; no primary forecasting balanced-accuracy contrast",
        ha="center",
        va="center",
        fontsize=10.3,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        0.500,
        0.061,
        "survived the registered temporal, falsification, multiplicity and transfer checks.",
        ha="center",
        va="center",
        fontsize=10.3,
        fontweight="bold",
        color=NAVY,
    )

    fig.savefig(PNG_PATH, dpi=300, pad_inches=0)
    fig.savefig(PDF_PATH, pad_inches=0)
    fig.savefig(ASSET_DIR / PNG_PATH.name, dpi=300, pad_inches=0)
    fig.savefig(ASSET_DIR / PDF_PATH.name, pad_inches=0)
    plt.close(fig)

    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
