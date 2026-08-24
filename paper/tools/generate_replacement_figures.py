from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm


OUT = Path(r"D:\SET50_direction_prediction_paper\paper\assets")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def box(ax, xy, width, height, text, fill, fontsize=8.5, lw=0.9):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=fill,
        edgecolor="#5f5f5f",
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, lw=0.9):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            color="#555555",
            shrinkA=2,
            shrinkB=2,
        )
    )


def generate_figure5():
    # The compact canvas keeps labels readable after the figure is scaled to journal column width.
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cream = "#f3f0e6"
    blue = "#cfdeed"
    green = "#dce9d2"
    grey = "#eeeeee"

    ax.text(0.25, 0.965, "A  Downstream SET50 forecasting audit", ha="center", va="top", fontsize=11, weight="bold")
    ax.text(0.75, 0.965, "B  Locked intrinsic LLM sentiment audit", ha="center", va="top", fontsize=11, weight="bold")
    ax.plot([0.5, 0.5], [0.07, 0.91], color="#aaaaaa", linewidth=0.8)

    # Panel A: a top-to-bottom layout with separated terminal boxes.
    box(ax, (0.035, 0.77), 0.19, 0.095, "Point-in-time\nmarket data", cream, fontsize=9)
    box(ax, (0.275, 0.77), 0.19, 0.095, "Dated financial\nnews", cream, fontsize=9)
    box(ax, (0.035, 0.59), 0.19, 0.095, "Numerical\nfeature set", green, fontsize=9)
    box(ax, (0.275, 0.59), 0.19, 0.095, "Expanding OOS\nLocal-NLP sentiment", green, fontsize=9)
    box(
        ax,
        (0.075, 0.37),
        0.35,
        0.115,
        "Five frozen architectures\nMarket-Only  |  Market + Local-NLP",
        blue,
        fontsize=9,
    )
    box(ax, (0.035, 0.145), 0.20, 0.105, "Paired SET50\nforecast comparison", blue, fontsize=8.8)
    box(
        ax,
        (0.265, 0.145),
        0.20,
        0.105,
        "Falsification arms\nshuffled | lagged\nnews-only | random",
        grey,
        fontsize=8,
    )

    arrow(ax, (0.13, 0.77), (0.13, 0.685))
    arrow(ax, (0.37, 0.77), (0.37, 0.685))
    arrow(ax, (0.13, 0.59), (0.205, 0.485))
    arrow(ax, (0.37, 0.59), (0.295, 0.485))
    arrow(ax, (0.20, 0.37), (0.135, 0.25))
    arrow(ax, (0.30, 0.37), (0.365, 0.25))
    ax.text(
        0.25,
        0.075,
        "Same dates, windows, seeds and training budget",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#444444",
    )

    # Panel B: one uninterrupted chain; controls are folded into the comparison box.
    box(ax, (0.635, 0.77), 0.23, 0.095, "Locked 2023\narticle-ticker pairs", cream, fontsize=9)
    box(ax, (0.60, 0.59), 0.13, 0.095, "Bull worker", green, fontsize=9)
    box(ax, (0.77, 0.59), 0.13, 0.095, "Bear worker", green, fontsize=9)
    box(ax, (0.67, 0.39), 0.16, 0.10, "Leader\nsentiment", blue, fontsize=9)
    box(
        ax,
        (0.615, 0.16),
        0.27,
        0.12,
        "Intrinsic sentiment comparison\nLeader vs SC1, SC3 and SC4\n(article-cluster uncertainty)",
        blue,
        fontsize=8.5,
    )

    arrow(ax, (0.75, 0.77), (0.665, 0.685))
    arrow(ax, (0.75, 0.77), (0.835, 0.685))
    arrow(ax, (0.665, 0.59), (0.72, 0.49))
    arrow(ax, (0.835, 0.59), (0.78, 0.49))
    arrow(ax, (0.75, 0.39), (0.75, 0.28))
    ax.text(
        0.75,
        0.075,
        "Intrinsic sentiment endpoint only",
        ha="center",
        va="center",
        fontsize=8.4,
        weight="bold",
        color="#444444",
    )

    fig.savefig(OUT / "figure5_separated_audits.png", dpi=320, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / "figure5_separated_audits.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def generate_figure7():
    models = ["LSTM", "CNN", "LSTM-CNN", "LSTM-Attention", "LSTM-CNN-Attention"]
    columns = ["VMD - Full-TA", "Observed news -\nMarket-Only", "Regime-SHAP -\nRegime-All", "SET100 -\nmatched SET50"]
    values = np.array(
        [
            [-0.33, -1.58, -0.10, -1.50],
            [-0.30, -0.01, +1.46, -1.19],
            [+0.35, -0.17, +0.05, -2.17],
            [-0.50, -1.89, -1.03, -0.95],
            [-0.60, -1.22, -0.74, -1.07],
        ]
    )

    fig = plt.figure(figsize=(11.8, 7.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.35], hspace=0.52)
    ax = fig.add_subplot(grid[0])
    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)
    image = ax.imshow(values, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), labels=columns)
    ax.set_yticks(np.arange(len(models)), labels=models)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=6)
    ax.set_title("A  Architecture-wise balanced-accuracy changes", loc="left", pad=34, weight="bold")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            text_color = "white" if abs(value) >= 1.35 else "black"
            ax.text(col, row, f"{value:+.2f}", ha="center", va="center", color=text_color, fontsize=10, weight="bold")

    for edge in np.arange(-0.5, values.shape[0], 1):
        ax.axhline(edge, color="white", linewidth=1.2)
    for edge in np.arange(-0.5, values.shape[1], 1):
        ax.axvline(edge, color="white", linewidth=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.025)
    colorbar.set_label("Paired BAcc change (percentage points)")
    colorbar.outline.set_linewidth(0.6)
    ax.text(
        0.5,
        -0.16,
        "Green indicates a positive point estimate; red indicates a negative point estimate.\nNo heatmap contrast survives Holm adjustment.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#444444",
    )

    ax2 = fig.add_subplot(grid[1])
    labels = ["vs SC3 (equal calls)", "vs SC4 (near cost)"]
    gains = [5.93, 6.00]
    ypos = np.arange(len(labels))
    bars = ax2.barh(ypos, gains, height=0.5, color="#78a866", edgecolor="#4f6f46", linewidth=0.8)
    ax2.set_yticks(ypos, labels=labels)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 7)
    ax2.set_xlabel("Leader intrinsic sentiment-accuracy gain (percentage points)")
    ax2.set_title("B  Leader intrinsic sentiment gain (separate endpoint)", loc="left", weight="bold")
    ax2.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax2.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax2.spines[spine].set_visible(False)
    for bar, gain in zip(bars, gains):
        ax2.text(gain + 0.10, bar.get_y() + bar.get_height() / 2, f"+{gain:.2f}  (Holm p < 0.001)", va="center", fontsize=9)

    fig.text(
        0.5,
        0.012,
        "Heatmap cells are point estimates. Corresponding 95% confidence intervals and Holm-adjusted p-values are reported in Tables 2, 3A, 4 and 5B.",
        ha="center",
        fontsize=8.2,
        color="#444444",
    )
    fig.savefig(OUT / "figure7_heatmap_summary.png", dpi=320, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(OUT / "figure7_heatmap_summary.pdf", bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


if __name__ == "__main__":
    generate_figure5()
    generate_figure7()
    print(OUT / "figure5_separated_audits.png")
    print(OUT / "figure7_heatmap_summary.png")
