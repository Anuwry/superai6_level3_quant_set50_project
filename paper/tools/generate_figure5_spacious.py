from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(r"D:\SET50_direction_prediction_paper")
OUT = ROOT / "paper" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUT / "figure5_separated_audits_v2.png"
PDF_PATH = OUT / "figure5_separated_audits_v2.pdf"


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


OUTLINE = "#666666"
ARROW = "#555555"
CREAM = "#F5F1E7"
GREEN = "#E2ECD7"
BLUE = "#D7E5F2"
GREY = "#F1F1F1"


def add_box(ax, x, y, width, height, label, fill, fontsize=9.5):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.022",
        facecolor=fill,
        edgecolor=OUTLINE,
        linewidth=1.05,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.05,
        color="#171717",
        zorder=3,
    )
    return patch


def add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.05,
            color=ARROW,
            shrinkA=1,
            shrinkB=1,
            connectionstyle="arc3,rad=0",
            zorder=1,
        )
    )


def main():
    # Keep the same approximate aspect ratio as the embedded manuscript image,
    # while creating substantially more breathing room between stages.
    fig = plt.figure(figsize=(11.2, 5.45))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.255,
        0.955,
        "A  Downstream SET50 forecasting audit",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.755,
        0.955,
        "B  Locked intrinsic LLM sentiment audit",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.plot([0.525, 0.525], [0.065, 0.905], color="#B5B5B5", linewidth=0.9)

    # Panel A: multimodal forecasting audit.
    add_box(ax, 0.035, 0.770, 0.190, 0.100, "Point-in-time\nmarket data", CREAM)
    add_box(ax, 0.290, 0.770, 0.190, 0.100, "Dated financial\nnews", CREAM)

    add_box(ax, 0.035, 0.565, 0.190, 0.100, "Numerical\nfeature set", GREEN)
    add_box(ax, 0.290, 0.565, 0.190, 0.100, "Expanding OOS\nLocal-NLP sentiment", GREEN)

    add_box(
        ax,
        0.095,
        0.340,
        0.325,
        0.115,
        "Five frozen architectures\nMarket-Only  |  Market + Local-NLP",
        BLUE,
        fontsize=9.4,
    )

    add_box(ax, 0.030, 0.105, 0.205, 0.105, "Paired SET50\nforecast comparison", BLUE, fontsize=9.3)
    add_box(
        ax,
        0.280,
        0.105,
        0.205,
        0.105,
        "Falsification arms\nshuffled | lagged\nnews-only | random",
        GREY,
        fontsize=8.8,
    )

    add_arrow(ax, (0.130, 0.770), (0.130, 0.666))
    add_arrow(ax, (0.385, 0.770), (0.385, 0.666))
    add_arrow(ax, (0.130, 0.565), (0.205, 0.456))
    add_arrow(ax, (0.385, 0.565), (0.310, 0.456))
    add_arrow(ax, (0.205, 0.340), (0.133, 0.211))
    add_arrow(ax, (0.310, 0.340), (0.383, 0.211))

    ax.text(
        0.258,
        0.043,
        "Same dates, windows, seeds and training budget",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#444444",
    )

    # Panel B: intrinsic role-structured sentiment audit.
    add_box(ax, 0.650, 0.770, 0.240, 0.100, "Locked 2023\narticle-ticker pairs", CREAM)

    add_box(ax, 0.575, 0.565, 0.145, 0.100, "Bull worker", GREEN)
    add_box(ax, 0.820, 0.565, 0.145, 0.100, "Bear worker", GREEN)

    add_box(ax, 0.690, 0.340, 0.160, 0.115, "Leader\nsentiment", BLUE)

    add_box(
        ax,
        0.615,
        0.105,
        0.310,
        0.115,
        "Intrinsic sentiment comparison\nLeader vs SC1, SC3 and SC4\n(article-cluster uncertainty)",
        BLUE,
        fontsize=9.0,
    )

    add_arrow(ax, (0.770, 0.770), (0.648, 0.666))
    add_arrow(ax, (0.770, 0.770), (0.893, 0.666))
    add_arrow(ax, (0.648, 0.565), (0.730, 0.456))
    add_arrow(ax, (0.893, 0.565), (0.810, 0.456))
    add_arrow(ax, (0.770, 0.340), (0.770, 0.221))

    ax.text(
        0.770,
        0.043,
        "Intrinsic sentiment endpoint only",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="#444444",
    )

    fig.savefig(PNG_PATH, dpi=500, pad_inches=0)
    fig.savefig(PDF_PATH, pad_inches=0)
    plt.close(fig)

    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
