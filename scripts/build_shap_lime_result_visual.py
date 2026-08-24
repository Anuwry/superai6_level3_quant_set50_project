"""Build a compact publication visual for the SHAP and LIME audit results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "manuscript_tables_v1"
OUTPUT_DIR = ROOT / "paper" / "assets"

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
REGIME_ORDER = ["Bull", "Sideway", "Bear"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shap = pd.read_csv(TABLE_DIR / "table_4_regime_shap.csv")
    shap = shap[
        (shap["inference_type"] == "four_fold_exact_sign_flip")
        & (shap["contrast"] == "regime_shap_reduction")
    ].set_index("model").loc[MODEL_ORDER]

    lime = pd.read_csv(TABLE_DIR / "supplement_lime_diagnostic.csv")
    lime_matrix = (
        lime.pivot(index="model", columns="regime", values="low_fidelity_fraction")
        .loc[MODEL_ORDER, REGIME_ORDER]
        .to_numpy()
        * 100.0
    )

    fig, (ax_effect, ax_lime) = plt.subplots(
        1,
        2,
        figsize=(13.6, 5.5),
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )
    fig.subplots_adjust(left=0.075, right=0.94, top=0.90, bottom=0.22, wspace=0.30)

    x = np.arange(len(MODEL_ORDER))
    effects = shap["point_estimate"].to_numpy(dtype=float)
    lower = shap["ci95_lower"].to_numpy(dtype=float)
    upper = shap["ci95_upper"].to_numpy(dtype=float)
    errors = np.vstack([effects - lower, upper - effects])
    colors = np.where(effects >= 0, "#4C956C", "#C65D57")

    bars = ax_effect.bar(
        x,
        effects,
        width=0.68,
        color=colors,
        edgecolor="#31404E",
        linewidth=0.6,
        zorder=2,
    )
    ax_effect.errorbar(
        x,
        effects,
        yerr=errors,
        fmt="none",
        ecolor="#263238",
        elinewidth=1.1,
        capsize=3.5,
        capthick=1.1,
        zorder=3,
    )
    ax_effect.axhline(0, color="#263238", linewidth=1.0, zorder=1)
    ax_effect.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=25, ha="right")
    ax_effect.set_ylabel("Change in balanced accuracy (percentage points)")
    ax_effect.set_title("A  Regime-SHAP feature reduction", loc="left", fontweight="bold", pad=10)
    ax_effect.grid(axis="y", color="#D9E2EC", linewidth=0.7, alpha=0.8, zorder=0)
    ax_effect.spines["top"].set_visible(False)
    ax_effect.spines["right"].set_visible(False)
    ax_effect.set_ylim(-3.45, 3.45)
    for bar, effect in zip(bars, effects, strict=True):
        va = "bottom" if effect >= 0 else "top"
        offset = 0.10 if effect >= 0 else -0.10
        ax_effect.text(
            bar.get_x() + bar.get_width() / 2,
            effect + offset,
            f"{effect:+.2f}",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            color="#263238",
        )

    norm = Normalize(vmin=60, vmax=80)
    image = ax_lime.imshow(lime_matrix, cmap="YlOrRd", norm=norm, aspect="auto")
    ax_lime.set_title("B  LIME low-fidelity rate", loc="left", fontweight="bold", pad=10)
    ax_lime.set_xticks(np.arange(len(REGIME_ORDER)), REGIME_ORDER)
    ax_lime.set_yticks(np.arange(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER])
    ax_lime.set_xlabel("Market regime")
    ax_lime.set_ylabel("Architecture")
    ax_lime.set_xticks(np.arange(-0.5, len(REGIME_ORDER), 1), minor=True)
    ax_lime.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax_lime.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
    ax_lime.tick_params(which="minor", bottom=False, left=False)
    for row in range(lime_matrix.shape[0]):
        for column in range(lime_matrix.shape[1]):
            value = lime_matrix[row, column]
            ax_lime.text(
                column,
                row,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color="white" if value >= 72.0 else "#263238",
            )
    colorbar = fig.colorbar(image, ax=ax_lime, fraction=0.046, pad=0.04)
    colorbar.set_label("Low-fidelity explanations (%)")

    for axis in (ax_effect, ax_lime):
        axis.tick_params(labelsize=9, colors="#263238")

    stem = OUTPUT_DIR / "figure10_shap_lime_result_audit"
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "shap_models": len(shap),
        "lime_cells": int(lime_matrix.size),
        "lime_low_fidelity_repeats": int(lime["low_fidelity_repeats"].sum()),
        "lime_total_repeats": int(lime["audit_repeats"].sum()),
    }
    print(summary)


if __name__ == "__main__":
    main()
