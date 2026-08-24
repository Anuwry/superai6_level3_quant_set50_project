from __future__ import annotations

import numpy as np
import pandas as pd

from models.track_c_shap_selection import derive_protocol_seed

OUTER_ARMS = (
    "Global-All",
    "Global3-All",
    "Global-SHAP",
    "Global-Spearman",
    "Regime-All",
    "Regime-SHAP",
    "Regime-Spearman",
)
REGIMES = ("bull", "sideway", "bear")


def capacity_matched_subseeds(base_seed: int) -> dict[str, int]:
    seeds = {
        regime: derive_protocol_seed(
            base_seed,
            "capacity_matched_expert",
            regime,
        )
        for regime in REGIMES
    }
    if len(set(seeds.values())) != len(REGIMES):
        raise RuntimeError("Capacity-matched subseeds are not unique")
    return seeds


def selected_feature_lookup(
    selected_features: pd.DataFrame,
) -> dict[tuple[str, str], tuple[str, ...]]:
    required = {
        "selector",
        "regime",
        "selected_top_k",
        "feature",
        "selector_rank",
    }
    missing = sorted(required.difference(selected_features.columns))
    if missing:
        raise ValueError(f"Selected features missing columns: {missing}")
    if selected_features.empty:
        raise ValueError("Selected features are empty")
    if selected_features.duplicated(["selector", "regime", "feature"]).any():
        raise ValueError("Selected features contain duplicates")
    result: dict[tuple[str, str], tuple[str, ...]] = {}
    for key, group in selected_features.groupby(
        ["selector", "regime"],
        sort=False,
    ):
        top_k_values = group["selected_top_k"].astype(int).unique()
        if len(top_k_values) != 1:
            raise ValueError(f"{key} has inconsistent selected_top_k")
        top_k = int(top_k_values[0])
        ordered = group.sort_values("selector_rank")
        ranks = ordered["selector_rank"].astype(int).tolist()
        if ranks != list(range(1, top_k + 1)):
            raise ValueError(f"{key} selector ranks are not consecutive")
        result[(str(key[0]), str(key[1]))] = tuple(
            ordered["feature"].astype(str)
        )
    expected = {
        (selector, regime)
        for selector in ("shap", "spearman")
        for regime in ("global", *REGIMES)
    }
    missing_keys = sorted(expected.difference(result))
    if missing_keys:
        raise ValueError(f"Selected feature groups missing: {missing_keys}")
    return result


def route_regime_predictions(
    regimes: np.ndarray,
    expert_predictions: dict[str, np.ndarray],
) -> np.ndarray:
    labels = np.asarray(regimes, dtype=object)
    if labels.ndim != 1:
        raise ValueError("regimes must be one-dimensional")
    missing_experts = sorted(set(REGIMES).difference(expert_predictions))
    if missing_experts:
        raise ValueError(f"Missing regime expert predictions: {missing_experts}")
    unknown_labels = sorted(set(labels).difference(REGIMES))
    if unknown_labels:
        raise ValueError(f"Unknown routing regimes: {unknown_labels}")
    predictions = {
        regime: np.asarray(expert_predictions[regime], dtype=np.float64)
        for regime in REGIMES
    }
    if any(values.shape != labels.shape for values in predictions.values()):
        raise ValueError("Expert prediction shapes do not match regimes")
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Expert predictions must be finite")
    routed = np.empty(len(labels), dtype=np.float64)
    for regime in REGIMES:
        mask = labels == regime
        routed[mask] = predictions[regime][mask]
    if not np.isfinite(routed).all():
        raise ValueError("Routed predictions contain non-finite values")
    return routed
