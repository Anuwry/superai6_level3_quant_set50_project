from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.stats import spearmanr


def aggregate_feature_attributions(values: np.ndarray) -> np.ndarray:
    attributions = np.asarray(values, dtype=float)
    if attributions.ndim != 3 or not np.isfinite(attributions).all():
        raise ValueError("SHAP attributions must be a finite 3D array")
    return attributions.sum(axis=1)


def delete_feature_trajectories(
    instance: np.ndarray,
    reference: np.ndarray,
    *,
    feature_indices: Iterable[int],
) -> np.ndarray:
    values = np.asarray(instance, dtype=float)
    baseline = np.asarray(reference, dtype=float)
    if values.ndim != 2 or baseline.shape != values.shape:
        raise ValueError("Instance and reference must have the same 2D shape")
    if not np.isfinite(values).all() or not np.isfinite(baseline).all():
        raise ValueError("Deletion inputs must be finite")
    indexes = tuple(int(index) for index in feature_indices)
    if any(index < 0 or index >= values.shape[1] for index in indexes):
        raise ValueError("Feature deletion index is out of range")
    result = values.copy()
    if indexes:
        result[:, indexes] = baseline[:, indexes]
    return result


def randomization_rank_correlation(
    trained_attribution: np.ndarray,
    randomized_attribution: np.ndarray,
) -> float:
    trained = np.abs(np.asarray(trained_attribution, dtype=float).reshape(-1))
    randomized = np.abs(
        np.asarray(randomized_attribution, dtype=float).reshape(-1)
    )
    if trained.shape != randomized.shape or len(trained) < 2:
        raise ValueError("Attribution vectors must have equal non-trivial shape")
    if not np.isfinite(trained).all() or not np.isfinite(randomized).all():
        raise ValueError("Attribution vectors must be finite")
    correlation = float(spearmanr(trained, randomized).statistic)
    return 0.0 if not np.isfinite(correlation) else correlation


def faithfulness_percentile(
    *,
    top_feature_effect: float,
    random_feature_effects: np.ndarray,
) -> float:
    random_effects = np.asarray(random_feature_effects, dtype=float).reshape(-1)
    if len(random_effects) < 1:
        raise ValueError("At least one random deletion effect is required")
    if not np.isfinite(top_feature_effect) or not np.isfinite(random_effects).all():
        raise ValueError("Faithfulness effects must be finite")
    return float(np.mean(random_effects <= float(top_feature_effect)))
