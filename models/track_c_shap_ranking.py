from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr

from models.baseline_common import FoldSpec


@dataclass(frozen=True)
class SelectionFoldTriplet:
    training_rank_spec: FoldSpec
    validation_spec: FoldSpec


def selection_fold_triplets(
    specs: Sequence[FoldSpec],
    *,
    first_outer_year: int,
) -> tuple[SelectionFoldTriplet, ...]:
    ordered = tuple(sorted(specs, key=lambda item: item.test_year))
    if len(ordered) < 2:
        raise ValueError("At least two selection folds are required")
    if any(item.test_year >= first_outer_year for item in ordered):
        raise ValueError("Selection folds must precede the first outer test")
    pairs: list[SelectionFoldTriplet] = []
    for current, following in pairwise(ordered):
        if following.test_year != current.test_year + 1:
            raise ValueError("Selection ranking and validation years must be consecutive")
        pairs.append(
            SelectionFoldTriplet(
                training_rank_spec=current,
                validation_spec=following,
            )
        )
    return tuple(pairs)


def compute_absolute_spearman(
    features: pd.DataFrame,
    target: np.ndarray,
) -> pd.Series:
    if features.empty:
        raise ValueError("features are empty")
    target_values = np.asarray(target, dtype=np.float64)
    if target_values.ndim != 1 or len(target_values) != len(features):
        raise ValueError("target must match feature rows")
    feature_values = features.to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all() or not np.isfinite(
        target_values
    ).all():
        raise ValueError("Spearman inputs must be finite")

    scores: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for column in features.columns:
            statistic = float(
                spearmanr(
                    features[column].to_numpy(dtype=float),
                    target_values,
                ).statistic
            )
            scores.append(abs(statistic) if np.isfinite(statistic) else 0.0)
    return pd.Series(scores, index=features.columns, dtype=float)


def validate_ranking_sequence_counts(
    available: int,
    *,
    required: int,
    cell: str,
) -> int:
    if isinstance(available, bool) or not isinstance(available, int):
        raise TypeError("available must be an integer")
    if isinstance(required, bool) or not isinstance(required, int) or required < 1:
        raise ValueError("required must be a positive integer")
    if available < required:
        raise ValueError(
            f"{cell}: available ranking sequences={available}; required={required}"
        )
    return available
