from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

DEFAULT_TEMPORAL_GAP = 5
DEFAULT_NEAREST_K = 5
COVERAGE_WEIGHTS = np.asarray([1.0, 1.0, 2.0], dtype=np.float64)
ControlStrategy = Literal["matched", "random"]


@dataclass(frozen=True)
class ControlMatches:
    anchor_indices: np.ndarray
    source_indices: np.ndarray
    coverage_distance: np.ndarray
    same_year: np.ndarray


def coverage_descriptors(news_sequences: np.ndarray) -> np.ndarray:
    """Return the frozen article/ticker/availability coverage descriptor."""

    values = np.asarray(news_sequences, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 1 or values.shape[2] < 8:
        raise ValueError("news_sequences must have shape (samples, window, >=8)")
    if not np.isfinite(values).all():
        raise ValueError("news_sequences contain non-finite values")
    if np.any(values[:, :, 5:8] < 0.0):
        raise ValueError("news coverage values must be non-negative")
    return np.column_stack(
        [
            np.log1p(values[:, :, 5].sum(axis=1)),
            np.log1p(values[:, :, 6].sum(axis=1)),
            values[:, :, 7].mean(axis=1),
        ]
    )


def _validated_arrays(
    dates: pd.DatetimeIndex | np.ndarray,
    regimes: np.ndarray,
    descriptors: np.ndarray,
    *,
    name: str,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    date_values = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise"))
    regime_values = np.asarray(regimes, dtype=object).reshape(-1)
    descriptor_values = np.asarray(descriptors, dtype=np.float64)
    if len(date_values) < 1:
        raise ValueError(f"{name} must be non-empty")
    if date_values.has_duplicates or not date_values.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be unique and increasing")
    if len(regime_values) != len(date_values):
        raise ValueError(f"{name} regimes do not align with dates")
    if descriptor_values.shape != (len(date_values), 3):
        raise ValueError(f"{name} descriptors must have shape (rows, 3)")
    if not np.isfinite(descriptor_values).all():
        raise ValueError(f"{name} descriptors contain non-finite values")
    if any(not str(value) for value in regime_values):
        raise ValueError(f"{name} regimes must be non-empty")
    return date_values, regime_values, descriptor_values


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _candidate_choice(
    *,
    anchor_index: int,
    candidates: np.ndarray,
    dates: pd.DatetimeIndex,
    descriptors: np.ndarray,
    nearest_k: int,
    rng: np.random.Generator,
    matched: bool,
) -> tuple[int, float, bool]:
    if matched:
        same_year = candidates[
            dates[candidates].year == dates[anchor_index].year
        ]
        candidate_pool = same_year if len(same_year) >= nearest_k else candidates
        distance = np.sum(
            np.abs(descriptors[candidate_pool] - descriptors[anchor_index])
            * COVERAGE_WEIGHTS,
            axis=1,
        )
        order = np.argsort(distance, kind="stable")[:nearest_k]
        shortlist = candidate_pool[order]
    else:
        shortlist = candidates
    selected = int(shortlist[int(rng.integers(0, len(shortlist)))])
    selected_distance = float(
        np.sum(
            np.abs(descriptors[selected] - descriptors[anchor_index])
            * COVERAGE_WEIGHTS
        )
    )
    return (
        selected,
        selected_distance,
        bool(dates[selected].year == dates[anchor_index].year),
    )


def match_training_controls(
    dates: pd.DatetimeIndex | np.ndarray,
    regimes: np.ndarray,
    descriptors: np.ndarray,
    *,
    seed: int,
    temporal_gap: int = DEFAULT_TEMPORAL_GAP,
    nearest_k: int = DEFAULT_NEAREST_K,
    strategy: ControlStrategy = "matched",
    required_anchor_indices: np.ndarray | None = None,
) -> ControlMatches:
    """Match label-independent controls within one chronological train split."""

    if strategy not in ("matched", "random"):
        raise ValueError("strategy must be 'matched' or 'random'")
    gap = _positive_integer(temporal_gap, name="temporal_gap")
    neighbours = _positive_integer(nearest_k, name="nearest_k")
    date_values, regime_values, descriptor_values = _validated_arrays(
        dates,
        regimes,
        descriptors,
        name="training",
    )
    if required_anchor_indices is None:
        anchors = np.arange(len(date_values), dtype=int)
    else:
        anchors = np.asarray(required_anchor_indices, dtype=int).reshape(-1)
        if (
            len(anchors) < 1
            or np.any(anchors < 0)
            or np.any(anchors >= len(date_values))
            or len(np.unique(anchors)) != len(anchors)
            or np.any(np.diff(anchors) <= 0)
        ):
            raise ValueError("required_anchor_indices must be unique and increasing")

    rng = np.random.default_rng(int(seed))
    kept_anchors: list[int] = []
    sources: list[int] = []
    distances: list[float] = []
    same_year_flags: list[bool] = []
    for anchor in anchors:
        eligible = np.arange(max(0, int(anchor) - gap + 1), dtype=int)
        if strategy == "matched":
            eligible = eligible[regime_values[eligible] == regime_values[anchor]]
        if len(eligible) == 0:
            if required_anchor_indices is not None:
                raise ValueError(
                    f"Required anchor {anchor} has no eligible past control source"
                )
            continue
        source, distance, same_year = _candidate_choice(
            anchor_index=int(anchor),
            candidates=eligible,
            dates=date_values,
            descriptors=descriptor_values,
            nearest_k=neighbours,
            rng=rng,
            matched=strategy == "matched",
        )
        kept_anchors.append(int(anchor))
        sources.append(source)
        distances.append(distance)
        same_year_flags.append(same_year)

    if not kept_anchors:
        raise ValueError("No eligible past control pairs were found")
    return ControlMatches(
        anchor_indices=np.asarray(kept_anchors, dtype=int),
        source_indices=np.asarray(sources, dtype=int),
        coverage_distance=np.asarray(distances, dtype=np.float64),
        same_year=np.asarray(same_year_flags, dtype=bool),
    )


def match_external_controls(
    anchor_dates: pd.DatetimeIndex | np.ndarray,
    anchor_regimes: np.ndarray,
    anchor_descriptors: np.ndarray,
    *,
    source_dates: pd.DatetimeIndex | np.ndarray,
    source_regimes: np.ndarray,
    source_descriptors: np.ndarray,
    seed: int,
    nearest_k: int = DEFAULT_NEAREST_K,
    strategy: ControlStrategy = "matched",
) -> ControlMatches:
    """Match every external anchor to a strictly earlier same-regime source."""

    if strategy not in ("matched", "random"):
        raise ValueError("strategy must be 'matched' or 'random'")
    neighbours = _positive_integer(nearest_k, name="nearest_k")
    anchors_d, anchors_r, anchors_x = _validated_arrays(
        anchor_dates,
        anchor_regimes,
        anchor_descriptors,
        name="anchor",
    )
    sources_d, sources_r, sources_x = _validated_arrays(
        source_dates,
        source_regimes,
        source_descriptors,
        name="source",
    )
    rng = np.random.default_rng(int(seed))
    source_indices: list[int] = []
    distances: list[float] = []
    same_year_flags: list[bool] = []
    for anchor, anchor_date in enumerate(anchors_d):
        is_earlier = sources_d < anchor_date
        candidates = np.flatnonzero(
            is_earlier & (sources_r == anchors_r[anchor])
            if strategy == "matched"
            else is_earlier
        )
        if len(candidates) == 0:
            qualifier = "same-regime " if strategy == "matched" else ""
            raise ValueError(
                f"Anchor {anchor} has no strictly earlier {qualifier}source"
            )
        if strategy == "matched":
            same_year = candidates[sources_d[candidates].year == anchor_date.year]
            pool = same_year if len(same_year) >= neighbours else candidates
            distance = np.sum(
                np.abs(sources_x[pool] - anchors_x[anchor]) * COVERAGE_WEIGHTS,
                axis=1,
            )
            order = np.argsort(distance, kind="stable")[:neighbours]
            shortlist = pool[order]
        else:
            shortlist = candidates
        selected = int(shortlist[int(rng.integers(0, len(shortlist)))])
        source_indices.append(selected)
        distances.append(
            float(
                np.sum(
                    np.abs(sources_x[selected] - anchors_x[anchor])
                    * COVERAGE_WEIGHTS
                )
            )
        )
        same_year_flags.append(bool(sources_d[selected].year == anchor_date.year))

    return ControlMatches(
        anchor_indices=np.arange(len(anchors_d), dtype=int),
        source_indices=np.asarray(source_indices, dtype=int),
        coverage_distance=np.asarray(distances, dtype=np.float64),
        same_year=np.asarray(same_year_flags, dtype=bool),
    )
