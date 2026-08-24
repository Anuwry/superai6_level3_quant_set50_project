from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.pit_fcg_controls import (
    coverage_descriptors,
    match_external_controls,
    match_training_controls,
)


def _control_fixture(rows: int = 24) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    dates = pd.bdate_range("2019-01-02", periods=rows)
    regimes = np.asarray((["bull", "sideway", "bear"] * 8)[:rows], dtype=object)
    descriptors = np.column_stack(
        [
            np.log1p(np.arange(rows) % 7),
            np.log1p((np.arange(rows) * 2) % 11),
            (np.arange(rows) % 5) / 5.0,
        ]
    )
    return dates, regimes, descriptors


def test_coverage_descriptors_use_registered_three_component_contract() -> None:
    news = np.zeros((2, 5, 8), dtype=np.float32)
    news[0, :, 5] = [0, 1, 2, 0, 1]
    news[0, :, 6] = [0, 2, 1, 0, 3]
    news[0, :, 7] = [0, 1, 1, 0, 1]

    result = coverage_descriptors(news)

    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[0], [np.log1p(4), np.log1p(6), 0.6])
    np.testing.assert_allclose(result[1], [0.0, 0.0, 0.0])


def test_matched_training_controls_are_deterministic_past_only_and_same_regime() -> None:
    dates, regimes, descriptors = _control_fixture()

    first = match_training_controls(
        dates,
        regimes,
        descriptors,
        seed=42,
        temporal_gap=5,
        nearest_k=5,
        strategy="matched",
    )
    repeated = match_training_controls(
        dates,
        regimes,
        descriptors,
        seed=42,
        temporal_gap=5,
        nearest_k=5,
        strategy="matched",
    )

    np.testing.assert_array_equal(first.anchor_indices, repeated.anchor_indices)
    np.testing.assert_array_equal(first.source_indices, repeated.source_indices)
    assert np.all(first.source_indices <= first.anchor_indices - 5)
    assert np.all(regimes[first.source_indices] == regimes[first.anchor_indices])
    assert np.all(dates[first.source_indices] < dates[first.anchor_indices])
    assert np.isfinite(first.coverage_distance).all()


def test_random_controls_preserve_registered_common_anchor_cohort() -> None:
    dates, regimes, descriptors = _control_fixture()
    matched = match_training_controls(
        dates,
        regimes,
        descriptors,
        seed=123,
        strategy="matched",
    )

    random_control = match_training_controls(
        dates,
        regimes,
        descriptors,
        seed=123,
        strategy="random",
        required_anchor_indices=matched.anchor_indices,
    )

    np.testing.assert_array_equal(
        random_control.anchor_indices,
        matched.anchor_indices,
    )
    assert np.all(random_control.source_indices <= random_control.anchor_indices - 5)


def test_external_controls_use_training_sources_only_and_cover_every_anchor() -> None:
    source_dates, source_regimes, source_descriptors = _control_fixture(18)
    anchor_dates = pd.bdate_range("2020-01-02", periods=6)
    anchor_regimes = np.asarray(["bull", "sideway", "bear"] * 2, dtype=object)
    anchor_descriptors = source_descriptors[-6:] + 0.01

    matches = match_external_controls(
        anchor_dates,
        anchor_regimes,
        anchor_descriptors,
        source_dates=source_dates,
        source_regimes=source_regimes,
        source_descriptors=source_descriptors,
        seed=456,
    )

    np.testing.assert_array_equal(matches.anchor_indices, np.arange(6))
    assert np.all(source_dates[matches.source_indices] < anchor_dates)
    assert np.all(
        source_regimes[matches.source_indices]
        == anchor_regimes[matches.anchor_indices]
    )


def test_external_random_controls_are_past_only_without_regime_matching() -> None:
    source_dates, source_regimes, source_descriptors = _control_fixture(18)
    anchor_dates = pd.bdate_range("2020-01-02", periods=6)
    anchor_regimes = np.asarray(["unseen"] * 6, dtype=object)

    matches = match_external_controls(
        anchor_dates,
        anchor_regimes,
        source_descriptors[-6:],
        source_dates=source_dates,
        source_regimes=source_regimes,
        source_descriptors=source_descriptors,
        seed=456,
        strategy="random",
    )

    assert len(matches.source_indices) == 6
    assert np.all(source_dates[matches.source_indices] < anchor_dates)


def test_external_controls_fail_closed_when_regime_has_no_training_source() -> None:
    source_dates, source_regimes, source_descriptors = _control_fixture(12)
    source_regimes[:] = "bull"

    with pytest.raises(ValueError, match="same-regime source"):
        match_external_controls(
            pd.DatetimeIndex([pd.Timestamp("2020-01-02")]),
            np.asarray(["bear"], dtype=object),
            np.zeros((1, 3)),
            source_dates=source_dates,
            source_regimes=source_regimes,
            source_descriptors=source_descriptors,
            seed=42,
        )


@pytest.mark.parametrize("strategy", ["future", "", "MATCHED"])
def test_unknown_control_strategy_fails_closed(strategy: str) -> None:
    dates, regimes, descriptors = _control_fixture()

    with pytest.raises(ValueError, match="strategy"):
        match_training_controls(
            dates,
            regimes,
            descriptors,
            seed=42,
            strategy=strategy,
        )
