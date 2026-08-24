from __future__ import annotations

import numpy as np
import pytest

from models.track_d_economics import (
    backtest_positions,
    deflated_sharpe_ratio,
    positions_from_probabilities,
)


def test_selective_positions_are_symmetric_and_support_abstention():
    probabilities = np.array([0.80, 0.55, 0.45, 0.20])

    long_short = positions_from_probabilities(
        probabilities,
        threshold=0.60,
        strategy="long_short",
    )
    long_flat = positions_from_probabilities(
        probabilities,
        threshold=0.60,
        strategy="long_flat",
    )

    assert long_short.tolist() == [1.0, 0.0, 0.0, -1.0]
    assert long_flat.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_probability_exactly_half_uses_registered_up_tie_break():
    positions = positions_from_probabilities(
        np.array([0.5]),
        threshold=0.5,
        strategy="long_short",
    )

    assert positions.tolist() == [1.0]


def test_intraday_backtest_charges_round_trip_cost_each_active_day():
    positions = np.array([0.0, 1.0, 1.0, -1.0])
    returns = np.array([0.01, 0.01, -0.02, 0.03])

    result = backtest_positions(
        positions,
        returns,
        cost_bps=10.0,
    )

    assert result["round_trip_units"].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert result["position_change"].tolist() == [0.0, 1.0, 0.0, 2.0]
    assert result["net_return"].tolist() == pytest.approx(
        [0.0, 0.009, -0.021, -0.031]
    )


def test_deflated_sharpe_is_a_probability():
    probability = deflated_sharpe_ratio(
        observed_sharpe=1.2,
        return_count=250,
        return_skewness=-0.2,
        return_kurtosis=4.0,
        trials=40,
        sharpe_std_across_trials=0.35,
    )

    assert 0.0 <= probability <= 1.0


def test_economics_rejects_non_finite_inputs():
    with pytest.raises(ValueError, match="finite"):
        backtest_positions(
            np.array([1.0, 0.0]),
            np.array([0.01, np.nan]),
            cost_bps=10.0,
        )
