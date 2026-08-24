from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.track_c_inference import (
    CONTRASTS,
    average_seed_predictions,
    build_paired_fold_contrasts,
    circular_moving_block_indices,
    fold_level_inference,
    holm_adjust,
    moving_block_bootstrap,
    paired_daily_effects,
)


def test_registered_contrasts_include_capacity_matched_routing():
    assert CONTRASTS["regime_routing"] == ("Regime-All", "Global3-All")
    assert len(CONTRASTS) == 5


def test_holm_adjust_is_monotone_in_sorted_pvalue_order():
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03, 0.20, 0.50]))

    np.testing.assert_allclose(adjusted, [0.05, 0.12, 0.12, 0.40, 0.50])
    assert np.all((0.0 <= adjusted) & (adjusted <= 1.0))


def test_seed_average_validates_dates_targets_and_averages_predictions():
    first = pd.DataFrame(
        {
            "Date": ["2022-01-03", "2022-01-04"],
            "Close_D": [100.0, 101.0],
            "y_true": [101.0, 100.0],
            "routing_regime": ["bull", "bear"],
            "y_pred": [102.0, 99.0],
        }
    )
    second = first.copy()
    second["y_pred"] = [100.0, 101.0]

    result = average_seed_predictions({42: first, 123: second})

    np.testing.assert_allclose(result["y_pred"], [101.0, 100.0])
    assert result["seeds_averaged"].eq(2).all()


def test_seed_average_rejects_different_dates():
    first = pd.DataFrame(
        {
            "Date": ["2022-01-03"],
            "Close_D": [100.0],
            "y_true": [101.0],
            "routing_regime": ["bull"],
            "y_pred": [102.0],
        }
    )
    second = first.copy()
    second["Date"] = ["2022-01-04"]

    with pytest.raises(ValueError, match="dates"):
        average_seed_predictions({42: first, 123: second})


def test_paired_fold_contrasts_use_treatment_minus_control():
    rows = []
    for arm, ba, rmse in (
        ("Regime-All", 0.60, 10.0),
        ("Global3-All", 0.55, 12.0),
    ):
        rows.append(
            {
                "model": "lstm",
                "fold": "fold_1",
                "test_year": 2022,
                "arm": arm,
                "balanced_accuracy": ba,
                "direction_accuracy": ba,
                "mcc": ba,
                "rmse": rmse,
                "mae": rmse - 1.0,
            }
        )

    paired = build_paired_fold_contrasts(pd.DataFrame(rows))
    result = paired.loc[paired["contrast"].eq("regime_routing")].iloc[0]

    assert result["balanced_accuracy_delta_pp"] == pytest.approx(5.0)
    assert result["rmse_delta"] == pytest.approx(-2.0)


def test_fold_inference_uses_four_folds_and_exact_sign_flip():
    paired = pd.DataFrame(
        {
            "model": ["lstm"] * 4,
            "contrast": ["regime_routing"] * 4,
            "fold": [f"fold_{index}" for index in range(1, 5)],
            "balanced_accuracy_delta_pp": [1.0, 1.0, 1.0, 1.0],
            "direction_accuracy_delta_pp": [2.0, 2.0, 2.0, 2.0],
            "mcc_delta": [0.1, 0.1, 0.1, 0.1],
            "rmse_delta": [-1.0, -1.0, -1.0, -1.0],
            "mae_delta": [-0.5, -0.5, -0.5, -0.5],
        }
    )

    result = fold_level_inference(paired)
    ba = result.loc[result["metric"].eq("balanced_accuracy_delta_pp")].iloc[0]

    assert ba["outer_folds"] == 4
    assert ba["mean_delta"] == pytest.approx(1.0)
    assert ba["exact_sign_flip_pvalue"] == pytest.approx(0.125)


def test_circular_block_indices_are_deterministic_and_in_range():
    first = circular_moving_block_indices(
        7,
        block_length=3,
        rng=np.random.default_rng(42),
    )
    second = circular_moving_block_indices(
        7,
        block_length=3,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(first, second)
    assert len(first) == 7
    assert np.all((0 <= first) & (first < 7))


def test_moving_block_bootstrap_constant_effect_has_degenerate_interval():
    result = moving_block_bootstrap(
        [np.full(20, 2.0), np.full(15, 2.0)],
        block_length=5,
        replicates=100,
        seed=42,
    )

    assert result["point_estimate"] == pytest.approx(2.0)
    assert result["ci95_lower"] == pytest.approx(2.0)
    assert result["ci95_upper"] == pytest.approx(2.0)
    assert result["two_sided_pvalue"] == pytest.approx(0.0)


def test_daily_balanced_accuracy_contributions_preserve_date_order():
    treatment = pd.DataFrame(
        {
            "Date": pd.date_range("2022-01-01", periods=4),
            "Close_D": [10.0] * 4,
            "y_true": [11.0, 9.0, 11.0, 9.0],
            "y_pred": [11.0, 9.0, 9.0, 9.0],
        }
    )
    control = treatment.copy()
    control["y_pred"] = [9.0, 9.0, 11.0, 11.0]

    _, balanced_delta = paired_daily_effects(treatment, control)

    np.testing.assert_allclose(
        balanced_delta,
        [100.0, 0.0, -100.0, 100.0],
    )
    assert balanced_delta.mean() == pytest.approx(25.0)
