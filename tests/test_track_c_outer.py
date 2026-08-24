from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import models.track_c_outer_runner as outer_runner
from models.track_c_outer import (
    OUTER_ARMS,
    capacity_matched_subseeds,
    route_regime_predictions,
    selected_feature_lookup,
)


def test_capacity_subseeds_are_stable_unique_and_shared_by_controls():
    first = capacity_matched_subseeds(42)
    second = capacity_matched_subseeds(42)

    assert first == second
    assert set(first) == {"bull", "sideway", "bear"}
    assert len(set(first.values())) == 3
    assert all(0 <= value < 2**31 for value in first.values())


def test_selected_feature_lookup_covers_shap_and_matched_spearman():
    frame = pd.DataFrame(
        [
            {
                "selector": selector,
                "regime": regime,
                "selected_top_k": 1,
                "feature": f"{selector}_{regime}",
                "selector_rank": 1,
            }
            for selector in ("shap", "spearman")
            for regime in ("global", "bull", "sideway", "bear")
        ]
    )

    result = selected_feature_lookup(frame)

    assert result[("shap", "global")] == ("shap_global",)
    assert result[("spearman", "bear")] == ("spearman_bear",)
    assert set(OUTER_ARMS) == {
        "Global-All",
        "Global3-All",
        "Global-SHAP",
        "Global-Spearman",
        "Regime-All",
        "Regime-SHAP",
        "Regime-Spearman",
    }


def test_selected_feature_lookup_fails_on_rank_gaps():
    frame = pd.DataFrame(
        {
            "selector": ["shap", "shap"],
            "regime": ["global", "global"],
            "selected_top_k": [2, 2],
            "feature": ["a", "b"],
            "selector_rank": [1, 3],
        }
    )

    with pytest.raises(ValueError, match="ranks"):
        selected_feature_lookup(frame)


def test_route_regime_predictions_uses_only_matching_expert():
    regimes = np.array(["bull", "sideway", "bear", "bull"])
    expert_predictions = {
        "bull": np.array([1.0, 10.0, 100.0, 2.0]),
        "sideway": np.array([3.0, 20.0, 300.0, 4.0]),
        "bear": np.array([5.0, 50.0, 500.0, 6.0]),
    }

    result = route_regime_predictions(regimes, expert_predictions)

    np.testing.assert_allclose(result, [1.0, 20.0, 500.0, 2.0])


def test_route_regime_predictions_rejects_missing_expert():
    with pytest.raises(ValueError, match="expert"):
        route_regime_predictions(
            np.array(["bull"]),
            {"bull": np.array([1.0])},
        )


def test_seed_averaging_fails_if_prediction_dates_differ(
    tmp_path,
    monkeypatch,
):
    for seed, dates in (
        (42, ["2022-01-03", "2022-01-04"]),
        (123, ["2022-01-03"]),
    ):
        directory = tmp_path / f"seed_{seed}"
        directory.mkdir(parents=True)
        pd.DataFrame(
            {
                "Date": dates,
                "Close_D": [100.0] * len(dates),
                "y_true": [101.0] * len(dates),
                "routing_regime": ["bull"] * len(dates),
                "y_pred": [100.5] * len(dates),
            }
        ).to_csv(directory / "predictions_Global-All.csv", index=False)

    monkeypatch.setattr(
        outer_runner,
        "_cell_dir",
        lambda _model, _fold, seed: tmp_path / f"seed_{seed}",
    )
    metrics = pd.DataFrame(
        {
            "model": ["lstm", "lstm"],
            "fold": ["fold_1", "fold_1"],
            "arm": ["Global-All", "Global-All"],
            "base_seed": [42, 123],
            "test_year": [2022, 2022],
        }
    )

    with pytest.raises(ValueError, match="dates"):
        outer_runner._seed_averaged_fold_metrics(metrics)
