from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from models.track_c_daily_regime import (
    DailyRegimeConfig,
    build_daily_regime_features,
    fit_fold_daily_regimes,
    run_daily_regime_labeling,
    score_memberships,
)


def _daily_feature_frame(
    *,
    start: str,
    signals: np.ndarray,
    initial_close: float = 100.0,
) -> pd.DataFrame:
    config = DailyRegimeConfig()
    dates = pd.bdate_range(start, periods=len(signals))
    volatility_20 = np.full(len(signals), 0.01)
    volatility_60 = np.full(len(signals), 0.012)
    columns: dict[str, object] = {
        "Date": dates,
        "Volatility_5": np.full(len(signals), 0.009),
        "Volatility_10": np.full(len(signals), 0.0095),
        "Volatility_20": volatility_20,
        "Volatility_60": volatility_60,
        "ADX_14": np.where(np.abs(signals) < 0.1, 12.0, 35.0),
    }
    for horizon, volatility_horizon in zip(
        config.horizons,
        config.volatility_horizons,
        strict=True,
    ):
        reference = volatility_20 if volatility_horizon == 20 else volatility_60
        columns[f"Return_{horizon}D"] = signals * reference * np.sqrt(horizon)
    one_day_return = np.asarray(columns["Return_1D"], dtype=float)
    previous_close = initial_close * np.cumprod(np.r_[1.0, 1.0 + one_day_return[:-1]])
    close = previous_close * (1.0 + one_day_return)
    columns.update(
        {
            "Close_D_lag1": previous_close,
            "Close_D": close,
            "Target_Next_Close": np.r_[close[1:], close[-1]],
        }
    )
    return pd.DataFrame(columns)


def _three_regime_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_signals = np.r_[
        np.full(45, 1.5),
        np.zeros(45),
        np.full(45, -1.5),
        np.zeros(20),
    ]
    train = _daily_feature_frame(
        start="2020-01-02",
        signals=train_signals,
    )
    test_signals = np.r_[
        np.zeros(15),
        np.full(15, 1.5),
        np.full(15, -1.5),
    ]
    test = _daily_feature_frame(
        start="2022-01-03",
        signals=test_signals,
        initial_close=float(train["Close_D"].iloc[-1]),
    )
    return train, test


def test_config_rejects_invalid_multi_timescale_contract():
    with pytest.raises(ValueError, match="same length"):
        DailyRegimeConfig(weights=(0.5, 0.5)).validate()
    with pytest.raises(ValueError, match="sum to one"):
        DailyRegimeConfig(weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)).validate()
    with pytest.raises(ValueError, match="sideway_quantile"):
        DailyRegimeConfig(sideway_quantile=0.05).validate()


def test_features_include_one_day_and_are_prefix_invariant():
    frame = _daily_feature_frame(
        start="2020-01-02",
        signals=np.array([1.2, 0.8, 0.0, -0.7, -1.1, 0.3]),
    )
    config = DailyRegimeConfig(smoothing_span=3)

    prefix = build_daily_regime_features(frame.iloc[:4], config=config)
    extended = build_daily_regime_features(frame, config=config).iloc[:4]

    assert "trend_z_1d" in prefix
    assert "semantic_score" in prefix
    pd.testing.assert_frame_equal(
        prefix.reset_index(drop=True),
        extended.reset_index(drop=True),
    )


def test_features_validate_required_columns_and_numeric_values():
    frame = _daily_feature_frame(
        start="2020-01-02",
        signals=np.array([1.0, 0.0, -1.0]),
    )

    with pytest.raises(ValueError, match="missing columns"):
        build_daily_regime_features(
            frame.drop(columns="ADX_14"),
            config=DailyRegimeConfig(),
        )
    invalid = frame.copy()
    invalid.loc[1, "Volatility_20"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        build_daily_regime_features(
            invalid,
            config=DailyRegimeConfig(),
        )


def test_memberships_match_semantic_threshold_and_sum_to_one():
    scores = np.array([-0.20, 0.0, 0.20])

    probabilities, labels = score_memberships(
        scores,
        threshold=0.10,
        temperature=0.35,
    )

    assert labels.tolist() == ["bear", "sideway", "bull"]
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(3))
    assert probabilities.argmax(axis=1).tolist() == [2, 1, 0]


def test_fold_fit_uses_current_daily_regime_and_training_only_threshold():
    train, test = _three_regime_frames()
    config = DailyRegimeConfig(sideway_quantile=0.35)

    original = fit_fold_daily_regimes(
        train,
        test,
        config=config,
        fold_name="fold_test",
    )
    changed_test = test.copy()
    for horizon in config.horizons:
        changed_test[f"Return_{horizon}D"] *= 100.0
    changed = fit_fold_daily_regimes(
        train,
        changed_test,
        config=config,
        fold_name="fold_test",
    )

    assert original.model_parameters["sideway_threshold"] == pytest.approx(
        changed.model_parameters["sideway_threshold"]
    )
    assert original.metadata["threshold_fit_scope"] == "fold_training_only"
    assert original.metadata["routing_target"] == (
        "current_regime_at_t_for_direction_t_plus_1"
    )
    probability_columns = [
        "prob_bull",
        "prob_sideway",
        "prob_bear",
    ]
    assert original.test_labels[probability_columns].sum(axis=1).to_numpy() == (
        pytest.approx(np.ones(len(test)))
    )
    assert set(original.test_labels["routing_regime"]).issubset(
        {"bull", "sideway", "bear"}
    )


def test_targets_do_not_affect_daily_regime_labels():
    train, test = _three_regime_frames()
    config = DailyRegimeConfig()

    original = fit_fold_daily_regimes(
        train,
        test,
        config=config,
        fold_name="fold_test",
    )
    changed_train = train.assign(Target_Next_Close=-999_999.0)
    changed_test = test.assign(Target_Next_Close=999_999.0)
    changed = fit_fold_daily_regimes(
        changed_train,
        changed_test,
        config=config,
        fold_name="fold_test",
    )

    pd.testing.assert_series_equal(
        original.train_labels["routing_regime"],
        changed.train_labels["routing_regime"],
    )
    pd.testing.assert_series_equal(
        original.test_labels["routing_regime"],
        changed.test_labels["routing_regime"],
    )
    assert original.metadata["target_columns_used"] is False


def test_fold_semantic_quality_gate_requires_ordered_returns_and_sideway():
    train, test = _three_regime_frames()

    result = fit_fold_daily_regimes(
        train,
        test,
        config=DailyRegimeConfig(),
        fold_name="fold_test",
    )

    profiles = result.semantic_profiles.set_index("regime")
    assert profiles.loc["bull", "mean_return_20d"] > 0.0
    assert profiles.loc["bear", "mean_return_20d"] < 0.0
    assert abs(profiles.loc["sideway", "mean_return_20d"]) < min(
        abs(profiles.loc["bull", "mean_return_20d"]),
        abs(profiles.loc["bear", "mean_return_20d"]),
    )
    assert result.metadata["quality_gate_passed"] is True


def test_run_writes_reproducible_daily_regime_artifacts(tmp_path):
    train, test = _three_regime_frames()
    source_dir = tmp_path / "folds"
    fold_dir = source_dir / "fold_1"
    fold_dir.mkdir(parents=True)
    train.to_csv(fold_dir / "train_2020_2021.csv", index=False)
    test.to_csv(fold_dir / "test_2022.csv", index=False)
    output_dir = tmp_path / "output"

    metadata = run_daily_regime_labeling(
        source_dir=source_dir,
        output_dir=output_dir,
        config=DailyRegimeConfig(),
        hmm_baseline_dir=None,
    )

    expected = {
        "causality_audit.csv",
        "fold_summary.csv",
        "regime_distribution.csv",
        "semantic_profiles.csv",
        "sensitivity_analysis.csv",
        "threshold_stability.csv",
        "run_metadata.json",
    }
    assert expected.issubset(path.name for path in output_dir.iterdir())
    assert (output_dir / "fold_1" / "train_regimes.csv").is_file()
    assert (output_dir / "fold_1" / "test_regimes.csv").is_file()
    assert (output_dir / "fold_1" / "model_parameters.json").is_file()
    saved = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["completed_folds"] == 1
    assert saved["protocol"]["horizons"] == [1, 3, 5, 10, 20, 60]
    assert saved["protocol"]["outer_test_used_for_thresholds"] is False
    assert saved["protocol"]["routing_target"] == (
        "current_regime_at_t_for_direction_t_plus_1"
    )
    causality = pd.read_csv(output_dir / "causality_audit.csv")
    assert causality["feature_max_abs_difference"].max() == pytest.approx(0.0)
    assert causality["membership_max_abs_difference"].max() == pytest.approx(0.0)
    assert causality["label_mismatches"].sum() == 0
    sensitivity = pd.read_csv(output_dir / "sensitivity_analysis.csv")
    assert set(sensitivity["variant"]) == {
        "sideway_quantile_0.30",
        "sideway_quantile_0.40",
        "smoothing_span_2",
        "smoothing_span_5",
    }
    assert sensitivity["agreement_with_selected"].between(0.0, 1.0).all()
    assert (
        sensitivity["adjusted_rand_with_selected"]
        .between(
            -1.0,
            1.0,
        )
        .all()
    )
