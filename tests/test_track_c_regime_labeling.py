from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from models.track_c_regime_labeling import (
    RegimeConfig,
    build_causal_observations,
    filter_gaussian_hmm,
    fit_fold_regimes,
    map_states_by_return,
    run_regime_labeling,
)


def _market_frame(
    *,
    start: str,
    returns: np.ndarray,
    initial_close: float = 100.0,
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(returns))
    previous = initial_close * np.exp(np.r_[0.0, np.cumsum(returns[:-1])])
    close = previous * np.exp(returns)
    return pd.DataFrame(
        {
            "Date": dates,
            "Close_D": close,
            "Close_D_lag1": previous,
            "Target_Next_Close": np.r_[close[1:], close[-1]],
        }
    )


def _three_regime_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(2026)
    train_returns = np.r_[
        rng.normal(0.006, 0.001, 70),
        rng.normal(0.0001, 0.0005, 70),
        rng.normal(-0.007, 0.002, 70),
    ]
    train = _market_frame(start="2020-01-02", returns=train_returns)
    test_returns = np.r_[
        rng.normal(0.005, 0.001, 15),
        rng.normal(-0.006, 0.002, 15),
    ]
    test = _market_frame(
        start="2022-01-03",
        returns=test_returns,
        initial_close=float(train["Close_D"].iloc[-1]),
    )
    return train, test


def test_causal_observations_do_not_change_when_future_rows_are_appended():
    returns = np.array([0.01, 0.02, -0.01, 0.005, -0.003, 0.004])
    frame = _market_frame(start="2020-01-02", returns=returns)

    prefix = build_causal_observations(frame.iloc[:4], span=3)
    extended = build_causal_observations(frame, span=3).iloc[:4]

    pd.testing.assert_frame_equal(
        prefix.reset_index(drop=True),
        extended.reset_index(drop=True),
    )


def test_causal_observations_use_lagged_close_and_validate_input():
    frame = _market_frame(
        start="2020-01-02",
        returns=np.array([0.01, -0.02, 0.03]),
    )

    observations = build_causal_observations(frame, span=3)

    assert observations["log_return"].to_numpy() == pytest.approx(
        np.log(frame["Close_D"] / frame["Close_D_lag1"])
    )
    assert np.isfinite(
        observations[["ewma_return", "ewma_volatility"]].to_numpy()
    ).all()
    with pytest.raises(ValueError, match="missing columns"):
        build_causal_observations(frame.drop(columns="Close_D_lag1"), span=3)
    with pytest.raises(ValueError, match="span"):
        build_causal_observations(frame, span=1)


def test_map_states_orders_training_return_means_without_manual_labels():
    raw_state_means = np.array(
        [
            [-0.01, 0.03],
            [0.0002, 0.01],
            [0.02, 0.02],
        ]
    )

    mapping = map_states_by_return(raw_state_means)

    assert mapping == {0: "bear", 1: "sideway", 2: "bull"}


def test_forward_filter_is_normalized_and_prefix_invariant():
    observations = np.array(
        [[-1.0], [-0.8], [0.9], [1.1], [0.0]],
        dtype=float,
    )
    start_probability = np.array([0.5, 0.5])
    transition_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    means = np.array([[-1.0], [1.0]])
    covariances = np.array([[[0.2]], [[0.2]]])

    prefix = filter_gaussian_hmm(
        observations[:3],
        start_probability=start_probability,
        transition_matrix=transition_matrix,
        means=means,
        covariances=covariances,
    )
    extended = filter_gaussian_hmm(
        observations,
        start_probability=start_probability,
        transition_matrix=transition_matrix,
        means=means,
        covariances=covariances,
    )

    assert extended.sum(axis=1) == pytest.approx(np.ones(len(observations)))
    assert extended[:3] == pytest.approx(prefix)


def test_fold_fit_uses_filtered_probabilities_and_next_day_routing():
    train, test = _three_regime_frames()
    config = RegimeConfig(
        span=5,
        restart_seeds=(7, 11),
        max_iterations=100,
    )

    result = fit_fold_regimes(train, test, config=config, fold_name="fold_test")

    assert len(result.train_labels) == len(train)
    assert len(result.test_labels) == len(test)
    probability_columns = [
        "prob_bull_filtered",
        "prob_sideway_filtered",
        "prob_bear_filtered",
    ]
    next_probability_columns = [
        "prob_bull_next",
        "prob_sideway_next",
        "prob_bear_next",
    ]
    assert result.test_labels[probability_columns].sum(axis=1).to_numpy() == (
        pytest.approx(np.ones(len(test)))
    )
    assert result.test_labels[next_probability_columns].sum(axis=1).to_numpy() == (
        pytest.approx(np.ones(len(test)))
    )
    assert set(result.test_labels["routing_regime"]).issubset(
        {"bull", "sideway", "bear"}
    )
    first_current = result.test_labels.loc[0, probability_columns].to_numpy(dtype=float)
    expected_next = first_current @ result.semantic_transition_matrix
    assert result.test_labels.loc[0, next_probability_columns].to_numpy(
        dtype=float
    ) == pytest.approx(expected_next)
    assert result.metadata["fit_scope"] == "fold_training_only"
    assert result.metadata["inference"] == "causal_forward_filter"
    assert result.metadata["quality_gate_passed"] is True
    assert {
        "delta_log_likelihood_from_best",
        "adjusted_rand_vs_selected",
    }.issubset(result.restart_diagnostics.columns)
    assert (
        result.restart_diagnostics["adjusted_rand_vs_selected"]
        .between(
            0.0,
            1.0,
        )
        .all()
    )


def test_fold_fit_does_not_use_target_values():
    train, test = _three_regime_frames()
    config = RegimeConfig(
        span=5,
        restart_seeds=(7,),
        max_iterations=100,
    )

    original = fit_fold_regimes(
        train,
        test,
        config=config,
        fold_name="fold_test",
    )
    changed_train = train.assign(Target_Next_Close=-999_999.0)
    changed_test = test.assign(Target_Next_Close=999_999.0)
    changed = fit_fold_regimes(
        changed_train,
        changed_test,
        config=config,
        fold_name="fold_test",
    )

    columns = [
        "regime_filtered",
        "routing_regime",
        "prob_bull_filtered",
        "prob_sideway_filtered",
        "prob_bear_filtered",
        "prob_bull_next",
        "prob_sideway_next",
        "prob_bear_next",
    ]
    pd.testing.assert_frame_equal(
        original.test_labels[columns],
        changed.test_labels[columns],
    )


def test_run_regime_labeling_writes_paper_audit_artifacts(tmp_path):
    train, test = _three_regime_frames()
    source_dir = tmp_path / "folds"
    fold_dir = source_dir / "fold_1"
    fold_dir.mkdir(parents=True)
    train.to_csv(fold_dir / "train_2020_2021.csv", index=False)
    test.to_csv(fold_dir / "test_2022.csv", index=False)
    output_dir = tmp_path / "outputs"
    config = RegimeConfig(
        span=5,
        restart_seeds=(7, 11),
        max_iterations=100,
    )

    metadata = run_regime_labeling(
        source_dir=source_dir,
        output_dir=output_dir,
        config=config,
    )

    expected_files = {
        "fold_summary.csv",
        "regime_distribution.csv",
        "state_profiles.csv",
        "transition_matrices.csv",
        "restart_diagnostics.csv",
        "run_metadata.json",
    }
    assert expected_files.issubset(
        {path.name for path in output_dir.iterdir() if path.is_file()}
    )
    assert (output_dir / "fold_1" / "train_regimes.csv").is_file()
    assert (output_dir / "fold_1" / "test_regimes.csv").is_file()
    saved = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert saved["protocol"]["state_count"] == 3
    assert saved["protocol"]["routing_target"] == "next_trading_day_regime"
    assert saved["protocol"]["minimum_training_regime_share"] == pytest.approx(0.05)
    assert saved["protocol"]["minimum_self_transition_probability"] == (
        pytest.approx(0.80)
    )
    assert saved["completed_folds"] == 1
    assert metadata["completed_folds"] == 1
