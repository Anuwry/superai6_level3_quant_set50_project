from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.pit_tldn import (
    CNN_WINDOW,
    DEBATE_FEATURES,
    LSTM_WINDOW,
    TOP_K,
    build_debate_features,
    expanding_temporal_splits,
    remove_disagreement_signal,
    top_feature_indices,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_protocol_contract_precedes_results() -> None:
    freeze = json.loads(
        (PROJECT_ROOT / "test" / "pit_tldn_freeze_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert freeze["protocol_id"] == "pit-tldn-inner-development-v1"
    assert freeze["result_access_at_freeze"] is False
    assert freeze["development_validation_years"] == [2020, 2021]
    assert freeze["outer_years_accessed_by_this_protocol"] == []
    assert freeze["windows"] == {"cnn_trend": 20, "lstm_price": 5}
    assert freeze["shap"]["top_k_per_worker"] == 30


def test_temporal_crossfit_is_expanding_and_strictly_purged() -> None:
    splits = expanding_temporal_splits(240)

    assert len(splits) == 3
    for earlier, later in zip(splits, splits[1:], strict=False):
        assert len(later.train_indices) > len(earlier.train_indices)
    for split in splits:
        assert split.train_indices[-1] + CNN_WINDOW < split.validation_indices[0]
        assert np.all(np.diff(split.train_indices) == 1)
        assert np.all(np.diff(split.validation_indices) == 1)


def test_debate_features_make_disagreement_explicit() -> None:
    cnn = np.asarray([0.9, 0.2], dtype=float)
    lstm = np.asarray([0.1, 0.4], dtype=float)
    context = np.asarray([[0.7, 0.2, 0.1, 0.3], [0.1, 0.7, 0.2, 0.5]])

    features = build_debate_features(cnn, lstm, context)
    ablated = remove_disagreement_signal(features)

    assert features.shape == (2, len(DEBATE_FEATURES))
    np.testing.assert_allclose(features[:, 2], [0.8, 0.2])
    np.testing.assert_allclose(ablated[:, 2], 0.0)
    np.testing.assert_allclose(ablated[:, [0, 1, 3, 4]], features[:, [0, 1, 3, 4]])


def test_top_feature_selection_is_deterministic_under_ties() -> None:
    importance = np.ones(40, dtype=float)

    selected = top_feature_indices(importance, top_k=TOP_K)

    np.testing.assert_array_equal(selected, np.arange(TOP_K))


def test_workers_and_leader_produce_finite_bounded_outputs() -> None:
    import tensorflow as tf

    from models.pit_tldn import (
        build_cnn_trend_worker,
        build_debate_leader,
        build_lstm_price_worker,
    )

    tf.keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)
    cnn_x = rng.normal(size=(8, CNN_WINDOW, TOP_K)).astype(np.float32)
    lstm_x = rng.normal(size=(8, LSTM_WINDOW, TOP_K)).astype(np.float32)
    labels = np.asarray([0, 1] * 4, dtype=np.float32)
    returns = rng.normal(size=8).astype(np.float32)

    cnn = build_cnn_trend_worker()
    cnn_loss = cnn.train_on_batch(cnn_x, labels, return_dict=True)
    cnn_probability = cnn(cnn_x, training=False).numpy().reshape(-1)
    assert np.isfinite(np.asarray(list(cnn_loss.values()), dtype=float)).all()

    lstm = build_lstm_price_worker()
    lstm_loss = lstm.train_on_batch(
        lstm_x,
        {"direction": labels, "next_return": returns},
        return_dict=True,
    )
    lstm_outputs = lstm(lstm_x, training=False)
    lstm_probability = lstm_outputs["direction"].numpy().reshape(-1)
    assert np.isfinite(np.asarray(list(lstm_loss.values()), dtype=float)).all()

    context = rng.uniform(size=(8, 4)).astype(np.float32)
    claims = build_debate_features(cnn_probability, lstm_probability, context)
    leader, diagnostics = build_debate_leader()
    leader_loss = leader.train_on_batch(claims, labels, return_dict=True)
    output = diagnostics(claims, training=False)

    probability = output["probability"].numpy()
    weight = output["cnn_weight"].numpy()
    correction = output["correction"].numpy()
    assert np.isfinite(np.asarray(list(leader_loss.values()), dtype=float)).all()
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    assert np.all((weight >= 0.0) & (weight <= 1.0))
    assert np.max(np.abs(correction)) <= 0.5 + 1e-7
