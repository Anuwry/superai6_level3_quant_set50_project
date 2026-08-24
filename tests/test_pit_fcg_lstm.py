from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _features(batch: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "numeric": rng.normal(size=(batch, 5, 122)).astype(np.float32),
        "news": rng.normal(size=(batch, 5, 8)).astype(np.float32),
        "context": rng.uniform(size=(batch, 4)).astype(np.float32),
        "placebo_news": rng.normal(size=(batch, 5, 8)).astype(np.float32),
    }


def test_frozen_protocol_contract_precedes_results() -> None:
    freeze = json.loads(
        (PROJECT_ROOT / "test" / "pit_fcg_lstm_freeze_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert freeze["protocol_id"] == "pit-fcg-lstm-inner-development-v1"
    assert freeze["result_access_at_freeze"] is False
    assert freeze["development_validation_years"] == [2020, 2021]
    assert freeze["window"] == 5
    assert freeze["seeds"] == [42, 123, 456, 789, 2025]
    assert freeze["outer_years_accessed_by_this_protocol"] == []


def test_builder_produces_bounded_residual_gate_and_finite_probability() -> None:
    from models.pit_fcg_lstm import build_pit_fcg_lstm_model

    model = build_pit_fcg_lstm_model()
    inputs = _features(3)
    outputs = model(
        {key: inputs[key] for key in ("numeric", "news", "context")},
        training=False,
    )

    assert outputs["probability"].shape == (3, 1)
    assert np.isfinite(outputs["probability"].numpy()).all()
    assert np.all((outputs["gate"].numpy() >= 0.0) & (outputs["gate"].numpy() <= 1.0))
    assert np.max(np.abs(outputs["residual"].numpy())) <= 1.0 + 1e-7


def test_inference_contract_does_not_require_or_read_placebo_news() -> None:
    from models.pit_fcg_lstm import build_pit_fcg_lstm_model

    model = build_pit_fcg_lstm_model()
    inputs = _features(4)
    inference_inputs = {
        key: inputs[key] for key in ("numeric", "news", "context")
    }

    first = model(inference_inputs, training=False)["probability"].numpy()
    inputs["placebo_news"] *= 1_000.0
    repeated = model(inference_inputs, training=False)["probability"].numpy()

    np.testing.assert_allclose(first, repeated, atol=0.0, rtol=0.0)


def test_zero_gate_recovers_anchor_logit_exactly() -> None:
    import tensorflow as tf

    from models.pit_fcg_lstm import combine_anchor_residual

    anchor = tf.constant([[0.5], [-1.2]], dtype=tf.float32)
    residual = tf.constant([[1.0], [-0.8]], dtype=tf.float32)

    combined = combine_anchor_residual(anchor, tf.zeros_like(anchor), residual)

    np.testing.assert_allclose(combined.numpy(), anchor.numpy())


def test_gate_target_requires_advantage_over_anchor_and_placebo() -> None:
    import tensorflow as tf

    from models.pit_fcg_lstm import falsification_gate_target

    labels = tf.constant([[1.0], [1.0]], dtype=tf.float32)
    anchor = tf.constant([[0.0], [2.0]], dtype=tf.float32)
    aligned = tf.constant([[2.0], [1.0]], dtype=tf.float32)
    placebo = tf.constant([[-2.0], [0.0]], dtype=tf.float32)

    target = falsification_gate_target(
        labels,
        anchor_logits=anchor,
        aligned_candidate_logits=aligned,
        placebo_candidate_logits=placebo,
        margin=0.01,
    )

    np.testing.assert_array_equal(target.numpy().reshape(-1), [1.0, 0.0])


def test_better_aligned_candidate_has_lower_falsification_rank_loss() -> None:
    import tensorflow as tf

    from models.pit_fcg_lstm import falsification_rank_loss

    labels = tf.ones((2, 1), dtype=tf.float32)
    good = falsification_rank_loss(
        labels,
        aligned_candidate_logits=tf.constant([[2.0], [1.5]]),
        placebo_candidate_logits=tf.constant([[-1.0], [-0.5]]),
        margin=0.01,
    )
    bad = falsification_rank_loss(
        labels,
        aligned_candidate_logits=tf.constant([[-1.0], [-0.5]]),
        placebo_candidate_logits=tf.constant([[2.0], [1.5]]),
        margin=0.01,
    )

    assert float(good) < float(bad)


def test_full_model_completes_one_finite_fcg_training_step() -> None:
    from models.pit_fcg_lstm import build_pit_fcg_lstm_model

    model = build_pit_fcg_lstm_model(use_fcg_loss=True)
    inputs = _features(8)
    labels = np.asarray([0, 1] * 4, dtype=np.float32)

    metrics = model.train_on_batch(inputs, labels, return_dict=True)

    assert set(metrics) >= {"loss", "final_bce", "rank_loss", "gate_loss"}
    assert np.isfinite(np.asarray(list(metrics.values()), dtype=float)).all()


def test_parameter_budget_is_within_fifteen_percent_of_direct_lstm() -> None:
    from models.pit_fcg_lstm import (
        build_direct_numeric_lstm,
        build_pit_fcg_lstm_model,
    )

    ours = build_pit_fcg_lstm_model()
    _ = ours(
        {
            key: value
            for key, value in _features(1).items()
            if key != "placebo_news"
        }
    )
    baseline = build_direct_numeric_lstm()
    relative_delta = (ours.count_params() - baseline.count_params()) / float(
        baseline.count_params()
    )

    assert relative_delta <= 0.15
    assert relative_delta > 0.0


@pytest.mark.parametrize(
    "numeric_shape,news_shape,context_features",
    [((0, 122), (5, 8), 4), ((5, 0), (5, 8), 4), ((5, 122), (0, 8), 4), ((5, 122), (5, 8), 0)],
)
def test_builder_rejects_invalid_shapes(
    numeric_shape: tuple[int, int],
    news_shape: tuple[int, int],
    context_features: int,
) -> None:
    from models.pit_fcg_lstm import build_pit_fcg_lstm_model

    with pytest.raises(ValueError):
        build_pit_fcg_lstm_model(
            numeric_shape=numeric_shape,
            news_shape=news_shape,
            context_features=context_features,
        )
