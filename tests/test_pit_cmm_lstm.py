from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from models.full_non_ta_experiments import build_lstm_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_protocol_contract_precedes_results() -> None:
    freeze = json.loads(
        (PROJECT_ROOT / "test" / "pit_cmm_lstm_freeze_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert freeze["protocol_id"] == "pit-cmm-lstm-exploratory-v1"
    assert freeze["result_access_at_freeze"] is False
    assert freeze["window"] == 5
    assert freeze["seeds"] == [42, 123, 456, 789, 2025]
    assert freeze["negative_control_consistency_implemented"] is False


def test_builder_produces_finite_next_close_predictions() -> None:
    from models.pit_cmm_lstm import build_pit_cmm_lstm_model

    model = build_pit_cmm_lstm_model((5, 38))
    sample = np.linspace(0.0, 1.0, num=3 * 5 * 38, dtype=np.float32).reshape(
        3, 5, 38
    )

    prediction = model(sample, training=False).numpy()

    assert prediction.shape == (3, 1)
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize("feature_count", [38, 88, 130])
def test_parameter_budget_is_matched_to_lstm(feature_count: int) -> None:
    from models.pit_cmm_lstm import build_pit_cmm_lstm_model

    ours = build_pit_cmm_lstm_model((5, feature_count))
    baseline = build_lstm_model((5, feature_count))
    relative_delta = abs(ours.count_params() - baseline.count_params()) / float(
        baseline.count_params()
    )

    assert relative_delta <= 0.15


def test_competing_memories_are_distinct_and_finite_after_update() -> None:
    import tensorflow as tf

    from models.pit_cmm_lstm import CompetitiveMatrixMemoryLSTMCell

    tf.keras.utils.set_random_seed(42)
    cell = CompetitiveMatrixMemoryLSTMCell(hidden_units=12, memory_rank=4)
    inputs = tf.reshape(tf.linspace(0.0, 1.0, 2 * 10), (2, 10))
    states = [
        tf.zeros((2, 12)),
        tf.zeros((2, 12)),
        tf.zeros((2, 16)),
        tf.zeros((2, 16)),
    ]

    output, next_states = cell(inputs, states, training=False)
    bullish = next_states[2].numpy()
    bearish = next_states[3].numpy()

    assert output.shape == (2, 12)
    assert bullish.shape == bearish.shape == (2, 16)
    assert np.isfinite(output.numpy()).all()
    assert np.isfinite(bullish).all()
    assert np.isfinite(bearish).all()
    assert not np.allclose(bullish, bearish)


def test_model_completes_one_finite_training_step() -> None:
    from models.pit_cmm_lstm import build_pit_cmm_lstm_model

    model = build_pit_cmm_lstm_model((5, 8))
    features = np.linspace(0.0, 1.0, num=8 * 5 * 8, dtype=np.float32).reshape(
        8, 5, 8
    )
    target = np.linspace(0.2, 0.8, num=8, dtype=np.float32)

    loss = float(model.train_on_batch(features, target))

    assert np.isfinite(loss)


def test_model_fit_preserves_static_matrix_state_shape() -> None:
    from models.pit_cmm_lstm import build_pit_cmm_lstm_model

    model = build_pit_cmm_lstm_model((5, 12))
    features = np.linspace(0.0, 1.0, num=9 * 5 * 12, dtype=np.float32).reshape(
        9, 5, 12
    )
    target = np.linspace(0.2, 0.8, num=9, dtype=np.float32)

    history = model.fit(
        features,
        target,
        epochs=1,
        batch_size=4,
        shuffle=False,
        verbose=0,
    )

    assert np.isfinite(history.history["loss"]).all()


@pytest.mark.parametrize("input_shape", [(0, 10), (5, 0), (5,), (5, 10, 2)])
def test_builder_rejects_invalid_input_shapes(input_shape: tuple[int, ...]) -> None:
    from models.pit_cmm_lstm import build_pit_cmm_lstm_model

    with pytest.raises(ValueError):
        build_pit_cmm_lstm_model(input_shape)  # type: ignore[arg-type]
