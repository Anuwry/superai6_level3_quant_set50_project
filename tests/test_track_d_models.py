from __future__ import annotations

import numpy as np
import pytest

from models.track_d_models import (
    build_track_d_model,
    make_direction_sequences,
    standardize_return_targets,
)


def test_direction_sequences_align_endpoint_labels():
    features = np.arange(30, dtype=float).reshape(10, 3)
    labels = np.arange(10, dtype=float) % 2

    x_values, y_values = make_direction_sequences(features, labels, window=4)

    assert x_values.shape == (7, 4, 3)
    assert y_values.tolist() == labels[3:].tolist()


def test_return_standardization_is_train_fitted_and_reversible():
    train = np.array([-0.02, 0.00, 0.01, 0.03])
    validation = np.array([0.02, -0.01])

    scaled_train, scaled_validation, metadata = standardize_return_targets(
        train,
        validation,
    )

    assert scaled_train.mean() == pytest.approx(0.0, abs=1e-7)
    restored = scaled_validation * metadata["std"] + metadata["mean"]
    assert restored.tolist() == pytest.approx(validation.tolist())


@pytest.mark.parametrize(
    "model_key,window",
    [
        ("lstm", 5),
        ("cnn", 20),
        ("lstm_cnn", 20),
        ("lstm_attention", 10),
        ("lstm_cnn_attention", 20),
    ],
)
@pytest.mark.parametrize("objective", ["direct", "multitask"])
def test_all_track_d_models_have_registered_output_contract(
    model_key,
    window,
    objective,
):
    model = build_track_d_model(
        model_key,
        input_shape=(window, 6),
        objective=objective,
    )
    values = model(np.zeros((2, window, 6), dtype=np.float32), training=False)

    if objective == "direct":
        assert tuple(values.shape) == (2, 1)
        assert np.all((values.numpy() >= 0.0) & (values.numpy() <= 1.0))
    else:
        assert set(values) == {"direction", "return"}
        assert tuple(values["direction"].shape) == (2, 1)
        assert tuple(values["return"].shape) == (2, 1)
