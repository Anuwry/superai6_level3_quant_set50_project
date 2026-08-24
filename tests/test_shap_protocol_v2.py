from __future__ import annotations

import numpy as np
import pytest

from models.shap_protocol_v2 import (
    MODEL_BUILDERS,
    build_original_change_model,
    evenly_spaced_indices,
    normalize_single_output_shap,
)
from models.track_a_final import TRACK_A_MODELS


def test_shap_protocol_registers_exactly_the_five_pipeline_models():
    assert tuple(MODEL_BUILDERS) == tuple(TRACK_A_MODELS)


def test_evenly_spaced_indices_are_deterministic_and_include_endpoints():
    first = evenly_spaced_indices(1000, 100)
    second = evenly_spaced_indices(1000, 100)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 100
    assert first[0] == 0
    assert first[-1] == 999


def test_original_change_wrapper_converts_both_terms_to_original_units():
    tf = pytest.importorskip("tensorflow")
    inputs = tf.keras.layers.Input(shape=(2, 2))
    scaled_level = tf.keras.layers.Lambda(
        lambda values: values[:, -1, 1:2]
    )(inputs)
    level_model = tf.keras.Model(inputs=inputs, outputs=scaled_level)
    change_model = build_original_change_model(
        level_model,
        close_feature_index=0,
        close_scale=0.5,
        close_offset=-1.0,
        target_scale=0.25,
        target_offset=-0.5,
    )
    values = np.array(
        [[[0.0, 0.0], [0.5, 0.25]]],
        dtype=np.float32,
    )

    result = change_model(values).numpy().reshape(-1)

    # predicted level=(0.25-(-0.5))/0.25=3;
    # current close=(0.5-(-1))/0.5=3.
    assert result.tolist() == pytest.approx([0.0])


def test_normalize_single_output_shap_accepts_trailing_output_axis():
    values = np.ones((2, 3, 4, 1), dtype=float)

    normalized = normalize_single_output_shap(values, (2, 3, 4))

    assert normalized.shape == (2, 3, 4)
