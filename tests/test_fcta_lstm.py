from __future__ import annotations

import numpy as np
import pytest

from models.fcta_lstm import (
    REGISTERED_VARIANTS,
    FCTAConfig,
    apply_counterfactual_deletions,
    build_fcta_model,
    counterfactual_distribution,
)


def test_fcta_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        FCTAConfig(feature_count=0)
    with pytest.raises(ValueError, match="variant"):
        FCTAConfig(feature_count=3, variant="unknown")
    with pytest.raises(ValueError, match="non-negative"):
        FCTAConfig(feature_count=3, faithfulness_weight=-0.1)


def test_counterfactual_deletions_remove_exactly_one_timestep() -> None:
    import tensorflow as tf

    values = tf.reshape(tf.range(24, dtype=tf.float32), (2, 3, 4)) + 1.0
    deleted, indicators = apply_counterfactual_deletions(values)

    assert tuple(deleted.shape) == (6, 3, 4)
    assert tuple(indicators.shape) == (6, 3, 1)
    deleted_values = deleted.numpy().reshape(2, 3, 3, 4)
    indicator_values = indicators.numpy().reshape(2, 3, 3)
    for sample in range(2):
        for removed_day in range(3):
            np.testing.assert_allclose(
                deleted_values[sample, removed_day, removed_day], 0.0
            )
            assert indicator_values[sample, removed_day, removed_day] == 1.0
            assert indicator_values[sample, removed_day].sum() == 1.0


def test_counterfactual_distribution_is_normalized_and_prefers_larger_change() -> None:
    import tensorflow as tf

    full = tf.constant([[0.5], [0.2]], dtype=tf.float32)
    deleted = tf.constant(
        [[0.5, 0.7, -0.5], [0.21, 0.19, 0.8]], dtype=tf.float32
    )
    distribution = counterfactual_distribution(full, deleted).numpy()

    np.testing.assert_allclose(distribution.sum(axis=1), 1.0, atol=1e-6)
    assert distribution[0].argmax() == 2
    assert distribution[1].argmax() == 2


@pytest.mark.parametrize("variant", REGISTERED_VARIANTS)
def test_every_variant_has_identical_inference_shape_and_parameter_count(
    variant: str,
) -> None:
    import tensorflow as tf

    config = FCTAConfig(window=5, feature_count=7, variant=variant)
    model = build_fcta_model(config)
    values = tf.zeros((3, 5, 7), dtype=tf.float32)
    outputs = model(values, training=False, return_attention=True)

    assert tuple(outputs["prediction"].shape) == (3, 1)
    assert tuple(outputs["attention"].shape) == (3, 5)
    np.testing.assert_allclose(
        outputs["attention"].numpy().sum(axis=1), 1.0, atol=1e-6
    )
    assert np.isfinite(outputs["prediction"].numpy()).all()


def test_registered_variants_are_capacity_matched() -> None:
    counts = []
    for variant in REGISTERED_VARIANTS:
        model = build_fcta_model(
            FCTAConfig(window=5, feature_count=7, variant=variant)
        )
        model(np.zeros((1, 5, 7), dtype=np.float32))
        counts.append(model.count_params())
    assert len(set(counts)) == 1


@pytest.mark.parametrize("variant", REGISTERED_VARIANTS)
def test_every_variant_completes_one_batch_training(variant: str) -> None:
    rng = np.random.default_rng(42)
    sequence = rng.normal(size=(12, 5, 7)).astype(np.float32)
    current = rng.uniform(0.3, 0.7, size=12).astype(np.float32)
    target = current + rng.normal(scale=0.02, size=12).astype(np.float32)
    packed_target = np.column_stack([target, current]).astype(np.float32)
    model = build_fcta_model(
        FCTAConfig(
            window=5,
            feature_count=7,
            variant=variant,
            direction_temperature=0.02,
        )
    )
    model.compile(optimizer="adam")

    history = model.fit(
        sequence,
        packed_target,
        batch_size=6,
        epochs=1,
        shuffle=False,
        verbose=0,
    )

    assert np.isfinite(history.history["loss"]).all()
    prediction = model.predict(sequence[:2], verbose=0).reshape(-1)
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()
