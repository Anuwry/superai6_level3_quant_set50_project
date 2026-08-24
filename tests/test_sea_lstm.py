from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from models.sea_lstm import (
    SEAConfig,
    SignedEvidenceHead,
    build_direction_model,
    compile_direction_model,
    verify_freeze_manifest,
)


def test_sea_config_rejects_invalid_dimensions_and_mode() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        SEAConfig(feature_count=0)
    with pytest.raises(ValueError, match="memory_mode"):
        SEAConfig(feature_count=3, memory_mode="unknown")


def test_signed_evidence_head_is_structurally_monotone() -> None:
    import tensorflow as tf

    head = SignedEvidenceHead(units=2)
    _ = head(tf.zeros((1, 4), dtype=tf.float32))
    head.up_raw.assign(tf.zeros_like(head.up_raw))
    head.down_raw.assign(tf.zeros_like(head.down_raw))
    head.bias.assign(tf.zeros_like(head.bias))

    baseline = float(head(tf.constant([[0.0, 0.0, 0.0, 0.0]])).numpy()[0, 0])
    more_up = float(head(tf.constant([[1.0, 0.0, 0.0, 0.0]])).numpy()[0, 0])
    more_down = float(head(tf.constant([[0.0, 0.0, 1.0, 0.0]])).numpy()[0, 0])

    assert more_up > baseline
    assert more_down < baseline


@pytest.mark.parametrize(
    ("mode", "expected_zero_slice"),
    (("positive_only", slice(3, 6)), ("negative_only", slice(0, 3))),
)
def test_single_memory_ablation_disables_registered_branch(
    mode: str,
    expected_zero_slice: slice,
) -> None:
    import tensorflow as tf

    model = build_direction_model(
        SEAConfig(window=4, feature_count=5, sea_units=3, memory_mode=mode),
        variant=mode,
        return_evidence=True,
    )
    evidence = model(tf.ones((2, 4, 5), dtype=tf.float32))["evidence"].numpy()

    np.testing.assert_allclose(evidence[:, expected_zero_slice], 0.0, atol=0.0)
    assert np.isfinite(evidence).all()


def test_sea_model_is_capacity_matched_and_produces_valid_probabilities() -> None:
    import tensorflow as tf

    standard = build_direction_model(
        SEAConfig(window=5, feature_count=130),
        variant="standard_lstm",
    )
    sea = build_direction_model(
        SEAConfig(window=5, feature_count=130),
        variant="sea_lstm",
    )
    parameter_ratio = sea.count_params() / standard.count_params()
    assert 0.95 <= parameter_ratio <= 1.05

    values = tf.zeros((3, 5, 130), dtype=tf.float32)
    probability = sea(values).numpy().reshape(-1)
    assert probability.shape == (3,)
    assert np.isfinite(probability).all()
    assert np.all((probability >= 0.0) & (probability <= 1.0))


def test_sea_model_one_batch_training_smoke() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(16, 5, 7)).astype(np.float32)
    y = (np.arange(16) % 2).astype(np.float32)
    model = build_direction_model(
        SEAConfig(window=5, feature_count=7, sea_units=3),
        variant="sea_lstm",
    )
    compile_direction_model(model, SEAConfig(window=5, feature_count=7, sea_units=3))

    history = model.fit(x, y, epochs=1, batch_size=8, shuffle=False, verbose=0)
    probability = model.predict(x[:2], verbose=0).reshape(-1)

    assert np.isfinite(history.history["loss"]).all()
    assert probability.shape == (2,)
    assert np.isfinite(probability).all()


def test_freeze_manifest_detects_input_change(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "freeze.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol_id": "sea-lstm-direct-2024-2025-v1",
                "frozen_before_outer_execution": True,
                "input_sha256": {"source.csv": digest},
            }
        ),
        encoding="utf-8",
    )

    assert verify_freeze_manifest(tmp_path, manifest)["passed"] is True
    source.write_text("value\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_freeze_manifest(tmp_path, manifest)
