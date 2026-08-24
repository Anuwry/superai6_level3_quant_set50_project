from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from models.pit_cdr_lstm import (
    CDRConfig,
    apply_endpoint_regime_masks,
    build_pit_cdr_models,
    build_relation_pairs,
    direction_metrics,
    regime_feature_masks,
    verify_freeze_manifest,
)


def test_regime_feature_masks_keep_frozen_news_and_selected_numeric() -> None:
    pool = ("a", "b", "c", "news_1", "news_2")
    selected = {
        "bull": ("a",),
        "sideway": ("a", "b", "c"),
        "bear": ("b", "c"),
    }

    masks = regime_feature_masks(
        pool,
        selected,
        news_features=("news_1", "news_2"),
    )

    np.testing.assert_array_equal(masks["bull"], [1, 0, 0, 1, 1])
    np.testing.assert_array_equal(masks["sideway"], [1, 1, 1, 1, 1])
    np.testing.assert_array_equal(masks["bear"], [0, 1, 1, 1, 1])

    sequences = np.ones((3, 2, len(pool)), dtype=np.float32)
    masked = apply_endpoint_regime_masks(
        sequences,
        np.array(["bull", "sideway", "bear"], dtype=object),
        masks,
    )
    np.testing.assert_array_equal(masked[0, 0], masks["bull"])
    np.testing.assert_array_equal(masked[1, 1], masks["sideway"])
    np.testing.assert_array_equal(masked[2, 0], masks["bear"])


def test_matched_relation_pairs_obey_label_regime_and_separation_contract() -> None:
    count = 90
    labels = (np.arange(count) % 2).astype(np.int8)
    regimes = np.asarray(
        (["bull"] * 30) + (["sideway"] * 30) + (["bear"] * 30),
        dtype=object,
    )
    state = np.column_stack(
        [
            np.linspace(-1.0, 1.0, count),
            np.sin(np.arange(count) / 7.0),
            np.cos(np.arange(count) / 11.0),
        ]
    )
    positions = np.arange(count)

    pairs = build_relation_pairs(
        labels,
        regimes,
        state,
        endpoint_positions=positions,
        minimum_separation=5,
        seed=42,
        strategy="matched",
    )

    hard = pairs.relation == "counter"
    transport = pairs.relation == "transport"
    assert hard.sum() == count
    assert transport.sum() == count
    assert np.all(labels[pairs.left[hard]] == 1)
    assert np.all(labels[pairs.right[hard]] == 0)
    assert np.all(regimes[pairs.left[hard]] == regimes[pairs.right[hard]])
    assert np.all(labels[pairs.left[transport]] == labels[pairs.right[transport]])
    assert np.all(regimes[pairs.left[transport]] != regimes[pairs.right[transport]])
    assert np.all(
        np.abs(positions[pairs.left] - positions[pairs.right]) >= 5
    )


def test_random_relation_pairs_remain_label_compatible() -> None:
    count = 90
    labels = (np.arange(count) % 2).astype(np.int8)
    regimes = np.asarray(
        (["bull"] * 30) + (["sideway"] * 30) + (["bear"] * 30),
        dtype=object,
    )
    state = np.column_stack([np.arange(count), np.arange(count) ** 2])
    positions = np.arange(count)

    first = build_relation_pairs(
        labels,
        regimes,
        state,
        endpoint_positions=positions,
        minimum_separation=5,
        seed=123,
        strategy="random",
    )
    repeated = build_relation_pairs(
        labels,
        regimes,
        state,
        endpoint_positions=positions,
        minimum_separation=5,
        seed=123,
        strategy="random",
    )

    np.testing.assert_array_equal(first.left, repeated.left)
    np.testing.assert_array_equal(first.right, repeated.right)
    hard = first.relation == "counter"
    transport = first.relation == "transport"
    assert np.all(labels[first.left[hard]] == 1)
    assert np.all(labels[first.right[hard]] == 0)
    assert np.all(labels[first.left[transport]] == labels[first.right[transport]])


def test_direction_metrics_excludes_zero_return_and_uses_half_threshold() -> None:
    result = direction_metrics(
        current_close=np.array([100.0, 100.0, 100.0, 100.0, 100.0]),
        next_close=np.array([101.0, 99.0, 100.0, 102.0, 98.0]),
        probability=np.array([0.8, 0.7, 0.9, 0.4, 0.2]),
    )

    assert result["observations"] == 5
    assert result["direction_evaluable"] == 4
    assert result["direction_accuracy"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["tn"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tp"] == 1


def test_shared_twin_training_does_not_increase_inference_parameters() -> None:
    config = CDRConfig(window=5, feature_count=7)
    training_model, inference_model = build_pit_cdr_models(config)

    assert training_model.get_layer("shared_encoder") is inference_model.get_layer(
        "shared_encoder"
    )
    assert training_model.get_layer("direction_logit") is inference_model.get_layer(
        "direction_logit"
    )
    assert inference_model.count_params() > 0
    probability = inference_model(np.zeros((2, 5, 7), dtype=np.float32))
    assert tuple(probability.shape) == (2, 1)


def test_freeze_manifest_detects_input_change(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "freeze.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol_id": "pit-cdr-lstm-direct-2024-2025-v1",
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
