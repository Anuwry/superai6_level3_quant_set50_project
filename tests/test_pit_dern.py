from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_registered_pit_dern_configuration_is_frozen() -> None:
    from models.pit_dern import (
        BATCH_SIZE,
        EMBEDDING_DIM,
        EPOCHS,
        MODEL_KEY,
        PROTOCOL_ID,
        RETRIEVAL_TOP_K_PER_CLASS,
        SEQUENCE_WINDOW,
    )

    assert PROTOCOL_ID == "pit-dern-exploratory-v1"
    assert MODEL_KEY == "pit_dern"
    assert SEQUENCE_WINDOW == 5
    assert EPOCHS == 20
    assert BATCH_SIZE == 32
    assert EMBEDDING_DIM == 16
    assert RETRIEVAL_TOP_K_PER_CLASS == 5


def test_point_in_time_memory_rejects_unavailable_label() -> None:
    from models.pit_dern import validate_point_in_time_memory

    memory_label_dates = pd.to_datetime(["2021-12-30", "2022-01-04"])
    query_dates = pd.to_datetime(["2022-01-04", "2022-01-05"])

    with pytest.raises(ValueError, match="strictly precede"):
        validate_point_in_time_memory(memory_label_dates, query_dates)


def test_point_in_time_memory_accepts_strictly_historical_labels() -> None:
    from models.pit_dern import validate_point_in_time_memory

    memory_label_dates = pd.to_datetime(["2021-12-29", "2021-12-30"])
    query_dates = pd.to_datetime(["2022-01-04", "2022-01-05"])

    validate_point_in_time_memory(memory_label_dates, query_dates)


def test_dual_evidence_retrieval_is_balanced_and_finite() -> None:
    from models.pit_dern import dual_evidence_retrieval

    memory_embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [-1.0, 0.0],
            [-0.9, 0.1],
            [-0.8, 0.2],
            [-0.7, 0.3],
        ]
    )
    memory_labels = np.asarray([1, 1, 1, 0, 0, 0, 0])
    memory_deltas = np.asarray([0.4, 0.3, 0.2, -0.4, -0.3, -0.2, -0.1])
    query_embeddings = np.asarray([[1.0, 0.0], [-1.0, 0.0]])
    memory_dates = pd.date_range("2020-01-01", periods=7)

    result = dual_evidence_retrieval(
        query_embeddings,
        memory_embeddings,
        memory_labels,
        memory_deltas,
        memory_dates=memory_dates,
        top_k_per_class=2,
    )

    assert result.probability.shape == (2,)
    assert result.scaled_delta.shape == (2,)
    assert np.isfinite(result.probability).all()
    assert np.isfinite(result.scaled_delta).all()
    assert result.probability[0] > 0.5
    assert result.probability[1] < 0.5
    assert result.up_dates.shape == (2, 2)
    assert result.down_dates.shape == (2, 2)


def test_dual_evidence_requires_both_classes_and_enough_neighbors() -> None:
    from models.pit_dern import dual_evidence_retrieval

    with pytest.raises(ValueError, match="both direction classes"):
        dual_evidence_retrieval(
            np.asarray([[1.0, 0.0]]),
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            np.asarray([1, 1]),
            np.asarray([0.1, 0.2]),
            memory_dates=pd.date_range("2020-01-01", periods=2),
            top_k_per_class=1,
        )


def test_transferability_blend_falls_back_when_evidence_is_weak() -> None:
    from models.pit_dern import blend_transferable_evidence

    result = blend_transferable_evidence(
        encoder_probability=np.asarray([0.8]),
        encoder_delta=np.asarray([0.3]),
        retrieval_probability=np.asarray([0.2]),
        retrieval_delta=np.asarray([-0.4]),
        best_similarity=np.asarray([0.25]),
    )

    assert result.gate[0] == pytest.approx(0.0)
    assert result.probability[0] == pytest.approx(0.8)
    assert result.scaled_delta[0] == pytest.approx(0.3)


def test_standard_and_shuffled_retrieval_are_deterministic() -> None:
    from models.pit_dern import permute_memory_outcomes, standard_retrieval

    memory_embeddings = np.eye(4, dtype=float)
    query_embeddings = memory_embeddings[:2]
    labels = np.asarray([1, 0, 1, 0])
    deltas = np.asarray([0.2, -0.2, 0.1, -0.1])

    first = permute_memory_outcomes(labels, deltas, seed=42)
    second = permute_memory_outcomes(labels, deltas, seed=42)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])

    retrieval = standard_retrieval(
        query_embeddings,
        memory_embeddings,
        labels,
        deltas,
        top_k=2,
    )
    assert retrieval.probability.shape == (2,)
    assert np.isfinite(retrieval.scaled_delta).all()


def test_regime_conditioning_preserves_unselected_features_at_floor() -> None:
    from models.pit_dern import build_regime_conditioned_features

    values = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    names = ("a", "b", "c")
    regimes = np.asarray(["bull", "bear"], dtype=object)
    probabilities = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
    selected = {
        "bull": ("a",),
        "sideway": ("b",),
        "bear": ("c",),
    }

    conditioned, output_names = build_regime_conditioned_features(
        values,
        feature_names=names,
        regimes=regimes,
        regime_probabilities=probabilities,
        selected_by_regime=selected,
        residual_floor=0.25,
    )

    assert output_names == (
        "a",
        "b",
        "c",
        "regime_prob_bull",
        "regime_prob_sideway",
        "regime_prob_bear",
    )
    assert conditioned.shape == (2, 6)
    assert conditioned[0, :3].tolist() == pytest.approx([1.0, 0.5, 0.75])
    assert conditioned[1, :3].tolist() == pytest.approx([1.0, 1.25, 6.0])


def test_model_builds_and_runs_one_graph_fit() -> None:
    tf = pytest.importorskip("tensorflow")
    from models.pit_dern import build_pit_dern_model

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(7)
    model = build_pit_dern_model((5, 8))
    rng = np.random.default_rng(7)
    x = rng.normal(size=(16, 5, 8)).astype(np.float32)
    direction = np.asarray([0, 1] * 8, dtype=np.float32)
    delta = np.where(direction > 0, 0.1, -0.1).astype(np.float32)

    model.fit(
        x,
        {
            "direction": direction,
            "scaled_delta": delta,
            "embedding": direction,
        },
        epochs=1,
        batch_size=8,
        shuffle=False,
        verbose=0,
    )
    prediction = model.predict(x[:3], verbose=0)

    assert prediction["direction"].shape == (3, 1)
    assert prediction["scaled_delta"].shape == (3, 1)
    assert prediction["embedding"].shape == (3, 16)
    assert np.isfinite(prediction["embedding"]).all()
