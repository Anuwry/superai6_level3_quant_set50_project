from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

PROTOCOL_ID = "pit-dern-exploratory-v1"
MODEL_KEY = "pit_dern"
MODEL_LABEL = "PIT-DERN"
SEQUENCE_WINDOW = 5
EPOCHS = 20
BATCH_SIZE = 32
PROJECTION_DIM = 24
MIXER_BLOCKS = 2
MIXER_EXPANSION = 2
EMBEDDING_DIM = 16
DROPOUT = 0.10
LEARNING_RATE = 0.001
CONTRASTIVE_TEMPERATURE = 0.10
CONTRASTIVE_LOSS_WEIGHT = 0.10
DELTA_LOSS_WEIGHT = 0.25
STANDARD_RETRIEVAL_TOP_K = 10
RETRIEVAL_TOP_K_PER_CLASS = 5
RETRIEVAL_TEMPERATURE = 0.20
MINIMUM_SIMILARITY_ANCHOR = 0.25
REGIME_RESIDUAL_FLOOR = 0.25
DECISION_THRESHOLD = 0.50

CONFIG: dict[str, object] = {
    "protocol_id": PROTOCOL_ID,
    "model": MODEL_LABEL,
    "sequence_window": SEQUENCE_WINDOW,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "projection_dim": PROJECTION_DIM,
    "mixer_blocks": MIXER_BLOCKS,
    "mixer_expansion": MIXER_EXPANSION,
    "embedding_dim": EMBEDDING_DIM,
    "dropout": DROPOUT,
    "learning_rate": LEARNING_RATE,
    "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
    "contrastive_loss_weight": CONTRASTIVE_LOSS_WEIGHT,
    "delta_loss_weight": DELTA_LOSS_WEIGHT,
    "standard_retrieval_top_k": STANDARD_RETRIEVAL_TOP_K,
    "retrieval_top_k_per_class": RETRIEVAL_TOP_K_PER_CLASS,
    "retrieval_temperature": RETRIEVAL_TEMPERATURE,
    "minimum_similarity_anchor": MINIMUM_SIMILARITY_ANCHOR,
    "regime_residual_floor": REGIME_RESIDUAL_FLOOR,
    "decision_threshold": DECISION_THRESHOLD,
    "hyperparameter_tuning": False,
    "shuffle": False,
}


@tf.keras.utils.register_keras_serializable(package="pit_dern")
class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        temperature: float = CONTRASTIVE_TEMPERATURE,
        name: str = "supervised_contrastive_loss",
    ) -> None:
        super().__init__(name=name, reduction="none")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        labels = tf.reshape(tf.cast(y_true, tf.int32), (-1,))
        embeddings = tf.math.l2_normalize(y_pred, axis=1)
        logits = tf.matmul(embeddings, embeddings, transpose_b=True)
        logits = logits / self.temperature
        logits = logits - tf.stop_gradient(
            tf.reduce_max(logits, axis=1, keepdims=True)
        )
        batch_size = tf.shape(labels)[0]
        non_self = 1.0 - tf.eye(batch_size, dtype=logits.dtype)
        positives = tf.cast(
            tf.equal(labels[:, None], labels[None, :]),
            logits.dtype,
        ) * non_self
        exp_logits = tf.exp(logits) * non_self
        log_probability = logits - tf.math.log(
            tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-12
        )
        positive_count = tf.reduce_sum(positives, axis=1)
        mean_positive_log_probability = tf.math.divide_no_nan(
            tf.reduce_sum(positives * log_probability, axis=1),
            positive_count,
        )
        return tf.where(
            positive_count > 0.0,
            -mean_positive_log_probability,
            tf.zeros_like(mean_positive_log_probability),
        )

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "temperature": self.temperature}


def _mixer_block(
    values: tf.Tensor,
    *,
    window: int,
    channels: int,
    block: int,
) -> tf.Tensor:
    temporal = tf.keras.layers.Permute(
        (2, 1),
        name=f"temporal_transpose_{block}",
    )(values)
    temporal_update = tf.keras.layers.LayerNormalization(
        name=f"temporal_norm_{block}"
    )(temporal)
    temporal_update = tf.keras.layers.Dense(
        window * MIXER_EXPANSION,
        activation="gelu",
        name=f"temporal_expand_{block}",
    )(temporal_update)
    temporal_update = tf.keras.layers.Dropout(
        DROPOUT,
        name=f"temporal_dropout_{block}",
    )(temporal_update)
    temporal_update = tf.keras.layers.Dense(
        window,
        name=f"temporal_contract_{block}",
    )(temporal_update)
    temporal = tf.keras.layers.Add(name=f"temporal_residual_{block}")(
        [temporal, temporal_update]
    )
    temporal = tf.keras.layers.Permute(
        (2, 1),
        name=f"temporal_restore_{block}",
    )(temporal)

    channel_update = tf.keras.layers.LayerNormalization(
        name=f"channel_norm_{block}"
    )(temporal)
    channel_update = tf.keras.layers.Dense(
        channels * MIXER_EXPANSION,
        activation="gelu",
        name=f"channel_expand_{block}",
    )(channel_update)
    channel_update = tf.keras.layers.Dropout(
        DROPOUT,
        name=f"channel_dropout_{block}",
    )(channel_update)
    channel_update = tf.keras.layers.Dense(
        channels,
        name=f"channel_contract_{block}",
    )(channel_update)
    return tf.keras.layers.Add(name=f"channel_residual_{block}")(
        [temporal, channel_update]
    )


def build_pit_dern_model(
    input_shape: tuple[int, int],
) -> tf.keras.Model:
    window, features = (int(input_shape[0]), int(input_shape[1]))
    if window < 1 or features < 1:
        raise ValueError("input_shape values must be positive")
    inputs = tf.keras.layers.Input(shape=(window, features), name="market_window")
    values = tf.keras.layers.Dense(
        PROJECTION_DIM,
        name="input_projection",
    )(inputs)
    for block in range(MIXER_BLOCKS):
        values = _mixer_block(
            values,
            window=window,
            channels=PROJECTION_DIM,
            block=block,
        )
    normalized = tf.keras.layers.LayerNormalization(name="encoder_norm")(values)
    average = tf.keras.layers.GlobalAveragePooling1D(name="temporal_average")(
        normalized
    )
    last = tf.keras.layers.Lambda(
        lambda tensor: tensor[:, -1, :],
        name="last_timestep",
    )(normalized)
    representation = tf.keras.layers.Concatenate(name="pooled_representation")(
        [average, last]
    )
    raw_embedding = tf.keras.layers.Dense(
        EMBEDDING_DIM,
        name="raw_embedding",
    )(representation)
    embedding = tf.keras.layers.UnitNormalization(
        axis=1,
        name="embedding",
    )(raw_embedding)
    direction = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="direction",
    )(embedding)
    scaled_delta = tf.keras.layers.Dense(
        1,
        name="scaled_delta",
    )(embedding)
    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "direction": direction,
            "scaled_delta": scaled_delta,
            "embedding": embedding,
        },
        name=MODEL_KEY,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "direction": tf.keras.losses.BinaryCrossentropy(),
            "scaled_delta": tf.keras.losses.Huber(),
            "embedding": SupervisedContrastiveLoss(),
        },
        loss_weights={
            "direction": 1.0,
            "scaled_delta": DELTA_LOSS_WEIGHT,
            "embedding": CONTRASTIVE_LOSS_WEIGHT,
        },
    )
    return model


@dataclass(frozen=True)
class RetrievalPrediction:
    probability: np.ndarray
    scaled_delta: np.ndarray


@dataclass(frozen=True)
class DualEvidencePrediction(RetrievalPrediction):
    best_similarity: np.ndarray
    up_similarity: np.ndarray
    down_similarity: np.ndarray
    up_dates: np.ndarray
    down_dates: np.ndarray


@dataclass(frozen=True)
class BlendedEvidencePrediction(RetrievalPrediction):
    gate: np.ndarray


def _normalized_embeddings(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] < 1:
        raise ValueError(f"{name} must be a two-dimensional embedding matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError(f"{name} contains a zero-length embedding")
    return result / norms


def _memory_arrays(
    memory_embeddings: np.ndarray,
    memory_labels: np.ndarray,
    memory_deltas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = _normalized_embeddings(
        memory_embeddings,
        name="memory_embeddings",
    )
    labels = np.asarray(memory_labels, dtype=np.int8).reshape(-1)
    deltas = np.asarray(memory_deltas, dtype=np.float64).reshape(-1)
    if len(embeddings) != len(labels) or len(labels) != len(deltas):
        raise ValueError("Memory embeddings, labels, and deltas must align")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Retrieval memory must contain both direction classes")
    if not np.isfinite(deltas).all():
        raise ValueError("memory_deltas contain non-finite values")
    return embeddings, labels, deltas


def _top_indices(similarities: np.ndarray, top_k: int) -> np.ndarray:
    if top_k < 1 or top_k > similarities.shape[1]:
        raise ValueError("top_k exceeds the available retrieval memory")
    candidates = np.argpartition(
        -similarities,
        kth=top_k - 1,
        axis=1,
    )[:, :top_k]
    candidate_scores = np.take_along_axis(similarities, candidates, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    return np.take_along_axis(candidates, order, axis=1)


def _softmax(values: np.ndarray, axis: int = 1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / np.sum(numerator, axis=axis, keepdims=True)


def standard_retrieval(
    query_embeddings: np.ndarray,
    memory_embeddings: np.ndarray,
    memory_labels: np.ndarray,
    memory_deltas: np.ndarray,
    *,
    top_k: int = STANDARD_RETRIEVAL_TOP_K,
    temperature: float = RETRIEVAL_TEMPERATURE,
) -> RetrievalPrediction:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    queries = _normalized_embeddings(query_embeddings, name="query_embeddings")
    memory, labels, deltas = _memory_arrays(
        memory_embeddings,
        memory_labels,
        memory_deltas,
    )
    similarities = queries @ memory.T
    indices = _top_indices(similarities, int(top_k))
    scores = np.take_along_axis(similarities, indices, axis=1)
    weights = _softmax(scores / temperature)
    retrieved_labels = labels[indices]
    retrieved_deltas = deltas[indices]
    return RetrievalPrediction(
        probability=np.sum(weights * retrieved_labels, axis=1),
        scaled_delta=np.sum(weights * retrieved_deltas, axis=1),
    )


def _class_evidence(
    similarities: np.ndarray,
    class_indices: np.ndarray,
    deltas: np.ndarray,
    dates: np.ndarray,
    *,
    top_k: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    class_similarities = similarities[:, class_indices]
    local_indices = _top_indices(class_similarities, top_k)
    global_indices = class_indices[local_indices]
    scores = np.take_along_axis(class_similarities, local_indices, axis=1)
    weights = _softmax(scores / temperature)
    class_delta = np.sum(weights * deltas[global_indices], axis=1)
    scaled = scores / temperature
    maximum = np.max(scaled, axis=1, keepdims=True)
    log_mean_evidence = (
        np.log(np.mean(np.exp(scaled - maximum), axis=1))
        + maximum.reshape(-1)
    )
    return log_mean_evidence, class_delta, scores, dates[global_indices]


def dual_evidence_retrieval(
    query_embeddings: np.ndarray,
    memory_embeddings: np.ndarray,
    memory_labels: np.ndarray,
    memory_deltas: np.ndarray,
    *,
    memory_dates: pd.DatetimeIndex | np.ndarray,
    top_k_per_class: int = RETRIEVAL_TOP_K_PER_CLASS,
    temperature: float = RETRIEVAL_TEMPERATURE,
) -> DualEvidencePrediction:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    queries = _normalized_embeddings(query_embeddings, name="query_embeddings")
    memory, labels, deltas = _memory_arrays(
        memory_embeddings,
        memory_labels,
        memory_deltas,
    )
    dates = pd.to_datetime(memory_dates, errors="raise").to_numpy()
    if dates.shape != (len(memory),):
        raise ValueError("memory_dates must align with retrieval memory")
    up_indices = np.flatnonzero(labels == 1)
    down_indices = np.flatnonzero(labels == 0)
    if min(len(up_indices), len(down_indices)) < top_k_per_class:
        raise ValueError("Both classes require enough neighbors for dual retrieval")
    similarities = queries @ memory.T
    up_log, up_delta, up_scores, up_dates = _class_evidence(
        similarities,
        up_indices,
        deltas,
        dates,
        top_k=top_k_per_class,
        temperature=temperature,
    )
    down_log, down_delta, down_scores, down_dates = _class_evidence(
        similarities,
        down_indices,
        deltas,
        dates,
        top_k=top_k_per_class,
        temperature=temperature,
    )
    probability = 1.0 / (1.0 + np.exp(np.clip(down_log - up_log, -50.0, 50.0)))
    scaled_delta = probability * up_delta + (1.0 - probability) * down_delta
    best_similarity = np.maximum(up_scores[:, 0], down_scores[:, 0])
    return DualEvidencePrediction(
        probability=probability,
        scaled_delta=scaled_delta,
        best_similarity=best_similarity,
        up_similarity=up_scores,
        down_similarity=down_scores,
        up_dates=up_dates,
        down_dates=down_dates,
    )


def blend_transferable_evidence(
    *,
    encoder_probability: np.ndarray,
    encoder_delta: np.ndarray,
    retrieval_probability: np.ndarray,
    retrieval_delta: np.ndarray,
    best_similarity: np.ndarray,
    minimum_similarity_anchor: float = MINIMUM_SIMILARITY_ANCHOR,
) -> BlendedEvidencePrediction:
    arrays = [
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in (
            encoder_probability,
            encoder_delta,
            retrieval_probability,
            retrieval_delta,
            best_similarity,
        )
    ]
    if len({values.shape for values in arrays}) != 1:
        raise ValueError("Evidence blend arrays must have identical shapes")
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("Evidence blend arrays must be finite")
    encoder_p, encoder_d, retrieval_p, retrieval_d, similarity = arrays
    if not 0.0 <= minimum_similarity_anchor < 1.0:
        raise ValueError("minimum_similarity_anchor must be in [0, 1)")
    similarity_quality = np.clip(
        (similarity - minimum_similarity_anchor)
        / (1.0 - minimum_similarity_anchor),
        0.0,
        1.0,
    )
    evidence_margin = np.abs(retrieval_p - 0.5) * 2.0
    gate = similarity_quality * evidence_margin
    return BlendedEvidencePrediction(
        probability=(1.0 - gate) * encoder_p + gate * retrieval_p,
        scaled_delta=(1.0 - gate) * encoder_d + gate * retrieval_d,
        gate=gate,
    )


def permute_memory_outcomes(
    labels: np.ndarray,
    deltas: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    label_values = np.asarray(labels).reshape(-1)
    delta_values = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if label_values.shape != delta_values.shape:
        raise ValueError("labels and deltas must align")
    permutation = np.random.default_rng(int(seed)).permutation(len(label_values))
    return label_values[permutation].copy(), delta_values[permutation].copy()


def validate_point_in_time_memory(
    memory_label_dates: pd.DatetimeIndex | np.ndarray,
    query_dates: pd.DatetimeIndex | np.ndarray,
) -> None:
    labels = pd.to_datetime(memory_label_dates, errors="raise")
    queries = pd.to_datetime(query_dates, errors="raise")
    if len(labels) < 1 or len(queries) < 1:
        raise ValueError("Memory labels and queries must be non-empty")
    if labels.max() >= queries.min():
        raise ValueError("All retrieval label dates must strictly precede queries")


def build_regime_conditioned_features(
    values: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    regimes: np.ndarray,
    regime_probabilities: np.ndarray,
    selected_by_regime: dict[str, tuple[str, ...]],
    residual_floor: float = REGIME_RESIDUAL_FLOOR,
) -> tuple[np.ndarray, tuple[str, ...]]:
    matrix = np.asarray(values, dtype=np.float64)
    labels = np.asarray(regimes, dtype=object).reshape(-1)
    probabilities = np.asarray(regime_probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError("values and feature_names do not align")
    if len(labels) != len(matrix) or probabilities.shape != (len(matrix), 3):
        raise ValueError("Regime context does not align with feature rows")
    if set(selected_by_regime) != {"bull", "sideway", "bear"}:
        raise ValueError("selected_by_regime must contain all three regimes")
    if not 0.0 <= residual_floor <= 1.0:
        raise ValueError("residual_floor must be in [0, 1]")
    if not np.isfinite(matrix).all() or not np.isfinite(probabilities).all():
        raise ValueError("Regime-conditioned inputs must be finite")
    unknown = sorted(set(labels).difference(selected_by_regime))
    if unknown:
        raise ValueError(f"Unknown regimes: {unknown}")
    feature_index = {name: index for index, name in enumerate(feature_names)}
    selected_indices: dict[str, np.ndarray] = {}
    for regime, selected in selected_by_regime.items():
        missing = sorted(set(selected).difference(feature_index))
        if missing:
            raise ValueError(f"Selected {regime} features are missing: {missing}")
        selected_indices[regime] = np.asarray(
            [feature_index[name] for name in selected],
            dtype=int,
        )
    weights = np.full_like(matrix, residual_floor, dtype=np.float64)
    for regime, indices in selected_indices.items():
        rows = np.flatnonzero(labels == regime)
        weights[np.ix_(rows, indices)] = 1.0
    conditioned = np.concatenate([matrix * weights, probabilities], axis=1)
    output_names = (
        *feature_names,
        "regime_prob_bull",
        "regime_prob_sideway",
        "regime_prob_bear",
    )
    return conditioned.astype(np.float32), output_names


def probability_signed_scaled_target(
    current_scaled_close: np.ndarray,
    probability: np.ndarray,
    scaled_delta: np.ndarray,
) -> np.ndarray:
    close = np.asarray(current_scaled_close, dtype=np.float64).reshape(-1)
    direction_probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    delta = np.asarray(scaled_delta, dtype=np.float64).reshape(-1)
    if len({close.shape, direction_probability.shape, delta.shape}) != 1:
        raise ValueError("Prediction arrays must have identical shapes")
    magnitude = np.maximum(np.abs(delta), 1e-8)
    sign = np.where(direction_probability >= DECISION_THRESHOLD, 1.0, -1.0)
    return close + sign * magnitude
