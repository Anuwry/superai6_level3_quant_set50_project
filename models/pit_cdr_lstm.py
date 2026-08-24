from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
)

PROTOCOL_ID = "pit-cdr-lstm-direct-2024-2025-v1"
REGIMES = ("bull", "sideway", "bear")


@dataclass(frozen=True)
class CDRConfig:
    window: int = 5
    feature_count: int = 130
    lstm_units: int = 16
    dense_units: int = 8
    learning_rate: float = 0.001
    brier_component_weight: float = 0.05
    counter_direction_weight: float = 0.20
    cross_state_weight: float = 0.05

    def __post_init__(self) -> None:
        integer_values = (
            self.window,
            self.feature_count,
            self.lstm_units,
            self.dense_units,
        )
        if any(isinstance(value, bool) or value < 1 for value in integer_values):
            raise ValueError("CDR dimensions must be positive integers")
        numeric_values = (
            self.learning_rate,
            self.brier_component_weight,
            self.counter_direction_weight,
            self.cross_state_weight,
        )
        if not np.isfinite(numeric_values).all() or any(
            value < 0.0 for value in numeric_values
        ):
            raise ValueError("CDR loss and optimizer values must be finite and non-negative")


@dataclass(frozen=True)
class RelationPairs:
    left: np.ndarray
    right: np.ndarray
    relation: np.ndarray

    def __post_init__(self) -> None:
        lengths = {len(self.left), len(self.right), len(self.relation)}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise ValueError("Relation-pair arrays must be non-empty and aligned")
        if set(np.asarray(self.relation, dtype=object)).difference(
            {"counter", "transport"}
        ):
            raise ValueError("Unknown relation type")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("PIT-CDR protocol ID does not match")
    if payload.get("frozen_before_outer_execution") is not True:
        raise ValueError("PIT-CDR protocol was not frozen before execution")
    hashes = payload.get("input_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("PIT-CDR freeze manifest has no input hashes")
    resolved_root = root.resolve()
    checked: dict[str, str] = {}
    for relative, expected in hashes.items():
        path = (resolved_root / str(relative)).resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError("Frozen input path escapes the project") from error
        if not path.is_file():
            raise FileNotFoundError(f"Frozen input is missing: {relative}")
        actual = _sha256(path)
        if actual.lower() != str(expected).lower():
            raise ValueError(f"Frozen input hash mismatch: {relative}")
        checked[str(relative)] = actual.upper()
    return {
        "passed": True,
        "protocol_id": PROTOCOL_ID,
        "checked_files": len(checked),
        "input_sha256": checked,
    }


def regime_feature_masks(
    feature_pool: Sequence[str],
    selected_by_regime: Mapping[str, Sequence[str]],
    *,
    news_features: Sequence[str],
) -> dict[str, np.ndarray]:
    features = tuple(str(value) for value in feature_pool)
    if not features or len(set(features)) != len(features):
        raise ValueError("feature_pool must be non-empty and unique")
    if set(selected_by_regime) != set(REGIMES):
        raise ValueError("selected_by_regime must contain exactly the three regimes")
    news = tuple(str(value) for value in news_features)
    if not news or len(set(news)) != len(news):
        raise ValueError("news_features must be non-empty and unique")
    missing_news = sorted(set(news).difference(features))
    if missing_news:
        raise ValueError(f"News features are missing from the pool: {missing_news}")
    feature_set = set(features)
    masks: dict[str, np.ndarray] = {}
    for regime in REGIMES:
        selected = tuple(str(value) for value in selected_by_regime[regime])
        if not selected or len(set(selected)) != len(selected):
            raise ValueError(f"Selected {regime} features must be non-empty and unique")
        missing = sorted(set(selected).difference(feature_set))
        if missing:
            raise ValueError(f"Selected {regime} features are missing: {missing}")
        active = set(selected).union(news)
        masks[regime] = np.asarray(
            [1.0 if feature in active else 0.0 for feature in features],
            dtype=np.float32,
        )
    return masks


def apply_endpoint_regime_masks(
    sequences: np.ndarray,
    regimes: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> np.ndarray:
    values = np.asarray(sequences, dtype=np.float32)
    labels = np.asarray(regimes, dtype=object).reshape(-1)
    if values.ndim != 3 or len(values) != len(labels):
        raise ValueError("Sequences and endpoint regimes must be three-dimensional/aligned")
    if set(masks) != set(REGIMES):
        raise ValueError("Masks must contain exactly the three regimes")
    unknown = sorted(set(labels).difference(REGIMES))
    if unknown:
        raise ValueError(f"Unknown endpoint regimes: {unknown}")
    matrix = np.vstack([np.asarray(masks[str(regime)]) for regime in labels])
    if matrix.shape != (len(values), values.shape[2]):
        raise ValueError("Regime-mask feature count does not match sequences")
    return values * matrix[:, np.newaxis, :]


def _validated_relation_inputs(
    labels: np.ndarray,
    regimes: np.ndarray,
    state: np.ndarray,
    endpoint_positions: np.ndarray,
    minimum_separation: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    binary = np.asarray(labels, dtype=np.int8).reshape(-1)
    regime_values = np.asarray(regimes, dtype=object).reshape(-1)
    state_values = np.asarray(state, dtype=np.float64)
    positions = np.asarray(endpoint_positions, dtype=np.int64).reshape(-1)
    lengths = {len(binary), len(regime_values), len(state_values), len(positions)}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
        raise ValueError("Relation inputs must contain aligned non-trivial rows")
    if state_values.ndim != 2 or state_values.shape[1] < 1:
        raise ValueError("Matching state must be a two-dimensional feature matrix")
    if set(np.unique(binary)).difference({0, 1}) or len(np.unique(binary)) != 2:
        raise ValueError("Relation labels must contain both binary classes")
    if set(regime_values).difference(REGIMES):
        raise ValueError("Relation regimes contain unknown values")
    if not np.isfinite(state_values).all():
        raise ValueError("Matching state contains non-finite values")
    if isinstance(minimum_separation, bool) or minimum_separation < 1:
        raise ValueError("minimum_separation must be a positive integer")
    mean = state_values.mean(axis=0)
    scale = state_values.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    standardized = (state_values - mean) / scale
    return binary, regime_values, standardized, positions


def build_relation_pairs(
    labels: np.ndarray,
    regimes: np.ndarray,
    state: np.ndarray,
    *,
    endpoint_positions: np.ndarray,
    minimum_separation: int,
    seed: int,
    strategy: str,
) -> RelationPairs:
    if strategy not in {"matched", "random"}:
        raise ValueError("strategy must be matched or random")
    binary, regime_values, standardized, positions = _validated_relation_inputs(
        labels,
        regimes,
        state,
        endpoint_positions,
        minimum_separation,
    )
    rng = np.random.default_rng(int(seed))
    left: list[int] = []
    right: list[int] = []
    relation: list[str] = []
    indices = np.arange(len(binary))

    def choose(anchor: int, eligible: np.ndarray) -> int:
        candidates = indices[eligible]
        if len(candidates) < 1:
            raise ValueError(f"No eligible {strategy} relation partner for row {anchor}")
        if strategy == "random":
            return int(rng.choice(candidates))
        distance = np.square(
            standardized[candidates] - standardized[anchor]
        ).sum(axis=1)
        return int(candidates[np.lexsort((candidates, distance))[0]])

    for anchor in indices:
        separated = np.abs(positions - positions[anchor]) >= minimum_separation
        counter_partner = choose(
            int(anchor),
            separated
            & (regime_values == regime_values[anchor])
            & (binary != binary[anchor]),
        )
        if binary[anchor] == 1:
            left.append(int(anchor))
            right.append(counter_partner)
        else:
            left.append(counter_partner)
            right.append(int(anchor))
        relation.append("counter")

        transport_partner = choose(
            int(anchor),
            separated
            & (regime_values != regime_values[anchor])
            & (binary == binary[anchor]),
        )
        left.append(int(anchor))
        right.append(transport_partner)
        relation.append("transport")

    return RelationPairs(
        left=np.asarray(left, dtype=np.int64),
        right=np.asarray(right, dtype=np.int64),
        relation=np.asarray(relation, dtype=object),
    )


def build_pit_cdr_models(config: CDRConfig):
    import tensorflow as tf

    encoder_input = tf.keras.layers.Input(
        shape=(config.window, config.feature_count),
        name="encoder_sequence",
    )
    encoded = tf.keras.layers.LSTM(config.lstm_units, name="lstm")(encoder_input)
    encoded = tf.keras.layers.Dense(
        config.dense_units,
        activation="relu",
        name="representation",
    )(encoded)
    encoder = tf.keras.Model(encoder_input, encoded, name="shared_encoder")
    direction_logit = tf.keras.layers.Dense(1, name="direction_logit")

    left_input = tf.keras.layers.Input(
        shape=(config.window, config.feature_count),
        name="left_sequence",
    )
    right_input = tf.keras.layers.Input(
        shape=(config.window, config.feature_count),
        name="right_sequence",
    )
    left_embedding = encoder(left_input)
    right_embedding = encoder(right_input)
    left_logit = direction_logit(left_embedding)
    right_logit = direction_logit(right_embedding)
    left_probability = tf.keras.layers.Activation(
        "sigmoid", name="left_probability"
    )(left_logit)
    right_probability = tf.keras.layers.Activation(
        "sigmoid", name="right_probability"
    )(right_logit)
    logit_difference = tf.keras.layers.Subtract(name="counter_logit_difference")(
        [left_logit, right_logit]
    )
    counter_rank = tf.keras.layers.Activation("sigmoid", name="counter_rank")(
        logit_difference
    )
    cosine_similarity = tf.keras.layers.Dot(
        axes=1,
        normalize=True,
        name="transport_cosine_similarity",
    )([left_embedding, right_embedding])
    transport_distance = tf.keras.layers.Lambda(
        lambda value: 1.0 - value,
        output_shape=(1,),
        name="transport_distance",
    )(cosine_similarity)
    training_model = tf.keras.Model(
        [left_input, right_input],
        {
            "left_probability": left_probability,
            "right_probability": right_probability,
            "counter_rank": counter_rank,
            "transport_distance": transport_distance,
        },
        name="pit_cdr_twin_training",
    )

    inference_input = tf.keras.layers.Input(
        shape=(config.window, config.feature_count),
        name="inference_sequence",
    )
    inference_probability = tf.keras.layers.Activation(
        "sigmoid", name="inference_probability"
    )(direction_logit(encoder(inference_input)))
    inference_model = tf.keras.Model(
        inference_input,
        inference_probability,
        name="pit_cdr_single_tower_inference",
    )
    return training_model, inference_model


def compile_pit_cdr_model(model: Any, config: CDRConfig) -> None:
    import tensorflow as tf

    bce = tf.keras.losses.BinaryCrossentropy()

    def pointwise_loss(y_true: Any, y_pred: Any) -> Any:
        base = bce(y_true, y_pred)
        brier = tf.reduce_mean(tf.square(y_true - y_pred))
        return base + config.brier_component_weight * brier

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss={
            "left_probability": pointwise_loss,
            "right_probability": pointwise_loss,
            "counter_rank": tf.keras.losses.BinaryCrossentropy(),
            "transport_distance": tf.keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "left_probability": 0.5,
            "right_probability": 0.5,
            "counter_rank": config.counter_direction_weight,
            "transport_distance": config.cross_state_weight,
        },
    )


def direction_metrics(
    *,
    current_close: np.ndarray,
    next_close: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | int]:
    close = np.asarray(current_close, dtype=np.float64).reshape(-1)
    future = np.asarray(next_close, dtype=np.float64).reshape(-1)
    scores = np.asarray(probability, dtype=np.float64).reshape(-1)
    if len(close) < 1 or close.shape != future.shape or close.shape != scores.shape:
        raise ValueError("Direction metric arrays must be non-empty and aligned")
    if not np.isfinite(np.column_stack([close, future, scores])).all():
        raise ValueError("Direction metric arrays contain non-finite values")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Direction probabilities must be in [0, 1]")
    true_direction = np.sign(future - close).astype(np.int8)
    eligible = true_direction != 0
    if not eligible.any():
        raise ValueError("No non-zero direction observations are available")
    labels = (true_direction[eligible] > 0).astype(np.int8)
    predictions = (scores[eligible] > 0.5).astype(np.int8)
    if len(np.unique(labels)) != 2:
        raise ValueError("Balanced accuracy requires both direction classes")
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "observations": len(close),
        "direction_evaluable": int(eligible.sum()),
        "direction_accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "predicted_up_share": float(predictions.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
