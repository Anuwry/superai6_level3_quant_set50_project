from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PROTOCOL_ID = "pit-tldn-inner-development-v1"
MODEL_KEY = "pit_tldn"
MODEL_LABEL = "PIT-TLDN"

CNN_WINDOW = 20
LSTM_WINDOW = 5
TOP_K = 30
CNN_FILTERS = 8
WORKER_DENSE_UNITS = 8
LSTM_UNITS = 16
RETURN_LOSS_WEIGHT = 0.25
LEADER_HIDDEN_UNITS = 4
LEADER_CORRECTION_CAP = 0.5
LEARNING_RATE = 0.001

DEBATE_FEATURES = (
    "cnn_logit",
    "lstm_logit",
    "absolute_disagreement",
    "cnn_confidence",
    "lstm_confidence",
    "prob_bull",
    "prob_sideway",
    "prob_bear",
    "routing_entropy",
)
DISAGREEMENT_INDEX = DEBATE_FEATURES.index("absolute_disagreement")


@dataclass(frozen=True)
class TemporalSplit:
    name: str
    train_indices: np.ndarray
    validation_indices: np.ndarray


def _positive_shape(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value)
    ):
        raise ValueError(f"{name} must contain two positive integers")
    return value


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Worker probabilities must be finite")
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def build_debate_features(
    cnn_probability: np.ndarray,
    lstm_probability: np.ndarray,
    context: np.ndarray,
) -> np.ndarray:
    cnn = np.asarray(cnn_probability, dtype=np.float64).reshape(-1)
    lstm = np.asarray(lstm_probability, dtype=np.float64).reshape(-1)
    regime = np.asarray(context, dtype=np.float64)
    if cnn.shape != lstm.shape or len(cnn) < 1:
        raise ValueError("Worker probabilities must be non-empty and aligned")
    if regime.shape != (len(cnn), 4):
        raise ValueError("Regime context must have shape (observations, 4)")
    if not np.isfinite(regime).all():
        raise ValueError("Regime context must be finite")
    result = np.column_stack(
        [
            clipped_logit(cnn),
            clipped_logit(lstm),
            np.abs(cnn - lstm),
            2.0 * np.abs(cnn - 0.5),
            2.0 * np.abs(lstm - 0.5),
            regime,
        ]
    ).astype(np.float32)
    if result.shape[1] != len(DEBATE_FEATURES) or not np.isfinite(result).all():
        raise RuntimeError("Debate features violate their frozen contract")
    return result


def remove_disagreement_signal(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(DEBATE_FEATURES):
        raise ValueError("Debate feature matrix has an invalid shape")
    result = values.copy()
    result[:, DISAGREEMENT_INDEX] = 0.0
    return result


def expanding_temporal_splits(
    observations: int,
    *,
    split_count: int = 3,
    initial_fraction: float = 0.55,
    purge_gap: int = CNN_WINDOW,
    minimum_train: int = 80,
) -> tuple[TemporalSplit, ...]:
    if isinstance(observations, bool) or not isinstance(observations, int) or observations < 1:
        raise ValueError("observations must be a positive integer")
    if split_count < 2 or not 0.0 < initial_fraction < 1.0:
        raise ValueError("Temporal split settings are invalid")
    validation_start = max(minimum_train + purge_gap, int(np.floor(observations * initial_fraction)))
    if validation_start >= observations:
        raise ValueError("Not enough observations for temporal cross-fitting")
    blocks = np.array_split(np.arange(validation_start, observations, dtype=int), split_count)
    splits: list[TemporalSplit] = []
    for index, validation in enumerate(blocks, start=1):
        if len(validation) < 1:
            raise ValueError("A temporal validation block is empty")
        train_stop = int(validation[0]) - purge_gap
        if train_stop < minimum_train:
            raise ValueError("A temporal training prefix is too short")
        train = np.arange(0, train_stop, dtype=int)
        if int(train[-1]) + purge_gap >= int(validation[0]):
            raise RuntimeError("Temporal cross-fit purge is not strict")
        splits.append(
            TemporalSplit(
                name=f"crossfit_{index}",
                train_indices=train,
                validation_indices=validation,
            )
        )
    return tuple(splits)


def top_feature_indices(importance: np.ndarray, *, top_k: int = TOP_K) -> np.ndarray:
    values = np.asarray(importance, dtype=np.float64).reshape(-1)
    if len(values) < top_k or top_k < 1 or not np.isfinite(values).all():
        raise ValueError("Feature importance cannot satisfy the registered top-k")
    order = np.lexsort((np.arange(len(values)), -values))
    selected = np.sort(order[:top_k]).astype(int)
    if len(np.unique(selected)) != top_k:
        raise RuntimeError("Selected feature indices are not unique")
    return selected


def build_cnn_trend_worker(
    input_shape: tuple[int, int] = (CNN_WINDOW, TOP_K),
):
    import tensorflow as tf

    shape = _positive_shape(input_shape, name="input_shape")
    inputs = tf.keras.layers.Input(shape=shape, name="trend_sequence")
    branches = [
        tf.keras.layers.Conv1D(
            filters=CNN_FILTERS,
            kernel_size=kernel,
            activation="relu",
            padding="causal",
            name=f"causal_conv_k{kernel}",
        )(inputs)
        for kernel in (2, 3, 5)
    ]
    pooled = [
        tf.keras.layers.GlobalAveragePooling1D(name=f"trend_pool_{index}")(branch)
        for index, branch in enumerate(branches, start=1)
    ]
    encoded = tf.keras.layers.Concatenate(name="multi_scale_trend")(pooled)
    encoded = tf.keras.layers.Dense(
        WORKER_DENSE_UNITS,
        activation="relu",
        name="trend_dense",
    )(encoded)
    direction = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="direction",
    )(encoded)
    model = tf.keras.Model(inputs=inputs, outputs=direction, name="cnn_trend_worker")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def build_lstm_price_worker(
    input_shape: tuple[int, int] = (LSTM_WINDOW, TOP_K),
):
    import tensorflow as tf

    shape = _positive_shape(input_shape, name="input_shape")
    inputs = tf.keras.layers.Input(shape=shape, name="price_sequence")
    encoded = tf.keras.layers.LSTM(LSTM_UNITS, name="price_lstm")(inputs)
    encoded = tf.keras.layers.Dense(
        WORKER_DENSE_UNITS,
        activation="relu",
        name="price_dense",
    )(encoded)
    direction = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="direction",
    )(encoded)
    next_return = tf.keras.layers.Dense(1, name="next_return")(encoded)
    model = tf.keras.Model(
        inputs=inputs,
        outputs={"direction": direction, "next_return": next_return},
        name="lstm_price_worker",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            "direction": "binary_crossentropy",
            "next_return": tf.keras.losses.Huber(),
        },
        loss_weights={"direction": 1.0, "next_return": RETURN_LOSS_WEIGHT},
        metrics={"direction": [tf.keras.metrics.BinaryAccuracy(name="accuracy")]},
    )
    return model


def worker_output_model(worker, *, output_layer: str):
    import tensorflow as tf

    if output_layer not in {"direction", "next_return"}:
        raise ValueError("Unknown worker attribution target")
    return tf.keras.Model(
        inputs=worker.input,
        outputs=worker.get_layer(output_layer).output,
        name=f"{worker.name}_{output_layer}_output",
    )


def build_debate_leader(
    *,
    input_features: int = len(DEBATE_FEATURES),
):
    import tensorflow as tf

    if isinstance(input_features, bool) or not isinstance(input_features, int) or input_features < 2:
        raise ValueError("input_features must be an integer greater than one")
    inputs = tf.keras.layers.Input(shape=(input_features,), name="debate_claims")
    hidden = tf.keras.layers.Dense(
        LEADER_HIDDEN_UNITS,
        activation="tanh",
        name="leader_hidden",
    )(inputs)
    weight = tf.keras.layers.Dense(1, activation="sigmoid", name="cnn_weight")(hidden)
    correction_unit = tf.keras.layers.Dense(
        1,
        activation="tanh",
        name="correction_unit",
    )(hidden)
    correction = tf.keras.layers.Lambda(
        lambda value: value * LEADER_CORRECTION_CAP,
        name="bounded_correction",
    )(correction_unit)
    cnn_logit = tf.keras.layers.Lambda(lambda value: value[:, 0:1], name="cnn_claim")(inputs)
    lstm_logit = tf.keras.layers.Lambda(lambda value: value[:, 1:2], name="lstm_claim")(inputs)
    combined = tf.keras.layers.Lambda(
        lambda values: values[0] * values[1] + (1.0 - values[0]) * values[2] + values[3],
        name="arbitrated_logit",
    )([weight, cnn_logit, lstm_logit, correction])
    probability = tf.keras.layers.Activation("sigmoid", name="probability")(combined)
    training_model = tf.keras.Model(inputs=inputs, outputs=probability, name="debate_leader")
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    diagnostics_model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "probability": probability,
            "cnn_weight": weight,
            "correction": correction,
        },
        name="debate_leader_diagnostics",
    )
    return training_model, diagnostics_model
