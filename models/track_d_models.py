from __future__ import annotations

from typing import Literal

import numpy as np

from models.attention_lstm import (
    ATTENTION_HEADS,
    ATTENTION_KEY_DIM,
)
from models.attention_lstm import (
    DENSE_UNITS as ATTENTION_DENSE_UNITS,
)
from models.attention_lstm import (
    LSTM_UNITS as ATTENTION_LSTM_UNITS,
)
from models.convolutional_neural_network import (
    CONV_FILTERS as CNN_FILTERS,
)
from models.convolutional_neural_network import (
    DENSE_UNITS as CNN_DENSE_UNITS,
)
from models.convolutional_neural_network import (
    KERNEL_SIZE as CNN_KERNEL_SIZE,
)
from models.lstm_cnn import (
    CONV_FILTERS as LSTM_CNN_FILTERS,
)
from models.lstm_cnn import (
    DENSE_UNITS as LSTM_CNN_DENSE_UNITS,
)
from models.lstm_cnn import (
    KERNEL_SIZE as LSTM_CNN_KERNEL_SIZE,
)
from models.lstm_cnn import (
    LSTM_UNITS as LSTM_CNN_UNITS,
)
from models.lstm_cnn_attention import (
    ATTENTION_HEADS as HYBRID_ATTENTION_HEADS,
)
from models.lstm_cnn_attention import (
    ATTENTION_KEY_DIM as HYBRID_ATTENTION_KEY_DIM,
)
from models.lstm_cnn_attention import (
    CONV_FILTERS as HYBRID_FILTERS,
)
from models.lstm_cnn_attention import (
    DENSE_UNITS as HYBRID_DENSE_UNITS,
)
from models.lstm_cnn_attention import (
    KERNEL_SIZE as HYBRID_KERNEL_SIZE,
)
from models.lstm_cnn_attention import (
    LSTM_UNITS as HYBRID_LSTM_UNITS,
)

Objective = Literal["direct", "multitask"]


def make_direction_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=float)
    target = np.asarray(labels, dtype=float).reshape(-1)
    if values.ndim != 2 or len(values) != len(target):
        raise ValueError("Feature and label sequence inputs are invalid")
    if window < 1 or window > len(values):
        raise ValueError("window is incompatible with feature rows")
    sequences = [
        values[index - window + 1 : index + 1]
        for index in range(window - 1, len(values))
    ]
    return (
        np.asarray(sequences, dtype=np.float32),
        target[window - 1 :].astype(np.float32),
    )


def standardize_return_targets(
    train_returns: np.ndarray,
    evaluation_returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    train = np.asarray(train_returns, dtype=float).reshape(-1)
    evaluation = np.asarray(evaluation_returns, dtype=float).reshape(-1)
    if not np.isfinite(train).all() or not np.isfinite(evaluation).all():
        raise ValueError("Return targets must be finite")
    mean = float(train.mean())
    std = float(train.std(ddof=0))
    if std <= np.finfo(float).eps:
        raise ValueError("Training returns have zero variance")
    return (
        ((train - mean) / std).astype(np.float32),
        ((evaluation - mean) / std).astype(np.float32),
        {"mean": mean, "std": std, "fit_scope": "train_only"},
    )


def _backbone(model_key: str, inputs):
    import tensorflow as tf

    if model_key == "lstm":
        encoded = tf.keras.layers.LSTM(16)(inputs)
        return tf.keras.layers.Dense(8, activation="relu")(encoded)
    if model_key == "cnn":
        encoded = tf.keras.layers.Conv1D(
            filters=CNN_FILTERS,
            kernel_size=CNN_KERNEL_SIZE,
            activation="relu",
            padding="causal",
        )(inputs)
        encoded = tf.keras.layers.GlobalAveragePooling1D()(encoded)
        return tf.keras.layers.Dense(CNN_DENSE_UNITS, activation="relu")(
            encoded
        )
    if model_key == "lstm_cnn":
        encoded = tf.keras.layers.LSTM(
            LSTM_CNN_UNITS,
            return_sequences=True,
        )(inputs)
        encoded = tf.keras.layers.Conv1D(
            filters=LSTM_CNN_FILTERS,
            kernel_size=LSTM_CNN_KERNEL_SIZE,
            activation="relu",
            padding="causal",
        )(encoded)
        encoded = tf.keras.layers.GlobalAveragePooling1D()(encoded)
        return tf.keras.layers.Dense(
            LSTM_CNN_DENSE_UNITS,
            activation="relu",
        )(encoded)
    if model_key == "lstm_attention":
        encoded = tf.keras.layers.LSTM(
            ATTENTION_LSTM_UNITS,
            return_sequences=True,
        )(inputs)
        encoded = tf.keras.layers.MultiHeadAttention(
            num_heads=ATTENTION_HEADS,
            key_dim=ATTENTION_KEY_DIM,
        )(encoded, encoded, use_causal_mask=True)
        encoded = tf.keras.layers.GlobalAveragePooling1D()(encoded)
        return tf.keras.layers.Dense(
            ATTENTION_DENSE_UNITS,
            activation="relu",
        )(encoded)
    if model_key == "lstm_cnn_attention":
        encoded = tf.keras.layers.LSTM(
            HYBRID_LSTM_UNITS,
            return_sequences=True,
        )(inputs)
        encoded = tf.keras.layers.Conv1D(
            filters=HYBRID_FILTERS,
            kernel_size=HYBRID_KERNEL_SIZE,
            activation="relu",
            padding="causal",
        )(encoded)
        encoded = tf.keras.layers.MultiHeadAttention(
            num_heads=HYBRID_ATTENTION_HEADS,
            key_dim=HYBRID_ATTENTION_KEY_DIM,
        )(encoded, encoded, use_causal_mask=True)
        encoded = tf.keras.layers.GlobalAveragePooling1D()(encoded)
        return tf.keras.layers.Dense(
            HYBRID_DENSE_UNITS,
            activation="relu",
        )(encoded)
    raise ValueError(f"Unknown Track D model: {model_key}")


def build_track_d_model(
    model_key: str,
    *,
    input_shape: tuple[int, int],
    objective: Objective,
    return_loss_weight: float = 0.25,
):
    import tensorflow as tf

    if objective not in {"direct", "multitask"}:
        raise ValueError(f"Unknown Track D objective: {objective}")
    inputs = tf.keras.layers.Input(shape=input_shape, name="market_sequence")
    encoded = _backbone(model_key, inputs)
    direction = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="direction",
    )(encoded)
    if objective == "direct":
        model = tf.keras.Model(inputs=inputs, outputs=direction)
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
        return model
    return_output = tf.keras.layers.Dense(1, name="return")(encoded)
    model = tf.keras.Model(
        inputs=inputs,
        outputs={"direction": direction, "return": return_output},
    )
    model.compile(
        optimizer="adam",
        loss={"direction": "binary_crossentropy", "return": "mse"},
        loss_weights={"direction": 1.0, "return": return_loss_weight},
        metrics={
            "direction": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
            "return": [tf.keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )
    return model
