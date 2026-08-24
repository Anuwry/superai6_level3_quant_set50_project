from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import tensorflow as tf

MODEL_KEY = "pit_cmm_lstm"
MODEL_LABEL = "PIT-CMM-LSTM"
PROTOCOL_ID = "pit-cmm-lstm-exploratory-v1"
SEQUENCE_WINDOW = 5
HIDDEN_UNITS = 12
MEMORY_RANK = 4
DENSE_UNITS = 8
EPOCHS = 20
BATCH_SIZE = 32

CONFIG: dict[str, object] = {
    "protocol_id": PROTOCOL_ID,
    "model": MODEL_LABEL,
    "evidence_status": "post_freeze_exploratory_architecture_extension",
    "hyperparameter_tuning": False,
    "model_parameters": {
        "sequence_window": SEQUENCE_WINDOW,
        "hidden_units": HIDDEN_UNITS,
        "memory_rank": MEMORY_RANK,
        "dense_units": DENSE_UNITS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "adam",
        "loss": "mse",
        "shuffle": False,
        "negative_control_consistency_implemented": False,
    },
}


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@tf.keras.utils.register_keras_serializable(package="set50_reliability")
class CompetitiveMatrixMemoryLSTMCell(tf.keras.layers.Layer):
    """LSTM cell with competing bullish and bearish low-rank memories."""

    def __init__(
        self,
        *,
        hidden_units: int = HIDDEN_UNITS,
        memory_rank: int = MEMORY_RANK,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_units = _positive_integer(
            hidden_units,
            name="hidden_units",
        )
        self.memory_rank = _positive_integer(
            memory_rank,
            name="memory_rank",
        )
        matrix_size = self.memory_rank * self.memory_rank
        self.state_size = (
            self.hidden_units,
            self.hidden_units,
            matrix_size,
            matrix_size,
        )
        self.output_size = self.hidden_units

        self.lstm_gates = tf.keras.layers.Dense(
            4 * self.hidden_units,
            name="lstm_gates",
        )
        self.evidence_weights = tf.keras.layers.Dense(
            2,
            name="competitive_evidence_weights",
        )
        self.memory_decay = tf.keras.layers.Dense(
            2,
            name="matrix_memory_decay",
        )
        self.key_projection = tf.keras.layers.Dense(
            self.memory_rank,
            activation="tanh",
            name="matrix_key",
        )
        self.query_projection = tf.keras.layers.Dense(
            self.memory_rank,
            activation="tanh",
            name="matrix_query",
        )
        self.value_projection = tf.keras.layers.Dense(
            2 * self.memory_rank,
            activation="tanh",
            name="bull_bear_values",
        )
        self.debate_projection = tf.keras.layers.Dense(
            self.hidden_units,
            activation="tanh",
            name="competitive_readout",
        )
        self.debate_scale: tf.Variable | None = None

    def build(self, input_shape: tf.TensorShape) -> None:
        input_features = input_shape[-1]
        if input_features is None:
            raise ValueError("The PIT-CMM input feature dimension must be known")
        joined_shape = tf.TensorShape(
            [None, int(input_features) + self.hidden_units]
        )
        for projection in (
            self.lstm_gates,
            self.evidence_weights,
            self.memory_decay,
            self.key_projection,
            self.query_projection,
            self.value_projection,
        ):
            projection.build(joined_shape)
        self.debate_projection.build(
            tf.TensorShape([None, 3 * self.memory_rank + 1])
        )
        self.debate_scale = self.add_weight(
            name="debate_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
        )
        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        states: Sequence[tf.Tensor],
        training: bool | None = None,
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, ...]]:
        del training
        previous_hidden, previous_cell, bullish_flat, bearish_flat = states
        joined = tf.concat([inputs, previous_hidden], axis=-1)

        gate_values = self.lstm_gates(joined)
        candidate, input_logit, forget_logit, output_logit = tf.split(
            gate_values,
            4,
            axis=-1,
        )
        input_gate = tf.sigmoid(input_logit)
        forget_gate = tf.sigmoid(forget_logit + 1.0)
        output_gate = tf.sigmoid(output_logit)
        cell_state = forget_gate * previous_cell + input_gate * tf.tanh(candidate)

        batch_size = tf.shape(inputs)[0]
        matrix_shape = (batch_size, self.memory_rank, self.memory_rank)
        bullish_previous = tf.reshape(bullish_flat, matrix_shape)
        bearish_previous = tf.reshape(bearish_flat, matrix_shape)

        evidence = tf.nn.softmax(self.evidence_weights(joined), axis=-1)
        memory_decay = tf.sigmoid(self.memory_decay(joined) + 1.0)
        key = tf.math.l2_normalize(self.key_projection(joined), axis=-1)
        query = tf.math.l2_normalize(self.query_projection(joined), axis=-1)
        bullish_value, bearish_value = tf.split(
            self.value_projection(joined),
            2,
            axis=-1,
        )
        rank_scale = tf.sqrt(tf.cast(self.memory_rank, inputs.dtype))
        bullish_update = tf.einsum(
            "bi,bj->bij",
            bullish_value,
            key,
        ) / rank_scale
        bearish_update = tf.einsum(
            "bi,bj->bij",
            bearish_value,
            key,
        ) / rank_scale
        bullish_memory = (
            memory_decay[:, 0, None, None] * bullish_previous
            + evidence[:, 0, None, None] * bullish_update
        )
        bearish_memory = (
            memory_decay[:, 1, None, None] * bearish_previous
            + evidence[:, 1, None, None] * bearish_update
        )

        bullish_read = tf.einsum("bij,bj->bi", bullish_memory, query)
        bearish_read = tf.einsum("bij,bj->bi", bearish_memory, query)
        signed_margin = evidence[:, :1] - evidence[:, 1:]
        debate_features = tf.concat(
            [
                bullish_read,
                -bearish_read,
                bullish_read - bearish_read,
                signed_margin,
            ],
            axis=-1,
        )
        debate_state = self.debate_projection(debate_features)
        if self.debate_scale is None:
            raise RuntimeError("Cell was called before it was built")
        hidden_state = output_gate * tf.tanh(
            cell_state + self.debate_scale * debate_state
        )

        matrix_size = self.memory_rank * self.memory_rank
        bullish_next_flat = tf.reshape(
            bullish_memory,
            (-1, matrix_size),
        )
        bearish_next_flat = tf.reshape(
            bearish_memory,
            (-1, matrix_size),
        )
        bullish_next_flat.set_shape((None, matrix_size))
        bearish_next_flat.set_shape((None, matrix_size))
        next_states = (
            hidden_state,
            cell_state,
            bullish_next_flat,
            bearish_next_flat,
        )
        return hidden_state, next_states

    def get_config(self) -> dict[str, object]:
        return {
            **super().get_config(),
            "hidden_units": self.hidden_units,
            "memory_rank": self.memory_rank,
        }


def _validate_input_shape(input_shape: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(input_shape, tuple) or len(input_shape) != 2:
        raise ValueError("input_shape must be a (window, features) tuple")
    window = _positive_integer(input_shape[0], name="window")
    features = _positive_integer(input_shape[1], name="features")
    return window, features


def build_pit_cmm_lstm_model(
    input_shape: tuple[int, int],
) -> tf.keras.Model:
    """Build the frozen v1 PIT-CMM-LSTM next-close regression model."""

    window, features = _validate_input_shape(input_shape)
    inputs = tf.keras.layers.Input(
        shape=(window, features),
        name="point_in_time_sequence",
    )
    recurrent_state = tf.keras.layers.RNN(
        CompetitiveMatrixMemoryLSTMCell(
            hidden_units=HIDDEN_UNITS,
            memory_rank=MEMORY_RANK,
            name="pit_cmm_cell",
        ),
        name="pit_cmm_recurrent_layer",
    )(inputs)
    hidden = tf.keras.layers.Dense(
        DENSE_UNITS,
        activation="relu",
        name="forecast_hidden",
    )(recurrent_state)
    outputs = tf.keras.layers.Dense(1, name="next_close")(hidden)
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=MODEL_KEY,
    )
    model.compile(optimizer="adam", loss="mse")
    return model
