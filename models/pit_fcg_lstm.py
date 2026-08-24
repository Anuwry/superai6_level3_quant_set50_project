from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tensorflow as tf

PROTOCOL_ID = "pit-fcg-lstm-inner-development-v1"
MODEL_KEY = "pit_fcg_lstm"
MODEL_LABEL = "PIT-FCG-LSTM"
WINDOW = 5
NUMERIC_FEATURES = 122
NEWS_FEATURES = 8
CONTEXT_FEATURES = 4
LSTM_UNITS = 16
ANCHOR_DENSE_UNITS = 8
CORRECTION_HIDDEN_UNITS = 4
RESIDUAL_LOGIT_CAP = 1.0
LEARNING_RATE = 0.001
FALSIFICATION_MARGIN = 0.01

LOSS_WEIGHTS = {
    "final_bce": 1.0,
    "anchor_bce": 0.25,
    "candidate_bce": 0.25,
    "rank_loss": 0.25,
    "gate_loss": 0.10,
    "placebo_loss": 0.05,
}


def _positive_shape(value: tuple[int, int], *, name: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a (window, features) tuple")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value):
        raise ValueError(f"{name} values must be positive integers")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def combine_anchor_residual(
    anchor_logits: tf.Tensor,
    gate: tf.Tensor,
    residual: tf.Tensor,
) -> tf.Tensor:
    return anchor_logits + gate * residual


def _per_sample_bce(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    targets = tf.cast(tf.reshape(labels, (-1, 1)), logits.dtype)
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits)


def falsification_gate_target(
    labels: tf.Tensor,
    *,
    anchor_logits: tf.Tensor,
    aligned_candidate_logits: tf.Tensor,
    placebo_candidate_logits: tf.Tensor,
    margin: float = FALSIFICATION_MARGIN,
) -> tf.Tensor:
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    anchor_loss = _per_sample_bce(labels, anchor_logits)
    aligned_loss = _per_sample_bce(labels, aligned_candidate_logits)
    placebo_loss = _per_sample_bce(labels, placebo_candidate_logits)
    improves_anchor = aligned_loss + margin <= anchor_loss
    beats_placebo = aligned_loss + margin <= placebo_loss
    return tf.stop_gradient(
        tf.cast(tf.logical_and(improves_anchor, beats_placebo), aligned_loss.dtype)
    )


def falsification_rank_loss(
    labels: tf.Tensor,
    *,
    aligned_candidate_logits: tf.Tensor,
    placebo_candidate_logits: tf.Tensor,
    margin: float = FALSIFICATION_MARGIN,
) -> tf.Tensor:
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    aligned_loss = _per_sample_bce(labels, aligned_candidate_logits)
    placebo_loss = tf.stop_gradient(
        _per_sample_bce(labels, placebo_candidate_logits)
    )
    return tf.reduce_mean(tf.nn.relu(margin + aligned_loss - placebo_loss))


@tf.keras.utils.register_keras_serializable(package="set50_reliability")
class PITFCGLSTM(tf.keras.Model):
    """Numerical LSTM anchor with a bounded falsification-calibrated news gate."""

    def __init__(
        self,
        *,
        numeric_shape: tuple[int, int] = (WINDOW, NUMERIC_FEATURES),
        news_shape: tuple[int, int] = (WINDOW, NEWS_FEATURES),
        context_features: int = CONTEXT_FEATURES,
        use_fcg_loss: bool = True,
        name: str = MODEL_KEY,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.numeric_shape_contract = _positive_shape(
            numeric_shape,
            name="numeric_shape",
        )
        self.news_shape_contract = _positive_shape(news_shape, name="news_shape")
        if self.numeric_shape_contract[0] != self.news_shape_contract[0]:
            raise ValueError("numeric and news windows must be identical")
        self.context_features = _positive_integer(
            context_features,
            name="context_features",
        )
        self.use_fcg_loss = bool(use_fcg_loss)

        self.numeric_encoder = tf.keras.layers.LSTM(
            LSTM_UNITS,
            name="numeric_lstm_anchor",
        )
        self.anchor_hidden = tf.keras.layers.Dense(
            ANCHOR_DENSE_UNITS,
            activation="relu",
            name="anchor_hidden",
        )
        self.anchor_logit = tf.keras.layers.Dense(1, name="anchor_logit")
        self.residual_hidden = tf.keras.layers.Dense(
            CORRECTION_HIDDEN_UNITS,
            activation="relu",
            name="residual_hidden",
        )
        self.residual_output = tf.keras.layers.Dense(
            1,
            activation="tanh",
            name="bounded_news_residual",
        )
        self.gate_hidden = tf.keras.layers.Dense(
            CORRECTION_HIDDEN_UNITS,
            activation="relu",
            name="gate_hidden",
        )
        self.gate_output = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="falsification_gate",
        )

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.final_bce_tracker = tf.keras.metrics.Mean(name="final_bce")
        self.anchor_bce_tracker = tf.keras.metrics.Mean(name="anchor_bce")
        self.candidate_bce_tracker = tf.keras.metrics.Mean(name="candidate_bce")
        self.rank_loss_tracker = tf.keras.metrics.Mean(name="rank_loss")
        self.gate_loss_tracker = tf.keras.metrics.Mean(name="gate_loss")
        self.placebo_loss_tracker = tf.keras.metrics.Mean(name="placebo_loss")
        self.gate_target_rate_tracker = tf.keras.metrics.Mean(
            name="gate_target_rate"
        )

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [
            self.total_loss_tracker,
            self.final_bce_tracker,
            self.anchor_bce_tracker,
            self.candidate_bce_tracker,
            self.rank_loss_tracker,
            self.gate_loss_tracker,
            self.placebo_loss_tracker,
            self.gate_target_rate_tracker,
        ]

    @staticmethod
    def _news_summary(news: tf.Tensor) -> tf.Tensor:
        return tf.concat(
            [tf.reduce_mean(news, axis=1), news[:, -1, :]],
            axis=-1,
        )

    def _correction(
        self,
        numeric_state: tf.Tensor,
        news: tf.Tensor,
        context: tf.Tensor,
        *,
        training: bool | None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        joined = tf.concat(
            [numeric_state, self._news_summary(news), context],
            axis=-1,
        )
        residual = RESIDUAL_LOGIT_CAP * self.residual_output(
            self.residual_hidden(joined, training=training),
            training=training,
        )
        gate = self.gate_output(
            self.gate_hidden(joined, training=training),
            training=training,
        )
        return residual, gate

    def call(
        self,
        inputs: Mapping[str, tf.Tensor],
        training: bool | None = None,
    ) -> dict[str, tf.Tensor]:
        required = {"numeric", "news", "context"}
        missing = sorted(required.difference(inputs))
        if missing:
            raise ValueError(f"PIT-FCG inputs are missing: {missing}")
        numeric_state = self.numeric_encoder(inputs["numeric"], training=training)
        anchor_logits = self.anchor_logit(
            self.anchor_hidden(numeric_state, training=training),
            training=training,
        )
        residual, gate = self._correction(
            numeric_state,
            inputs["news"],
            inputs["context"],
            training=training,
        )
        final_logits = combine_anchor_residual(anchor_logits, gate, residual)
        return {
            "probability": tf.sigmoid(final_logits),
            "final_logits": final_logits,
            "anchor_logits": anchor_logits,
            "residual": residual,
            "gate": gate,
            "numeric_state": numeric_state,
        }

    def _loss_terms(
        self,
        labels: tf.Tensor,
        outputs: Mapping[str, tf.Tensor],
        *,
        placebo_news: tf.Tensor | None,
        context: tf.Tensor,
        training: bool,
    ) -> dict[str, tf.Tensor]:
        final_bce = tf.reduce_mean(
            _per_sample_bce(labels, outputs["final_logits"])
        )
        anchor_bce = tf.reduce_mean(
            _per_sample_bce(labels, outputs["anchor_logits"])
        )
        aligned_candidate = outputs["anchor_logits"] + outputs["residual"]
        candidate_bce = tf.reduce_mean(
            _per_sample_bce(labels, aligned_candidate)
        )
        zero = tf.zeros((), dtype=final_bce.dtype)
        rank_loss = gate_loss = placebo_loss = gate_target_rate = zero
        if self.use_fcg_loss:
            if placebo_news is None:
                raise ValueError("FCG training requires placebo_news")
            placebo_residual, placebo_gate = self._correction(
                outputs["numeric_state"],
                placebo_news,
                context,
                training=training,
            )
            placebo_candidate = outputs["anchor_logits"] + placebo_residual
            rank_loss = falsification_rank_loss(
                labels,
                aligned_candidate_logits=aligned_candidate,
                placebo_candidate_logits=placebo_candidate,
            )
            gate_target = falsification_gate_target(
                labels,
                anchor_logits=outputs["anchor_logits"],
                aligned_candidate_logits=aligned_candidate,
                placebo_candidate_logits=placebo_candidate,
            )
            gate_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    gate_target,
                    outputs["gate"],
                )
            )
            placebo_loss = tf.reduce_mean(
                tf.square(placebo_gate * placebo_residual)
            )
            gate_target_rate = tf.reduce_mean(gate_target)
        total = (
            LOSS_WEIGHTS["final_bce"] * final_bce
            + LOSS_WEIGHTS["anchor_bce"] * anchor_bce
            + LOSS_WEIGHTS["candidate_bce"] * candidate_bce
            + LOSS_WEIGHTS["rank_loss"] * rank_loss
            + LOSS_WEIGHTS["gate_loss"] * gate_loss
            + LOSS_WEIGHTS["placebo_loss"] * placebo_loss
        )
        return {
            "loss": total,
            "final_bce": final_bce,
            "anchor_bce": anchor_bce,
            "candidate_bce": candidate_bce,
            "rank_loss": rank_loss,
            "gate_loss": gate_loss,
            "placebo_loss": placebo_loss,
            "gate_target_rate": gate_target_rate,
        }

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        inputs, labels, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported by PIT-FCG-LSTM")
        if not isinstance(inputs, Mapping):
            raise TypeError("PIT-FCG training inputs must be a mapping")
        aligned_inputs = {
            key: inputs[key] for key in ("numeric", "news", "context")
        }
        with tf.GradientTape() as tape:
            outputs = self(aligned_inputs, training=True)
            terms = self._loss_terms(
                labels,
                outputs,
                placebo_news=inputs.get("placebo_news"),
                context=inputs["context"],
                training=True,
            )
        gradients = tape.gradient(terms["loss"], self.trainable_variables)
        valid_gradients = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients,
                self.trainable_variables,
                strict=True,
            )
            if gradient is not None
        ]
        self.optimizer.apply_gradients(valid_gradients)
        for metric in self.metrics:
            metric.update_state(terms[metric.name])
        return {metric.name: metric.result() for metric in self.metrics}

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "numeric_shape": self.numeric_shape_contract,
            "news_shape": self.news_shape_contract,
            "context_features": self.context_features,
            "use_fcg_loss": self.use_fcg_loss,
        }


def build_pit_fcg_lstm_model(
    *,
    numeric_shape: tuple[int, int] = (WINDOW, NUMERIC_FEATURES),
    news_shape: tuple[int, int] = (WINDOW, NEWS_FEATURES),
    context_features: int = CONTEXT_FEATURES,
    use_fcg_loss: bool = True,
) -> PITFCGLSTM:
    model = PITFCGLSTM(
        numeric_shape=numeric_shape,
        news_shape=news_shape,
        context_features=context_features,
        use_fcg_loss=use_fcg_loss,
    )
    model(
        {
            "numeric": tf.zeros((1, *numeric_shape), dtype=tf.float32),
            "news": tf.zeros((1, *news_shape), dtype=tf.float32),
            "context": tf.zeros((1, context_features), dtype=tf.float32),
        }
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    )
    return model


def build_direct_numeric_lstm(
    input_shape: tuple[int, int] = (WINDOW, NUMERIC_FEATURES),
) -> tf.keras.Model:
    shape = _positive_shape(input_shape, name="input_shape")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=shape),
            tf.keras.layers.LSTM(LSTM_UNITS),
            tf.keras.layers.Dense(ANCHOR_DENSE_UNITS, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="direct_numeric_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )
    return model


def build_concat_lstm(
    input_shape: tuple[int, int] = (WINDOW, NUMERIC_FEATURES + NEWS_FEATURES),
) -> tf.keras.Model:
    shape = _positive_shape(input_shape, name="input_shape")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=shape),
            tf.keras.layers.LSTM(LSTM_UNITS),
            tf.keras.layers.Dense(ANCHOR_DENSE_UNITS, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="concat_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )
    return model
