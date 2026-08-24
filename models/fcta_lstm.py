from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

PROTOCOL_ID = "fcta-lstm-retrospective-2024-2025-v1"
REGISTERED_VARIANTS = (
    "attention_control",
    "direction_consistency",
    "mask_augmentation",
    "fcta_lstm",
)


@dataclass(frozen=True)
class FCTAConfig:
    window: int = 5
    feature_count: int = 130
    lstm_units: int = 16
    attention_heads: int = 2
    attention_key_dim: int = 8
    dense_units: int = 8
    learning_rate: float = 0.001
    direction_weight: float = 0.10
    mask_augmentation_weight: float = 0.25
    faithfulness_weight: float = 0.10
    direction_temperature: float = 0.01
    variant: str = "fcta_lstm"

    def __post_init__(self) -> None:
        dimensions = (
            self.window,
            self.feature_count,
            self.lstm_units,
            self.attention_heads,
            self.attention_key_dim,
            self.dense_units,
        )
        if any(isinstance(value, bool) or value < 1 for value in dimensions):
            raise ValueError("FCTA dimensions must be positive integers")
        weights = (
            self.learning_rate,
            self.direction_weight,
            self.mask_augmentation_weight,
            self.faithfulness_weight,
        )
        if not np.isfinite(weights).all() or any(value < 0.0 for value in weights):
            raise ValueError("FCTA optimizer and loss weights must be finite and non-negative")
        if not np.isfinite(self.direction_temperature) or self.direction_temperature <= 0:
            raise ValueError("direction_temperature must be finite and positive")
        if self.variant not in REGISTERED_VARIANTS:
            raise ValueError(f"variant must be one of {REGISTERED_VARIANTS}")


def apply_counterfactual_deletions(
    inputs: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return every leave-one-timestep-out sequence and its mask indicator."""
    values = tf.convert_to_tensor(inputs)
    tf.debugging.assert_rank(values, 3)
    batch = tf.shape(values)[0]
    window = tf.shape(values)[1]
    feature_count = tf.shape(values)[2]
    expanded = tf.repeat(values[:, None, :, :], repeats=window, axis=1)
    deletion = tf.eye(window, dtype=values.dtype)[None, :, :, None]
    deletion = tf.broadcast_to(deletion, (batch, window, window, 1))
    deleted = expanded * (1.0 - deletion)
    return (
        tf.reshape(deleted, (batch * window, window, feature_count)),
        tf.reshape(deletion, (batch * window, window, 1)),
    )


def counterfactual_distribution(
    full_prediction: tf.Tensor,
    deleted_prediction: tf.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tf.Tensor:
    """Normalize deletion sensitivity without a tuned temperature."""
    full = tf.convert_to_tensor(full_prediction)
    deleted = tf.convert_to_tensor(deleted_prediction)
    influence = tf.abs(deleted - full)
    scale = tf.reduce_mean(influence, axis=1, keepdims=True) + epsilon
    return tf.nn.softmax(tf.stop_gradient(influence / scale), axis=1)


@tf.keras.utils.register_keras_serializable(package="set50")
class FCTALSTM(tf.keras.Model):
    """LSTM attention whose training attention is aligned to deletion effects."""

    def __init__(self, config: FCTAConfig, **kwargs: Any) -> None:
        super().__init__(name=f"{config.variant}_model", **kwargs)
        self.config = config
        self.lstm = tf.keras.layers.LSTM(
            config.lstm_units,
            return_sequences=True,
            name="lstm",
        )
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=config.attention_heads,
            key_dim=config.attention_key_dim,
            name="temporal_attention",
        )
        self.pool = tf.keras.layers.GlobalAveragePooling1D(name="temporal_pool")
        self.representation = tf.keras.layers.Dense(
            config.dense_units,
            activation="relu",
            name="representation",
        )
        self.output_head = tf.keras.layers.Dense(1, name="scaled_next_close")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.regression_tracker = tf.keras.metrics.Mean(name="regression_loss")
        self.direction_tracker = tf.keras.metrics.Mean(name="direction_loss")
        self.mask_tracker = tf.keras.metrics.Mean(name="mask_loss")
        self.faithfulness_tracker = tf.keras.metrics.Mean(name="faithfulness_loss")

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        return [
            self.loss_tracker,
            self.regression_tracker,
            self.direction_tracker,
            self.mask_tracker,
            self.faithfulness_tracker,
        ]

    def _forward(
        self,
        inputs: tf.Tensor,
        deletion_indicator: tf.Tensor,
        *,
        training: bool,
    ) -> dict[str, tf.Tensor]:
        augmented = tf.concat([inputs, deletion_indicator], axis=-1)
        sequence = self.lstm(augmented, training=training)
        attended, scores = self.attention(
            sequence,
            sequence,
            use_causal_mask=True,
            return_attention_scores=True,
            training=training,
        )
        temporal_importance = tf.reduce_mean(scores, axis=(1, 2))
        temporal_importance /= tf.reduce_sum(
            temporal_importance, axis=1, keepdims=True
        )
        represented = self.representation(self.pool(attended), training=training)
        prediction = self.output_head(represented, training=training)
        return {"prediction": prediction, "attention": temporal_importance}

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        return_attention: bool = False,
    ) -> tf.Tensor | dict[str, tf.Tensor]:
        values = tf.convert_to_tensor(inputs)
        indicator = tf.zeros(
            (tf.shape(values)[0], tf.shape(values)[1], 1), dtype=values.dtype
        )
        outputs = self._forward(values, indicator, training=training)
        return outputs if return_attention else outputs["prediction"]

    def counterfactual_outputs(
        self,
        inputs: tf.Tensor,
        *,
        training: bool = False,
    ) -> tf.Tensor:
        values = tf.convert_to_tensor(inputs)
        deleted, indicator = apply_counterfactual_deletions(values)
        prediction = self._forward(
            deleted,
            indicator,
            training=training,
        )["prediction"]
        return tf.reshape(prediction, (tf.shape(values)[0], self.config.window))

    def _direction_loss(
        self,
        target: tf.Tensor,
        current: tf.Tensor,
        prediction: tf.Tensor,
    ) -> tf.Tensor:
        label = tf.cast(target > current, tf.float32)
        logits = (prediction - current) / self.config.direction_temperature
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)
        )

    def train_step(self, data: Any) -> dict[str, tf.Tensor]:
        inputs, packed_target, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(
            data
        )
        if sample_weight is not None:
            raise ValueError("FCTA-LSTM does not accept sample weights")
        target = tf.cast(packed_target[:, 0:1], tf.float32)
        current = tf.cast(packed_target[:, 1:2], tf.float32)
        with tf.GradientTape() as tape:
            full = self(inputs, training=True, return_attention=True)
            prediction = full["prediction"]
            regression_loss = tf.reduce_mean(tf.square(target - prediction))
            direction_loss = tf.constant(0.0, dtype=regression_loss.dtype)
            mask_loss = tf.constant(0.0, dtype=regression_loss.dtype)
            faithfulness_loss = tf.constant(0.0, dtype=regression_loss.dtype)
            total_loss = regression_loss

            uses_direction = self.config.variant != "attention_control"
            uses_counterfactual = self.config.variant in (
                "mask_augmentation",
                "fcta_lstm",
            )
            uses_faithfulness = self.config.variant == "fcta_lstm"
            if uses_direction:
                direction_loss = self._direction_loss(target, current, prediction)
                total_loss += self.config.direction_weight * direction_loss
            if uses_counterfactual or uses_faithfulness:
                deleted_prediction = self.counterfactual_outputs(
                    inputs, training=True
                )
                if uses_counterfactual:
                    repeated_target = tf.repeat(target, self.config.window, axis=1)
                    repeated_current = tf.repeat(current, self.config.window, axis=1)
                    deleted_regression = tf.reduce_mean(
                        tf.square(repeated_target - deleted_prediction)
                    )
                    deleted_direction = self._direction_loss(
                        repeated_target,
                        repeated_current,
                        deleted_prediction,
                    )
                    mask_loss = deleted_regression + (
                        self.config.direction_weight * deleted_direction
                    )
                    total_loss += self.config.mask_augmentation_weight * mask_loss
                if uses_faithfulness:
                    importance = counterfactual_distribution(
                        prediction, deleted_prediction
                    )
                    attention = tf.clip_by_value(full["attention"], 1e-7, 1.0)
                    importance = tf.clip_by_value(importance, 1e-7, 1.0)
                    faithfulness_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            importance * tf.math.log(importance / attention), axis=1
                        )
                    )
                    total_loss += self.config.faithfulness_weight * faithfulness_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(total_loss)
        self.regression_tracker.update_state(regression_loss)
        self.direction_tracker.update_state(direction_loss)
        self.mask_tracker.update_state(mask_loss)
        self.faithfulness_tracker.update_state(faithfulness_loss)
        return {metric.name: metric.result() for metric in self.metrics}

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "config": self.config.__dict__}


def build_fcta_model(config: FCTAConfig) -> FCTALSTM:
    model = FCTALSTM(config)
    model(
        tf.zeros((1, config.window, config.feature_count), dtype=tf.float32),
        training=False,
    )
    return model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("FCTA-LSTM protocol ID does not match")
    if payload.get("frozen_before_outer_execution") is not True:
        raise ValueError("FCTA-LSTM protocol was not frozen before execution")
    hashes = payload.get("input_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("FCTA-LSTM freeze manifest has no input hashes")
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
