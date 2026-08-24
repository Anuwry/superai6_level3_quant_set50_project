from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

PROTOCOL_ID = "sea-lstm-direct-2024-2025-v1"
MEMORY_MODES = ("dual", "positive_only", "negative_only")
VARIANT_MEMORY_MODES = {
    "sea_lstm": "dual",
    "positive_memory_only": "positive_only",
    "negative_memory_only": "negative_only",
    "positive_only": "positive_only",
    "negative_only": "negative_only",
}


@dataclass(frozen=True)
class SEAConfig:
    window: int = 5
    feature_count: int = 130
    standard_lstm_units: int = 16
    standard_dense_units: int = 8
    sea_units: int = 15
    learning_rate: float = 0.001
    brier_component_weight: float = 0.05
    memory_mode: str = "dual"

    def __post_init__(self) -> None:
        dimensions = (
            self.window,
            self.feature_count,
            self.standard_lstm_units,
            self.standard_dense_units,
            self.sea_units,
        )
        if any(isinstance(value, bool) or value < 1 for value in dimensions):
            raise ValueError("SEA dimensions must be positive integers")
        numeric = (self.learning_rate, self.brier_component_weight)
        if not np.isfinite(numeric).all() or any(value < 0.0 for value in numeric):
            raise ValueError("SEA optimizer and loss values must be finite and non-negative")
        if self.memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {MEMORY_MODES}")


@tf.keras.utils.register_keras_serializable(package="set50")
class SignedEvidenceCell(tf.keras.layers.Layer):
    def __init__(self, units: int, memory_mode: str = "dual", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if isinstance(units, bool) or units < 1:
            raise ValueError("units must be a positive integer")
        if memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {MEMORY_MODES}")
        self.units = int(units)
        self.memory_mode = str(memory_mode)
        self.state_size = (self.units, self.units)
        self.output_size = 2 * self.units

    def build(self, input_shape: tf.TensorShape) -> None:
        input_count = int(input_shape[-1])
        gate_count = 4 * self.units
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_count, gate_count),
            initializer="glorot_uniform",
        )
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=(2 * self.units, gate_count),
            initializer="orthogonal",
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(gate_count,),
            initializer="zeros",
        )
        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        states: tuple[tf.Tensor, tf.Tensor],
    ) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
        previous_up, previous_down = states
        previous = tf.concat([previous_up, previous_down], axis=-1)
        projected = (
            tf.matmul(inputs, self.kernel)
            + tf.matmul(previous, self.recurrent_kernel)
            + self.bias
        )
        forget_raw, input_raw, output_raw, evidence_raw = tf.split(
            projected, 4, axis=-1
        )
        forget_gate = tf.sigmoid(forget_raw)
        input_gate = tf.sigmoid(input_raw)
        output_gate = tf.sigmoid(output_raw)
        evidence = tf.tanh(evidence_raw)
        positive = tf.nn.relu(evidence)
        negative = tf.nn.relu(-evidence)
        if self.memory_mode == "positive_only":
            negative = tf.zeros_like(negative)
        elif self.memory_mode == "negative_only":
            positive = tf.zeros_like(positive)
        up_state = forget_gate * previous_up + input_gate * positive
        down_state = forget_gate * previous_down + input_gate * negative
        output = tf.concat(
            [output_gate * tf.tanh(up_state), output_gate * tf.tanh(down_state)],
            axis=-1,
        )
        return output, (up_state, down_state)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "units": self.units,
            "memory_mode": self.memory_mode,
        }


@tf.keras.utils.register_keras_serializable(package="set50")
class SignedEvidenceHead(tf.keras.layers.Layer):
    def __init__(self, units: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if isinstance(units, bool) or units < 1:
            raise ValueError("units must be a positive integer")
        self.units = int(units)

    def build(self, input_shape: tf.TensorShape) -> None:
        if int(input_shape[-1]) != 2 * self.units:
            raise ValueError("Signed-evidence input width must equal two times units")
        self.up_raw = self.add_weight(
            name="up_raw", shape=(self.units, 1), initializer="zeros"
        )
        self.down_raw = self.add_weight(
            name="down_raw", shape=(self.units, 1), initializer="zeros"
        )
        self.bias = self.add_weight(name="bias", shape=(1,), initializer="zeros")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        up_evidence, down_evidence = tf.split(inputs, 2, axis=-1)
        up_weight = tf.nn.softplus(self.up_raw)
        down_weight = tf.nn.softplus(self.down_raw)
        return (
            tf.matmul(up_evidence, up_weight)
            - tf.matmul(down_evidence, down_weight)
            + self.bias
        )

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "units": self.units}


def _standard_lstm_model(config: SEAConfig) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(
        shape=(config.window, config.feature_count), name="sequence"
    )
    encoded = tf.keras.layers.LSTM(config.standard_lstm_units, name="lstm")(inputs)
    represented = tf.keras.layers.Dense(
        config.standard_dense_units,
        activation="relu",
        name="representation",
    )(encoded)
    probability = tf.keras.layers.Dense(
        1, activation="sigmoid", name="probability"
    )(represented)
    return tf.keras.Model(inputs, probability, name="standard_lstm_control")


def _sea_model(
    config: SEAConfig,
    *,
    memory_mode: str,
    return_evidence: bool,
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(
        shape=(config.window, config.feature_count), name="sequence"
    )
    cell = SignedEvidenceCell(
        config.sea_units,
        memory_mode=memory_mode,
        name="signed_evidence_cell",
    )
    evidence = tf.keras.layers.RNN(cell, name="signed_evidence_recurrence")(inputs)
    logit = SignedEvidenceHead(config.sea_units, name="signed_evidence_head")(evidence)
    probability = tf.keras.layers.Activation("sigmoid", name="probability")(logit)
    outputs: tf.Tensor | dict[str, tf.Tensor] = probability
    if return_evidence:
        outputs = {"probability": probability, "evidence": evidence}
    return tf.keras.Model(inputs, outputs, name=f"sea_lstm_{memory_mode}")


def build_direction_model(
    config: SEAConfig,
    *,
    variant: str,
    return_evidence: bool = False,
) -> tf.keras.Model:
    if variant == "standard_lstm":
        if return_evidence:
            raise ValueError("standard_lstm has no signed evidence output")
        return _standard_lstm_model(config)
    if variant not in VARIANT_MEMORY_MODES:
        raise ValueError(f"Unknown SEA-LSTM variant: {variant}")
    mode = VARIANT_MEMORY_MODES[variant]
    return _sea_model(config, memory_mode=mode, return_evidence=return_evidence)


def compile_direction_model(model: tf.keras.Model, config: SEAConfig) -> None:
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy()

    def direction_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        base = binary_crossentropy(y_true, y_pred)
        brier = tf.reduce_mean(tf.square(y_true - y_pred))
        return base + config.brier_component_weight * brier

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=direction_loss,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("SEA-LSTM protocol ID does not match")
    if payload.get("frozen_before_outer_execution") is not True:
        raise ValueError("SEA-LSTM protocol was not frozen before execution")
    hashes = payload.get("input_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("SEA-LSTM freeze manifest has no input hashes")
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
