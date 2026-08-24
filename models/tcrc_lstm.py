from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

PROTOCOL_ID = "tcrc-lstm-posthoc-development-v1"
MODEL_KEY = "tcrc_lstm"
MODEL_LABEL = "TCRC-LSTM"

VARIANTS = (
    "lstm_anchor",
    "cnn_residual",
    "latent_turn_gate",
    "supervised_turn_gate",
    "tcrc_full",
)


@dataclass(frozen=True)
class TCRCConfig:
    window: int = 20
    lstm_window: int = 5
    lstm_units: int = 16
    cnn_filters: int = 8
    representation_units: int = 8
    correction_units: int = 8
    correction_cap: float = 2.0
    minimum_temperature: float = 0.05
    return_loss_weight: float = 0.75
    direction_loss_weight: float = 1.0
    anchor_loss_weight: float = 0.25
    turn_loss_weight: float = 0.25
    correction_penalty_weight: float = 0.01

    def __post_init__(self) -> None:
        integer_fields = (
            self.window,
            self.lstm_window,
            self.lstm_units,
            self.cnn_filters,
            self.representation_units,
            self.correction_units,
        )
        if any(isinstance(value, bool) or value < 1 for value in integer_fields):
            raise ValueError("TCRC integer configuration values must be positive")
        if self.lstm_window > self.window:
            raise ValueError("lstm_window must not exceed window")
        positive_fields = (
            self.correction_cap,
            self.minimum_temperature,
            self.return_loss_weight,
            self.direction_loss_weight,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive_fields):
            raise ValueError("TCRC scale and primary loss weights must be positive")
        nonnegative_fields = (
            self.anchor_loss_weight,
            self.turn_loss_weight,
            self.correction_penalty_weight,
        )
        if any(
            not np.isfinite(value) or value < 0.0 for value in nonnegative_fields
        ):
            raise ValueError("TCRC auxiliary loss weights must be non-negative")


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size < 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def turning_point_targets(
    current_close: np.ndarray,
    next_close: np.ndarray,
    *,
    previous_close: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return next-day reversal labels and a mask excluding tied moves."""

    current = _finite_vector(current_close, name="current_close")
    future = _finite_vector(next_close, name="next_close")
    if current.shape != future.shape:
        raise ValueError("current_close and next_close must align")
    if previous_close is None or not np.isfinite(previous_close):
        raise ValueError("A finite previous_close is required for causal alignment")
    previous = np.concatenate(
        [np.asarray([float(previous_close)]), current[:-1]]
    )
    current_move = current - previous
    next_move = future - current
    valid = (current_move != 0.0) & (next_move != 0.0)
    turning = ((current_move * next_move) < 0.0).astype(np.float32)
    turning[~valid] = 0.0
    return turning, valid.astype(bool)


def balanced_positive_weight(labels: np.ndarray) -> float:
    values = np.asarray(labels, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("labels must be a finite binary vector")
    unique = set(np.unique(values))
    if not unique.issubset({0.0, 1.0}):
        raise ValueError("labels must be binary")
    positives = int(np.count_nonzero(values == 1.0))
    negatives = int(np.count_nonzero(values == 0.0))
    if positives == 0 or negatives == 0:
        raise ValueError("labels must contain both classes")
    return float(negatives / positives)


def _variant_flags(variant: str) -> tuple[bool, bool, bool, bool]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown TCRC variant: {variant}")
    uses_residual = variant != "lstm_anchor"
    uses_gate = variant in {
        "latent_turn_gate",
        "supervised_turn_gate",
        "tcrc_full",
    }
    supervises_turn = variant in {"supervised_turn_gate", "tcrc_full"}
    uses_attention = variant == "tcrc_full"
    return uses_residual, uses_gate, supervises_turn, uses_attention


def build_tcrc_lstm(
    *,
    input_features: int,
    variant: str,
    return_mean: float,
    return_std: float,
    config: TCRCConfig | None = None,
):
    import torch
    from torch import nn
    from torch.nn import functional as functional

    resolved = config or TCRCConfig()
    if isinstance(input_features, bool) or input_features < 1:
        raise ValueError("input_features must be a positive integer")
    if not np.isfinite(return_mean):
        raise ValueError("return_mean must be finite")
    if not np.isfinite(return_std) or return_std <= 0.0:
        raise ValueError("return_std must be positive")
    flags = _variant_flags(variant)

    class TCRCLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.variant = variant
            self.flags = flags
            self.config = resolved
            self.lstm = nn.LSTM(
                input_size=input_features,
                hidden_size=resolved.lstm_units,
                batch_first=True,
            )
            self.causal_convolutions = nn.ModuleList(
                nn.Conv1d(input_features, resolved.cnn_filters, kernel)
                for kernel in (2, 3, 5)
            )
            cnn_width = resolved.cnn_filters * len(self.causal_convolutions)
            self.cnn_projection = nn.Sequential(
                nn.Linear(cnn_width, resolved.representation_units),
                nn.ReLU(),
            )
            self.anchor_projection = nn.Sequential(
                nn.Linear(resolved.lstm_units, resolved.representation_units),
                nn.ReLU(),
            )
            self.anchor_head = nn.Linear(resolved.representation_units, 1)
            self.turn_head = nn.Linear(resolved.representation_units, 1)
            self.attention_key = nn.Linear(
                resolved.lstm_units,
                resolved.lstm_units,
                bias=False,
            )
            self.attention_query = nn.Linear(
                resolved.representation_units,
                resolved.lstm_units,
                bias=False,
            )
            self.correction_head = nn.Sequential(
                nn.Linear(
                    resolved.lstm_units + resolved.representation_units,
                    resolved.correction_units,
                ),
                nn.Tanh(),
                nn.Linear(resolved.correction_units, 1),
                nn.Tanh(),
            )
            self.log_temperature = nn.Parameter(torch.zeros(()))
            self.register_buffer(
                "return_mean",
                torch.tensor(float(return_mean), dtype=torch.float32),
            )
            self.register_buffer(
                "return_std",
                torch.tensor(float(return_std), dtype=torch.float32),
            )

        def _cnn_state(self, values: torch.Tensor) -> torch.Tensor:
            channels_first = values.transpose(1, 2)
            pooled = []
            for convolution in self.causal_convolutions:
                kernel = int(convolution.kernel_size[0])
                causal = functional.pad(channels_first, (kernel - 1, 0))
                encoded = functional.relu(convolution(causal))
                pooled.append(encoded.mean(dim=2))
            return self.cnn_projection(torch.cat(pooled, dim=1))

        def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
            if values.ndim != 3:
                raise ValueError("TCRC input must have shape (batch, window, features)")
            if values.shape[1:] != (resolved.window, input_features):
                raise ValueError("TCRC input shape violates the registered contract")
            if not torch.isfinite(values).all():
                raise ValueError("TCRC input contains non-finite values")

            lstm_values = values[:, -resolved.lstm_window :, :]
            lstm_states, _ = self.lstm(lstm_values)
            last_state = lstm_states[:, -1, :]
            cnn_state = self._cnn_state(values)
            turn_logit = self.turn_head(cnn_state).reshape(-1)
            turn_probability = torch.sigmoid(turn_logit)

            keys = self.attention_key(lstm_states)
            query = self.attention_query(cnn_state).unsqueeze(2)
            scores = torch.bmm(keys, query).squeeze(2)
            scores = scores / float(resolved.lstm_units) ** 0.5
            learned_attention = torch.softmax(scores, dim=1)
            attended_state = torch.sum(
                lstm_states * learned_attention.unsqueeze(2),
                dim=1,
            )
            _, _, _, uses_attention = self.flags
            if uses_attention:
                q = turn_probability.unsqueeze(1)
                anchor_state = (1.0 - q) * last_state + q * attended_state
                attention_weights = learned_attention
            else:
                anchor_state = last_state
                attention_weights = torch.zeros_like(learned_attention)
                attention_weights[:, -1] = 1.0

            anchor_representation = self.anchor_projection(anchor_state)
            anchor_return = self.anchor_head(anchor_representation).reshape(-1)
            correction_input = torch.cat([anchor_state, cnn_state], dim=1)
            correction = self.correction_head(correction_input).reshape(-1)
            correction = correction * resolved.correction_cap

            uses_residual, uses_gate, _, _ = self.flags
            if not uses_residual:
                gate = torch.zeros_like(turn_probability)
            elif uses_gate:
                gate = turn_probability
            else:
                gate = torch.ones_like(turn_probability)
            standardized_return = anchor_return + gate * correction
            raw_return_percent = (
                standardized_return * self.return_std + self.return_mean
            )
            temperature = (
                functional.softplus(self.log_temperature)
                + resolved.minimum_temperature
            )
            direction_logit = raw_return_percent / temperature
            return {
                "standardized_return": standardized_return,
                "raw_return_percent": raw_return_percent,
                "direction_logit": direction_logit,
                "direction_probability": torch.sigmoid(direction_logit),
                "turn_logit": turn_logit,
                "turn_probability": turn_probability,
                "anchor_standardized_return": anchor_return,
                "correction": correction,
                "gate": gate,
                "attention_weights": attention_weights,
            }

    return TCRCLSTM()


def compute_tcrc_loss(
    outputs: Mapping[str, object],
    targets: Mapping[str, object],
    *,
    variant: str,
    direction_positive_weight: float,
    turn_positive_weight: float,
    config: TCRCConfig | None = None,
) -> dict[str, object]:
    import torch
    from torch.nn import functional as functional

    resolved = config or TCRCConfig()
    _, _, supervises_turn, _ = _variant_flags(variant)
    if direction_positive_weight <= 0.0 or turn_positive_weight <= 0.0:
        raise ValueError("Positive class weights must be greater than zero")
    required_outputs = {
        "standardized_return",
        "direction_logit",
        "turn_logit",
        "anchor_standardized_return",
        "correction",
        "gate",
    }
    required_targets = {
        "standardized_return",
        "direction",
        "turn",
        "turn_valid",
    }
    if not required_outputs.issubset(outputs):
        raise ValueError("TCRC outputs are incomplete")
    if not required_targets.issubset(targets):
        raise ValueError("TCRC targets are incomplete")

    return_target = targets["standardized_return"]
    direction_target = targets["direction"]
    turn_target = targets["turn"]
    turn_valid = targets["turn_valid"].bool()
    return_loss = functional.smooth_l1_loss(
        outputs["standardized_return"],
        return_target,
    )
    direction_weight = torch.as_tensor(
        direction_positive_weight,
        dtype=outputs["direction_logit"].dtype,
        device=outputs["direction_logit"].device,
    )
    direction_loss = functional.binary_cross_entropy_with_logits(
        outputs["direction_logit"],
        direction_target,
        pos_weight=direction_weight,
    )
    if variant == "lstm_anchor":
        anchor_loss = torch.zeros_like(return_loss)
    else:
        anchor_loss = functional.smooth_l1_loss(
            outputs["anchor_standardized_return"],
            return_target,
        )
    if supervises_turn:
        if not torch.any(turn_valid):
            raise ValueError("At least one valid turning-point target is required")
        turn_weight = torch.as_tensor(
            turn_positive_weight,
            dtype=outputs["turn_logit"].dtype,
            device=outputs["turn_logit"].device,
        )
        turn_loss = functional.binary_cross_entropy_with_logits(
            outputs["turn_logit"][turn_valid],
            turn_target[turn_valid],
            pos_weight=turn_weight,
        )
    else:
        turn_loss = torch.zeros_like(return_loss)
    correction_penalty = torch.mean(
        torch.abs(outputs["gate"] * outputs["correction"])
    )
    loss = (
        resolved.return_loss_weight * return_loss
        + resolved.direction_loss_weight * direction_loss
        + resolved.anchor_loss_weight * anchor_loss
        + resolved.turn_loss_weight * turn_loss
        + resolved.correction_penalty_weight * correction_penalty
    )
    return {
        "loss": loss,
        "return_loss": return_loss,
        "direction_loss": direction_loss,
        "anchor_loss": anchor_loss,
        "turn_loss": turn_loss,
        "correction_penalty": correction_penalty,
    }
