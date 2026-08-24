from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from models.attention_lstm import build_attention_lstm_model
from models.baseline_common import PROJECT_ROOT
from models.convolutional_neural_network import build_cnn_model
from models.full_non_ta_experiments import build_lstm_model
from models.lstm_cnn import build_lstm_cnn_model
from models.lstm_cnn_attention import build_lstm_cnn_attention_model
from models.track_a_final import TRACK_A_MODELS

BACKGROUND_CAP = 100
EXPLANATION_CAP = 128
NSAMPLES = 200
RANDOM_SEED = 42
SMOKE_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "track_c"
    / "shap_protocol_v2"
    / "explainer_smoke.json"
)

MODEL_BUILDERS: dict[str, Callable[[tuple[int, int]], object]] = {
    "lstm": build_lstm_model,
    "cnn": build_cnn_model,
    "lstm_cnn": build_lstm_cnn_model,
    "lstm_attention": build_attention_lstm_model,
    "lstm_cnn_attention": build_lstm_cnn_attention_model,
}


def evenly_spaced_indices(total: int, cap: int) -> np.ndarray:
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError("total must be a positive integer")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError("cap must be a positive integer")
    count = min(total, cap)
    if count == total:
        return np.arange(total, dtype=int)
    return np.linspace(0, total - 1, num=count, dtype=int)


def build_original_change_model(
    scaled_level_model,
    *,
    close_feature_index: int,
    close_scale: float,
    close_offset: float,
    target_scale: float,
    target_offset: float,
):
    """Wrap a scaled next-close model with a differentiable change output."""

    import tensorflow as tf

    if close_feature_index < 0:
        raise ValueError("close_feature_index must be non-negative")
    if close_scale == 0.0 or target_scale == 0.0:
        raise ValueError("scaler scale values must be non-zero")
    input_shape = tuple(scaled_level_model.input_shape[1:])
    inputs = tf.keras.layers.Input(shape=input_shape)
    scaled_level = scaled_level_model(inputs)
    original_level = (
        scaled_level - float(target_offset)
    ) / float(target_scale)
    scaled_close = inputs[
        :,
        -1,
        close_feature_index : close_feature_index + 1,
    ]
    original_close = (
        scaled_close - float(close_offset)
    ) / float(close_scale)
    change = original_level - original_close
    return tf.keras.Model(inputs=inputs, outputs=change)


def normalize_single_output_shap(
    values: np.ndarray,
    expected_input_shape: tuple[int, ...],
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape == (*expected_input_shape, 1):
        result = result[..., 0]
    if result.shape != expected_input_shape:
        raise ValueError(
            f"Unexpected SHAP shape {result.shape}; "
            f"expected {expected_input_shape}"
        )
    if not np.isfinite(result).all():
        raise ValueError("SHAP values contain non-finite values")
    return result


def run_explainer_smoke(
    *,
    output_path: Path = SMOKE_OUTPUT,
    window: int = 3,
    features: int = 6,
    nsamples: int = 20,
) -> dict[str, object]:
    """Verify GradientExplainer compatibility without producing a ranking."""

    import shap
    import tensorflow as tf

    if tuple(MODEL_BUILDERS) != tuple(TRACK_A_MODELS):
        raise ValueError("SHAP smoke model registry differs from Track A")
    rng = np.random.default_rng(RANDOM_SEED)
    background = rng.uniform(0.0, 1.0, size=(4, window, features)).astype(
        np.float32
    )
    explained = rng.uniform(0.0, 1.0, size=(2, window, features)).astype(
        np.float32
    )
    results: list[dict[str, object]] = []
    for model_key, builder in MODEL_BUILDERS.items():
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(RANDOM_SEED)
        level_model = builder((window, features))
        change_model = build_original_change_model(
            level_model,
            close_feature_index=0,
            close_scale=0.25,
            close_offset=-0.5,
            target_scale=0.2,
            target_offset=-0.4,
        )
        explainer = shap.GradientExplainer(change_model, background)
        first = normalize_single_output_shap(
            explainer.shap_values(
                explained,
                nsamples=nsamples,
                rseed=RANDOM_SEED,
            ),
            explained.shape,
        )
        second = normalize_single_output_shap(
            explainer.shap_values(
                explained,
                nsamples=nsamples,
                rseed=RANDOM_SEED,
            ),
            explained.shape,
        )
        repeat_max_abs_diff = float(np.max(np.abs(first - second)))
        results.append(
            {
                "model": model_key,
                "input_shape": list(explained.shape),
                "shap_shape": list(first.shape),
                "finite": True,
                "repeat_max_abs_diff": repeat_max_abs_diff,
                "repeat_exact": bool(np.array_equal(first, second)),
            }
        )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_version": "track-c-shap-point-in-time-v2",
        "ranking_generated": False,
        "explainer": "shap.GradientExplainer",
        "attribution_target": "predicted next-close minus current close in original units",
        "smoke_nsamples": int(nsamples),
        "registered_nsamples": NSAMPLES,
        "random_seed": RANDOM_SEED,
        "results": results,
        "all_models_passed": len(results) == len(TRACK_A_MODELS),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    run_explainer_smoke()
