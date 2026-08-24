from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from models.baseline_common import PROJECT_ROOT
from models.shap_protocol_v2 import (
    MODEL_BUILDERS,
    build_original_change_model,
    evenly_spaced_indices,
)
from models.track_a_final import TRACK_A_MODELS

LIME_PROTOCOL_VERSION = "track-c-dual-xai-lime-v1"
REGISTERED_REGIMES = ("Bull", "Sideway", "Bear")
LIME_SMOKE_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "track_c"
    / "dual_xai_lime_v1"
    / "lime_smoke.json"
)


@dataclass(frozen=True)
class LimeAuditConfig:
    protocol_version: str = LIME_PROTOCOL_VERSION
    n_perturbations: int = 1024
    presence_probability: float = 0.5
    repeat_seeds: tuple[int, ...] = (42, 123, 456, 789, 2025)
    samples_per_regime_fold: int = 6
    top_k: int = 10
    ridge_alpha: float = 1.0
    kernel_width_multiplier: float = 0.75
    minimum_fidelity_r2: float = 0.70
    selection_role: str = "post_selection_audit_only"

    def __post_init__(self) -> None:
        if self.protocol_version != LIME_PROTOCOL_VERSION:
            raise ValueError("Unexpected LIME protocol version")
        if self.n_perturbations < 2:
            raise ValueError("n_perturbations must be at least two")
        if not 0.0 < self.presence_probability < 1.0:
            raise ValueError("presence_probability must be between zero and one")
        if not self.repeat_seeds or len(set(self.repeat_seeds)) != len(
            self.repeat_seeds
        ):
            raise ValueError("repeat_seeds must be non-empty and unique")
        if self.samples_per_regime_fold < 1:
            raise ValueError("samples_per_regime_fold must be positive")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative")
        if self.kernel_width_multiplier <= 0.0:
            raise ValueError("kernel_width_multiplier must be positive")
        if not -1.0 <= self.minimum_fidelity_r2 <= 1.0:
            raise ValueError("minimum_fidelity_r2 must be between -1 and one")
        if self.selection_role != "post_selection_audit_only":
            raise ValueError("LIME cannot be assigned a feature-selection role")


@dataclass(frozen=True)
class LimeNeighborhood:
    masks: np.ndarray
    perturbed: np.ndarray
    weights: np.ndarray
    background_indices: np.ndarray


@dataclass(frozen=True)
class LimeSurrogate:
    coefficients: np.ndarray
    intercept: float
    fidelity_r2: float
    local_prediction: float


@dataclass(frozen=True)
class LimeExplanation:
    surrogate: LimeSurrogate
    black_box_prediction: float
    inference_seconds: float
    runtime_seconds: float


def _finite_array(
    values: np.ndarray,
    *,
    name: str,
    dimensions: int,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def generate_grouped_neighborhood(
    instance: np.ndarray,
    background: np.ndarray,
    *,
    n_perturbations: int,
    seed: int,
    presence_probability: float = 0.5,
    kernel_width_multiplier: float = 0.75,
) -> LimeNeighborhood:
    """Generate a LIME neighborhood grouped by feature across all lags."""

    observed = _finite_array(instance, name="instance", dimensions=2)
    reference = _finite_array(background, name="background", dimensions=3)
    if tuple(reference.shape[1:]) != tuple(observed.shape):
        raise ValueError("background sequence shape must match instance")
    if len(reference) < 1:
        raise ValueError("background must contain at least one sequence")
    if isinstance(n_perturbations, bool) or n_perturbations < 2:
        raise ValueError("n_perturbations must be at least two")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0.0 < presence_probability < 1.0:
        raise ValueError("presence_probability must be between zero and one")
    if kernel_width_multiplier <= 0.0:
        raise ValueError("kernel_width_multiplier must be positive")

    rng = np.random.default_rng(seed)
    n_features = observed.shape[1]
    masks = rng.binomial(
        1,
        presence_probability,
        size=(n_perturbations, n_features),
    ).astype(np.float32)
    masks[0] = 1.0
    background_indices = rng.integers(
        0,
        len(reference),
        size=n_perturbations,
        dtype=np.int64,
    )
    replacements = reference[background_indices]
    perturbed = np.where(
        masks[:, np.newaxis, :].astype(bool),
        observed[np.newaxis, :, :],
        replacements,
    ).astype(np.float32)
    perturbed[0] = observed

    distances = np.linalg.norm(1.0 - masks, axis=1)
    kernel_width = kernel_width_multiplier * np.sqrt(float(n_features))
    weights = np.sqrt(
        np.exp(-np.square(distances) / np.square(kernel_width))
    )
    weights[0] = 1.0
    return LimeNeighborhood(
        masks=masks,
        perturbed=perturbed,
        weights=weights.astype(np.float64),
        background_indices=background_indices,
    )


def fit_lime_surrogate(
    masks: np.ndarray,
    outputs: np.ndarray,
    weights: np.ndarray,
    *,
    ridge_alpha: float,
) -> LimeSurrogate:
    interpretable = _finite_array(masks, name="masks", dimensions=2).astype(
        np.float64
    )
    predictions = np.asarray(outputs, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    if predictions.ndim != 1 or sample_weights.ndim != 1:
        raise ValueError("outputs and weights must be one-dimensional")
    if not (
        len(interpretable) == len(predictions) == len(sample_weights)
    ):
        raise ValueError("LIME neighborhood arrays must have equal rows")
    if not np.isfinite(predictions).all() or not np.isfinite(
        sample_weights
    ).all():
        raise ValueError("LIME outputs and weights must be finite")
    if np.any(sample_weights <= 0.0):
        raise ValueError("LIME weights must be positive")
    if ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be non-negative")

    surrogate = Ridge(alpha=float(ridge_alpha), fit_intercept=True)
    surrogate.fit(
        interpretable,
        predictions,
        sample_weight=sample_weights,
    )
    fidelity = surrogate.score(
        interpretable,
        predictions,
        sample_weight=sample_weights,
    )
    local_prediction = surrogate.predict(
        np.ones((1, interpretable.shape[1]), dtype=np.float64)
    )[0]
    coefficients = np.asarray(surrogate.coef_, dtype=np.float64)
    if not (
        np.isfinite(coefficients).all()
        and np.isfinite(fidelity)
        and np.isfinite(local_prediction)
    ):
        raise ValueError("LIME surrogate produced non-finite output")
    return LimeSurrogate(
        coefficients=coefficients,
        intercept=float(surrogate.intercept_),
        fidelity_r2=float(fidelity),
        local_prediction=float(local_prediction),
    )


def explain_instance_with_lime(
    predict_change: Callable[[np.ndarray], np.ndarray],
    instance: np.ndarray,
    background: np.ndarray,
    *,
    n_perturbations: int,
    seed: int,
    ridge_alpha: float,
    presence_probability: float = 0.5,
    kernel_width_multiplier: float = 0.75,
) -> LimeExplanation:
    started = time.perf_counter()
    neighborhood = generate_grouped_neighborhood(
        instance,
        background,
        n_perturbations=n_perturbations,
        seed=seed,
        presence_probability=presence_probability,
        kernel_width_multiplier=kernel_width_multiplier,
    )
    inference_started = time.perf_counter()
    outputs = np.asarray(
        predict_change(neighborhood.perturbed),
        dtype=np.float64,
    )
    inference_seconds = time.perf_counter() - inference_started
    if outputs.shape != (n_perturbations,):
        raise ValueError(
            "predict_change must return one value per perturbed sequence"
        )
    if not np.isfinite(outputs).all():
        raise ValueError("predict_change returned non-finite values")
    surrogate = fit_lime_surrogate(
        neighborhood.masks,
        outputs,
        neighborhood.weights,
        ridge_alpha=ridge_alpha,
    )
    return LimeExplanation(
        surrogate=surrogate,
        black_box_prediction=float(outputs[0]),
        inference_seconds=float(inference_seconds),
        runtime_seconds=float(time.perf_counter() - started),
    )


def aggregate_local_shap(values: np.ndarray) -> np.ndarray:
    attributions = np.asarray(values, dtype=np.float64)
    if attributions.ndim != 3:
        raise ValueError("local SHAP values must have sample, lag, feature axes")
    if not np.isfinite(attributions).all():
        raise ValueError("local SHAP values must be finite")
    return attributions.sum(axis=1)


def _top_indices(values: np.ndarray, top_k: int) -> set[int]:
    if top_k > len(values):
        raise ValueError("top_k cannot exceed the number of features")
    order = np.argsort(-np.abs(values), kind="stable")
    return {int(index) for index in order[:top_k]}


def compare_local_explanations(
    shap_values: np.ndarray,
    lime_values: np.ndarray,
    *,
    top_k: int,
) -> dict[str, float]:
    shap_array = np.asarray(shap_values, dtype=np.float64)
    lime_array = np.asarray(lime_values, dtype=np.float64)
    if shap_array.ndim != 1 or lime_array.ndim != 1:
        raise ValueError("Local explanations must be one-dimensional")
    if shap_array.shape != lime_array.shape:
        raise ValueError("SHAP and LIME feature counts differ")
    if not np.isfinite(shap_array).all() or not np.isfinite(lime_array).all():
        raise ValueError("Local explanations must be finite")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    correlation = float(
        spearmanr(np.abs(shap_array), np.abs(lime_array)).statistic
    )
    if not np.isfinite(correlation):
        correlation = 0.0
    shap_top = _top_indices(shap_array, top_k)
    lime_top = _top_indices(lime_array, top_k)
    union = shap_top | lime_top
    jaccard = float(len(shap_top & lime_top) / len(union))
    nonzero = (shap_array != 0.0) & (lime_array != 0.0)
    sign_agreement = (
        float(np.mean(np.sign(shap_array[nonzero]) == np.sign(lime_array[nonzero])))
        if np.any(nonzero)
        else float("nan")
    )
    return {
        "spearman_abs": correlation,
        "top_k_jaccard": jaccard,
        "sign_agreement_nonzero": sign_agreement,
        "n_sign_compared": float(np.sum(nonzero)),
    }


def lime_repeat_stability(
    coefficients: np.ndarray,
    *,
    top_k: int,
) -> dict[str, float | int]:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("coefficients must have repeat and feature axes")
    if values.shape[0] < 2:
        raise ValueError("At least two LIME repeats are required")
    if not np.isfinite(values).all():
        raise ValueError("LIME repeat coefficients must be finite")
    if top_k < 1 or top_k > values.shape[1]:
        raise ValueError("top_k is outside the feature range")

    correlations: list[float] = []
    overlaps: list[float] = []
    for left_index, right_index in combinations(range(len(values)), 2):
        left = values[left_index]
        right = values[right_index]
        correlation = float(
            spearmanr(np.abs(left), np.abs(right)).statistic
        )
        correlations.append(correlation if np.isfinite(correlation) else 0.0)
        left_top = _top_indices(left, top_k)
        right_top = _top_indices(right, top_k)
        overlaps.append(
            float(
                len(left_top & right_top)
                / len(left_top | right_top)
            )
        )
    feature_std = np.std(values, axis=0, ddof=1)
    feature_mean_abs = np.mean(np.abs(values), axis=0)
    relative_std = feature_std / np.maximum(feature_mean_abs, 1e-12)
    return {
        "n_repeats": len(values),
        "n_pairs": len(correlations),
        "pairwise_spearman_abs_median": float(np.median(correlations)),
        "pairwise_spearman_abs_min": float(np.min(correlations)),
        "pairwise_top_k_jaccard_median": float(np.median(overlaps)),
        "pairwise_top_k_jaccard_min": float(np.min(overlaps)),
        "mean_feature_coefficient_std": float(np.mean(feature_std)),
        "median_feature_coefficient_std": float(np.median(feature_std)),
        "max_feature_coefficient_std": float(np.max(feature_std)),
        "median_feature_relative_std": float(np.median(relative_std)),
    }


def select_regime_audit_indices(
    regimes: np.ndarray,
    *,
    samples_per_regime: int,
) -> Mapping[str, tuple[int, ...]]:
    values = np.asarray(regimes, dtype=object)
    if values.ndim != 1:
        raise ValueError("regimes must be one-dimensional")
    if samples_per_regime < 1:
        raise ValueError("samples_per_regime must be positive")
    unknown = sorted(set(values).difference(REGISTERED_REGIMES))
    if unknown:
        raise ValueError(f"Unknown regimes: {unknown}")

    selected: dict[str, tuple[int, ...]] = {}
    for regime in REGISTERED_REGIMES:
        eligible = np.flatnonzero(values == regime)
        if len(eligible) < samples_per_regime:
            raise ValueError(
                f"{regime} has {len(eligible)} rows; "
                f"{samples_per_regime} required"
            )
        within_regime = evenly_spaced_indices(
            len(eligible),
            samples_per_regime,
        )
        selected[regime] = tuple(
            int(index) for index in eligible[within_regime]
        )
    return selected


def run_lime_explainer_smoke(
    *,
    output_path: Path = LIME_SMOKE_OUTPUT,
    window: int = 3,
    features: int = 6,
    n_perturbations: int = 64,
) -> dict[str, object]:
    """Check grouped LIME compatibility without producing empirical ranks."""

    import tensorflow as tf

    if tuple(MODEL_BUILDERS) != tuple(TRACK_A_MODELS):
        raise ValueError("LIME smoke model registry differs from Track A")
    rng = np.random.default_rng(42)
    background = rng.uniform(
        0.0,
        1.0,
        size=(4, window, features),
    ).astype(np.float32)
    instance = rng.uniform(
        0.0,
        1.0,
        size=(window, features),
    ).astype(np.float32)
    results: list[dict[str, object]] = []
    for model_key, builder in MODEL_BUILDERS.items():
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        level_model = builder((window, features))
        change_model = build_original_change_model(
            level_model,
            close_feature_index=0,
            close_scale=0.25,
            close_offset=-0.5,
            target_scale=0.2,
            target_offset=-0.4,
        )

        def predict_change(
            sequences: np.ndarray,
            _change_model=change_model,
        ) -> np.ndarray:
            output = _change_model(
                np.asarray(sequences, dtype=np.float32),
                training=False,
            )
            return np.asarray(output.numpy(), dtype=float).reshape(-1)

        first = explain_instance_with_lime(
            predict_change,
            instance,
            background,
            n_perturbations=n_perturbations,
            seed=42,
            ridge_alpha=1.0,
        )
        second = explain_instance_with_lime(
            predict_change,
            instance,
            background,
            n_perturbations=n_perturbations,
            seed=42,
            ridge_alpha=1.0,
        )
        difference = float(
            np.max(
                np.abs(
                    first.surrogate.coefficients
                    - second.surrogate.coefficients
                )
            )
        )
        results.append(
            {
                "model": model_key,
                "input_shape": [window, features],
                "n_perturbations": int(n_perturbations),
                "coefficient_shape": list(
                    first.surrogate.coefficients.shape
                ),
                "finite": True,
                "fidelity_r2": first.surrogate.fidelity_r2,
                "repeat_max_abs_diff": difference,
                "repeat_exact": bool(
                    np.array_equal(
                        first.surrogate.coefficients,
                        second.surrogate.coefficients,
                    )
                ),
            }
        )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_version": LIME_PROTOCOL_VERSION,
        "ranking_generated": False,
        "outer_explanations_generated": False,
        "method": "temporally grouped LIME weighted Ridge surrogate",
        "attribution_target": (
            "predicted next-close minus current close in original units"
        ),
        "results": results,
        "all_models_passed": len(results) == len(TRACK_A_MODELS),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    run_lime_explainer_smoke()
