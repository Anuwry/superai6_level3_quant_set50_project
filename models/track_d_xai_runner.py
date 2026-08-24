from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_common import (
    DATE_COLUMN,
    PROJECT_ROOT,
    TARGET_COLUMN,
    discover_folds,
    load_fold,
    sequence_history_features,
)
from models.convolutional_neural_network import make_test_sequences
from models.shap_protocol_v2 import (
    evenly_spaced_indices,
    normalize_single_output_shap,
)
from models.track_a_final import TRACK_A_MODELS
from models.track_d_models import build_track_d_model, make_direction_sequences
from models.track_d_protocol import (
    TrackDConfig,
    direction_labels,
    verify_freeze_manifest,
)
from models.track_d_xai import (
    aggregate_feature_attributions,
    delete_feature_trajectories,
    faithfulness_percentile,
    randomization_rank_correlation,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_d_q2"
CELL_DIR = OUTPUT_DIR / "cells"
FORWARD_DIR = PROJECT_ROOT / "data-track-d" / "forward_2026"
FORWARD_SCALED_DIR = PROJECT_ROOT / "data-track-d" / "forward_2026_nn"
RANDOMIZATION_BASE_SEED = 20_260_731


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _derived_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little") % (2**31 - 1)


def _forward_folds():
    original_specs = discover_folds(FORWARD_DIR)
    scaled_specs = discover_folds(FORWARD_SCALED_DIR)
    if len(original_specs) != 1 or len(scaled_specs) != 1:
        raise ValueError("XAI requires exactly one forward fold")
    original = load_fold(original_specs[0])
    scaled = load_fold(scaled_specs[0])
    if original.feature_columns != scaled.feature_columns:
        raise ValueError("XAI original and scaled feature pools differ")
    if original.spec.test_year != 2026 or scaled.spec.test_year != 2026:
        raise ValueError("XAI requires the registered 2026 fold")
    return original, scaled


def _training_sequences(original, scaled, *, window: int):
    train_labels, eligible = direction_labels(
        original.train[TARGET_COLUMN].to_numpy(dtype=float),
        original.train["Close_D"].to_numpy(dtype=float),
    )
    sequences, labels = make_direction_sequences(
        scaled.train.loc[:, scaled.feature_columns].to_numpy(dtype=float),
        train_labels,
        window=window,
    )
    endpoint_eligible = eligible[window - 1 :]
    return sequences[endpoint_eligible], labels[endpoint_eligible]


def _test_sequences(scaled, *, window: int) -> np.ndarray:
    return make_test_sequences(
        sequence_history_features(scaled),
        scaled.test.loc[:, scaled.feature_columns].to_numpy(dtype=float),
        window,
    ).astype(np.float32)


def _explain(model, background, explained, *, nsamples: int, seed: int):
    import shap

    started = time.perf_counter()
    explainer = shap.GradientExplainer(model, background)
    raw = explainer.shap_values(
        explained,
        nsamples=nsamples,
        rseed=seed,
    )
    values = normalize_single_output_shap(raw, explained.shape)
    return aggregate_feature_attributions(values), float(
        time.perf_counter() - started
    )


def run_xai_smoke(
    *,
    output_path: Path = OUTPUT_DIR / "xai_smoke.json",
    nsamples: int = 20,
) -> dict[str, object]:
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
    rng = np.random.default_rng(RANDOMIZATION_BASE_SEED)
    background = rng.normal(size=(4, 3, 6)).astype(np.float32)
    explained = rng.normal(size=(2, 3, 6)).astype(np.float32)
    rows: list[dict[str, object]] = []
    for model_key in TrackDConfig().models:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(RANDOMIZATION_BASE_SEED)
        model = build_track_d_model(
            model_key,
            input_shape=(3, 6),
            objective="direct",
        )
        values, runtime_seconds = _explain(
            model,
            background,
            explained,
            nsamples=nsamples,
            seed=RANDOMIZATION_BASE_SEED,
        )
        rows.append(
            {
                "model": model_key,
                "shape": list(values.shape),
                "finite": bool(np.isfinite(values).all()),
                "runtime_seconds": runtime_seconds,
            }
        )
    payload = {
        "created_at_utc": _utc_now(),
        "forward_data_accessed": False,
        "explainer": "shap.GradientExplainer",
        "nsamples": nsamples,
        "results": rows,
        "all_models_passed": all(row["finite"] for row in rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _attribution_frame(
    *,
    model_key: str,
    condition: str,
    dates: np.ndarray,
    selected_indices: np.ndarray,
    features: list[str],
    values: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for local_index, test_index in enumerate(selected_indices):
        for feature_index, feature in enumerate(features):
            rows.append(
                {
                    "model": model_key,
                    "condition": condition,
                    "test_row_index": int(test_index),
                    "date": str(dates[test_index]),
                    "feature_index": feature_index,
                    "feature": feature,
                    "signed_attribution": float(values[local_index, feature_index]),
                    "absolute_attribution": float(
                        abs(values[local_index, feature_index])
                    ),
                }
            )
    return pd.DataFrame(rows)


def _deletion_audit(
    *,
    model,
    model_key: str,
    explained: np.ndarray,
    selected_indices: np.ndarray,
    dates: np.ndarray,
    trained_values: np.ndarray,
    reference: np.ndarray,
    top_k: int,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    started = time.perf_counter()
    baseline_probability = np.asarray(
        model.predict(explained, verbose=0), dtype=float
    ).reshape(-1)
    summary_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []
    for local_index, test_index in enumerate(selected_indices):
        top_features = np.argsort(-np.abs(trained_values[local_index]))[:top_k]
        variants = [
            delete_feature_trajectories(
                explained[local_index],
                reference,
                feature_indices=top_features,
            )
        ]
        rng = np.random.default_rng(
            _derived_seed(RANDOMIZATION_BASE_SEED, model_key, int(test_index))
        )
        random_sets: list[np.ndarray] = []
        for _ in range(repeats):
            feature_set = np.sort(
                rng.choice(
                    explained.shape[2],
                    size=top_k,
                    replace=False,
                )
            )
            random_sets.append(feature_set)
            variants.append(
                delete_feature_trajectories(
                    explained[local_index],
                    reference,
                    feature_indices=feature_set,
                )
            )
        altered = np.asarray(variants, dtype=np.float32)
        probabilities = np.asarray(
            model.predict(altered, verbose=0), dtype=float
        ).reshape(-1)
        effects = np.abs(probabilities - baseline_probability[local_index])
        top_effect = float(effects[0])
        random_effects = effects[1:]
        percentile = faithfulness_percentile(
            top_feature_effect=top_effect,
            random_feature_effects=random_effects,
        )
        summary_rows.append(
            {
                "model": model_key,
                "test_row_index": int(test_index),
                "date": str(dates[test_index]),
                "top_k": top_k,
                "top_feature_indices": ";".join(map(str, top_features)),
                "baseline_probability": float(baseline_probability[local_index]),
                "top_deletion_effect": top_effect,
                "random_repeats": repeats,
                "random_effect_mean": float(random_effects.mean()),
                "random_effect_std": float(random_effects.std(ddof=0)),
                "random_effect_max": float(random_effects.max()),
                "faithfulness_percentile": percentile,
            }
        )
        for repeat, (feature_set, effect) in enumerate(
            zip(random_sets, random_effects, strict=True)
        ):
            random_rows.append(
                {
                    "model": model_key,
                    "test_row_index": int(test_index),
                    "date": str(dates[test_index]),
                    "repeat": repeat,
                    "feature_indices": ";".join(map(str, feature_set)),
                    "deletion_effect": float(effect),
                }
            )
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(random_rows),
        float(time.perf_counter() - started),
    )


def _run_model(model_key: str, original, scaled, config: TrackDConfig):
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
    window = int(config.windows[model_key])
    features = list(scaled.feature_columns)
    x_train, y_train = _training_sequences(original, scaled, window=window)
    test_sequences = _test_sequences(scaled, window=window)
    background_indices = evenly_spaced_indices(len(x_train), config.xai_background)
    selected_indices = evenly_spaced_indices(
        len(test_sequences), config.xai_instances
    )
    background = np.asarray(x_train[background_indices], dtype=np.float32)
    explained = np.asarray(test_sequences[selected_indices], dtype=np.float32)
    reference = np.broadcast_to(
        background.mean(axis=0), explained[0].shape
    ).copy()
    dates = original.test[DATE_COLUMN].dt.strftime("%Y-%m-%d").to_numpy()
    weights_path = (
        CELL_DIR
        / "forward"
        / model_key
        / "direct"
        / "seed_42"
        / "model.weights.h5"
    )
    prediction_path = weights_path.parent / "predictions.csv"
    if not weights_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(f"Direct seed-42 forward cell is incomplete: {model_key}")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    trained_model = build_track_d_model(
        model_key,
        input_shape=(window, len(features)),
        objective="direct",
    )
    trained_model.load_weights(weights_path)
    reproduced = np.asarray(
        trained_model.predict(test_sequences, verbose=0), dtype=float
    ).reshape(-1)
    saved_probability = pd.read_csv(prediction_path)["probability"].to_numpy(
        dtype=float
    )
    reproduction_error = float(np.max(np.abs(reproduced - saved_probability)))
    if reproduction_error > 1e-6:
        raise ValueError(
            f"{model_key} forward weight reproduction error is {reproduction_error}"
        )
    shap_seed = _derived_seed("track-d-shap", model_key)
    trained_values, trained_seconds = _explain(
        trained_model,
        background,
        explained,
        nsamples=config.xai_nsamples,
        seed=shap_seed,
    )

    tf.keras.utils.set_random_seed(_derived_seed(RANDOMIZATION_BASE_SEED, model_key))
    random_model = build_track_d_model(
        model_key,
        input_shape=(window, len(features)),
        objective="direct",
    )
    random_values, random_seconds = _explain(
        random_model,
        background,
        explained,
        nsamples=config.xai_nsamples,
        seed=shap_seed,
    )

    tf.keras.utils.set_random_seed(42)
    permuted_model = build_track_d_model(
        model_key,
        input_shape=(window, len(features)),
        objective="direct",
    )
    permutation_rng = np.random.default_rng(
        _derived_seed(RANDOMIZATION_BASE_SEED, model_key, "labels")
    )
    permuted_labels = permutation_rng.permutation(y_train)
    parameters = TRACK_A_MODELS[model_key].parameters
    permuted_started = time.perf_counter()
    permuted_model.fit(
        x_train,
        permuted_labels,
        epochs=int(parameters["epochs"]),
        batch_size=int(parameters["batch_size"]),
        shuffle=False,
        verbose=0,
    )
    permuted_fit_seconds = float(time.perf_counter() - permuted_started)
    permuted_values, permuted_seconds = _explain(
        permuted_model,
        background,
        explained,
        nsamples=config.xai_nsamples,
        seed=shap_seed,
    )
    attributions = pd.concat(
        [
            _attribution_frame(
                model_key=model_key,
                condition=condition,
                dates=dates,
                selected_indices=selected_indices,
                features=features,
                values=values,
            )
            for condition, values in (
                ("trained", trained_values),
                ("random_initialization", random_values),
                ("permuted_labels", permuted_values),
            )
        ],
        ignore_index=True,
    )
    randomization_rows: list[dict[str, object]] = []
    for local_index, test_index in enumerate(selected_indices):
        for condition, values in (
            ("random_initialization", random_values),
            ("permuted_labels", permuted_values),
        ):
            randomization_rows.append(
                {
                    "model": model_key,
                    "test_row_index": int(test_index),
                    "date": str(dates[test_index]),
                    "comparison": condition,
                    "absolute_rank_spearman": randomization_rank_correlation(
                        trained_values[local_index], values[local_index]
                    ),
                }
            )
    deletion, random_deletions, deletion_seconds = _deletion_audit(
        model=trained_model,
        model_key=model_key,
        explained=explained,
        selected_indices=selected_indices,
        dates=dates,
        trained_values=trained_values,
        reference=reference,
        top_k=config.xai_top_k,
        repeats=config.xai_random_deletion_repeats,
    )
    runtime = {
        "model": model_key,
        "stage": "xai_sanity",
        "instances": len(selected_indices),
        "features": len(features),
        "trained_shap_seconds": trained_seconds,
        "random_initialization_shap_seconds": random_seconds,
        "permuted_label_fit_seconds": permuted_fit_seconds,
        "permuted_label_shap_seconds": permuted_seconds,
        "deletion_seconds": deletion_seconds,
        "weight_reproduction_max_abs_error": reproduction_error,
    }
    return (
        attributions,
        pd.DataFrame(randomization_rows),
        deletion,
        random_deletions,
        runtime,
    )


def run_xai_audit() -> dict[str, object]:
    config = TrackDConfig()
    verify_freeze_manifest(OUTPUT_DIR / "freeze_manifest.json")
    original, scaled = _forward_folds()
    cell_frames: dict[str, list[pd.DataFrame]] = {
        "attributions": [],
        "randomization": [],
        "deletion": [],
        "random_deletions": [],
    }
    runtime_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for model_key in config.models:
        model_dir = CELL_DIR / "xai" / model_key
        paths = {
            "attributions": model_dir / "attributions.csv",
            "randomization": model_dir / "randomization.csv",
            "deletion": model_dir / "deletion.csv",
            "random_deletions": model_dir / "random_deletions.csv",
        }
        runtime_path = model_dir / "runtime.json"
        if all(path.is_file() for path in paths.values()) and runtime_path.is_file():
            results = tuple(pd.read_csv(paths[key]) for key in paths)
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        else:
            results = _run_model(model_key, original, scaled, config)
            *frames, runtime = results
            model_dir.mkdir(parents=True, exist_ok=True)
            for key, frame in zip(paths, frames, strict=True):
                frame.to_csv(paths[key], index=False)
            runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
            results = tuple(frames)
        for key, frame in zip(cell_frames, results, strict=True):
            cell_frames[key].append(frame)
        runtime_rows.append(runtime)
    outputs = {
        "attributions": OUTPUT_DIR / "xai_attributions.csv",
        "randomization": OUTPUT_DIR / "xai_randomization_summary.csv",
        "deletion": OUTPUT_DIR / "xai_deletion_summary.csv",
        "random_deletions": OUTPUT_DIR / "xai_random_deletion_effects.csv",
    }
    for key, frames in cell_frames.items():
        pd.concat(frames, ignore_index=True).to_csv(outputs[key], index=False)
    runtime_path = OUTPUT_DIR / "runtime_summary.csv"
    prior_runtime = pd.read_csv(runtime_path) if runtime_path.is_file() else pd.DataFrame()
    if not prior_runtime.empty and "stage" in prior_runtime:
        prior_runtime = prior_runtime.loc[prior_runtime["stage"] != "xai_sanity"]
    pd.concat([prior_runtime, pd.DataFrame(runtime_rows)], ignore_index=True).to_csv(
        runtime_path, index=False
    )
    metadata = {
        "protocol_version": config.protocol_version,
        "completed_at_utc": _utc_now(),
        "models": list(config.models),
        "instances_per_model": config.xai_instances,
        "background_sequences": config.xai_background,
        "nsamples": config.xai_nsamples,
        "top_k": config.xai_top_k,
        "random_deletion_repeats": config.xai_random_deletion_repeats,
        "deterministic_environment": {
            "TF_DETERMINISTIC_OPS": os.environ.get("TF_DETERMINISTIC_OPS"),
            "TF_ENABLE_ONEDNN_OPTS": os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (OUTPUT_DIR / "xai_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    run_xai_audit()
