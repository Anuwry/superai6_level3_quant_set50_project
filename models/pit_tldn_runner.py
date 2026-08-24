from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT_EARLY = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "numba"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from models.baseline_common import PROJECT_ROOT, package_versions
from models.pit_fcg_development import prepare_inner_fold
from models.pit_fcg_runner import classification_metrics, load_development_frame
from models.pit_tldn import (
    CNN_WINDOW,
    LSTM_WINDOW,
    PROTOCOL_ID,
    TOP_K,
    build_cnn_trend_worker,
    build_debate_features,
    build_debate_leader,
    build_lstm_price_worker,
    expanding_temporal_splits,
    remove_disagreement_signal,
    top_feature_indices,
    worker_output_model,
)
from models.shap_protocol_v2 import evenly_spaced_indices, normalize_single_output_shap
from models.track_c_shap_selection import aggregate_shap_importance

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pit_tldn_inner_development_v1"
FREEZE_FILE = PROJECT_ROOT / "test" / "pit_tldn_freeze_v1.json"
INNER_YEARS = (2020, 2021)
SEEDS = (42, 123, 456, 789, 2025)
WORKER_EPOCHS = 15
LEADER_EPOCHS = 30
BATCH_SIZE = 32
SHAP_BACKGROUND_CAP = 48
SHAP_EXPLANATION_CAP = 64
SHAP_NSAMPLES = 100
SHAP_SEED = 31415
VARIANTS = (
    "cnn_trend_shap",
    "lstm_price_shap",
    "simple_average_shap",
    "leader_no_disagreement_shap",
    "pit_tldn_all_features",
    "pit_tldn",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_inputs() -> dict[str, Any]:
    payload = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("PIT-TLDN freeze protocol id is incorrect")
    hashes = payload.get("input_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("PIT-TLDN freeze has no input hashes")
    checked: dict[str, str] = {}
    root = PROJECT_ROOT.resolve()
    for relative, expected in hashes.items():
        path = (root / str(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("Frozen input path escapes the project") from error
        if not path.is_file():
            raise FileNotFoundError(f"Frozen input is missing: {relative}")
        actual = _sha256(path)
        if actual != str(expected):
            raise ValueError(f"Frozen input hash mismatch: {relative}")
        checked[str(relative)] = actual
    return {
        "passed": True,
        "protocol_id": PROTOCOL_ID,
        "checked_files": len(checked),
        "freeze_file_sha256": _sha256(FREEZE_FILE),
        "input_sha256": checked,
    }


def _derived_seed(seed: int, *parts: object) -> int:
    material = "|".join([PROTOCOL_ID, str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") % (2**31)


def _configure_tensorflow(seed: int):
    import tensorflow as tf

    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    return tf


def _standardized_return_targets(
    current_close: np.ndarray,
    next_close: np.ndarray,
    fit_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    returns = np.asarray(next_close, dtype=float) / np.asarray(current_close, dtype=float) - 1.0
    fitted = returns[np.asarray(fit_indices, dtype=int)]
    mean = float(fitted.mean())
    std = float(fitted.std(ddof=0))
    if not np.isfinite([mean, std]).all() or std <= np.finfo(float).eps:
        raise ValueError("Training next-return target has invalid scale")
    return ((returns - mean) / std).astype(np.float32), {
        "mean": mean,
        "std": std,
        "fit_scope": "worker_training_indices_only",
    }


def _fit_cnn(
    sequences: np.ndarray,
    labels: np.ndarray,
    *,
    train_indices: np.ndarray,
    prediction_indices: np.ndarray,
    feature_indices: np.ndarray,
    seed: int,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    tf = _configure_tensorflow(seed)
    tf.keras.backend.clear_session()
    _configure_tensorflow(seed)
    train_x = sequences[train_indices][:, :, feature_indices]
    prediction_x = sequences[prediction_indices][:, :, feature_indices]
    model = build_cnn_trend_worker((CNN_WINDOW, len(feature_indices)))
    started = time.perf_counter()
    history = model.fit(
        train_x,
        labels[train_indices],
        epochs=WORKER_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probability = model.predict(prediction_x, verbose=0).reshape(-1)
    inference_seconds = time.perf_counter() - started
    return model, probability, {
        "worker": "cnn_trend",
        "fit_seconds": float(fit_seconds),
        "inference_seconds": float(inference_seconds),
        "trainable_parameters": int(model.count_params()),
        "training_sequences": len(train_indices),
        "prediction_sequences": len(prediction_indices),
        "features": len(feature_indices),
        "final_training_loss": float(history.history["loss"][-1]),
    }


def _fit_lstm(
    sequences: np.ndarray,
    labels: np.ndarray,
    standardized_returns: np.ndarray,
    *,
    train_indices: np.ndarray,
    prediction_indices: np.ndarray,
    feature_indices: np.ndarray,
    seed: int,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    tf = _configure_tensorflow(seed)
    tf.keras.backend.clear_session()
    _configure_tensorflow(seed)
    train_x = sequences[train_indices][:, -LSTM_WINDOW:, feature_indices]
    prediction_x = sequences[prediction_indices][:, -LSTM_WINDOW:, feature_indices]
    model = build_lstm_price_worker((LSTM_WINDOW, len(feature_indices)))
    started = time.perf_counter()
    history = model.fit(
        train_x,
        {
            "direction": labels[train_indices],
            "next_return": standardized_returns[train_indices],
        },
        epochs=WORKER_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probability = model(prediction_x, training=False)["direction"].numpy().reshape(-1)
    inference_seconds = time.perf_counter() - started
    return model, probability, {
        "worker": "lstm_price",
        "fit_seconds": float(fit_seconds),
        "inference_seconds": float(inference_seconds),
        "trainable_parameters": int(model.count_params()),
        "training_sequences": len(train_indices),
        "prediction_sequences": len(prediction_indices),
        "features": len(feature_indices),
        "final_training_loss": float(history.history["loss"][-1]),
    }


def _explain_worker(worker, sequences: np.ndarray, *, output_layer: str, seed: int) -> tuple[np.ndarray, float]:
    import shap

    background = sequences[evenly_spaced_indices(len(sequences), SHAP_BACKGROUND_CAP)]
    explained = sequences[evenly_spaced_indices(len(sequences), SHAP_EXPLANATION_CAP)]
    explainer = shap.GradientExplainer(
        worker_output_model(worker, output_layer=output_layer),
        background,
    )
    started = time.perf_counter()
    raw = explainer.shap_values(explained, nsamples=SHAP_NSAMPLES, rseed=seed)
    seconds = time.perf_counter() - started
    values = normalize_single_output_shap(raw, explained.shape)
    return aggregate_shap_importance(values), float(seconds)


def _mask_paths(output_dir: Path, inner_year: int, split_name: str) -> tuple[Path, Path]:
    directory = output_dir / "shap_masks" / f"inner_{inner_year}"
    return directory / f"{split_name}.json", directory / f"{split_name}.csv"


def _ensure_masks(
    *,
    output_dir: Path,
    inner_year: int,
    split_name: str,
    sequences: np.ndarray,
    labels: np.ndarray,
    current_close: np.ndarray,
    next_close: np.ndarray,
    endpoint_dates: pd.DatetimeIndex,
    train_indices: np.ndarray,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    metadata_path, ranking_path = _mask_paths(output_dir, inner_year, split_name)
    if metadata_path.is_file() and ranking_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ranking = pd.read_csv(ranking_path)
        if (
            metadata.get("protocol_id") != PROTOCOL_ID
            or metadata.get("training_last_index") != int(train_indices[-1])
            or metadata.get("feature_names_sha256")
            != hashlib.sha256("|".join(feature_names).encode("utf-8")).hexdigest()
        ):
            raise ValueError(f"Cached SHAP mask contract changed: {split_name}")
        cnn = ranking.loc[ranking["selected_cnn"].astype(bool), "feature_index"].to_numpy(dtype=int)
        lstm = ranking.loc[ranking["selected_lstm"].astype(bool), "feature_index"].to_numpy(dtype=int)
        if len(cnn) != TOP_K or len(lstm) != TOP_K:
            raise ValueError("Cached SHAP mask has the wrong size")
        return cnn, lstm, [], {**metadata, "cache_hit": True}

    tf = _configure_tensorflow(_derived_seed(SHAP_SEED, inner_year, split_name))
    all_features = np.arange(sequences.shape[2], dtype=int)
    standardized_returns, return_scaler = _standardized_return_targets(
        current_close,
        next_close,
        train_indices,
    )
    cnn_model, _, cnn_runtime = _fit_cnn(
        sequences,
        labels,
        train_indices=train_indices,
        prediction_indices=train_indices[-1:],
        feature_indices=all_features,
        seed=_derived_seed(SHAP_SEED, inner_year, split_name, "cnn"),
    )
    cnn_sequences = sequences[train_indices]
    cnn_importance, cnn_shap_seconds = _explain_worker(
        cnn_model,
        cnn_sequences,
        output_layer="direction",
        seed=_derived_seed(SHAP_SEED, inner_year, split_name, "cnn_shap"),
    )
    del cnn_model
    tf.keras.backend.clear_session()
    gc.collect()

    lstm_model, _, lstm_runtime = _fit_lstm(
        sequences,
        labels,
        standardized_returns,
        train_indices=train_indices,
        prediction_indices=train_indices[-1:],
        feature_indices=all_features,
        seed=_derived_seed(SHAP_SEED, inner_year, split_name, "lstm"),
    )
    lstm_sequences = sequences[train_indices][:, -LSTM_WINDOW:, :]
    lstm_importance, lstm_shap_seconds = _explain_worker(
        lstm_model,
        lstm_sequences,
        output_layer="next_return",
        seed=_derived_seed(SHAP_SEED, inner_year, split_name, "lstm_shap"),
    )
    del lstm_model
    tf.keras.backend.clear_session()
    gc.collect()

    cnn_indices = top_feature_indices(cnn_importance)
    lstm_indices = top_feature_indices(lstm_importance)
    ranking = pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_names), dtype=int),
            "feature": feature_names,
            "cnn_shap_importance": cnn_importance,
            "lstm_shap_importance": lstm_importance,
            "selected_cnn": np.isin(np.arange(len(feature_names)), cnn_indices),
            "selected_lstm": np.isin(np.arange(len(feature_names)), lstm_indices),
        }
    )
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "created_at": _utc_now(),
        "inner_year": int(inner_year),
        "split": split_name,
        "training_first_index": int(train_indices[0]),
        "training_last_index": int(train_indices[-1]),
        "training_first_date": str(endpoint_dates[train_indices[0]].date()),
        "training_last_date": str(endpoint_dates[train_indices[-1]].date()),
        "training_sequences": len(train_indices),
        "selection_seed": SHAP_SEED,
        "top_k": TOP_K,
        "cnn_features": [feature_names[index] for index in cnn_indices],
        "lstm_features": [feature_names[index] for index in lstm_indices],
        "cnn_shap_seconds": cnn_shap_seconds,
        "lstm_shap_seconds": lstm_shap_seconds,
        "return_target_scaler": return_scaler,
        "feature_names_sha256": hashlib.sha256("|".join(feature_names).encode("utf-8")).hexdigest(),
        "cache_hit": False,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(ranking_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    runtimes = [
        {**cnn_runtime, "phase": "shap_mask_fit", "split": split_name, "shap_seconds": cnn_shap_seconds},
        {**lstm_runtime, "phase": "shap_mask_fit", "split": split_name, "shap_seconds": lstm_shap_seconds},
    ]
    return cnn_indices, lstm_indices, runtimes, metadata


def _fit_claim_pair(
    *,
    sequences: np.ndarray,
    labels: np.ndarray,
    current_close: np.ndarray,
    next_close: np.ndarray,
    train_indices: np.ndarray,
    prediction_indices: np.ndarray,
    cnn_features: np.ndarray,
    lstm_features: np.ndarray,
    seed: int,
    split_name: str,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    standardized_returns, _ = _standardized_return_targets(
        current_close,
        next_close,
        train_indices,
    )
    cnn_model, cnn_probability, cnn_runtime = _fit_cnn(
        sequences,
        labels,
        train_indices=train_indices,
        prediction_indices=prediction_indices,
        feature_indices=cnn_features,
        seed=_derived_seed(seed, split_name, feature_mode, "cnn"),
    )
    del cnn_model
    gc.collect()
    lstm_model, lstm_probability, lstm_runtime = _fit_lstm(
        sequences,
        labels,
        standardized_returns,
        train_indices=train_indices,
        prediction_indices=prediction_indices,
        feature_indices=lstm_features,
        seed=_derived_seed(seed, split_name, feature_mode, "lstm"),
    )
    del lstm_model
    gc.collect()
    runtimes = [
        {**cnn_runtime, "phase": "worker_fit", "split": split_name, "feature_mode": feature_mode},
        {**lstm_runtime, "phase": "worker_fit", "split": split_name, "feature_mode": feature_mode},
    ]
    return cnn_probability, lstm_probability, runtimes


def _fit_leader(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    *,
    seed: int,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    tf = _configure_tensorflow(_derived_seed(seed, variant, "leader"))
    tf.keras.backend.clear_session()
    _configure_tensorflow(_derived_seed(seed, variant, "leader"))
    model, diagnostics = build_debate_leader(input_features=train_features.shape[1])
    started = time.perf_counter()
    history = model.fit(
        train_features,
        train_labels,
        epochs=LEADER_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    outputs = diagnostics(validation_features, training=False)
    inference_seconds = time.perf_counter() - started
    result = (
        outputs["probability"].numpy().reshape(-1),
        outputs["cnn_weight"].numpy().reshape(-1),
        outputs["correction"].numpy().reshape(-1),
        {
            "phase": "leader_fit",
            "split": "oof_to_inner_validation",
            "feature_mode": variant,
            "worker": "debate_leader",
            "fit_seconds": float(fit_seconds),
            "inference_seconds": float(inference_seconds),
            "trainable_parameters": int(model.count_params()),
            "training_sequences": len(train_labels),
            "prediction_sequences": len(validation_features),
            "features": train_features.shape[1],
            "final_training_loss": float(history.history["loss"][-1]),
            "shap_seconds": 0.0,
        },
    )
    del model, diagnostics
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def cell_directory(output_dir: Path, inner_year: int, seed: int) -> Path:
    return output_dir / "cells" / f"inner_{inner_year}" / f"seed_{seed}"


def _cell_files(directory: Path) -> tuple[Path, ...]:
    return tuple(
        directory / name
        for name in (
            "metrics.csv",
            "predictions.csv",
            "diagnostics.csv",
            "runtime_components.csv",
            "mask_audit.csv",
            "oof_claims.csv",
            "run_metadata.json",
            "integrity_audit.json",
        )
    )


def cell_complete(output_dir: Path, inner_year: int, seed: int) -> bool:
    directory = cell_directory(output_dir, inner_year, seed)
    try:
        if not all(path.is_file() for path in _cell_files(directory)):
            return False
        metrics = pd.read_csv(directory / "metrics.csv")
        audit = json.loads((directory / "integrity_audit.json").read_text(encoding="utf-8"))
        return bool(
            audit.get("passed") is True
            and set(metrics["variant"].astype(str)) == set(VARIANTS)
            and len(metrics) == len(VARIANTS)
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def run_cell(
    *,
    inner_year: int,
    seed: int,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    if inner_year not in INNER_YEARS or seed not in SEEDS:
        raise ValueError("Cell is outside the frozen year/seed grid")
    if cell_complete(output_dir, inner_year, seed) and not force:
        return {"status": "skipped_complete", "inner_year": inner_year, "seed": seed}
    freeze_audit = verify_frozen_inputs()
    frame, feature_names, _ = load_development_frame()
    prepared = prepare_inner_fold(
        frame,
        numeric_features=feature_names,
        validation_year=inner_year,
        seed=seed,
        window=CNN_WINDOW,
    )
    started = time.perf_counter()
    sequences = prepared.train.numeric
    labels = prepared.train.labels
    all_features = np.arange(sequences.shape[2], dtype=int)
    oof_rows: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    splits = expanding_temporal_splits(len(labels))
    for split in splits:
        cnn_mask, lstm_mask, selection_runtime, mask_metadata = _ensure_masks(
            output_dir=output_dir,
            inner_year=inner_year,
            split_name=split.name,
            sequences=sequences,
            labels=labels,
            current_close=prepared.train.current_close,
            next_close=prepared.train.next_close,
            endpoint_dates=prepared.train.endpoint_dates,
            train_indices=split.train_indices,
            feature_names=feature_names,
        )
        runtime_rows.extend(selection_runtime)
        mask_rows.append(
            {
                "split": split.name,
                "training_first_index": int(split.train_indices[0]),
                "training_last_index": int(split.train_indices[-1]),
                "validation_first_index": int(split.validation_indices[0]),
                "validation_last_index": int(split.validation_indices[-1]),
                "purge_rows": int(split.validation_indices[0] - split.train_indices[-1] - 1),
                "cnn_features": "|".join(feature_names[index] for index in cnn_mask),
                "lstm_features": "|".join(feature_names[index] for index in lstm_mask),
                "mask_cache_hit": bool(mask_metadata["cache_hit"]),
            }
        )
        selected_cnn, selected_lstm, selected_runtime = _fit_claim_pair(
            sequences=sequences,
            labels=labels,
            current_close=prepared.train.current_close,
            next_close=prepared.train.next_close,
            train_indices=split.train_indices,
            prediction_indices=split.validation_indices,
            cnn_features=cnn_mask,
            lstm_features=lstm_mask,
            seed=seed,
            split_name=split.name,
            feature_mode="shap",
        )
        all_cnn, all_lstm, all_runtime = _fit_claim_pair(
            sequences=sequences,
            labels=labels,
            current_close=prepared.train.current_close,
            next_close=prepared.train.next_close,
            train_indices=split.train_indices,
            prediction_indices=split.validation_indices,
            cnn_features=all_features,
            lstm_features=all_features,
            seed=seed,
            split_name=split.name,
            feature_mode="all_features",
        )
        runtime_rows.extend([*selected_runtime, *all_runtime])
        index = split.validation_indices
        oof_rows.append(
            pd.DataFrame(
                {
                    "split": split.name,
                    "sequence_index": index,
                    "Date": prepared.train.endpoint_dates[index],
                    "y_true": labels[index].astype(np.int8),
                    "cnn_shap": selected_cnn,
                    "lstm_shap": selected_lstm,
                    "cnn_all": all_cnn,
                    "lstm_all": all_lstm,
                    "prob_bull": prepared.train.context[index, 0],
                    "prob_sideway": prepared.train.context[index, 1],
                    "prob_bear": prepared.train.context[index, 2],
                    "routing_entropy": prepared.train.context[index, 3],
                }
            )
        )
    oof = pd.concat(oof_rows, ignore_index=True).sort_values("sequence_index").reset_index(drop=True)
    oof_context = oof.loc[:, ["prob_bull", "prob_sideway", "prob_bear", "routing_entropy"]].to_numpy(dtype=np.float32)
    selected_oof_features = build_debate_features(oof["cnn_shap"], oof["lstm_shap"], oof_context)
    all_oof_features = build_debate_features(oof["cnn_all"], oof["lstm_all"], oof_context)

    final_indices = np.arange(len(labels), dtype=int)
    cnn_mask, lstm_mask, selection_runtime, mask_metadata = _ensure_masks(
        output_dir=output_dir,
        inner_year=inner_year,
        split_name="final_train",
        sequences=sequences,
        labels=labels,
        current_close=prepared.train.current_close,
        next_close=prepared.train.next_close,
        endpoint_dates=prepared.train.endpoint_dates,
        train_indices=final_indices,
        feature_names=feature_names,
    )
    runtime_rows.extend(selection_runtime)
    mask_rows.append(
        {
            "split": "final_train",
            "training_first_index": 0,
            "training_last_index": int(final_indices[-1]),
            "validation_first_index": np.nan,
            "validation_last_index": np.nan,
            "purge_rows": np.nan,
            "cnn_features": "|".join(feature_names[index] for index in cnn_mask),
            "lstm_features": "|".join(feature_names[index] for index in lstm_mask),
            "mask_cache_hit": bool(mask_metadata["cache_hit"]),
        }
    )
    combined_sequences = np.concatenate([sequences, prepared.validation.numeric], axis=0)
    combined_labels = np.concatenate([labels, prepared.validation.labels])
    combined_close = np.concatenate([prepared.train.current_close, prepared.validation.current_close])
    combined_next = np.concatenate([prepared.train.next_close, prepared.validation.next_close])
    prediction_indices = np.arange(len(sequences), len(combined_sequences), dtype=int)
    selected_cnn, selected_lstm, selected_runtime = _fit_claim_pair(
        sequences=combined_sequences,
        labels=combined_labels,
        current_close=combined_close,
        next_close=combined_next,
        train_indices=final_indices,
        prediction_indices=prediction_indices,
        cnn_features=cnn_mask,
        lstm_features=lstm_mask,
        seed=seed,
        split_name="final_train",
        feature_mode="shap",
    )
    all_cnn, all_lstm, all_runtime = _fit_claim_pair(
        sequences=combined_sequences,
        labels=combined_labels,
        current_close=combined_close,
        next_close=combined_next,
        train_indices=final_indices,
        prediction_indices=prediction_indices,
        cnn_features=all_features,
        lstm_features=all_features,
        seed=seed,
        split_name="final_train",
        feature_mode="all_features",
    )
    runtime_rows.extend([*selected_runtime, *all_runtime])
    validation_context = prepared.validation.context
    selected_validation_features = build_debate_features(selected_cnn, selected_lstm, validation_context)
    all_validation_features = build_debate_features(all_cnn, all_lstm, validation_context)
    y_oof = oof["y_true"].to_numpy(dtype=np.float32)
    no_dis_probability, no_dis_weight, no_dis_correction, no_dis_runtime = _fit_leader(
        remove_disagreement_signal(selected_oof_features),
        y_oof,
        remove_disagreement_signal(selected_validation_features),
        seed=seed,
        variant="leader_no_disagreement_shap",
    )
    all_probability, all_weight, all_correction, all_leader_runtime = _fit_leader(
        all_oof_features,
        y_oof,
        all_validation_features,
        seed=seed,
        variant="pit_tldn_all_features",
    )
    final_probability, final_weight, final_correction, final_leader_runtime = _fit_leader(
        selected_oof_features,
        y_oof,
        selected_validation_features,
        seed=seed,
        variant="pit_tldn",
    )
    runtime_rows.extend([no_dis_runtime, all_leader_runtime, final_leader_runtime])
    probability_by_variant = {
        "cnn_trend_shap": selected_cnn,
        "lstm_price_shap": selected_lstm,
        "simple_average_shap": (selected_cnn + selected_lstm) / 2.0,
        "leader_no_disagreement_shap": no_dis_probability,
        "pit_tldn_all_features": all_probability,
        "pit_tldn": final_probability,
    }
    leader_diagnostics = {
        "leader_no_disagreement_shap": (no_dis_weight, no_dis_correction),
        "pit_tldn_all_features": (all_weight, all_correction),
        "pit_tldn": (final_weight, final_correction),
    }
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    disagreement = np.abs(selected_cnn - selected_lstm)
    for variant in VARIANTS:
        probability = np.asarray(probability_by_variant[variant], dtype=float)
        metric_rows.append(
            {
                "inner_fold": f"inner_{inner_year}",
                "validation_year": inner_year,
                "seed": seed,
                "variant": variant,
                **classification_metrics(prepared.validation.labels, probability),
            }
        )
        weight, correction = leader_diagnostics.get(
            variant,
            (np.full(len(probability), np.nan), np.full(len(probability), np.nan)),
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "inner_fold": f"inner_{inner_year}",
                    "validation_year": inner_year,
                    "seed": seed,
                    "variant": variant,
                    "Date": prepared.validation.endpoint_dates,
                    "y_true": prepared.validation.labels.astype(np.int8),
                    "probability": probability,
                    "cnn_probability": selected_cnn,
                    "lstm_probability": selected_lstm,
                    "worker_disagreement": disagreement,
                    "cnn_weight": weight,
                    "correction": correction,
                }
            )
        )
        diagnostic_rows.append(
            {
                "inner_fold": f"inner_{inner_year}",
                "validation_year": inner_year,
                "seed": seed,
                "variant": variant,
                "mean_worker_disagreement": float(disagreement.mean()),
                "mean_cnn_weight": float(np.nanmean(weight)) if np.isfinite(weight).any() else np.nan,
                "mean_absolute_correction": float(np.nanmean(np.abs(correction))) if np.isfinite(correction).any() else np.nan,
                "oof_claims": len(oof),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    runtime = pd.DataFrame(runtime_rows)
    mask_audit = pd.DataFrame(mask_rows)
    oof["inner_fold"] = f"inner_{inner_year}"
    oof["seed"] = seed
    crossfit_purged = bool(
        all(split.train_indices[-1] + CNN_WINDOW < split.validation_indices[0] for split in splits)
    )
    finite_predictions = bool(np.isfinite(predictions["probability"]).all())
    integrity = {
        "passed": bool(
            freeze_audit["passed"]
            and crossfit_purged
            and finite_predictions
            and predictions.groupby("variant").size().nunique() == 1
            and len(metrics) == len(VARIANTS)
            and oof["sequence_index"].is_unique
            and prepared.train.endpoint_dates.max() < prepared.validation.endpoint_dates.min()
        ),
        "protocol_id": PROTOCOL_ID,
        "freeze_audit": freeze_audit,
        "crossfit_purge_passed": crossfit_purged,
        "oof_indices_unique": bool(oof["sequence_index"].is_unique),
        "train_precedes_validation": bool(prepared.train.endpoint_dates.max() < prepared.validation.endpoint_dates.min()),
        "finite_predictions": finite_predictions,
        "outer_years_accessed": [],
        "metric_rows": len(metrics),
        "prediction_rows": len(predictions),
        "oof_claim_rows": len(oof),
    }
    if not integrity["passed"]:
        raise RuntimeError("PIT-TLDN cell failed its integrity audit")
    directory = cell_directory(output_dir, inner_year, seed)
    directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(directory / "metrics.csv", index=False)
    predictions.to_csv(directory / "predictions.csv", index=False)
    diagnostics.to_csv(directory / "diagnostics.csv", index=False)
    runtime.to_csv(directory / "runtime_components.csv", index=False)
    mask_audit.to_csv(directory / "mask_audit.csv", index=False)
    oof.to_csv(directory / "oof_claims.csv", index=False)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "created_at": _utc_now(),
        "inner_year": inner_year,
        "seed": seed,
        "worker_epochs": WORKER_EPOCHS,
        "leader_epochs": LEADER_EPOCHS,
        "batch_size": BATCH_SIZE,
        "cell_wall_seconds": float(time.perf_counter() - started),
        "outer_years_accessed": [],
        "incremental_api_cost_usd": 0,
    }
    (directory / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (directory / "integrity_audit.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    return {**metadata, "status": "completed"}


def _collect(output_dir: Path, name: str) -> pd.DataFrame:
    paths = sorted((output_dir / "cells").glob(f"inner_*/seed_*/{name}"))
    if not paths:
        raise FileNotFoundError(f"No PIT-TLDN {name} files were found")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _seed_averaged_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["inner_fold", "validation_year", "variant", "Date", "y_true"]
    result = predictions.groupby(keys, sort=False, as_index=False).agg(
        probability=("probability", "mean"),
        worker_disagreement=("worker_disagreement", "mean"),
        cnn_weight=("cnn_weight", "mean"),
        correction=("correction", "mean"),
        seeds=("seed", "nunique"),
    )
    if not result["seeds"].eq(len(SEEDS)).all():
        raise ValueError("Seed averaging does not contain all frozen seeds")
    return result


def _promotion_decision(summary: pd.DataFrame, fold_metrics: pd.DataFrame, predictions: pd.DataFrame, *, integrity_passed: bool) -> dict[str, Any]:
    scores = summary.set_index("variant")["balanced_accuracy_mean"]
    ours = float(scores["pit_tldn"])
    comparison_names = (
        "simple_average_shap",
        "leader_no_disagreement_shap",
        "pit_tldn_all_features",
    )
    mean_beats = {name: bool(ours > float(scores[name])) for name in comparison_names}
    pivot = fold_metrics.pivot(index="validation_year", columns="variant", values="balanced_accuracy")
    fold_delta = pivot["pit_tldn"] - pivot["simple_average_shap"]
    ours_predictions = predictions.loc[predictions["variant"].eq("pit_tldn")]
    weights = ours_predictions.groupby("validation_year")["cnn_weight"].mean()
    disagreement = float(ours_predictions["worker_disagreement"].mean())
    conditions = {
        "integrity_passed": bool(integrity_passed),
        "beats_all_required_mean_ablations": bool(all(mean_beats.values())),
        "simple_average_delta_nonnegative_each_inner_fold": bool((fold_delta >= 0.0).all()),
        "mean_worker_disagreement_at_least_0_02": bool(disagreement >= 0.02),
        "leader_weight_noncollapsed_each_inner_fold": bool(weights.between(0.05, 0.95, inclusive="neither").all()),
    }
    return {
        "passed": bool(all(conditions.values())),
        "conditions": conditions,
        "mean_ablation_comparisons": mean_beats,
        "balanced_accuracy_delta_vs_simple_average_by_year": {str(int(year)): float(value) for year, value in fold_delta.items()},
        "mean_worker_disagreement": disagreement,
        "mean_cnn_weight_by_year": {str(int(year)): float(value) for year, value in weights.items()},
    }


def aggregate_experiment(*, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    incomplete = [(year, seed) for year in INNER_YEARS for seed in SEEDS if not cell_complete(output_dir, year, seed)]
    if incomplete:
        raise ValueError(f"PIT-TLDN has incomplete cells: {incomplete}")
    metrics = _collect(output_dir, "metrics.csv")
    predictions = _collect(output_dir, "predictions.csv")
    diagnostics = _collect(output_dir, "diagnostics.csv")
    runtime = _collect(output_dir, "runtime_components.csv")
    mask_audit = _collect(output_dir, "mask_audit.csv")
    oof = _collect(output_dir, "oof_claims.csv")
    metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    predictions.to_csv(output_dir / "predictions_by_seed.csv", index=False)
    diagnostics.to_csv(output_dir / "diagnostics_by_seed.csv", index=False)
    runtime.to_csv(output_dir / "runtime_components_all_cells.csv", index=False)
    mask_audit.to_csv(output_dir / "mask_audit_all_cells.csv", index=False)
    oof.to_csv(output_dir / "oof_claims_all_cells.csv", index=False)
    averaged = _seed_averaged_predictions(predictions)
    averaged.to_csv(output_dir / "predictions_seed_averaged.csv", index=False)
    fold_rows: list[dict[str, Any]] = []
    for (fold, year, variant), group in averaged.groupby(["inner_fold", "validation_year", "variant"], sort=False):
        fold_rows.append(
            {
                "inner_fold": fold,
                "validation_year": int(year),
                "variant": variant,
                **classification_metrics(group["y_true"], group["probability"]),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output_dir / "inner_fold_metrics.csv", index=False)
    summary = fold_metrics.groupby("variant", sort=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        mcc_mean=("mcc", "mean"),
        binary_crossentropy_mean=("binary_crossentropy", "mean"),
        brier_score_mean=("brier_score", "mean"),
        inner_folds=("inner_fold", "nunique"),
    ).reindex(VARIANTS).reset_index()
    summary.to_csv(output_dir / "inner_summary.csv", index=False)
    runtime_summary = runtime.groupby(["phase", "feature_mode", "worker"], dropna=False, sort=False).agg(
        fits=("fit_seconds", "size"),
        fit_seconds_total=("fit_seconds", "sum"),
        inference_seconds_total=("inference_seconds", "sum"),
        shap_seconds_total=("shap_seconds", "sum"),
        trainable_parameters=("trainable_parameters", "first"),
    ).reset_index()
    runtime_summary.to_csv(output_dir / "runtime_summary.csv", index=False)
    integrity_passed = bool(
        len(metrics) == len(INNER_YEARS) * len(SEEDS) * len(VARIANTS)
        and set(metrics["variant"]) == set(VARIANTS)
        and np.isfinite(averaged["probability"]).all()
        and mask_audit.loc[mask_audit["split"].ne("final_train"), "purge_rows"].ge(CNN_WINDOW).all()
    )
    promotion = _promotion_decision(summary, fold_metrics, averaged, integrity_passed=integrity_passed)
    (output_dir / "promotion_decision.json").write_text(json.dumps(promotion, indent=2), encoding="utf-8")
    integrity = {
        "passed": integrity_passed,
        "protocol_id": PROTOCOL_ID,
        "completed_cells": len(INNER_YEARS) * len(SEEDS),
        "metric_rows": len(metrics),
        "prediction_rows": len(predictions),
        "seed_averaged_prediction_rows": len(averaged),
        "outer_years_accessed": [],
        "promotion_passed": bool(promotion["passed"]),
    }
    (output_dir / "integrity_audit.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "completed_at": _utc_now(),
        "evidence_status": "post_benchmark_pre_result_exploratory_inner_development",
        "validation_years": list(INNER_YEARS),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "outer_years_accessed": [],
        "incremental_api_cost_usd": 0,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(["numpy", "pandas", "scikit-learn", "tensorflow", "shap"]),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "promotion": promotion, "integrity": integrity}


def run_all_cells(
    *,
    output_dir: Path = OUTPUT_DIR,
    years: Iterable[int] = INNER_YEARS,
    seeds: Iterable[int] = SEEDS,
    force: bool = False,
) -> None:
    for year in years:
        for seed in seeds:
            command = [
                sys.executable,
                "-m",
                "models.pit_tldn_runner",
                "cell",
                "--inner-year",
                str(year),
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
            ]
            if force:
                command.append("--force")
            print(f"Running PIT-TLDN inner year {year}, seed {seed}", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen PIT-TLDN inner-development protocol.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    cell = subparsers.add_parser("cell")
    cell.add_argument("--inner-year", type=int, required=True)
    cell.add_argument("--seed", type=int, required=True)
    cell.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    cell.add_argument("--force", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    run.add_argument("--inner-year", type=int, action="append")
    run.add_argument("--seed", type=int, action="append")
    run.add_argument("--force", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    subparsers.add_parser("verify-freeze")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    args = _parser().parse_args(argv)
    if args.command == "cell":
        result = run_cell(inner_year=args.inner_year, seed=args.seed, output_dir=args.output_dir, force=args.force)
        print(json.dumps(result, indent=2))
        return result
    if args.command == "run":
        run_all_cells(
            output_dir=args.output_dir,
            years=INNER_YEARS if args.inner_year is None else args.inner_year,
            seeds=SEEDS if args.seed is None else args.seed,
            force=args.force,
        )
        return None
    if args.command == "aggregate":
        result = aggregate_experiment(output_dir=args.output_dir)
        print(json.dumps(result, indent=2))
        return result
    result = verify_frozen_inputs()
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
