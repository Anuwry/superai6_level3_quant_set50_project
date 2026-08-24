from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from models.attention_lstm import CONFIG as LSTM_ATTENTION_CONFIG
from models.attention_lstm import predict_fold as predict_lstm_attention_fold
from models.baseline_common import (
    PROJECT_ROOT,
    RANDOM_SEED,
    FoldData,
    discover_folds,
    evaluate_predictions,
    load_fold,
    package_versions,
    predictions_frame,
)
from models.convolutional_neural_network import CONFIG as CNN_CONFIG
from models.convolutional_neural_network import predict_fold as predict_cnn_fold
from models.full_non_ta_experiments import (
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    predict_lstm_fold,
)
from models.full_ta_feature_pool import (
    FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR,
    FULL_TA_POINT_IN_TIME_NN_DATA_FOLDS_DIR,
)
from models.lstm_cnn import CONFIG as LSTM_CNN_CONFIG
from models.lstm_cnn import predict_fold as predict_lstm_cnn_fold
from models.lstm_cnn_attention import BENCHMARK_SEEDS
from models.lstm_cnn_attention import CONFIG as LSTM_CNN_ATTENTION_CONFIG
from models.lstm_cnn_attention import predict_fold as predict_lstm_cnn_attention_fold
from models.neural_network_folds import inverse_scaled_target
from models.point_in_time_data import CONTEXT_FILE_NAME
from models.track_a_analysis import (
    build_compact_paper_table,
    build_paired_deltas,
    build_paper_table,
    build_runtime_summary,
    exact_sign_flip_pvalue,
    summarize_final_results,
)
from models.track_a_data import (
    FULL_TA_POINT_IN_TIME_SELECTION_DIR,
    FULL_TA_POINT_IN_TIME_SELECTION_NN_DIR,
    FULL_TA_VMD_POINT_IN_TIME_SELECTION_DIR,
    FULL_TA_VMD_POINT_IN_TIME_SELECTION_NN_DIR,
    prepare_track_a_selection_data,
)
from models.vmd_feature_pool import (
    FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
    FULL_TA_VMD_POINT_IN_TIME_NN_DATA_FOLDS_DIR,
    VMDConfig,
)

__all__ = ["exact_sign_flip_pvalue"]

SEQUENCE_WINDOWS = (1, 3, 5, 10, 20)
SELECTION_YEARS = (2018, 2019, 2020, 2021)
FIRST_OUTER_TEST_YEAR = 2022
FINAL_SEEDS = BENCHMARK_SEEDS
SELECTION_SEED = RANDOM_SEED
METRIC_COLUMNS = (
    "rmse",
    "mae",
    "mape",
    "r2",
    "direction_accuracy",
    "balanced_accuracy",
    "mcc",
    "direction_coverage",
)
TRACK_A_PROTOCOL_VERSION = "track_a_point_in_time_v2"
TRACK_A_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "track_a_final_point_in_time_v2"
)
SELECTION_OUTPUT_DIR = TRACK_A_OUTPUT_DIR / "window_selection"
OUTER_TEST_OUTPUT_DIR = TRACK_A_OUTPUT_DIR / "outer_test"
SELECTION_METRICS_FILE = TRACK_A_OUTPUT_DIR / "selection_metrics.csv"
SELECTION_SUMMARY_FILE = TRACK_A_OUTPUT_DIR / "selection_summary.csv"
LOCKED_WINDOWS_FILE = TRACK_A_OUTPUT_DIR / "locked_windows.csv"
FINAL_METRICS_FILE = TRACK_A_OUTPUT_DIR / "final_metrics_by_seed_fold.csv"
FINAL_PERFORMANCE_FILE = TRACK_A_OUTPUT_DIR / "final_performance.csv"
PAIRED_DELTAS_FILE = TRACK_A_OUTPUT_DIR / "paired_deltas_by_seed_fold.csv"
PAIRED_SUMMARY_FILE = TRACK_A_OUTPUT_DIR / "paired_ablation_summary.csv"
PAPER_TABLE_FILE = TRACK_A_OUTPUT_DIR / "paper_track_a_table.csv"
PAPER_COMPACT_TABLE_FILE = TRACK_A_OUTPUT_DIR / "paper_track_a_compact.csv"
RUNTIME_SUMMARY_FILE = TRACK_A_OUTPUT_DIR / "runtime_summary.csv"
RUNTIME_ENVIRONMENT_FILE = TRACK_A_OUTPUT_DIR / "runtime_environment.json"
INPUT_MANIFEST_FILE = TRACK_A_OUTPUT_DIR / "input_manifest.json"


@dataclass(frozen=True)
class TrackAModel:
    label: str
    parameters: dict[str, object]


@dataclass(frozen=True)
class FeatureSet:
    label: str
    selection_original_dir: Path
    selection_scaled_dir: Path
    outer_original_dir: Path
    outer_scaled_dir: Path


def _parameters(config: dict[str, object]) -> dict[str, object]:
    values = dict(config.get("model_parameters", {}))
    values.pop("sequence_length", None)
    values.pop("scaled_data_dir", None)
    return values


TRACK_A_MODELS: dict[str, TrackAModel] = {
    "lstm": TrackAModel(
        "LSTM",
        {
            "epochs": LSTM_EPOCHS,
            "batch_size": LSTM_BATCH_SIZE,
            "lstm_units": 16,
            "dense_units": 8,
            "optimizer": "adam",
            "loss": "mse",
            "shuffle": False,
        },
    ),
    "cnn": TrackAModel("CNN", _parameters(CNN_CONFIG)),
    "lstm_cnn": TrackAModel("LSTM-CNN", _parameters(LSTM_CNN_CONFIG)),
    "lstm_attention": TrackAModel(
        "LSTM-Attention",
        _parameters(LSTM_ATTENTION_CONFIG),
    ),
    "lstm_cnn_attention": TrackAModel(
        "LSTM-CNN-Attention",
        _parameters(LSTM_CNN_ATTENTION_CONFIG),
    ),
}

TRACK_A_FEATURE_SETS: dict[str, FeatureSet] = {
    "full_ta": FeatureSet(
        "Full TA",
        FULL_TA_POINT_IN_TIME_SELECTION_DIR,
        FULL_TA_POINT_IN_TIME_SELECTION_NN_DIR,
        FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR,
        FULL_TA_POINT_IN_TIME_NN_DATA_FOLDS_DIR,
    ),
    "full_ta_vmd": FeatureSet(
        "Full TA + causal rolling VMD",
        FULL_TA_VMD_POINT_IN_TIME_SELECTION_DIR,
        FULL_TA_VMD_POINT_IN_TIME_SELECTION_NN_DIR,
        FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
        FULL_TA_VMD_POINT_IN_TIME_NN_DATA_FOLDS_DIR,
    ),
}


def validate_model_keys(model_keys: Iterable[str]) -> tuple[str, ...]:
    keys = tuple(model_keys)
    unknown = sorted(set(keys).difference(TRACK_A_MODELS))
    if unknown:
        raise ValueError(f"Unknown Track A models: {unknown}")
    if not keys:
        raise ValueError("At least one Track A model is required")
    return keys


def validate_sequence_window(sequence_window: int) -> int:
    if (
        isinstance(sequence_window, bool)
        or not isinstance(sequence_window, int)
        or sequence_window not in SEQUENCE_WINDOWS
    ):
        raise ValueError(f"sequence_window must be one of {SEQUENCE_WINDOWS}")
    return sequence_window


def predict_model(
    model_key: str,
    fold: FoldData,
    *,
    sequence_window: int,
    seed: int,
) -> np.ndarray:
    window = validate_sequence_window(sequence_window)
    if model_key == "lstm":
        prediction = predict_lstm_fold(
            fold,
            window,
            random_seed=seed,
        )
    elif model_key == "cnn":
        prediction = predict_cnn_fold(
            fold,
            window,
            random_seed=seed,
        )
    elif model_key == "lstm_cnn":
        prediction = predict_lstm_cnn_fold(
            fold,
            window,
            random_seed=seed,
        )
    elif model_key == "lstm_attention":
        prediction = predict_lstm_attention_fold(
            fold,
            window,
            random_seed=seed,
        )
    elif model_key == "lstm_cnn_attention":
        prediction = predict_lstm_cnn_attention_fold(
            fold,
            sequence_length=window,
            random_seed=seed,
        )
    else:
        raise ValueError(f"Unknown Track A model: {model_key}")
    return np.asarray(prediction, dtype=float)


def _set_tensorflow_seed(seed: int) -> None:
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)


def prepare_runtime_environment() -> dict[str, object]:
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
    _ = tf.constant(0.0).numpy()
    devices = [
        {"name": item.name, "device_type": item.device_type}
        for item in tf.config.list_physical_devices()
    ]
    payload = {
        "protocol_version": TRACK_A_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "tensorflow": tf.__version__,
        "physical_devices": devices,
        "gpu_available": any(item["device_type"] == "GPU" for item in devices),
        "runtime_clock": "time.perf_counter",
        "runtime_scope": "model build + training + evaluation-period inference",
        "framework_startup_excluded": True,
        "deterministic_operations_enabled": True,
        "selection_seed": SELECTION_SEED,
        "final_seeds": list(FINAL_SEEDS),
    }
    TRACK_A_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RUNTIME_ENVIRONMENT_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return payload


def _configuration_dir(
    stage: str,
    model_key: str,
    feature_set_key: str,
    sequence_window: int,
    seed: int,
) -> Path:
    root = (
        SELECTION_OUTPUT_DIR if stage == "window_selection" else OUTER_TEST_OUTPUT_DIR
    )
    return (
        root
        / model_key
        / feature_set_key
        / f"window_{sequence_window}"
        / f"seed_{seed}"
    )


def _data_dirs(feature_set: FeatureSet, stage: str) -> tuple[Path, Path]:
    if stage == "window_selection":
        return (
            feature_set.selection_scaled_dir,
            feature_set.selection_original_dir,
        )
    if stage == "outer_test":
        return feature_set.outer_scaled_dir, feature_set.outer_original_dir
    raise ValueError(f"Unknown Track A stage: {stage}")


def _load_scaler_metadata(scaled_spec) -> dict[str, object]:
    path = scaled_spec.train_path.parent / "minmax_scaler.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_configuration(
    output_dir: Path,
    metrics: list[dict[str, object]],
    predictions: dict[str, pd.DataFrame],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(
        output_dir / "metrics_by_fold.csv",
        index=False,
    )
    for fold_name, prediction in predictions.items():
        prediction.to_csv(
            output_dir / f"predictions_{fold_name}.csv",
            index=False,
        )
    with (output_dir / "run_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)


def run_configuration(
    stage: str,
    model_key: str,
    feature_set_key: str,
    sequence_window: int,
    seed: int,
    *,
    force: bool = False,
) -> pd.DataFrame:
    if model_key not in TRACK_A_MODELS:
        raise ValueError(f"Unknown Track A model: {model_key}")
    if feature_set_key not in TRACK_A_FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set_key}")
    window = validate_sequence_window(sequence_window)
    output_dir = _configuration_dir(
        stage,
        model_key,
        feature_set_key,
        window,
        seed,
    )
    metrics_path = output_dir / "metrics_by_fold.csv"
    if metrics_path.exists() and not force:
        print(
            "Reusing "
            f"{stage}: {model_key}, {feature_set_key}, "
            f"window={window}, seed={seed}"
        )
        return pd.read_csv(metrics_path)

    model = TRACK_A_MODELS[model_key]
    feature_set = TRACK_A_FEATURE_SETS[feature_set_key]
    scaled_dir, original_dir = _data_dirs(feature_set, stage)
    scaled_specs = discover_folds(scaled_dir)
    original_specs = discover_folds(original_dir)
    fold_metrics: list[dict[str, object]] = []
    fold_predictions: dict[str, pd.DataFrame] = {}
    experiment_started = time.perf_counter()

    for scaled_spec, original_spec in zip(
        scaled_specs,
        original_specs,
        strict=True,
    ):
        if scaled_spec.fold != original_spec.fold:
            raise ValueError("Scaled and original fold names do not match")
        scaled_fold = load_fold(scaled_spec)
        original_fold = load_fold(original_spec)
        scaler_metadata = _load_scaler_metadata(scaled_spec)
        _set_tensorflow_seed(seed)
        model_started = time.perf_counter()
        scaled_prediction = predict_model(
            model_key,
            scaled_fold,
            sequence_window=window,
            seed=seed,
        )
        runtime_seconds = time.perf_counter() - model_started
        if scaled_prediction.shape != (len(scaled_fold.test),):
            raise ValueError(
                f"{model_key} produced shape {scaled_prediction.shape}; "
                f"expected {(len(scaled_fold.test),)}"
            )
        prediction = inverse_scaled_target(
            scaled_prediction,
            scaler_metadata,
        )
        metrics = {
            "stage": stage,
            "model": model_key,
            "feature_set": feature_set_key,
            "sequence_window": window,
            "seed": seed,
            **evaluate_predictions(original_fold, prediction),
            "runtime_seconds": runtime_seconds,
        }
        fold_metrics.append(metrics)
        fold_predictions[scaled_spec.fold] = predictions_frame(
            original_fold,
            prediction,
        )

    fit_predict_seconds = float(
        sum(float(row["runtime_seconds"]) for row in fold_metrics)
    )
    metadata = {
        "protocol_version": TRACK_A_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "model_key": model_key,
        "model": model.label,
        "feature_set": feature_set_key,
        "feature_set_label": feature_set.label,
        "sequence_window": window,
        "seed": seed,
        "scaled_data_dir": str(scaled_dir),
        "original_units_data_dir": str(original_dir),
        "selection_years": list(SELECTION_YEARS),
        "outer_test_years": [2022, 2023, 2024, 2025],
        "test_data_used_for_window_selection": False,
        "selection_rule": (
            "highest mean validation balanced accuracy across both feature "
            "sets and all pretest folds; higher direction accuracy, lower "
            "RMSE, then shorter window"
        ),
        "direction_metric_contract": {
            "task": "binary Up versus Down",
            "actual_ties": "excluded from binary metrics",
            "predicted_no_change": "abstention",
            "coverage_denominator": "non-tied actual observations",
        },
        "sequence_boundary_context": {
            "file": CONTEXT_FILE_NAME,
            "role": (
                "feature-only history between supervised train and "
                "evaluation; excluded from model.fit"
            ),
            "scaling": "transformed with train-fitted scaler without refit",
        },
        "model_parameters": model.parameters,
        "vmd": {
            "enabled": feature_set_key == "full_ta_vmd",
            "window": VMDConfig().window_size,
            "causal": True,
        },
        "runtime": {
            "clock": "time.perf_counter",
            "scope": "model build + training + evaluation-period inference",
            "fit_predict_seconds": fit_predict_seconds,
            "total_configuration_seconds": (time.perf_counter() - experiment_started),
        },
        "packages": package_versions(["numpy", "pandas", "scikit-learn", "tensorflow"]),
    }
    _save_configuration(
        output_dir,
        fold_metrics,
        fold_predictions,
        metadata,
    )
    return pd.DataFrame(fold_metrics)


def select_locked_windows(
    selection_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "stage",
        "model",
        "feature_set",
        "sequence_window",
        "seed",
        "fold",
        "test_year",
        *METRIC_COLUMNS,
        "runtime_seconds",
    }
    missing = sorted(required.difference(selection_metrics.columns))
    if missing:
        raise ValueError(f"Selection metrics are missing columns: {missing}")
    if not selection_metrics["stage"].eq("window_selection").all():
        raise ValueError("Selection metrics contain a non-selection stage")
    if (selection_metrics["test_year"] >= FIRST_OUTER_TEST_YEAR).any():
        raise ValueError("Window selection contains outer test data")
    if not selection_metrics["seed"].eq(SELECTION_SEED).all():
        raise ValueError("Window selection must use the fixed selection seed")

    summary = (
        selection_metrics.groupby(
            ["model", "sequence_window"],
            sort=False,
        )
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            direction_accuracy_mean=("direction_accuracy", "mean"),
            direction_accuracy_std=("direction_accuracy", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
            selection_feature_sets=("feature_set", "nunique"),
            selection_years=("test_year", "nunique"),
            selection_runs=("fold", "size"),
        )
        .reset_index()
    )
    expected_features = len(TRACK_A_FEATURE_SETS)
    expected_years = len(SELECTION_YEARS)
    if not summary["selection_feature_sets"].eq(expected_features).all():
        raise ValueError("Every window requires both Track A feature sets")
    if not summary["selection_years"].eq(expected_years).all():
        raise ValueError("Every window requires all pretest selection years")

    locked_rows: list[dict[str, object]] = []
    for model_key, model_summary in summary.groupby("model", sort=False):
        selected = model_summary.sort_values(
            [
                "balanced_accuracy_mean",
                "direction_accuracy_mean",
                "rmse_mean",
                "sequence_window",
            ],
            ascending=[False, False, True, True],
        ).iloc[0]
        locked_rows.append(
            {
                "model": model_key,
                "selected_sequence_window": int(selected["sequence_window"]),
                "selection_balanced_accuracy_mean": float(
                    selected["balanced_accuracy_mean"]
                ),
                "selection_balanced_accuracy_std": float(
                    selected["balanced_accuracy_std"]
                ),
                "selection_direction_accuracy_mean": float(
                    selected["direction_accuracy_mean"]
                ),
                "selection_direction_accuracy_std": float(
                    selected["direction_accuracy_std"]
                ),
                "selection_rmse_mean": float(selected["rmse_mean"]),
                "selection_rmse_std": float(selected["rmse_std"]),
                "selection_feature_sets": int(selected["selection_feature_sets"]),
                "selection_years": int(selected["selection_years"]),
                "selection_seed": SELECTION_SEED,
            }
        )
    return pd.DataFrame(locked_rows), summary


def collect_selection_metrics(
    model_keys: Iterable[str] = TRACK_A_MODELS,
    *,
    force: bool = False,
) -> pd.DataFrame:
    frames = [
        run_configuration(
            "window_selection",
            model_key,
            feature_set_key,
            window,
            SELECTION_SEED,
            force=force,
        )
        for model_key in validate_model_keys(model_keys)
        for feature_set_key in TRACK_A_FEATURE_SETS
        for window in SEQUENCE_WINDOWS
    ]
    return pd.concat(frames, ignore_index=True)


def run_window_selection(
    model_keys: Iterable[str] = TRACK_A_MODELS,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prepare_track_a_selection_data(
        selection_years=SELECTION_YEARS,
        first_test_year=FIRST_OUTER_TEST_YEAR,
    )
    metrics = collect_selection_metrics(model_keys, force=force)
    locked, summary = select_locked_windows(metrics)
    TRACK_A_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(SELECTION_METRICS_FILE, index=False)
    summary.to_csv(SELECTION_SUMMARY_FILE, index=False)
    locked.to_csv(LOCKED_WINDOWS_FILE, index=False)
    return metrics, summary, locked


def load_locked_windows() -> pd.DataFrame:
    if not LOCKED_WINDOWS_FILE.exists():
        raise FileNotFoundError(
            "Locked windows are missing; run Track A window selection first"
        )
    locked = pd.read_csv(LOCKED_WINDOWS_FILE)
    expected = set(TRACK_A_MODELS)
    if set(locked["model"]) != expected:
        raise ValueError("Locked windows do not cover all Track A models")
    return locked


def collect_final_metrics(
    locked_windows: pd.DataFrame,
    *,
    force: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in locked_windows.itertuples(index=False):
        model_key = str(row.model)
        window = int(row.selected_sequence_window)
        for feature_set_key in TRACK_A_FEATURE_SETS:
            for seed in FINAL_SEEDS:
                frames.append(
                    run_configuration(
                        "outer_test",
                        model_key,
                        feature_set_key,
                        window,
                        seed,
                        force=force,
                    )
                )
    return pd.concat(frames, ignore_index=True)


def run_final_outer_test(
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    locked = load_locked_windows()
    metrics = collect_final_metrics(locked, force=force)
    performance, paired_summary = summarize_final_results(metrics)
    paired = build_paired_deltas(metrics)
    paper_table = build_paper_table(performance, paired_summary)
    compact_table = build_compact_paper_table(
        performance,
        paired_summary,
    )
    if not SELECTION_METRICS_FILE.exists():
        raise FileNotFoundError(
            "Selection metrics are required for the runtime summary"
        )
    selection_metrics = pd.read_csv(SELECTION_METRICS_FILE)
    runtime_summary = build_runtime_summary(selection_metrics, metrics)
    TRACK_A_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(FINAL_METRICS_FILE, index=False)
    performance.to_csv(FINAL_PERFORMANCE_FILE, index=False)
    paired.to_csv(PAIRED_DELTAS_FILE, index=False)
    paired_summary.to_csv(PAIRED_SUMMARY_FILE, index=False)
    paper_table.to_csv(PAPER_TABLE_FILE, index=False)
    compact_table.to_csv(PAPER_COMPACT_TABLE_FILE, index=False)
    runtime_summary.to_csv(RUNTIME_SUMMARY_FILE, index=False)
    return metrics, performance, paired_summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_input_manifest() -> dict[str, object]:
    data_paths = (
        {
            spec.train_path
            for feature_set in TRACK_A_FEATURE_SETS.values()
            for data_dir in (
                feature_set.selection_original_dir,
                feature_set.outer_original_dir,
            )
            for spec in discover_folds(data_dir)
        }
        | {
            spec.test_path
            for feature_set in TRACK_A_FEATURE_SETS.values()
            for data_dir in (
                feature_set.selection_original_dir,
                feature_set.outer_original_dir,
            )
            for spec in discover_folds(data_dir)
        }
    )
    context_paths = {
        spec.train_path.parent / CONTEXT_FILE_NAME
        for feature_set in TRACK_A_FEATURE_SETS.values()
        for data_dir in (
            feature_set.selection_original_dir,
            feature_set.outer_original_dir,
        )
        for spec in discover_folds(data_dir)
        if (spec.train_path.parent / CONTEXT_FILE_NAME).is_file()
    }
    paths = sorted(
        data_paths | context_paths,
        key=str,
    )
    payload = {
        "protocol_version": TRACK_A_PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_years": list(SELECTION_YEARS),
        "outer_test_years": [2022, 2023, 2024, 2025],
        "selection_uses_outer_test": False,
        "boundary_context_contract": (
            "context files are excluded from supervised fitting and used "
            "only to preserve consecutive evaluation sequences"
        ),
        "files": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
    }
    TRACK_A_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with INPUT_MANIFEST_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return payload


def main() -> object:
    parser = argparse.ArgumentParser(
        description="Complete leakage-free Track A VMD validation."
    )
    parser.add_argument(
        "stage",
        choices=["data", "selection", "outer-test", "all"],
        default="all",
        nargs="?",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prepare_runtime_environment()
    if args.stage == "data":
        result = prepare_track_a_selection_data(
            selection_years=SELECTION_YEARS,
            first_test_year=FIRST_OUTER_TEST_YEAR,
        )
        save_input_manifest()
        return result
    if args.stage == "selection":
        result = run_window_selection(force=args.force)
        save_input_manifest()
        return result
    if args.stage == "outer-test":
        result = run_final_outer_test(force=args.force)
        save_input_manifest()
        return result

    run_window_selection(force=args.force)
    result = run_final_outer_test(force=args.force)
    save_input_manifest()
    return result


if __name__ == "__main__":
    main()
