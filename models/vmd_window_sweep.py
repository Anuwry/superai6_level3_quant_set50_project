from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from models.attention_lstm import (
    CONFIG as LSTM_ATTENTION_CONFIG,
    predict_fold as predict_lstm_attention_fold,
)
from models.baseline_common import PROJECT_ROOT, RANDOM_SEED, FoldData
from models.convolutional_neural_network import (
    CONFIG as CNN_CONFIG,
    predict_fold as predict_cnn_fold,
)
from models.full_non_ta_experiments import (
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    predict_lstm_fold,
)
from models.lstm_cnn import (
    CONFIG as LSTM_CNN_CONFIG,
    predict_fold as predict_lstm_cnn_fold,
)
from models.lstm_cnn_attention import (
    CONFIG as LSTM_CNN_ATTENTION_CONFIG,
    predict_fold as predict_lstm_cnn_attention_fold,
)
from models.vmd_experiments import (
    FULL_TA_VMD_OUTPUT_DIR,
    VMD_RESULT_DIRS,
    run_vmd_sequence_model,
)
from models.vmd_feature_pool import (
    FULL_TA_VMD_NN_DATA_FOLDS_DIR,
    VMDConfig,
)

SEQUENCE_WINDOWS = (1, 3, 5, 10, 20)
NEW_SEQUENCE_WINDOWS = (1, 3, 5, 10)
METRIC_COLUMNS = ("rmse", "mae", "mape", "r2", "direction_accuracy")
WINDOW_SWEEP_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "full_ta_vmd_window_sweep"
)
WINDOW_FOLD_METRICS_FILE = (
    WINDOW_SWEEP_OUTPUT_DIR / "metrics_by_model_window_fold.csv"
)
WINDOW_SUMMARY_FILE = (
    WINDOW_SWEEP_OUTPUT_DIR / "metrics_by_model_window.csv"
)
BEST_WINDOWS_FILE = WINDOW_SWEEP_OUTPUT_DIR / "best_windows_by_model.csv"
WINDOW_RUNTIME_FILE = WINDOW_SWEEP_OUTPUT_DIR / "runtime_by_model_window.csv"
RUNTIME_ENVIRONMENT_FILE = (
    WINDOW_SWEEP_OUTPUT_DIR / "runtime_environment.json"
)
PAPER_BEST_WINDOWS_FILE = (
    WINDOW_SWEEP_OUTPUT_DIR / "paper_best_direction_windows.csv"
)

SequencePredictor = Callable[[FoldData, int], np.ndarray]


@dataclass(frozen=True)
class WindowSweepModel:
    label: str
    predictor: SequencePredictor
    parameters: dict[str, object]


def _model_parameters(base_config: dict[str, object]) -> dict[str, object]:
    parameters = dict(base_config.get("model_parameters", {}))
    parameters.pop("sequence_length", None)
    parameters["scaled_data_dir"] = str(FULL_TA_VMD_NN_DATA_FOLDS_DIR)
    return parameters


def _predict_lstm_cnn_attention_seed_42(
    fold: FoldData,
    sequence_length: int,
) -> np.ndarray:
    return predict_lstm_cnn_attention_fold(
        fold,
        sequence_length=sequence_length,
        random_seed=RANDOM_SEED,
    )


def predict_with_reproducible_seed(
    predictor: SequencePredictor,
    fold: FoldData,
    sequence_length: int,
) -> np.ndarray:
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    return predictor(fold, sequence_length)


def _reproducible_predictor(
    predictor: SequencePredictor,
) -> SequencePredictor:
    return lambda fold, sequence_length: predict_with_reproducible_seed(
        predictor,
        fold,
        sequence_length,
    )


WINDOW_SWEEP_MODELS: dict[str, WindowSweepModel] = {
    "lstm": WindowSweepModel(
        label="Keras LSTM",
        predictor=_reproducible_predictor(predict_lstm_fold),
        parameters={
            "epochs": LSTM_EPOCHS,
            "batch_size": LSTM_BATCH_SIZE,
            "lstm_units": 16,
            "dense_units": 8,
            "optimizer": "adam",
            "loss": "mse",
            "shuffle": False,
            "scaled_data_dir": str(FULL_TA_VMD_NN_DATA_FOLDS_DIR),
        },
    ),
    "cnn": WindowSweepModel(
        label="Keras 1D CNN",
        predictor=_reproducible_predictor(predict_cnn_fold),
        parameters=_model_parameters(CNN_CONFIG),
    ),
    "lstm_cnn": WindowSweepModel(
        label="Keras LSTM-CNN",
        predictor=_reproducible_predictor(predict_lstm_cnn_fold),
        parameters=_model_parameters(LSTM_CNN_CONFIG),
    ),
    "lstm_attention": WindowSweepModel(
        label="Keras LSTM-Attention",
        predictor=_reproducible_predictor(predict_lstm_attention_fold),
        parameters=_model_parameters(LSTM_ATTENTION_CONFIG),
    ),
    "lstm_cnn_attention": WindowSweepModel(
        label="Keras LSTM-CNN-Attention",
        predictor=_reproducible_predictor(
            _predict_lstm_cnn_attention_seed_42
        ),
        parameters=_model_parameters(LSTM_CNN_ATTENTION_CONFIG),
    ),
}


def validate_sequence_window(window: object) -> int:
    if (
        isinstance(window, bool)
        or not isinstance(window, int)
        or window not in SEQUENCE_WINDOWS
    ):
        raise ValueError(
            f"sequence window must be one of {SEQUENCE_WINDOWS}; got {window!r}"
        )
    return window


def prepare_runtime_environment() -> dict[str, object]:
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
    _ = tf.constant(0.0).numpy()
    devices = [
        {
            "name": device.name,
            "device_type": device.device_type,
        }
        for device in tf.config.list_physical_devices()
    ]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "tensorflow": tf.__version__,
        "physical_devices": devices,
        "gpu_available": any(
            device["device_type"] == "GPU" for device in devices
        ),
        "runtime_clock": "time.perf_counter",
        "runtime_scope": "model build + training + test inference",
        "framework_startup_excluded": True,
        "deterministic_operations_enabled": True,
        "notes": (
            "TensorFlow is imported and warmed before timed model runs; "
            "data loading, inverse scaling, and metric calculation are excluded."
        ),
    }
    WINDOW_SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RUNTIME_ENVIRONMENT_FILE.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return payload


def _sweep_metrics_path(model_key: str, sequence_window: int) -> Path:
    return (
        WINDOW_SWEEP_OUTPUT_DIR
        / model_key
        / f"window_{sequence_window}"
        / "metrics_by_fold.csv"
    )


def run_model_window(
    model_key: str,
    sequence_window: int,
    *,
    force: bool = False,
) -> pd.DataFrame:
    if model_key not in WINDOW_SWEEP_MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    window = validate_sequence_window(sequence_window)
    metrics_path = _sweep_metrics_path(model_key, window)
    if metrics_path.exists() and not force:
        print(f"Reusing completed sweep: {model_key}, window={window}")
        return pd.read_csv(metrics_path)

    model = WINDOW_SWEEP_MODELS[model_key]
    return run_vmd_sequence_model(
        model_key=model_key,
        model_name=f"window_{window}",
        model_label=model.label,
        predictor=model.predictor,
        sequence_length=window,
        model_parameters=model.parameters,
        output_dir=WINDOW_SWEEP_OUTPUT_DIR / model_key,
        experiment="full_ta_vmd_sequence_window_sweep",
    )


def run_requested_windows(
    model_keys: Iterable[str] = WINDOW_SWEEP_MODELS,
    sequence_windows: Iterable[int] = NEW_SEQUENCE_WINDOWS,
    *,
    force: bool = False,
) -> dict[tuple[str, int], pd.DataFrame]:
    requested_models = tuple(model_keys)
    requested_windows = tuple(
        validate_sequence_window(window) for window in sequence_windows
    )
    unknown_models = [
        model for model in requested_models if model not in WINDOW_SWEEP_MODELS
    ]
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")
    return {
        (model, window): run_model_window(model, window, force=force)
        for model in requested_models
        for window in requested_windows
    }


def _saved_metrics_path(model_key: str, sequence_window: int) -> Path:
    sweep_path = _sweep_metrics_path(model_key, sequence_window)
    if sweep_path.exists():
        return sweep_path
    if sequence_window == 20:
        return (
            FULL_TA_VMD_OUTPUT_DIR
            / VMD_RESULT_DIRS[model_key]
            / "metrics_by_fold.csv"
        )
    return sweep_path


def collect_window_fold_metrics() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_key in WINDOW_SWEEP_MODELS:
        for window in SEQUENCE_WINDOWS:
            path = _saved_metrics_path(model_key, window)
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing sweep metrics for {model_key}, window={window}: {path}"
                )
            metrics = pd.read_csv(path)
            if len(metrics) != 4 or metrics["fold"].nunique() != 4:
                raise ValueError(
                    f"Expected four folds for {model_key}, window={window}"
                )
            if not np.isfinite(
                metrics.loc[:, METRIC_COLUMNS].to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"Non-finite metrics for {model_key}, window={window}"
                )
            if (
                "runtime_seconds" not in metrics.columns
                or not np.isfinite(metrics["runtime_seconds"]).all()
                or (metrics["runtime_seconds"] <= 0.0).any()
            ):
                raise ValueError(
                    f"Missing or invalid runtime for {model_key}, window={window}"
                )
            frame = metrics.copy()
            frame.insert(0, "sequence_window", window)
            frame.insert(0, "model", model_key)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_window_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "sequence_window",
        "fold",
        "runtime_seconds",
        *METRIC_COLUMNS,
    }
    missing = sorted(required.difference(fold_metrics.columns))
    if missing:
        raise ValueError(f"Window metrics are missing columns: {missing}")

    summary = (
        fold_metrics.groupby(["model", "sequence_window"], sort=False)[
            list(METRIC_COLUMNS)
        ]
        .agg(["mean", "std"])
    )
    summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in summary.columns
    ]
    summary = summary.reset_index()
    runtime_summary = (
        fold_metrics.groupby(["model", "sequence_window"], sort=False)[
            "runtime_seconds"
        ]
        .agg(["mean", "std", "sum"])
        .rename(
            columns={
                "mean": "runtime_seconds_mean",
                "std": "runtime_seconds_std",
                "sum": "runtime_seconds_total",
            }
        )
        .reset_index()
    )
    summary = summary.merge(
        runtime_summary,
        on=["model", "sequence_window"],
        how="left",
        validate="one_to_one",
    )
    controls = summary.loc[
        summary["sequence_window"] == 20,
        ["model", "rmse_mean", "direction_accuracy_mean"],
    ].rename(
        columns={
            "rmse_mean": "window_20_rmse",
            "direction_accuracy_mean": "window_20_direction_accuracy",
        }
    )
    if controls["model"].nunique() != summary["model"].nunique():
        raise ValueError("Every model requires a window-20 control")
    summary = summary.merge(controls, on="model", how="left", validate="many_to_one")
    summary["rmse_delta_vs_window_20"] = (
        summary["rmse_mean"] - summary["window_20_rmse"]
    )
    summary["direction_accuracy_delta_vs_window_20_pp"] = (
        summary["direction_accuracy_mean"]
        - summary["window_20_direction_accuracy"]
    ) * 100.0
    summary["direction_accuracy_rank"] = (
        summary.groupby("model")["direction_accuracy_mean"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    summary["rmse_rank"] = (
        summary.groupby("model")["rmse_mean"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    model_order = {model: index for index, model in enumerate(WINDOW_SWEEP_MODELS)}
    summary["_model_order"] = summary["model"].map(model_order)
    return (
        summary.sort_values(["_model_order", "sequence_window"])
        .drop(columns=["_model_order"])
        .reset_index(drop=True)
    )


def select_best_windows(summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "sequence_window",
        "rmse_mean",
        "direction_accuracy_mean",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Window summary is missing columns: {missing}")

    rows: list[dict[str, float | int | str]] = []
    for model_key in summary["model"].drop_duplicates():
        model_summary = summary.loc[summary["model"] == model_key]
        best_direction = model_summary.sort_values(
            [
                "direction_accuracy_mean",
                "rmse_mean",
                "sequence_window",
            ],
            ascending=[False, True, True],
        ).iloc[0]
        best_rmse = model_summary.sort_values(
            [
                "rmse_mean",
                "direction_accuracy_mean",
                "sequence_window",
            ],
            ascending=[True, False, True],
        ).iloc[0]
        rows.append(
            {
                "model": model_key,
                "best_direction_window": int(best_direction["sequence_window"]),
                "best_direction_accuracy": float(
                    best_direction["direction_accuracy_mean"]
                ),
                "best_direction_window_rmse": float(best_direction["rmse_mean"]),
                "best_rmse_window": int(best_rmse["sequence_window"]),
                "best_rmse": float(best_rmse["rmse_mean"]),
                "best_rmse_window_direction_accuracy": float(
                    best_rmse["direction_accuracy_mean"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_paper_ready_table(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary.loc[
        summary["direction_accuracy_rank"] == 1
    ].copy()
    controls = summary.loc[
        summary["sequence_window"] == 20,
        ["model", "runtime_seconds_mean"],
    ].rename(
        columns={
            "runtime_seconds_mean": "window_20_runtime_seconds_mean"
        }
    )
    selected = selected.merge(
        controls,
        on="model",
        how="left",
        validate="one_to_one",
    )
    selected["runtime_reduction_vs_window_20_percent"] = (
        1.0
        - selected["runtime_seconds_mean"]
        / selected["window_20_runtime_seconds_mean"]
    ) * 100.0
    selected["seed"] = RANDOM_SEED
    selected["vmd_window"] = VMDConfig().window_size
    return selected.loc[
        :,
        [
            "model",
            "seed",
            "vmd_window",
            "sequence_window",
            "direction_accuracy_mean",
            "direction_accuracy_std",
            "direction_accuracy_delta_vs_window_20_pp",
            "rmse_mean",
            "rmse_std",
            "rmse_delta_vs_window_20",
            "runtime_seconds_mean",
            "runtime_seconds_std",
            "runtime_seconds_total",
            "window_20_runtime_seconds_mean",
            "runtime_reduction_vs_window_20_percent",
        ],
    ].rename(columns={"sequence_window": "selected_sequence_window"})


def build_window_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_metrics = collect_window_fold_metrics()
    summary = summarize_window_metrics(fold_metrics)
    best = select_best_windows(summary)
    paper_table = build_paper_ready_table(summary)
    WINDOW_SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(WINDOW_FOLD_METRICS_FILE, index=False)
    summary.to_csv(WINDOW_SUMMARY_FILE, index=False)
    best.to_csv(BEST_WINDOWS_FILE, index=False)
    paper_table.to_csv(PAPER_BEST_WINDOWS_FILE, index=False)
    summary.loc[
        :,
        [
            "model",
            "sequence_window",
            "runtime_seconds_mean",
            "runtime_seconds_std",
            "runtime_seconds_total",
        ],
    ].to_csv(WINDOW_RUNTIME_FILE, index=False)
    return fold_metrics, summary, best


def main() -> object:
    parser = argparse.ArgumentParser(
        description="Sweep model sequence windows on Full TA + causal VMD."
    )
    parser.add_argument(
        "--model",
        choices=[*WINDOW_SWEEP_MODELS, "all", "comparison"],
        default="all",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=list(NEW_SEQUENCE_WINDOWS),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.model == "comparison":
        _, summary, best = build_window_comparison()
        print(summary)
        print(best)
        return best

    prepare_runtime_environment()
    model_keys = (
        tuple(WINDOW_SWEEP_MODELS)
        if args.model == "all"
        else (args.model,)
    )
    results = run_requested_windows(
        model_keys=model_keys,
        sequence_windows=args.windows,
        force=args.force,
    )
    if args.model == "all":
        _, summary, best = build_window_comparison()
        print(summary)
        print(best)
    return results


if __name__ == "__main__":
    main()
