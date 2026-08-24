from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from models.attention_lstm import (
    CONFIG as LSTM_ATTENTION_CONFIG,
    SEQUENCE_LENGTH as LSTM_ATTENTION_SEQUENCE_LENGTH,
    predict_fold as predict_lstm_attention_fold,
)
from models.baseline_common import (
    PROJECT_ROOT,
    RANDOM_SEED,
    FoldData,
    discover_folds,
    evaluate_predictions,
    load_fold,
    predictions_frame,
    print_metrics,
    save_run_outputs,
)
from models.convolutional_neural_network import (
    CONFIG as CNN_CONFIG,
    SEQUENCE_LENGTH as CNN_SEQUENCE_LENGTH,
    predict_fold as predict_cnn_fold,
)
from models.full_non_ta_experiments import (
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_SEQUENCE_LENGTH,
    predict_lstm_fold,
)
from models.lstm_cnn import (
    CONFIG as LSTM_CNN_CONFIG,
    SEQUENCE_LENGTH as LSTM_CNN_SEQUENCE_LENGTH,
    predict_fold as predict_lstm_cnn_fold,
)
from models.lstm_cnn_attention import (
    CONFIG as LSTM_CNN_ATTENTION_CONFIG,
    SEQUENCE_LENGTH as LSTM_CNN_ATTENTION_SEQUENCE_LENGTH,
    predict_fold as predict_lstm_cnn_attention_fold,
)
from models.neural_network_folds import inverse_scaled_target
from models.vmd_feature_pool import (
    FULL_TA_VMD_DATA_FOLDS_DIR,
    FULL_TA_VMD_NN_DATA_FOLDS_DIR,
    VMDConfig,
    create_full_ta_vmd_folds,
    create_scaled_full_ta_vmd_nn_folds,
)

FULL_TA_VMD_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_ta_vmd_feature_pool"
FULL_TA_BASELINE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_ta_feature_pool"
COMPARISON_FILE = FULL_TA_VMD_OUTPUT_DIR / "benchmark_comparison.csv"
COMPARISON_BY_FOLD_FILE = (
    FULL_TA_VMD_OUTPUT_DIR / "benchmark_comparison_by_fold.csv"
)

SequencePredictor = Callable[[FoldData, int], np.ndarray]


def ensure_vmd_data() -> None:
    if not FULL_TA_VMD_DATA_FOLDS_DIR.exists():
        create_full_ta_vmd_folds()
    if not FULL_TA_VMD_NN_DATA_FOLDS_DIR.exists():
        create_scaled_full_ta_vmd_nn_folds()


def load_vmd_scaler_metadata(fold_name: str) -> dict[str, object]:
    path = FULL_TA_VMD_NN_DATA_FOLDS_DIR / fold_name / "minmax_scaler.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _model_parameters(base_config: dict[str, object]) -> dict[str, object]:
    parameters = dict(base_config.get("model_parameters", {}))
    parameters["scaled_data_dir"] = str(FULL_TA_VMD_NN_DATA_FOLDS_DIR)
    return parameters


def run_vmd_sequence_model(
    model_key: str,
    model_name: str,
    model_label: str,
    predictor: SequencePredictor,
    sequence_length: int,
    model_parameters: dict[str, object],
    output_dir: Path = FULL_TA_VMD_OUTPUT_DIR,
    experiment: str = "full_ta_plus_causal_rolling_vmd",
) -> pd.DataFrame:
    ensure_vmd_data()
    experiment_started = time.perf_counter()
    metrics: list[dict[str, float | str | int]] = []
    predictions: dict[str, pd.DataFrame] = {}
    fold_pairs = zip(
        discover_folds(FULL_TA_VMD_NN_DATA_FOLDS_DIR),
        discover_folds(FULL_TA_VMD_DATA_FOLDS_DIR),
        strict=True,
    )
    for scaled_spec, original_spec in tqdm(
        fold_pairs,
        desc=model_name,
        unit="fold",
    ):
        if scaled_spec.fold != original_spec.fold:
            raise ValueError("Scaled and original VMD folds are misaligned")
        scaled_fold = load_fold(scaled_spec)
        original_fold = load_fold(original_spec)
        metadata = load_vmd_scaler_metadata(scaled_spec.fold)
        model_started = time.perf_counter()
        scaled_prediction = np.asarray(
            predictor(scaled_fold, sequence_length),
            dtype=float,
        )
        runtime_seconds = time.perf_counter() - model_started
        prediction = inverse_scaled_target(scaled_prediction, metadata)
        fold_metrics = evaluate_predictions(original_fold, prediction)
        fold_metrics["runtime_seconds"] = runtime_seconds
        metrics.append(fold_metrics)
        predictions[scaled_spec.fold] = predictions_frame(original_fold, prediction)

    total_experiment_seconds = time.perf_counter() - experiment_started
    total_fit_predict_seconds = float(
        sum(float(fold["runtime_seconds"]) for fold in metrics)
    )
    config = {
        "experiment": experiment,
        "model_key": model_key,
        "model": model_label,
        "data": str(FULL_TA_VMD_NN_DATA_FOLDS_DIR),
        "original_units_data": str(FULL_TA_VMD_DATA_FOLDS_DIR),
        "random_seed": RANDOM_SEED,
        "hyperparameter_tuning": False,
        "comparison_base": "Full TA with the same fixed model hyperparameters",
        "scaling": "MinMaxScaler fitted on each train fold only",
        "vmd": {
            **asdict(VMDConfig()),
            "input_signal": "Close_D",
            "causal_scope": "rolling past-only including current row",
            "denoising_rule": "remove the highest-center-frequency mode",
        },
        "runtime": {
            "clock": "time.perf_counter",
            "scope": "model build + training + test inference",
            "unit": "seconds",
            "per_fold_column": "runtime_seconds",
            "total_fit_predict_seconds": total_fit_predict_seconds,
            "total_experiment_seconds": total_experiment_seconds,
        },
        **model_parameters,
        "sequence_length": sequence_length,
    }
    save_run_outputs(
        model_name,
        metrics,
        predictions,
        config,
        ["numpy", "pandas", "scikit-learn", "tensorflow"],
        output_dir=output_dir,
    )
    metrics_frame = pd.DataFrame(metrics)
    print_metrics(metrics_frame)
    return metrics_frame


def run_lstm_vmd() -> pd.DataFrame:
    return run_vmd_sequence_model(
        "lstm",
        "lstm_full_ta_vmd",
        "Keras LSTM",
        predict_lstm_fold,
        LSTM_SEQUENCE_LENGTH,
        {
            "epochs": LSTM_EPOCHS,
            "batch_size": LSTM_BATCH_SIZE,
            "lstm_units": 16,
            "dense_units": 8,
            "optimizer": "adam",
            "loss": "mse",
            "shuffle": False,
        },
    )


def run_cnn_vmd() -> pd.DataFrame:
    return run_vmd_sequence_model(
        "cnn",
        "cnn_full_ta_vmd",
        "Keras 1D CNN",
        predict_cnn_fold,
        CNN_SEQUENCE_LENGTH,
        _model_parameters(CNN_CONFIG),
    )


def run_lstm_cnn_vmd() -> pd.DataFrame:
    return run_vmd_sequence_model(
        "lstm_cnn",
        "lstm_cnn_full_ta_vmd",
        "Keras LSTM-CNN",
        predict_lstm_cnn_fold,
        LSTM_CNN_SEQUENCE_LENGTH,
        _model_parameters(LSTM_CNN_CONFIG),
    )


def run_lstm_attention_vmd() -> pd.DataFrame:
    return run_vmd_sequence_model(
        "lstm_attention",
        "lstm_attention_full_ta_vmd",
        "Keras LSTM-Attention",
        predict_lstm_attention_fold,
        LSTM_ATTENTION_SEQUENCE_LENGTH,
        _model_parameters(LSTM_ATTENTION_CONFIG),
    )


def _predict_lstm_cnn_attention_seed_42(
    fold: FoldData,
    sequence_length: int,
) -> np.ndarray:
    return predict_lstm_cnn_attention_fold(
        fold,
        sequence_length=sequence_length,
        random_seed=RANDOM_SEED,
    )


def run_lstm_cnn_attention_vmd() -> pd.DataFrame:
    return run_vmd_sequence_model(
        "lstm_cnn_attention",
        "lstm_cnn_attention_full_ta_vmd",
        "Keras LSTM-CNN-Attention",
        _predict_lstm_cnn_attention_seed_42,
        LSTM_CNN_ATTENTION_SEQUENCE_LENGTH,
        _model_parameters(LSTM_CNN_ATTENTION_CONFIG),
    )


VMD_MODEL_RUNNERS: dict[str, Callable[[], pd.DataFrame]] = {
    "lstm": run_lstm_vmd,
    "cnn": run_cnn_vmd,
    "lstm_cnn": run_lstm_cnn_vmd,
    "lstm_attention": run_lstm_attention_vmd,
    "lstm_cnn_attention": run_lstm_cnn_attention_vmd,
}

VMD_RESULT_DIRS = {
    "lstm": "lstm_full_ta_vmd",
    "cnn": "cnn_full_ta_vmd",
    "lstm_cnn": "lstm_cnn_full_ta_vmd",
    "lstm_attention": "lstm_attention_full_ta_vmd",
    "lstm_cnn_attention": "lstm_cnn_attention_full_ta_vmd",
}

FULL_TA_BASELINE_DIRS = {
    "lstm": "lstm_full_ta",
    "cnn": "cnn_full_ta",
    "lstm_cnn": "lstm_cnn_full_ta",
    "lstm_attention": "attention_lstm_full_ta",
    "lstm_cnn_attention": "lstm_cnn_attention_full_ta",
}


def _read_baseline_metrics(model_key: str) -> pd.DataFrame:
    model_dir = FULL_TA_BASELINE_OUTPUT_DIR / FULL_TA_BASELINE_DIRS[model_key]
    if model_key == "lstm_cnn_attention":
        path = model_dir / "metrics_by_seed_and_fold.csv"
        metrics = pd.read_csv(path)
        return metrics.loc[metrics["seed"] == RANDOM_SEED].copy()
    return pd.read_csv(model_dir / "metrics_by_fold.csv")


def build_benchmark_comparison() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    metric_columns = ["rmse", "mae", "mape", "r2", "direction_accuracy"]
    for model_key in VMD_MODEL_RUNNERS:
        baseline = _read_baseline_metrics(model_key)
        vmd_path = (
            FULL_TA_VMD_OUTPUT_DIR
            / VMD_RESULT_DIRS[model_key]
            / "metrics_by_fold.csv"
        )
        vmd = pd.read_csv(vmd_path)
        paired = baseline.loc[:, ["fold", *metric_columns]].merge(
            vmd.loc[:, ["fold", "n_train", "n_test", *metric_columns]],
            on="fold",
            how="inner",
            validate="one_to_one",
            suffixes=("_baseline", "_vmd"),
        )
        if len(paired) != 4:
            raise ValueError(f"{model_key} does not have four comparable folds")
        for _, fold in paired.iterrows():
            fold_rows.append(
                {
                    "model": model_key,
                    "seed": RANDOM_SEED,
                    "fold": fold["fold"],
                    "n_train": int(fold["n_train"]),
                    "n_test": int(fold["n_test"]),
                    **{
                        f"baseline_{metric}": float(fold[f"{metric}_baseline"])
                        for metric in metric_columns
                    },
                    **{
                        f"vmd_{metric}": float(fold[f"{metric}_vmd"])
                        for metric in metric_columns
                    },
                    "rmse_delta_vmd_minus_baseline": float(
                        fold["rmse_vmd"] - fold["rmse_baseline"]
                    ),
                    "mae_delta_vmd_minus_baseline": float(
                        fold["mae_vmd"] - fold["mae_baseline"]
                    ),
                    "direction_accuracy_delta_pp": float(
                        (
                            fold["direction_accuracy_vmd"]
                            - fold["direction_accuracy_baseline"]
                        )
                        * 100.0
                    ),
                }
            )
        baseline_mean = baseline.loc[:, metric_columns].mean()
        vmd_mean = vmd.loc[:, metric_columns].mean()
        rows.append(
            {
                "model": model_key,
                "seed": RANDOM_SEED,
                "baseline_feature_set": "Full TA",
                "vmd_feature_set": "Full TA + causal rolling VMD",
                **{
                    f"baseline_{metric}": float(baseline_mean[metric])
                    for metric in metric_columns
                },
                **{
                    f"vmd_{metric}": float(vmd_mean[metric])
                    for metric in metric_columns
                },
                "rmse_delta_vmd_minus_baseline": float(
                    vmd_mean["rmse"] - baseline_mean["rmse"]
                ),
                "mae_delta_vmd_minus_baseline": float(
                    vmd_mean["mae"] - baseline_mean["mae"]
                ),
                "direction_accuracy_delta_pp": float(
                    (vmd_mean["direction_accuracy"] - baseline_mean["direction_accuracy"])
                    * 100.0
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison_by_fold = pd.DataFrame(fold_rows)
    FULL_TA_VMD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_FILE, index=False)
    comparison_by_fold.to_csv(COMPARISON_BY_FOLD_FILE, index=False)
    return comparison


def run_all_vmd_models() -> dict[str, pd.DataFrame]:
    results = {model: runner() for model, runner in VMD_MODEL_RUNNERS.items()}
    comparison = build_benchmark_comparison()
    print(comparison)
    return results


def main() -> pd.DataFrame | dict[str, pd.DataFrame]:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe Full TA + VMD neural-network benchmarks."
    )
    parser.add_argument(
        "--model",
        choices=[*VMD_MODEL_RUNNERS, "all", "comparison"],
        default="all",
    )
    args = parser.parse_args()
    if args.model == "all":
        return run_all_vmd_models()
    if args.model == "comparison":
        comparison = build_benchmark_comparison()
        print(comparison)
        return comparison
    return VMD_MODEL_RUNNERS[args.model]()


if __name__ == "__main__":
    main()
