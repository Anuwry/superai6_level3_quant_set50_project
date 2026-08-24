from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from models.attention_lstm import (
    ATTENTION_HEADS as ATTENTION_LSTM_HEADS,
    ATTENTION_KEY_DIM as ATTENTION_LSTM_KEY_DIM,
    BATCH_SIZE as ATTENTION_LSTM_BATCH_SIZE,
    DENSE_UNITS as ATTENTION_LSTM_DENSE_UNITS,
    EPOCHS as ATTENTION_LSTM_EPOCHS,
    LSTM_UNITS as ATTENTION_LSTM_UNITS,
    SEQUENCE_LENGTH as ATTENTION_LSTM_SEQUENCE_LENGTH,
    predict_fold as predict_attention_lstm_fold,
)
from models.attention_lstm_cnn import (
    ATTENTION_HEADS as ATTENTION_LSTM_CNN_HEADS,
    ATTENTION_KEY_DIM as ATTENTION_LSTM_CNN_KEY_DIM,
    BATCH_SIZE as ATTENTION_LSTM_CNN_BATCH_SIZE,
    CONV_FILTERS as ATTENTION_LSTM_CNN_FILTERS,
    DENSE_UNITS as ATTENTION_LSTM_CNN_DENSE_UNITS,
    EPOCHS as ATTENTION_LSTM_CNN_EPOCHS,
    KERNEL_SIZE as ATTENTION_LSTM_CNN_KERNEL_SIZE,
    LSTM_UNITS as ATTENTION_LSTM_CNN_UNITS,
    SEQUENCE_LENGTH as ATTENTION_LSTM_CNN_SEQUENCE_LENGTH,
    predict_fold as predict_attention_lstm_cnn_fold,
)
from models.baseline_common import (
    DATA_FOLDS_DIR,
    PROJECT_ROOT,
    TARGET_COLUMN,
    FoldData,
    discover_folds,
    evaluate_predictions,
    load_fold,
    predictions_frame,
    print_metrics,
    run_model_on_folds,
    save_run_outputs,
)
from models.convolutional_neural_network import (
    BATCH_SIZE as CNN_BATCH_SIZE,
    CONV_FILTERS,
    DENSE_UNITS as CNN_DENSE_UNITS,
    EPOCHS as CNN_EPOCHS,
    KERNEL_SIZE,
    SEQUENCE_LENGTH as CNN_SEQUENCE_LENGTH,
    predict_fold as predict_cnn_fold,
)
from models.full_non_ta_experiments import (
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_SEQUENCE_LENGTH,
    predict_lightgbm_fold,
    predict_lstm_fold,
    predict_ridge_fold,
    predict_xgboost_fold,
)
from models.full_ta_feature_pool import (
    FULL_TA_DATA_FOLDS_DIR,
    FULL_TA_NN_DATA_FOLDS_DIR,
    create_full_ta_folds,
    create_scaled_full_ta_nn_folds,
)
from models.lstm_cnn import (
    BATCH_SIZE as LSTM_CNN_BATCH_SIZE,
    CONV_FILTERS as LSTM_CNN_CONV_FILTERS,
    DENSE_UNITS as LSTM_CNN_DENSE_UNITS,
    EPOCHS as LSTM_CNN_EPOCHS,
    KERNEL_SIZE as LSTM_CNN_KERNEL_SIZE,
    LSTM_UNITS as LSTM_CNN_UNITS,
    SEQUENCE_LENGTH as LSTM_CNN_SEQUENCE_LENGTH,
    predict_fold as predict_lstm_cnn_fold,
)
from models.lstm_cnn_attention import (
    BENCHMARK_SEEDS as LSTM_CNN_ATTENTION_SEEDS,
    CONFIG as LSTM_CNN_ATTENTION_CONFIG,
    run_multi_seed_benchmark as run_lstm_cnn_attention_multi_seed_benchmark,
)
from models.neural_network_folds import inverse_scaled_target

FULL_TA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_ta_feature_pool"


def ensure_full_ta_data() -> None:
    if not FULL_TA_DATA_FOLDS_DIR.exists():
        create_full_ta_folds()
    if not FULL_TA_NN_DATA_FOLDS_DIR.exists():
        create_scaled_full_ta_nn_folds()


def run_tabular_full_ta(
    model_name: str,
    model_label: str,
    predict_fold,
    packages: list[str],
    scaling: str,
) -> pd.DataFrame:
    ensure_full_ta_data()
    config = {
        "experiment": "full_non_ta_plus_paper_aligned_ta",
        "model": model_label,
        "data": str(FULL_TA_DATA_FOLDS_DIR),
        "hyperparameter_tuning": False,
        "scaling": scaling,
    }
    metrics = run_model_on_folds(
        model_name,
        predict_fold,
        config,
        packages,
        data_dir=FULL_TA_DATA_FOLDS_DIR,
        output_dir=FULL_TA_OUTPUT_DIR,
    )
    print_metrics(metrics)
    return metrics


def run_ridge_full_ta() -> pd.DataFrame:
    return run_tabular_full_ta(
        "ridge_regression_full_ta",
        "Ridge",
        predict_ridge_fold,
        ["numpy", "pandas", "scikit-learn"],
        "StandardScaler fitted inside each train fold only",
    )


def run_xgboost_full_ta() -> pd.DataFrame:
    return run_tabular_full_ta(
        "xgboost_full_ta",
        "XGBRegressor",
        predict_xgboost_fold,
        ["numpy", "pandas", "scikit-learn", "xgboost"],
        "none",
    )


def run_lightgbm_full_ta() -> pd.DataFrame:
    return run_tabular_full_ta(
        "lightgbm_full_ta",
        "LGBMRegressor",
        predict_lightgbm_fold,
        ["numpy", "pandas", "scikit-learn", "lightgbm"],
        "none",
    )


def predict_autogluon_full_ta_fold(fold: FoldData) -> np.ndarray:
    from autogluon.tabular import TabularPredictor

    model_path = FULL_TA_OUTPUT_DIR / "autogluon_full_ta" / "models" / fold.spec.fold
    if model_path.exists():
        shutil.rmtree(model_path)
    train_data = fold.train.drop(columns=["Date"]).copy()
    test_data = fold.test.drop(columns=["Date", TARGET_COLUMN]).copy()
    predictor = TabularPredictor(
        label=TARGET_COLUMN,
        problem_type="regression",
        path=str(model_path),
        verbosity=1,
    )
    predictor.fit(
        train_data=train_data,
        presets="medium_quality",
        time_limit=120,
        hyperparameters="default",
        hyperparameter_tune_kwargs=None,
    )
    return predictor.predict(test_data).to_numpy(dtype=float)


def run_autogluon_full_ta() -> pd.DataFrame:
    ensure_full_ta_data()
    config = {
        "experiment": "full_non_ta_plus_paper_aligned_ta",
        "model": "AutoGluon TabularPredictor",
        "data": str(FULL_TA_DATA_FOLDS_DIR),
        "hyperparameter_tuning": False,
        "presets": "medium_quality",
        "time_limit": 120,
        "hyperparameters": "default",
        "scaling": "AutoGluon internal preprocessing only",
    }
    metrics = run_model_on_folds(
        "autogluon_full_ta",
        predict_autogluon_full_ta_fold,
        config,
        ["numpy", "pandas", "scikit-learn", "autogluon"],
        data_dir=FULL_TA_DATA_FOLDS_DIR,
        output_dir=FULL_TA_OUTPUT_DIR,
    )
    print_metrics(metrics)
    return metrics


def load_full_ta_scaler_metadata(fold_name: str) -> dict[str, object]:
    path = FULL_TA_NN_DATA_FOLDS_DIR / fold_name / "minmax_scaler.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def run_lstm_full_ta() -> pd.DataFrame:
    ensure_full_ta_data()
    metrics = []
    predictions = {}
    fold_pairs = list(
        zip(
            discover_folds(FULL_TA_NN_DATA_FOLDS_DIR),
            discover_folds(FULL_TA_DATA_FOLDS_DIR),
        )
    )
    for scaled_spec, original_spec in tqdm(fold_pairs, desc="lstm_full_ta", unit="fold"):
        scaled_fold = load_fold(scaled_spec)
        original_fold = load_fold(original_spec)
        metadata = load_full_ta_scaler_metadata(scaled_spec.fold)
        scaled_prediction = predict_lstm_fold(scaled_fold, LSTM_SEQUENCE_LENGTH)
        prediction = inverse_scaled_target(scaled_prediction, metadata)
        metrics.append(evaluate_predictions(original_fold, prediction))
        predictions[scaled_spec.fold] = predictions_frame(original_fold, prediction)
    config = {
        "experiment": "full_non_ta_plus_paper_aligned_ta",
        "model": "Keras LSTM",
        "data": str(FULL_TA_NN_DATA_FOLDS_DIR),
        "original_units_data": str(FULL_TA_DATA_FOLDS_DIR),
        "hyperparameter_tuning": False,
        "scaling": "MinMaxScaler fitted on each train fold only",
        "sequence_length": LSTM_SEQUENCE_LENGTH,
        "epochs": LSTM_EPOCHS,
        "batch_size": LSTM_BATCH_SIZE,
        "lstm_units": 16,
        "dense_units": 8,
        "optimizer": "adam",
        "loss": "mse",
        "shuffle": False,
    }
    save_run_outputs(
        "lstm_full_ta",
        metrics,
        predictions,
        config,
        ["numpy", "pandas", "scikit-learn", "tensorflow"],
        output_dir=FULL_TA_OUTPUT_DIR,
    )
    metrics_frame = pd.DataFrame(metrics)
    print_metrics(metrics_frame)
    return metrics_frame


def run_cnn_full_ta() -> pd.DataFrame:
    ensure_full_ta_data()
    metrics = []
    predictions = {}
    fold_pairs = zip(
        discover_folds(FULL_TA_NN_DATA_FOLDS_DIR),
        discover_folds(FULL_TA_DATA_FOLDS_DIR),
        strict=True,
    )
    for scaled_spec, original_spec in tqdm(fold_pairs, desc="cnn_full_ta", unit="fold"):
        scaled_fold = load_fold(scaled_spec)
        original_fold = load_fold(original_spec)
        metadata = load_full_ta_scaler_metadata(scaled_spec.fold)
        scaled_prediction = predict_cnn_fold(scaled_fold, CNN_SEQUENCE_LENGTH)
        prediction = inverse_scaled_target(scaled_prediction, metadata)
        metrics.append(evaluate_predictions(original_fold, prediction))
        predictions[scaled_spec.fold] = predictions_frame(original_fold, prediction)
    config = {
        "experiment": "full_non_ta_plus_paper_aligned_ta",
        "model": "Keras 1D CNN",
        "data": str(FULL_TA_NN_DATA_FOLDS_DIR),
        "original_units_data": str(FULL_TA_DATA_FOLDS_DIR),
        "hyperparameter_tuning": False,
        "scaling": "MinMaxScaler fitted on each train fold only",
        "sequence_length": CNN_SEQUENCE_LENGTH,
        "epochs": CNN_EPOCHS,
        "batch_size": CNN_BATCH_SIZE,
        "conv_filters": CONV_FILTERS,
        "kernel_size": KERNEL_SIZE,
        "padding": "causal",
        "pooling": "GlobalAveragePooling1D",
        "dense_units": CNN_DENSE_UNITS,
        "optimizer": "adam",
        "loss": "mse",
        "shuffle": False,
    }
    save_run_outputs(
        "cnn_full_ta",
        metrics,
        predictions,
        config,
        ["numpy", "pandas", "scikit-learn", "tensorflow"],
        output_dir=FULL_TA_OUTPUT_DIR,
    )
    metrics_frame = pd.DataFrame(metrics)
    print_metrics(metrics_frame)
    return metrics_frame


def run_lstm_cnn_full_ta() -> pd.DataFrame:
    ensure_full_ta_data()
    metrics = []
    predictions = {}
    fold_pairs = zip(
        discover_folds(FULL_TA_NN_DATA_FOLDS_DIR),
        discover_folds(FULL_TA_DATA_FOLDS_DIR),
        strict=True,
    )
    for scaled_spec, original_spec in tqdm(
        fold_pairs,
        desc="lstm_cnn_full_ta",
        unit="fold",
    ):
        scaled_fold = load_fold(scaled_spec)
        original_fold = load_fold(original_spec)
        metadata = load_full_ta_scaler_metadata(scaled_spec.fold)
        scaled_prediction = predict_lstm_cnn_fold(
            scaled_fold,
            LSTM_CNN_SEQUENCE_LENGTH,
        )
        prediction = inverse_scaled_target(scaled_prediction, metadata)
        metrics.append(evaluate_predictions(original_fold, prediction))
        predictions[scaled_spec.fold] = predictions_frame(original_fold, prediction)
    config = {
        "experiment": "full_non_ta_plus_paper_aligned_ta",
        "model": "Keras LSTM-CNN",
        "data": str(FULL_TA_NN_DATA_FOLDS_DIR),
        "original_units_data": str(FULL_TA_DATA_FOLDS_DIR),
        "hyperparameter_tuning": False,
        "scaling": "MinMaxScaler fitted on each train fold only",
        "sequence_length": LSTM_CNN_SEQUENCE_LENGTH,
        "epochs": LSTM_CNN_EPOCHS,
        "batch_size": LSTM_CNN_BATCH_SIZE,
        "lstm_units": LSTM_CNN_UNITS,
        "conv_filters": LSTM_CNN_CONV_FILTERS,
        "kernel_size": LSTM_CNN_KERNEL_SIZE,
        "padding": "causal",
        "pooling": "GlobalAveragePooling1D",
        "dense_units": LSTM_CNN_DENSE_UNITS,
        "layer_order": [
            "LSTM",
            "Conv1D",
            "GlobalAveragePooling1D",
            "Dense",
        ],
        "optimizer": "adam",
        "loss": "mse",
        "shuffle": False,
    }
    save_run_outputs(
        "lstm_cnn_full_ta",
        metrics,
        predictions,
        config,
        ["numpy", "pandas", "scikit-learn", "tensorflow"],
        output_dir=FULL_TA_OUTPUT_DIR,
    )
    metrics_frame = pd.DataFrame(metrics)
    print_metrics(metrics_frame)
    return metrics_frame


def run_attention_sequence_full_ta(
    model_name: str,
    model_label: str,
    predictor: Callable[[FoldData, int], np.ndarray],
    sequence_length: int,
    model_parameters: dict[str, object],
) -> pd.DataFrame:
    ensure_full_ta_data()
    metrics = []
    predictions = {}
    fold_pairs = zip(
        discover_folds(FULL_TA_NN_DATA_FOLDS_DIR),
        discover_folds(FULL_TA_DATA_FOLDS_DIR),
        strict=True,
    )
    for scaled_spec, original_spec in tqdm(
        fold_pairs,
        desc=model_name,
        unit="fold",
    ):
        scaled_fold = load_fold(scaled_spec)
        original_fold = load_fold(original_spec)
        metadata = load_full_ta_scaler_metadata(scaled_spec.fold)
        scaled_prediction = predictor(scaled_fold, sequence_length)
        prediction = inverse_scaled_target(scaled_prediction, metadata)
        metrics.append(evaluate_predictions(original_fold, prediction))
        predictions[scaled_spec.fold] = predictions_frame(original_fold, prediction)
    config = {
        "experiment": "full_non_ta_plus_paper_aligned_ta",
        "model": model_label,
        "data": str(FULL_TA_NN_DATA_FOLDS_DIR),
        "original_units_data": str(FULL_TA_DATA_FOLDS_DIR),
        "hyperparameter_tuning": False,
        "scaling": "MinMaxScaler fitted on each train fold only",
        "sequence_length": sequence_length,
        **model_parameters,
        "optimizer": "adam",
        "loss": "mse",
        "shuffle": False,
    }
    save_run_outputs(
        model_name,
        metrics,
        predictions,
        config,
        ["numpy", "pandas", "scikit-learn", "tensorflow"],
        output_dir=FULL_TA_OUTPUT_DIR,
    )
    metrics_frame = pd.DataFrame(metrics)
    print_metrics(metrics_frame)
    return metrics_frame


def run_attention_lstm_full_ta() -> pd.DataFrame:
    return run_attention_sequence_full_ta(
        "attention_lstm_full_ta",
        "Keras Attention-LSTM",
        predict_attention_lstm_fold,
        ATTENTION_LSTM_SEQUENCE_LENGTH,
        {
            "epochs": ATTENTION_LSTM_EPOCHS,
            "batch_size": ATTENTION_LSTM_BATCH_SIZE,
            "lstm_units": ATTENTION_LSTM_UNITS,
            "attention_heads": ATTENTION_LSTM_HEADS,
            "attention_key_dim": ATTENTION_LSTM_KEY_DIM,
            "causal_attention": True,
            "pooling": "GlobalAveragePooling1D",
            "dense_units": ATTENTION_LSTM_DENSE_UNITS,
            "layer_order": [
                "LSTM",
                "MultiHeadAttention",
                "GlobalAveragePooling1D",
                "Dense",
            ],
        },
    )


def run_attention_lstm_cnn_full_ta() -> pd.DataFrame:
    return run_attention_sequence_full_ta(
        "attention_lstm_cnn_full_ta",
        "Keras Attention-LSTM-CNN",
        predict_attention_lstm_cnn_fold,
        ATTENTION_LSTM_CNN_SEQUENCE_LENGTH,
        {
            "epochs": ATTENTION_LSTM_CNN_EPOCHS,
            "batch_size": ATTENTION_LSTM_CNN_BATCH_SIZE,
            "lstm_units": ATTENTION_LSTM_CNN_UNITS,
            "attention_heads": ATTENTION_LSTM_CNN_HEADS,
            "attention_key_dim": ATTENTION_LSTM_CNN_KEY_DIM,
            "causal_attention": True,
            "conv_filters": ATTENTION_LSTM_CNN_FILTERS,
            "kernel_size": ATTENTION_LSTM_CNN_KERNEL_SIZE,
            "padding": "causal",
            "pooling": "GlobalAveragePooling1D",
            "dense_units": ATTENTION_LSTM_CNN_DENSE_UNITS,
            "layer_order": [
                "LSTM",
                "MultiHeadAttention",
                "Conv1D",
                "GlobalAveragePooling1D",
                "Dense",
            ],
        },
    )


def run_lstm_cnn_attention_full_ta(
    seeds: Iterable[int] = LSTM_CNN_ATTENTION_SEEDS,
) -> pd.DataFrame:
    ensure_full_ta_data()
    config = {
        **LSTM_CNN_ATTENTION_CONFIG,
        "experiment": "full_non_ta_plus_paper_aligned_ta",
        "data": str(FULL_TA_NN_DATA_FOLDS_DIR),
        "original_units_data": str(FULL_TA_DATA_FOLDS_DIR),
        "model_parameters": {
            **LSTM_CNN_ATTENTION_CONFIG["model_parameters"],
            "scaled_data_dir": str(FULL_TA_NN_DATA_FOLDS_DIR),
        },
    }
    return run_lstm_cnn_attention_multi_seed_benchmark(
        "lstm_cnn_attention_full_ta",
        FULL_TA_NN_DATA_FOLDS_DIR,
        FULL_TA_DATA_FOLDS_DIR,
        load_full_ta_scaler_metadata,
        FULL_TA_OUTPUT_DIR,
        config,
        seeds=seeds,
    )


def run_chronos_full_ta_reference() -> pd.DataFrame:
    from models.chronos import CONFIG, predict_fold

    config = {
        **CONFIG,
        "experiment": "full_ta_feature_pool_reference",
        "data": str(DATA_FOLDS_DIR),
        "feature_set": "Close_D only; original Chronos T5 Tiny does not accept covariates",
    }
    metrics = run_model_on_folds(
        "chronos_t5_tiny_zero_shot_greedy_full_ta_reference",
        predict_fold,
        config,
        ["numpy", "pandas", "torch", "transformers", "chronos-forecasting"],
        data_dir=DATA_FOLDS_DIR,
        output_dir=FULL_TA_OUTPUT_DIR,
    )
    print_metrics(metrics)
    return metrics
