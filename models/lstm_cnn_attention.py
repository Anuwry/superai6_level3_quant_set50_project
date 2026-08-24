from __future__ import annotations

import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]

from models.baseline_common import (
    DATA_FOLDS_DIR,
    OUTPUT_DIR,
    RANDOM_SEED,
    FoldData,
    discover_folds,
    evaluate_predictions,
    load_fold,
    package_versions,
    predictions_frame,
    save_run_outputs,
    split_xy,
    sequence_history_features,
)
from models.convolutional_neural_network import make_sequences, make_test_sequences
from models.neural_network_folds import (
    NN_DATA_FOLDS_DIR,
    create_neural_network_folds,
    inverse_scaled_target,
    load_scaler_metadata,
)

MODEL_NAME = "lstm_cnn_attention"
SEQUENCE_LENGTH = 20
EPOCHS = 20
BATCH_SIZE = 32
LSTM_UNITS = 16
CONV_FILTERS = 32
KERNEL_SIZE = 3
ATTENTION_HEADS = 2
ATTENTION_KEY_DIM = 8
DENSE_UNITS = 8
BENCHMARK_SEEDS = (42, 123, 456, 789, 2025)
METRIC_COLUMNS = ("rmse", "mae", "mape", "r2", "direction_accuracy")
PACKAGES = ["numpy", "pandas", "scikit-learn", "tensorflow"]

CONFIG = {
    "experiment": "naive_baseline",
    "model": "Keras LSTM-CNN-Attention",
    "paper_inspired_layer_order": True,
    "exact_paper_architecture_available": False,
    "hyperparameter_tuning": False,
    "multi_seed_evaluation": True,
    "benchmark_seeds": list(BENCHMARK_SEEDS),
    "model_parameters": {
        "sequence_length": SEQUENCE_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lstm_units": LSTM_UNITS,
        "conv_filters": CONV_FILTERS,
        "kernel_size": KERNEL_SIZE,
        "padding": "causal",
        "attention_heads": ATTENTION_HEADS,
        "attention_key_dim": ATTENTION_KEY_DIM,
        "causal_attention": True,
        "pooling": "GlobalAveragePooling1D",
        "dense_units": DENSE_UNITS,
        "layer_order": [
            "LSTM",
            "Conv1D",
            "MultiHeadAttention",
            "GlobalAveragePooling1D",
            "Dense",
        ],
        "optimizer": "adam",
        "loss": "mse",
        "shuffle": False,
        "scaling": "MinMaxScaler fitted on each train fold only",
        "scaled_data_dir": str(NN_DATA_FOLDS_DIR),
    },
}

ScalerMetadataLoader = Callable[[str], dict[str, object]]
SeedPredictor = Callable[[FoldData, int, int], np.ndarray]


def validate_benchmark_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    seed_values = tuple(seeds)
    if not seed_values:
        raise ValueError("At least one benchmark seed is required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seed_values):
        raise ValueError("Benchmark seeds must be integers")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("Benchmark seeds must be unique")
    return seed_values


def set_reproducible_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)


def build_lstm_cnn_attention_model(input_shape: tuple[int, int]):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=input_shape)
    lstm_sequence = tf.keras.layers.LSTM(
        LSTM_UNITS,
        return_sequences=True,
    )(inputs)
    convolution_sequence = tf.keras.layers.Conv1D(
        filters=CONV_FILTERS,
        kernel_size=KERNEL_SIZE,
        activation="relu",
        padding="causal",
    )(lstm_sequence)
    attention_sequence = tf.keras.layers.MultiHeadAttention(
        num_heads=ATTENTION_HEADS,
        key_dim=ATTENTION_KEY_DIM,
    )(
        convolution_sequence,
        convolution_sequence,
        use_causal_mask=True,
    )
    context = tf.keras.layers.GlobalAveragePooling1D()(attention_sequence)
    hidden = tf.keras.layers.Dense(DENSE_UNITS, activation="relu")(context)
    outputs = tf.keras.layers.Dense(1)(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def predict_fold(
    fold: FoldData,
    sequence_length: int = SEQUENCE_LENGTH,
    random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    set_reproducible_seed(random_seed)
    x_train, y_train, x_test, _ = split_xy(fold)
    train_features = x_train.to_numpy(dtype=float)
    test_features = x_test.to_numpy(dtype=float)
    x_sequence, y_sequence = make_sequences(
        train_features,
        y_train.to_numpy(dtype=float),
        sequence_length,
    )
    model = build_lstm_cnn_attention_model(
        (sequence_length, train_features.shape[1])
    )
    model.fit(
        x_sequence,
        y_sequence,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    test_sequences = make_test_sequences(
        sequence_history_features(fold),
        test_features,
        sequence_length,
    )
    prediction = model.predict(test_sequences, verbose=0).reshape(-1)
    if not np.isfinite(prediction).all():
        raise ValueError("LSTM-CNN-Attention produced non-finite predictions")
    return prediction


def summarize_seed_metrics(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = sorted({"seed", "fold", *METRIC_COLUMNS}.difference(metrics.columns))
    if missing:
        raise ValueError(f"Metrics are missing columns: {missing}")
    by_seed = (
        metrics.groupby("seed", sort=True)[list(METRIC_COLUMNS)]
        .mean()
        .reset_index()
    )
    summary_values: dict[str, float] = {}
    for column in METRIC_COLUMNS:
        summary_values[f"{column}_mean"] = float(by_seed[column].mean())
        summary_values[f"{column}_std"] = float(by_seed[column].std(ddof=1))
    return by_seed, pd.DataFrame([summary_values])


def save_multi_seed_summary(
    model_dir: Path,
    model_name: str,
    metrics: pd.DataFrame,
    config: dict[str, object],
    seeds: tuple[int, ...],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(model_dir / "metrics_by_seed_and_fold.csv", index=False)
    by_seed, mean_std = summarize_seed_metrics(metrics)
    by_seed.to_csv(model_dir / "metrics_by_seed.csv", index=False)
    mean_std.to_csv(model_dir / "metrics_mean_std_across_seeds.csv", index=False)

    fold_summary = metrics.groupby("fold", sort=True)[list(METRIC_COLUMNS)].agg(
        ["mean", "std"]
    )
    fold_summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in fold_summary.columns
    ]
    fold_summary.reset_index().to_csv(
        model_dir / "metrics_by_fold_mean_std.csv",
        index=False,
    )
    payload = {
        "model_name": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
        "benchmark_seeds": list(seeds),
        "packages": package_versions(PACKAGES),
        "summary_mean_std_across_seeds": mean_std.iloc[0].to_dict(),
    }
    with (model_dir / "multi_seed_run_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(payload, file, indent=2)


def run_multi_seed_benchmark(
    model_name: str,
    scaled_data_dir: Path,
    original_data_dir: Path,
    scaler_metadata_loader: ScalerMetadataLoader,
    output_dir: Path,
    config: dict[str, object],
    seeds: Iterable[int] = BENCHMARK_SEEDS,
    predictor: SeedPredictor = predict_fold,
) -> pd.DataFrame:
    seed_values = validate_benchmark_seeds(seeds)
    model_dir = output_dir / model_name
    all_metrics: list[dict[str, float | str | int]] = []
    scaled_specs = discover_folds(scaled_data_dir)
    original_specs = discover_folds(original_data_dir)

    for seed in tqdm(seed_values, desc=model_name, unit="seed"):
        seed_metrics: list[dict[str, float | str | int]] = []
        seed_predictions: dict[str, pd.DataFrame] = {}
        for scaled_spec, original_spec in zip(
            scaled_specs,
            original_specs,
            strict=True,
        ):
            if scaled_spec.fold != original_spec.fold:
                raise ValueError("Scaled and original fold names do not match")
            scaled_fold = load_fold(scaled_spec)
            original_fold = load_fold(original_spec)
            metadata = scaler_metadata_loader(scaled_spec.fold)
            scaled_prediction = np.asarray(
                predictor(scaled_fold, SEQUENCE_LENGTH, seed),
                dtype=float,
            )
            if scaled_prediction.shape != (len(scaled_fold.test),):
                raise ValueError(
                    f"{model_name} produced invalid prediction shape for "
                    f"{scaled_spec.fold}: {scaled_prediction.shape}"
                )
            prediction = inverse_scaled_target(scaled_prediction, metadata)
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"{model_name} produced non-finite predictions for "
                    f"{scaled_spec.fold}"
                )
            metric = {"seed": seed, **evaluate_predictions(original_fold, prediction)}
            seed_metrics.append(metric)
            all_metrics.append(metric)
            seed_predictions[scaled_spec.fold] = predictions_frame(
                original_fold,
                prediction,
            )
        seed_config = {
            **config,
            "random_seed": seed,
            "benchmark_seeds": list(seed_values),
        }
        save_run_outputs(
            f"seed_{seed}",
            seed_metrics,
            seed_predictions,
            seed_config,
            PACKAGES,
            output_dir=model_dir,
        )

    metrics_frame = pd.DataFrame(all_metrics)
    summary_config = {
        **config,
        "multi_seed_evaluation": True,
        "benchmark_seeds": list(seed_values),
    }
    save_multi_seed_summary(
        model_dir,
        model_name,
        metrics_frame,
        summary_config,
        seed_values,
    )
    by_seed, mean_std = summarize_seed_metrics(metrics_frame)
    print(by_seed)
    print(mean_std)
    return metrics_frame


def main(seeds: Iterable[int] = BENCHMARK_SEEDS) -> pd.DataFrame:
    if not NN_DATA_FOLDS_DIR.exists():
        create_neural_network_folds()
    config = {
        **CONFIG,
        "data": str(NN_DATA_FOLDS_DIR),
        "original_units_data": str(DATA_FOLDS_DIR),
    }
    return run_multi_seed_benchmark(
        MODEL_NAME,
        NN_DATA_FOLDS_DIR,
        DATA_FOLDS_DIR,
        lambda fold_name: load_scaler_metadata(fold_name, NN_DATA_FOLDS_DIR),
        OUTPUT_DIR,
        config,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
