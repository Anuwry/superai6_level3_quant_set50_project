from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]

from models.baseline_common import (
    DATA_FOLDS_DIR,
    RANDOM_SEED,
    FoldData,
    discover_folds,
    evaluate_predictions,
    load_fold,
    predictions_frame,
    print_metrics,
    save_run_outputs,
    split_xy,
    sequence_history_features,
)
from models.neural_network_folds import (
    NN_DATA_FOLDS_DIR,
    create_neural_network_folds,
    inverse_scaled_target,
    load_scaler_metadata,
)

MODEL_NAME = "cnn"
SEQUENCE_LENGTH = 20
EPOCHS = 20
BATCH_SIZE = 32
CONV_FILTERS = 32
KERNEL_SIZE = 3
DENSE_UNITS = 8
CONFIG = {
    "experiment": "naive_baseline",
    "model": "Keras 1D CNN",
    "hyperparameter_tuning": False,
    "model_parameters": {
        "sequence_length": SEQUENCE_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "conv_filters": CONV_FILTERS,
        "kernel_size": KERNEL_SIZE,
        "padding": "causal",
        "pooling": "GlobalAveragePooling1D",
        "dense_units": DENSE_UNITS,
        "optimizer": "adam",
        "loss": "mse",
        "shuffle": False,
        "scaling": "MinMaxScaler fitted on each train fold only",
        "scaled_data_dir": str(NN_DATA_FOLDS_DIR),
    },
}


def make_sequences(
    features: np.ndarray,
    target: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_values = np.asarray(features, dtype=float)
    target_values = np.asarray(target, dtype=float)
    if feature_values.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if target_values.ndim != 1:
        raise ValueError("target must be a one-dimensional array")
    if len(feature_values) != len(target_values):
        raise ValueError("features and target must contain the same number of rows")
    if sequence_length < 1 or sequence_length > len(feature_values):
        raise ValueError("sequence_length must be between 1 and the number of feature rows")

    x_values = [
        feature_values[index - sequence_length + 1 : index + 1]
        for index in range(sequence_length - 1, len(feature_values))
    ]
    y_values = target_values[sequence_length - 1 :]
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


def make_test_sequences(
    train_features: np.ndarray,
    test_features: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    train_values = np.asarray(train_features, dtype=float)
    test_values = np.asarray(test_features, dtype=float)
    if train_values.ndim != 2 or test_values.ndim != 2:
        raise ValueError("train_features and test_features must be two-dimensional arrays")
    if train_values.shape[1] != test_values.shape[1]:
        raise ValueError("train and test feature counts must match")
    if sequence_length < 1 or sequence_length > len(train_values) + 1:
        raise ValueError("sequence_length requires sufficient training history")

    combined = np.vstack([train_values, test_values])
    train_length = len(train_values)
    sequences = []
    for offset in range(len(test_values)):
        end = train_length + offset + 1
        sequences.append(combined[end - sequence_length : end])
    return np.asarray(sequences, dtype=np.float32)


def set_reproducible_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def build_cnn_model(input_shape: tuple[int, int]):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(
                filters=CONV_FILTERS,
                kernel_size=KERNEL_SIZE,
                activation="relu",
                padding="causal",
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(DENSE_UNITS, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def predict_fold(
    fold: FoldData,
    sequence_length: int = SEQUENCE_LENGTH,
    random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    if random_seed == RANDOM_SEED:
        set_reproducible_seed()
    else:
        set_reproducible_seed(random_seed)
    x_train, y_train, x_test, _ = split_xy(fold)
    train_features = x_train.to_numpy(dtype=float)
    test_features = x_test.to_numpy(dtype=float)
    x_sequence, y_sequence = make_sequences(
        train_features,
        y_train.to_numpy(dtype=float),
        sequence_length,
    )
    model = build_cnn_model((sequence_length, train_features.shape[1]))
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
    return model.predict(test_sequences, verbose=0).reshape(-1)


def main():
    import pandas as pd

    if not NN_DATA_FOLDS_DIR.exists():
        create_neural_network_folds()
    metrics = []
    predictions = {}
    fold_pairs = zip(
        discover_folds(NN_DATA_FOLDS_DIR),
        discover_folds(DATA_FOLDS_DIR),
        strict=True,
    )
    for scaled_spec, original_spec in fold_pairs:
        scaled_fold = load_fold(scaled_spec)
        original_fold = load_fold(original_spec)
        metadata = load_scaler_metadata(scaled_spec.fold)
        scaled_prediction = predict_fold(scaled_fold)
        prediction = inverse_scaled_target(scaled_prediction, metadata)
        metrics.append(evaluate_predictions(original_fold, prediction))
        predictions[scaled_spec.fold] = predictions_frame(original_fold, prediction)
    metrics_frame = pd.DataFrame(metrics)
    save_run_outputs(
        MODEL_NAME,
        metrics,
        predictions,
        CONFIG,
        ["numpy", "pandas", "scikit-learn", "tensorflow"],
    )
    print_metrics(metrics_frame)
    return metrics_frame


if __name__ == "__main__":
    main()
