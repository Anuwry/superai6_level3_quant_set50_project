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
    TARGET_COLUMN,
    FoldData,
    discover_folds,
    evaluate_predictions,
    load_fold,
    predictions_frame,
    print_metrics,
    save_run_outputs,
    split_xy,
)
from models.neural_network_folds import (
    NN_DATA_FOLDS_DIR,
    create_neural_network_folds,
    inverse_scaled_target,
    load_scaler_metadata,
)

MODEL_NAME = "lstm"
SEQUENCE_LENGTH = 20
EPOCHS = 20
BATCH_SIZE = 32
CONFIG = {
    "experiment": "naive_baseline",
    "model": "Keras LSTM",
    "hyperparameter_tuning": False,
    "model_parameters": {
        "sequence_length": SEQUENCE_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lstm_units": 16,
        "dense_units": 8,
        "optimizer": "adam",
        "loss": "mse",
        "shuffle": False,
        "scaling": "MinMaxScaler fitted on each train fold only",
        "scaled_data_dir": str(NN_DATA_FOLDS_DIR),
    },
}


def make_sequences(features: np.ndarray, target: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    for index in range(sequence_length - 1, len(features)):
        x_values.append(features[index - sequence_length + 1 : index + 1])
        y_values.append(float(target[index]))
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


def set_reproducible_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    import tensorflow as tf

    tf.random.set_seed(RANDOM_SEED)


def predict_fold(fold: FoldData) -> np.ndarray:
    import tensorflow as tf

    set_reproducible_seed()
    x_train, y_train, x_test, _ = split_xy(fold)
    train_features = x_train.to_numpy(dtype=float)
    combined_features = np.vstack([x_train.to_numpy(dtype=float), x_test.to_numpy(dtype=float)])
    train_target = y_train.to_numpy(dtype=float)
    x_seq, y_seq = make_sequences(train_features, train_target, SEQUENCE_LENGTH)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, train_features.shape[1])),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_seq, y_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, verbose=0)
    test_sequences = []
    train_length = len(x_train)
    for offset in range(len(x_test)):
        end = train_length + offset + 1
        start = end - SEQUENCE_LENGTH
        test_sequences.append(combined_features[start:end])
    x_test_seq = np.asarray(test_sequences, dtype=np.float32)
    return model.predict(x_test_seq, verbose=0).reshape(-1)


def main():
    import pandas as pd

    if not NN_DATA_FOLDS_DIR.exists():
        create_neural_network_folds()
    metrics = []
    predictions = {}
    for scaled_spec, original_spec in zip(discover_folds(NN_DATA_FOLDS_DIR), discover_folds(DATA_FOLDS_DIR)):
        scaled_fold = load_fold(scaled_spec)
        original_fold = load_fold(original_spec)
        metadata = load_scaler_metadata(scaled_spec.fold)
        scaled_prediction = predict_fold(scaled_fold)
        y_pred = inverse_scaled_target(scaled_prediction, metadata)
        metrics.append(evaluate_predictions(original_fold, y_pred))
        predictions[scaled_spec.fold] = predictions_frame(original_fold, y_pred)
    metrics_frame = pd.DataFrame(metrics)
    save_run_outputs(MODEL_NAME, metrics, predictions, CONFIG, ["numpy", "pandas", "scikit-learn", "tensorflow"])
    print_metrics(metrics_frame)
    return metrics_frame


if __name__ == "__main__":
    main()
