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
from models.convolutional_neural_network import make_sequences, make_test_sequences
from models.neural_network_folds import (
    NN_DATA_FOLDS_DIR,
    create_neural_network_folds,
    inverse_scaled_target,
    load_scaler_metadata,
)

MODEL_NAME = "attention_lstm"
SEQUENCE_LENGTH = 20
EPOCHS = 20
BATCH_SIZE = 32
LSTM_UNITS = 16
ATTENTION_HEADS = 2
ATTENTION_KEY_DIM = 8
DENSE_UNITS = 8
CONFIG = {
    "experiment": "naive_baseline",
    "model": "Keras Attention-LSTM",
    "hyperparameter_tuning": False,
    "model_parameters": {
        "sequence_length": SEQUENCE_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lstm_units": LSTM_UNITS,
        "attention_heads": ATTENTION_HEADS,
        "attention_key_dim": ATTENTION_KEY_DIM,
        "causal_attention": True,
        "pooling": "GlobalAveragePooling1D",
        "dense_units": DENSE_UNITS,
        "layer_order": [
            "LSTM",
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


def set_reproducible_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def build_attention_lstm_model(input_shape: tuple[int, int]):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=input_shape)
    lstm_sequence = tf.keras.layers.LSTM(
        LSTM_UNITS,
        return_sequences=True,
    )(inputs)
    attention_sequence = tf.keras.layers.MultiHeadAttention(
        num_heads=ATTENTION_HEADS,
        key_dim=ATTENTION_KEY_DIM,
    )(
        lstm_sequence,
        lstm_sequence,
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
    model = build_attention_lstm_model(
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
