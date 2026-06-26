from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    run_model_on_folds,
    save_run_outputs,
    split_xy,
)
from models.full_non_ta_feature_pool import (
    FULL_NON_TA_DATA_FOLDS_DIR,
    FULL_NON_TA_NN_DATA_FOLDS_DIR,
    create_full_non_ta_folds,
    create_scaled_full_non_ta_nn_folds,
)
from models.neural_network_folds import inverse_scaled_target, load_scaler_metadata

FULL_NON_TA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_non_ta_feature_pool"
LSTM_WINDOWS = [5, 10, 20, 40, 60]
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 32


def ensure_full_non_ta_data() -> None:
    if not FULL_NON_TA_DATA_FOLDS_DIR.exists():
        create_full_non_ta_folds()
    if not FULL_NON_TA_NN_DATA_FOLDS_DIR.exists():
        create_scaled_full_non_ta_nn_folds()


def make_ridge_full_non_ta_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, fit_intercept=True, solver="auto")),
        ]
    )


def predict_ridge_fold(fold: FoldData) -> np.ndarray:
    x_train, y_train, x_test, _ = split_xy(fold)
    model = make_ridge_full_non_ta_model()
    model.fit(x_train, y_train)
    return model.predict(x_test)


def run_ridge_full_non_ta() -> pd.DataFrame:
    ensure_full_non_ta_data()
    config = {
        "experiment": "full_non_ta_feature_pool",
        "model": "Ridge",
        "data": str(FULL_NON_TA_DATA_FOLDS_DIR),
        "hyperparameter_tuning": False,
        "scaling": "StandardScaler fitted inside each train fold only",
        "model_parameters": {"alpha": 1.0, "fit_intercept": True, "solver": "auto"},
    }
    metrics = run_model_on_folds(
        "ridge_regression_full_non_ta",
        predict_ridge_fold,
        config,
        ["numpy", "pandas", "scikit-learn"],
        data_dir=FULL_NON_TA_DATA_FOLDS_DIR,
        output_dir=FULL_NON_TA_OUTPUT_DIR,
    )
    print_metrics(metrics)
    return metrics


def predict_xgboost_fold(fold: FoldData) -> np.ndarray:
    from xgboost import XGBRegressor

    x_train, y_train, x_test, _ = split_xy(fold)
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, verbose=False)
    return model.predict(x_test)


def run_xgboost_full_non_ta() -> pd.DataFrame:
    ensure_full_non_ta_data()
    config = {
        "experiment": "full_non_ta_feature_pool",
        "model": "XGBRegressor",
        "data": str(FULL_NON_TA_DATA_FOLDS_DIR),
        "hyperparameter_tuning": False,
        "scaling": "none",
    }
    metrics = run_model_on_folds(
        "xgboost_full_non_ta",
        predict_xgboost_fold,
        config,
        ["numpy", "pandas", "scikit-learn", "xgboost"],
        data_dir=FULL_NON_TA_DATA_FOLDS_DIR,
        output_dir=FULL_NON_TA_OUTPUT_DIR,
    )
    print_metrics(metrics)
    return metrics


def predict_lightgbm_fold(fold: FoldData) -> np.ndarray:
    from lightgbm import LGBMRegressor

    x_train, y_train, x_test, _ = split_xy(fold)
    model = LGBMRegressor(random_state=RANDOM_SEED, verbosity=-1)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def run_lightgbm_full_non_ta() -> pd.DataFrame:
    ensure_full_non_ta_data()
    config = {
        "experiment": "full_non_ta_feature_pool",
        "model": "LGBMRegressor",
        "data": str(FULL_NON_TA_DATA_FOLDS_DIR),
        "hyperparameter_tuning": False,
        "scaling": "none",
    }
    metrics = run_model_on_folds(
        "lightgbm_full_non_ta",
        predict_lightgbm_fold,
        config,
        ["numpy", "pandas", "scikit-learn", "lightgbm"],
        data_dir=FULL_NON_TA_DATA_FOLDS_DIR,
        output_dir=FULL_NON_TA_OUTPUT_DIR,
    )
    print_metrics(metrics)
    return metrics


def predict_autogluon_fold(fold: FoldData) -> np.ndarray:
    from autogluon.tabular import TabularPredictor

    model_path = FULL_NON_TA_OUTPUT_DIR / "autogluon_full_non_ta" / "models" / fold.spec.fold
    if model_path.exists():
        shutil.rmtree(model_path)
    train_data = fold.train.drop(columns=["Date"]).copy()
    test_data = fold.test.drop(columns=["Date", TARGET_COLUMN]).copy()
    predictor = TabularPredictor(label=TARGET_COLUMN, problem_type="regression", path=str(model_path), verbosity=1)
    predictor.fit(
        train_data=train_data,
        presets="medium_quality",
        time_limit=120,
        hyperparameters="default",
        hyperparameter_tune_kwargs=None,
    )
    return predictor.predict(test_data).to_numpy(dtype=float)


def run_autogluon_full_non_ta() -> pd.DataFrame:
    ensure_full_non_ta_data()
    config = {
        "experiment": "full_non_ta_feature_pool",
        "model": "AutoGluon TabularPredictor",
        "data": str(FULL_NON_TA_DATA_FOLDS_DIR),
        "hyperparameter_tuning": False,
        "presets": "medium_quality",
        "time_limit": 120,
        "hyperparameters": "default",
        "scaling": "AutoGluon internal preprocessing only",
    }
    metrics = run_model_on_folds(
        "autogluon_full_non_ta",
        predict_autogluon_fold,
        config,
        ["numpy", "pandas", "scikit-learn", "autogluon"],
        data_dir=FULL_NON_TA_DATA_FOLDS_DIR,
        output_dir=FULL_NON_TA_OUTPUT_DIR,
    )
    print_metrics(metrics)
    return metrics


def set_lstm_seed() -> None:
    import random

    import tensorflow as tf

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


def make_lstm_sequences(features: np.ndarray, target: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    x_values = []
    y_values = []
    for index in range(window - 1, len(features)):
        x_values.append(features[index - window + 1 : index + 1])
        y_values.append(float(target[index]))
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


def predict_lstm_fold(fold: FoldData, window: int) -> np.ndarray:
    import tensorflow as tf

    set_lstm_seed()
    x_train, y_train, x_test, _ = split_xy(fold)
    train_features = x_train.to_numpy(dtype=float)
    combined_features = np.vstack([x_train.to_numpy(dtype=float), x_test.to_numpy(dtype=float)])
    x_seq, y_seq = make_lstm_sequences(train_features, y_train.to_numpy(dtype=float), window)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, train_features.shape[1])),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_seq, y_seq, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, shuffle=False, verbose=0)
    test_sequences = []
    train_length = len(x_train)
    for offset in range(len(x_test)):
        end = train_length + offset + 1
        start = end - window
        test_sequences.append(combined_features[start:end])
    return model.predict(np.asarray(test_sequences, dtype=np.float32), verbose=0).reshape(-1)


def run_lstm_full_non_ta_window(window: int) -> pd.DataFrame:
    if window not in LSTM_WINDOWS:
        raise ValueError(f"Unsupported LSTM window: {window}")
    ensure_full_non_ta_data()
    metrics = []
    predictions = {}
    for scaled_spec, original_spec in zip(
        discover_folds(FULL_NON_TA_NN_DATA_FOLDS_DIR),
        discover_folds(FULL_NON_TA_DATA_FOLDS_DIR),
    ):
        scaled_fold = load_fold(scaled_spec)
        original_fold = load_fold(original_spec)
        metadata = load_scaler_metadata_for_full_non_ta(scaled_spec.fold)
        scaled_prediction = predict_lstm_fold(scaled_fold, window)
        y_pred = inverse_scaled_target(scaled_prediction, metadata)
        metrics.append(evaluate_predictions(original_fold, y_pred))
        predictions[scaled_spec.fold] = predictions_frame(original_fold, y_pred)
    model_name = f"lstm_full_non_ta_window_{window}"
    config = {
        "experiment": "full_non_ta_feature_pool",
        "model": "Keras LSTM",
        "data": str(FULL_NON_TA_NN_DATA_FOLDS_DIR),
        "original_units_data": str(FULL_NON_TA_DATA_FOLDS_DIR),
        "hyperparameter_tuning": False,
        "scaling": "MinMaxScaler fitted on each train fold only",
        "window": window,
        "all_windows": LSTM_WINDOWS,
        "epochs": LSTM_EPOCHS,
        "batch_size": LSTM_BATCH_SIZE,
    }
    save_run_outputs(
        model_name,
        metrics,
        predictions,
        config,
        ["numpy", "pandas", "scikit-learn", "tensorflow"],
        output_dir=FULL_NON_TA_OUTPUT_DIR,
    )
    metrics_frame = pd.DataFrame(metrics)
    print_metrics(metrics_frame)
    return metrics_frame


def load_scaler_metadata_for_full_non_ta(fold_name: str) -> dict[str, object]:
    path = FULL_NON_TA_NN_DATA_FOLDS_DIR / fold_name / "minmax_scaler.json"
    with path.open("r", encoding="utf-8") as file:
        import json

        return json.load(file)


def run_lstm_full_non_ta_all_windows() -> dict[int, pd.DataFrame]:
    return {window: run_lstm_full_non_ta_window(window) for window in LSTM_WINDOWS}


def run_chronos_full_non_ta_reference() -> pd.DataFrame:
    from models.chronos import CONFIG, predict_fold

    config = {
        **CONFIG,
        "experiment": "full_non_ta_feature_pool_reference",
        "data": str(DATA_FOLDS_DIR),
        "feature_set": "Close_D only; Chronos does not use Full Non-TA multivariate features",
    }
    metrics = run_model_on_folds(
        "chronos_t5_tiny_zero_shot_greedy_reference",
        predict_fold,
        config,
        ["numpy", "pandas", "torch", "transformers", "chronos-forecasting"],
        data_dir=DATA_FOLDS_DIR,
        output_dir=FULL_NON_TA_OUTPUT_DIR,
    )
    print_metrics(metrics)
    return metrics
