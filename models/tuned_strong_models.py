from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from models.baseline_common import (
    CLOSE_COLUMN,
    DATE_COLUMN,
    PROJECT_ROOT,
    RANDOM_SEED,
    TARGET_COLUMN,
    FoldData,
    evaluate_predictions,
    load_fold,
    predictions_frame,
    print_metrics,
    save_run_outputs,
    split_xy,
)
from models.full_ta_feature_pool import (
    FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR,
    FULL_TA_POINT_IN_TIME_NN_DATA_FOLDS_DIR,
)
from models.neural_network_folds import inverse_scaled_target
from models.point_in_time_data import LABEL_DATE_COLUMN
from models.time_based_validation import (
    FULL_TA_TUNING_DIR,
    FULL_TA_TUNING_NN_DIR,
    ValidationFoldSpec,
    discover_validation_folds,
)

OPTUNA_TRIALS = {
    "ridge": 25,
    "xgboost": 75,
    "lightgbm": 75,
    "lstm": 40,
}
AUTOGLUON_TABULAR_TIME_LIMIT = 300
AUTOGLUON_TIMESERIES_TIME_LIMIT = 600
CHRONOS2_SMALL_MODEL_ID = "autogluon/chronos-2-small"
CHRONOS2_BASE_MODEL_ID = "amazon/chronos-2"
OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "tuned_strong_models_point_in_time_v2"
)


def cuda_available() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except OSError:
        return False


def load_validation_frames(
    spec: ValidationFoldSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(spec.train_path, parse_dates=[DATE_COLUMN]),
        pd.read_csv(spec.validation_path, parse_dates=[DATE_COLUMN]),
        pd.read_csv(spec.test_path, parse_dates=[DATE_COLUMN]),
    )


def split_frame_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return (
        frame.drop(
            columns=[DATE_COLUMN, LABEL_DATE_COLUMN, TARGET_COLUMN],
            errors="ignore",
        ),
        frame[TARGET_COLUMN],
    )


def outer_fold_for_name(fold_name: str, scaled: bool = False) -> FoldData:
    from models.baseline_common import discover_folds

    data_dir = (
        FULL_TA_POINT_IN_TIME_NN_DATA_FOLDS_DIR
        if scaled
        else FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR
    )
    spec = next(spec for spec in discover_folds(data_dir) if spec.fold == fold_name)
    return load_fold(spec)


def save_study(study, model_name: str, fold_name: str) -> None:
    import optuna

    fold_dir = OUTPUT_DIR / model_name / "optuna" / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(fold_dir / "trials.csv", index=False)
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        return
    payload = {
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_trial_number": int(study.best_trial.number),
        "user_attrs": study.best_trial.user_attrs,
    }
    with (fold_dir / "best_trial.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def create_study(model_name: str, fold_name: str):
    import optuna

    fold_dir = OUTPUT_DIR / model_name / "optuna" / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    database_path = fold_dir / "study.db"
    return optuna.create_study(
        study_name=f"{model_name}_{fold_name}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        storage=f"sqlite:///{database_path.as_posix()}",
        load_if_exists=True,
    )


def completed_trial_count(study) -> int:
    import optuna

    return sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )


def remaining_trial_count(study, target_trials: int) -> int:
    return max(0, target_trials - completed_trial_count(study))


def tuning_status() -> pd.DataFrame:
    rows = []
    for model_key, target_trials in OPTUNA_TRIALS.items():
        model_name = f"{model_key}_optuna_full_ta"
        for fold_name in [f"fold_{index}" for index in range(1, 5)]:
            database_path = OUTPUT_DIR / model_name / "optuna" / fold_name / "study.db"
            if not database_path.exists():
                rows.append(
                    {
                        "model": model_key,
                        "fold": fold_name,
                        "completed_trials": 0,
                        "target_trials": target_trials,
                        "best_validation_rmse": np.nan,
                        "best_params": None,
                    }
                )
                continue
            study = create_study(model_name, fold_name)
            completed = completed_trial_count(study)
            rows.append(
                {
                    "model": model_key,
                    "fold": fold_name,
                    "completed_trials": completed,
                    "target_trials": target_trials,
                    "best_validation_rmse": float(study.best_value) if completed else np.nan,
                    "best_params": study.best_params if completed else None,
                }
            )
    return pd.DataFrame(rows)


def run_tabular_optuna(
    model_name: str,
    trial_count: int,
    objective_factory: Callable,
    final_predictor: Callable,
    packages: list[str],
) -> pd.DataFrame:
    metrics = []
    predictions = {}
    best_parameters = {}
    runtime_by_fold: dict[str, dict[str, float]] = {}
    specs = discover_validation_folds(FULL_TA_TUNING_DIR)
    for spec in tqdm(specs, desc=model_name, unit="fold"):
        train, validation, _ = load_validation_frames(spec)
        study = create_study(model_name, spec.fold)
        remaining = remaining_trial_count(study, trial_count)
        tuning_started = time.perf_counter()
        if remaining:
            study.optimize(
                objective_factory(train, validation),
                n_trials=remaining,
                show_progress_bar=True,
                callbacks=[
                    lambda current_study, trial, fold_name=spec.fold: save_study(
                        current_study,
                        model_name,
                        fold_name,
                    )
                ],
            )
        tuning_runtime_seconds = time.perf_counter() - tuning_started
        save_study(study, model_name, spec.fold)
        outer_fold = outer_fold_for_name(spec.fold)
        final_started = time.perf_counter()
        prediction = np.asarray(
            final_predictor(study.best_params, outer_fold),
            dtype=float,
        )
        final_runtime_seconds = time.perf_counter() - final_started
        trial_runtime_seconds = float(
            sum(
                trial.duration.total_seconds()
                for trial in study.trials
                if trial.duration is not None
            )
        )
        metrics.append(
            {
                **evaluate_predictions(outer_fold, prediction),
                "runtime_seconds": final_runtime_seconds,
                "tuning_runtime_seconds_this_run": (
                    tuning_runtime_seconds
                ),
                "tuning_trial_runtime_seconds_total": (
                    trial_runtime_seconds
                ),
            }
        )
        predictions[spec.fold] = predictions_frame(outer_fold, prediction)
        best_parameters[spec.fold] = study.best_params
        runtime_by_fold[spec.fold] = {
            "final_fit_inference_seconds": final_runtime_seconds,
            "tuning_seconds_this_run": tuning_runtime_seconds,
            "completed_trial_duration_seconds": trial_runtime_seconds,
        }
    config = {
        "experiment": "full_ta_optuna_tuning",
        "model": model_name,
        "objective": "validation_rmse",
        "trials_per_fold": trial_count,
        "sampler": "TPESampler",
        "sampler_seed": RANDOM_SEED,
        "outer_test_used_for_tuning": False,
        "best_parameters": best_parameters,
        "cuda_requested": True,
        "cuda_available": cuda_available(),
        "runtime_by_fold": runtime_by_fold,
        "runtime_clock": "time.perf_counter",
    }
    save_run_outputs(model_name, metrics, predictions, config, packages, output_dir=OUTPUT_DIR)
    frame = pd.DataFrame(metrics)
    print_metrics(frame)
    return frame


def ridge_objective_factory(train: pd.DataFrame, validation: pd.DataFrame):
    x_train, y_train = split_frame_xy(train)
    x_validation, y_validation = split_frame_xy(validation)

    def objective(trial) -> float:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=trial.suggest_float("alpha", 1e-5, 1e5, log=True))),
            ]
        )
        model.fit(x_train, y_train)
        return float(mean_squared_error(y_validation, model.predict(x_validation)) ** 0.5)

    return objective


def ridge_final_predictor(params: dict[str, object], fold: FoldData) -> np.ndarray:
    x_train, y_train, x_test, _ = split_xy(fold)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=float(params["alpha"]))),
        ]
    )
    model.fit(x_train, y_train)
    return model.predict(x_test)


def run_ridge_optuna() -> pd.DataFrame:
    return run_tabular_optuna(
        "ridge_optuna_full_ta",
        OPTUNA_TRIALS["ridge"],
        ridge_objective_factory,
        ridge_final_predictor,
        ["numpy", "pandas", "scikit-learn", "optuna"],
    )


def xgboost_parameters(trial) -> dict[str, object]:
    parameters = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if cuda_available():
        parameters["device"] = "cuda"
    return parameters


def xgboost_objective_factory(train: pd.DataFrame, validation: pd.DataFrame):
    x_train, y_train = split_frame_xy(train)
    x_validation, y_validation = split_frame_xy(validation)

    def objective(trial) -> float:
        from xgboost import XGBRegressor

        model = XGBRegressor(**xgboost_parameters(trial))
        model.fit(x_train, y_train, verbose=False)
        return float(mean_squared_error(y_validation, model.predict(x_validation)) ** 0.5)

    return objective


def xgboost_final_predictor(params: dict[str, object], fold: FoldData) -> np.ndarray:
    from xgboost import XGBRegressor

    x_train, y_train, x_test, _ = split_xy(fold)
    fixed = {
        **params,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if cuda_available():
        fixed["device"] = "cuda"
    model = XGBRegressor(**fixed)
    model.fit(x_train, y_train, verbose=False)
    return model.predict(x_test)


def run_xgboost_optuna() -> pd.DataFrame:
    return run_tabular_optuna(
        "xgboost_optuna_full_ta",
        OPTUNA_TRIALS["xgboost"],
        xgboost_objective_factory,
        xgboost_final_predictor,
        ["numpy", "pandas", "scikit-learn", "xgboost", "optuna"],
    )


def lightgbm_parameters(trial) -> dict[str, object]:
    return {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }


def lightgbm_objective_factory(train: pd.DataFrame, validation: pd.DataFrame):
    x_train, y_train = split_frame_xy(train)
    x_validation, y_validation = split_frame_xy(validation)

    def objective(trial) -> float:
        from lightgbm import LGBMRegressor

        parameters = lightgbm_parameters(trial)
        if cuda_available():
            parameters["device_type"] = "gpu"
        model = LGBMRegressor(**parameters)
        try:
            model.fit(x_train, y_train)
        except Exception as error:
            if parameters.pop("device_type", None) != "gpu":
                raise error
            model = LGBMRegressor(**parameters)
            model.fit(x_train, y_train)
            trial.set_user_attr("device_fallback", "cpu")
        return float(mean_squared_error(y_validation, model.predict(x_validation)) ** 0.5)

    return objective


def lightgbm_final_predictor(params: dict[str, object], fold: FoldData) -> np.ndarray:
    from lightgbm import LGBMRegressor

    x_train, y_train, x_test, _ = split_xy(fold)
    fixed = {
        **params,
        "objective": "regression",
        "metric": "rmse",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if cuda_available():
        fixed["device_type"] = "gpu"
    model = LGBMRegressor(**fixed)
    try:
        model.fit(x_train, y_train)
    except Exception as error:
        if fixed.pop("device_type", None) != "gpu":
            raise error
        model = LGBMRegressor(**fixed)
        model.fit(x_train, y_train)
    return model.predict(x_test)


def run_lightgbm_optuna() -> pd.DataFrame:
    return run_tabular_optuna(
        "lightgbm_optuna_full_ta",
        OPTUNA_TRIALS["lightgbm"],
        lightgbm_objective_factory,
        lightgbm_final_predictor,
        ["numpy", "pandas", "scikit-learn", "lightgbm", "optuna"],
    )


def make_sequences(
    features: np.ndarray,
    target: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = []
    y_values = []
    for index in range(sequence_length - 1, len(features)):
        x_values.append(features[index - sequence_length + 1 : index + 1])
        y_values.append(target[index])
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


def make_forecast_sequences(
    history_features: np.ndarray,
    forecast_features: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    combined = np.vstack([history_features, forecast_features])
    history_length = len(history_features)
    return np.asarray(
        [
            combined[
                history_length + offset + 1 - sequence_length : history_length + offset + 1
            ]
            for offset in range(len(forecast_features))
        ],
        dtype=np.float32,
    )


def build_lstm_model(trial, input_shape):
    import tensorflow as tf

    units = trial.suggest_categorical("lstm_units", [16, 32, 64, 128])
    dense_units = trial.suggest_categorical("dense_units", [8, 16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(dense_units, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def run_lstm_optuna() -> pd.DataFrame:
    import optuna
    import tensorflow as tf

    metrics = []
    predictions = {}
    best_parameters = {}
    scaled_specs = discover_validation_folds(FULL_TA_TUNING_NN_DIR)
    for spec in tqdm(scaled_specs, desc="lstm_optuna_full_ta", unit="fold"):
        train, validation, _ = load_validation_frames(spec)
        x_train, y_train = split_frame_xy(train)
        x_validation, y_validation = split_frame_xy(validation)

        def objective(trial) -> float:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(RANDOM_SEED)
            sequence_length = trial.suggest_categorical("sequence_length", [5, 10, 20, 40, 60])
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            train_x, train_y = make_sequences(
                x_train.to_numpy(dtype=float),
                y_train.to_numpy(dtype=float),
                sequence_length,
            )
            validation_x = make_forecast_sequences(
                x_train.to_numpy(dtype=float),
                x_validation.to_numpy(dtype=float),
                sequence_length,
            )
            model = build_lstm_model(trial, (sequence_length, x_train.shape[1]))
            callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            )
            history = model.fit(
                train_x,
                train_y,
                validation_data=(validation_x, y_validation.to_numpy(dtype=np.float32)),
                epochs=100,
                batch_size=batch_size,
                shuffle=False,
                verbose=0,
                callbacks=[callback],
            )
            best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
            trial.set_user_attr("best_epoch", best_epoch)
            prediction = model.predict(validation_x, verbose=0).reshape(-1)
            return float(mean_squared_error(y_validation, prediction) ** 0.5)

        study = create_study("lstm_optuna_full_ta", spec.fold)
        remaining = remaining_trial_count(study, OPTUNA_TRIALS["lstm"])
        if remaining:
            study.optimize(
                objective,
                n_trials=remaining,
                show_progress_bar=True,
                callbacks=[
                    lambda current_study, trial, fold_name=spec.fold: save_study(
                        current_study,
                        "lstm_optuna_full_ta",
                        fold_name,
                    )
                ],
            )
        save_study(study, "lstm_optuna_full_ta", spec.fold)
        outer_scaled = outer_fold_for_name(spec.fold, scaled=True)
        outer_original = outer_fold_for_name(spec.fold)
        x_outer_train, y_outer_train, x_outer_test, _ = split_xy(outer_scaled)
        parameters = study.best_params
        sequence_length = int(parameters["sequence_length"])
        batch_size = int(parameters["batch_size"])
        final_x, final_y = make_sequences(
            x_outer_train.to_numpy(dtype=float),
            y_outer_train.to_numpy(dtype=float),
            sequence_length,
        )
        test_x = make_forecast_sequences(
            x_outer_train.to_numpy(dtype=float),
            x_outer_test.to_numpy(dtype=float),
            sequence_length,
        )

        class FixedTrial:
            def suggest_categorical(self, name, choices):
                return parameters[name]

            def suggest_float(self, name, low, high, log=False):
                return parameters[name]

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(RANDOM_SEED)
        model = build_lstm_model(FixedTrial(), (sequence_length, x_outer_train.shape[1]))
        model.fit(
            final_x,
            final_y,
            epochs=int(study.best_trial.user_attrs["best_epoch"]),
            batch_size=batch_size,
            shuffle=False,
            verbose=0,
        )
        scaled_prediction = model.predict(test_x, verbose=0).reshape(-1)
        metadata_path = (
            FULL_TA_POINT_IN_TIME_NN_DATA_FOLDS_DIR
            / spec.fold
            / "minmax_scaler.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prediction = inverse_scaled_target(scaled_prediction, metadata)
        metrics.append(evaluate_predictions(outer_original, prediction))
        predictions[spec.fold] = predictions_frame(outer_original, prediction)
        best_parameters[spec.fold] = {
            **parameters,
            "best_epoch": study.best_trial.user_attrs["best_epoch"],
        }
    config = {
        "experiment": "full_ta_optuna_tuning",
        "model": "LSTM",
        "objective": "validation_rmse",
        "trials_per_fold": OPTUNA_TRIALS["lstm"],
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "outer_test_used_for_tuning": False,
        "best_parameters": best_parameters,
        "cuda_requested": True,
        "tensorflow_devices": [device.name for device in tf.config.list_physical_devices()],
    }
    save_run_outputs(
        "lstm_optuna_full_ta",
        metrics,
        predictions,
        config,
        ["numpy", "pandas", "scikit-learn", "tensorflow", "optuna"],
        output_dir=OUTPUT_DIR,
    )
    frame = pd.DataFrame(metrics)
    print_metrics(frame)
    return frame


def run_autogluon_tabular_strong() -> pd.DataFrame:
    from autogluon.tabular import TabularPredictor
    from models.baseline_common import discover_folds

    model_name = "autogluon_tabular_high_quality"
    metrics = []
    predictions = {}
    for spec in tqdm(
        discover_folds(FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR),
        desc=model_name,
        unit="fold",
    ):
        fold = load_fold(spec)
        model_path = OUTPUT_DIR / model_name / "models" / spec.fold
        if model_path.exists():
            shutil.rmtree(model_path)
        predictor = TabularPredictor(
            label=TARGET_COLUMN,
            problem_type="regression",
            eval_metric="root_mean_squared_error",
            path=str(model_path),
            verbosity=2,
        )
        predictor.fit(
            train_data=fold.train.drop(
                columns=[DATE_COLUMN, LABEL_DATE_COLUMN],
                errors="ignore",
            ),
            presets="high_quality",
            time_limit=AUTOGLUON_TABULAR_TIME_LIMIT,
            num_gpus=1 if cuda_available() else 0,
        )
        prediction = predictor.predict(
            fold.test.drop(
                columns=[DATE_COLUMN, LABEL_DATE_COLUMN, TARGET_COLUMN],
                errors="ignore",
            )
        ).to_numpy(dtype=float)
        metrics.append(evaluate_predictions(fold, prediction))
        predictions[spec.fold] = predictions_frame(fold, prediction)
    config = {
        "experiment": "strong_models",
        "model": "AutoGluon Tabular",
        "presets": "high_quality",
        "time_limit_per_fold": AUTOGLUON_TABULAR_TIME_LIMIT,
        "cuda_requested": True,
    }
    save_run_outputs(
        model_name,
        metrics,
        predictions,
        config,
        ["autogluon.tabular", "numpy", "pandas"],
        output_dir=OUTPUT_DIR,
    )
    frame = pd.DataFrame(metrics)
    print_metrics(frame)
    return frame


def timeseries_frame(frame: pd.DataFrame):
    from autogluon.timeseries import TimeSeriesDataFrame

    result = pd.DataFrame(
        {
            "item_id": "SET50",
            "timestamp": pd.date_range("2000-01-01", periods=len(frame), freq="D"),
            "target": frame[CLOSE_COLUMN].to_numpy(dtype=float),
        }
    )
    return TimeSeriesDataFrame.from_data_frame(
        result,
        id_column="item_id",
        timestamp_column="timestamp",
    )


def run_autogluon_timeseries_strong() -> pd.DataFrame:
    from autogluon.timeseries import TimeSeriesPredictor
    from models.baseline_common import discover_folds

    model_name = "autogluon_timeseries_high_quality"
    metrics = []
    predictions = {}
    for spec in tqdm(
        discover_folds(FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR),
        desc=model_name,
        unit="fold",
    ):
        fold = load_fold(spec)
        model_path = OUTPUT_DIR / model_name / "models" / spec.fold
        if model_path.exists():
            shutil.rmtree(model_path)
        predictor = TimeSeriesPredictor(
            prediction_length=1,
            target="target",
            eval_metric="RMSE",
            path=str(model_path),
            verbosity=2,
        )
        predictor.fit(
            train_data=timeseries_frame(fold.train),
            presets="high_quality",
            time_limit=AUTOGLUON_TIMESERIES_TIME_LIMIT,
        )
        close_history = fold.train[[CLOSE_COLUMN]].copy()
        fold_predictions = []
        for offset in tqdm(range(len(fold.test)), desc=f"{model_name}_{spec.fold}", unit="day"):
            context = pd.concat(
                [close_history, fold.test.iloc[: offset + 1][[CLOSE_COLUMN]]],
                ignore_index=True,
            )
            forecast = predictor.predict(timeseries_frame(context))
            fold_predictions.append(float(forecast["mean"].iloc[-1]))
        prediction = np.asarray(fold_predictions)
        metrics.append(evaluate_predictions(fold, prediction))
        predictions[spec.fold] = predictions_frame(fold, prediction)
    config = {
        "experiment": "foundation_automl_upper_bound",
        "model": "AutoGluon TimeSeries",
        "presets": "high_quality",
        "time_limit_per_fold": AUTOGLUON_TIMESERIES_TIME_LIMIT,
        "prediction_length": 1,
        "walk_forward": True,
        "feature_set": "Close_D",
        "device_selection": "AutoGluon internal",
    }
    save_run_outputs(
        model_name,
        metrics,
        predictions,
        config,
        ["autogluon.timeseries", "numpy", "pandas"],
        output_dir=OUTPUT_DIR,
    )
    frame = pd.DataFrame(metrics)
    print_metrics(frame)
    return frame


def chronos_context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        column
        for column in frame.columns
        if column not in {DATE_COLUMN, LABEL_DATE_COLUMN, TARGET_COLUMN}
    ]
    result = frame.loc[:, feature_columns].copy()
    result.insert(0, "timestamp", pd.date_range("2000-01-01", periods=len(frame), freq="D"))
    result.insert(0, "item_id", "SET50")
    result = result.rename(columns={CLOSE_COLUMN: "target"})
    return result


def run_chronos2_zero_shot(model_id: str, model_name: str) -> pd.DataFrame:
    from chronos import Chronos2Pipeline
    from models.baseline_common import discover_folds

    device = "cuda" if cuda_available() else "cpu"
    pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device)
    metrics = []
    predictions = {}
    for spec in tqdm(
        discover_folds(FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR),
        desc=model_name,
        unit="fold",
    ):
        fold = load_fold(spec)
        fold_predictions = []
        for offset in tqdm(range(len(fold.test)), desc=f"{model_name}_{spec.fold}", unit="day"):
            context = pd.concat([fold.train, fold.test.iloc[: offset + 1]], ignore_index=True)
            forecast = pipeline.predict_df(
                chronos_context_frame(context),
                prediction_length=1,
                quantile_levels=[0.5],
                id_column="item_id",
                timestamp_column="timestamp",
                target="target",
            )
            fold_predictions.append(float(forecast["predictions"].iloc[-1]))
        prediction = np.asarray(fold_predictions)
        metrics.append(evaluate_predictions(fold, prediction))
        predictions[spec.fold] = predictions_frame(fold, prediction)
    config = {
        "experiment": "foundation_automl_upper_bound",
        "model": "Chronos-2",
        "model_id": model_id,
        "zero_shot": True,
        "fine_tuning": False,
        "prediction_length": 1,
        "walk_forward": True,
        "covariates": "Full TA historical covariates",
        "device": device,
    }
    save_run_outputs(
        model_name,
        metrics,
        predictions,
        config,
        ["chronos-forecasting", "torch", "numpy", "pandas"],
        output_dir=OUTPUT_DIR,
    )
    frame = pd.DataFrame(metrics)
    print_metrics(frame)
    return frame


def run_chronos2_small_zero_shot() -> pd.DataFrame:
    return run_chronos2_zero_shot(CHRONOS2_SMALL_MODEL_ID, "chronos2_small_zero_shot")


def run_chronos2_base_zero_shot() -> pd.DataFrame:
    return run_chronos2_zero_shot(CHRONOS2_BASE_MODEL_ID, "chronos2_base_120m_zero_shot")


def run_chronos2_ensemble_strong() -> pd.DataFrame:
    from autogluon.timeseries import TimeSeriesPredictor
    from models.baseline_common import discover_folds

    model_name = "chronos2_official_ensemble"
    metrics = []
    predictions = {}
    for spec in tqdm(
        discover_folds(FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR),
        desc=model_name,
        unit="fold",
    ):
        fold = load_fold(spec)
        model_path = OUTPUT_DIR / model_name / "models" / spec.fold
        if model_path.exists():
            shutil.rmtree(model_path)
        predictor = TimeSeriesPredictor(
            prediction_length=1,
            target="target",
            eval_metric="RMSE",
            path=str(model_path),
            verbosity=2,
        )
        predictor.fit(
            train_data=timeseries_frame(fold.train),
            presets="chronos2_ensemble",
            time_limit=AUTOGLUON_TIMESERIES_TIME_LIMIT,
        )
        fold_predictions = []
        for offset in tqdm(range(len(fold.test)), desc=f"{model_name}_{spec.fold}", unit="day"):
            context = pd.concat(
                [fold.train[[CLOSE_COLUMN]], fold.test.iloc[: offset + 1][[CLOSE_COLUMN]]],
                ignore_index=True,
            )
            forecast = predictor.predict(timeseries_frame(context))
            fold_predictions.append(float(forecast["mean"].iloc[-1]))
        prediction = np.asarray(fold_predictions)
        metrics.append(evaluate_predictions(fold, prediction))
        predictions[spec.fold] = predictions_frame(fold, prediction)
    config = {
        "experiment": "foundation_automl_upper_bound",
        "model": "AutoGluon Chronos-2 Official Ensemble",
        "presets": "chronos2_ensemble",
        "time_limit_per_fold": AUTOGLUON_TIMESERIES_TIME_LIMIT,
        "members": "Chronos-2 Base zero-shot and fine-tuned Chronos-2 Small",
        "outer_test_used_for_training": False,
        "prediction_length": 1,
        "walk_forward": True,
        "device_selection": "AutoGluon internal",
    }
    save_run_outputs(
        model_name,
        metrics,
        predictions,
        config,
        ["autogluon.timeseries", "chronos-forecasting", "torch", "numpy", "pandas"],
        output_dir=OUTPUT_DIR,
    )
    frame = pd.DataFrame(metrics)
    print_metrics(frame)
    return frame
