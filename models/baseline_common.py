from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from tqdm.auto import tqdm

from models.point_in_time_data import CONTEXT_FILE_NAME, LABEL_DATE_COLUMN

DATE_COLUMN = "Date"
TARGET_COLUMN = "Target_Next_Close"
CLOSE_COLUMN = "Close_D"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDS_DIR = PROJECT_ROOT / "data-folds"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_naive"
RANDOM_SEED = 42


@dataclass(frozen=True)
class FoldSpec:
    fold: str
    train_path: Path
    test_path: Path
    train_start_year: int
    train_end_year: int
    test_year: int


@dataclass(frozen=True)
class FoldData:
    spec: FoldSpec
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    context: pd.DataFrame | None = None


def discover_folds(data_dir: Path = DATA_FOLDS_DIR) -> list[FoldSpec]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data folder: {data_dir}")
    specs: list[FoldSpec] = []
    for fold_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        train_files = sorted(fold_dir.glob("train_*.csv"))
        test_files = sorted(fold_dir.glob("test_*.csv"))
        if len(train_files) != 1 or len(test_files) != 1:
            raise ValueError(f"Expected one train and one test file in {fold_dir}")
        train_name = train_files[0].stem.replace("train_", "")
        test_name = test_files[0].stem.replace("test_", "")
        train_start, train_end = [int(part) for part in train_name.split("_")]
        test_year = int(test_name)
        specs.append(
            FoldSpec(
                fold=fold_dir.name,
                train_path=train_files[0],
                test_path=test_files[0],
                train_start_year=train_start,
                train_end_year=train_end,
                test_year=test_year,
            )
        )
    if not specs:
        raise ValueError(f"No folds found in {data_dir}")
    return specs


def load_fold(spec: FoldSpec) -> FoldData:
    train = read_frame(spec.train_path)
    test = read_frame(spec.test_path)
    validate_temporal_split(train, test, spec)
    feature_columns = get_feature_columns(train)
    if feature_columns != get_feature_columns(test):
        raise ValueError(f"Feature columns differ in {spec.fold}")
    context_path = spec.train_path.parent / CONTEXT_FILE_NAME
    context = read_frame(context_path) if context_path.is_file() else None
    if context is not None:
        validate_test_context(
            train,
            context,
            test,
            feature_columns,
            spec,
        )
    return FoldData(
        spec=spec,
        train=train,
        test=test,
        feature_columns=feature_columns,
        context=context,
    )


def validate_test_context(
    train: pd.DataFrame,
    context: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    spec: FoldSpec,
) -> None:
    """Validate feature-only history that is excluded from supervised fitting."""

    if context.empty:
        raise ValueError(f"{spec.fold} has an empty test context")
    if get_feature_columns(context) != feature_columns:
        raise ValueError(f"{spec.fold} context feature columns differ")
    if train[DATE_COLUMN].max() >= context[DATE_COLUMN].min():
        raise ValueError(f"{spec.fold} context overlaps supervised training")
    if context[DATE_COLUMN].max() >= test[DATE_COLUMN].min():
        raise ValueError(f"{spec.fold} context overlaps evaluation dates")
    combined_dates = pd.concat(
        [
            train[DATE_COLUMN],
            context[DATE_COLUMN],
            test[DATE_COLUMN],
        ],
        ignore_index=True,
    )
    if combined_dates.duplicated().any():
        raise ValueError(f"{spec.fold} context contains duplicate split dates")


def sequence_history_features(fold: FoldData) -> np.ndarray:
    """Return supervised train features plus unsupervised boundary context."""

    frames = [fold.train.loc[:, fold.feature_columns]]
    if fold.context is not None:
        frames.append(fold.context.loc[:, fold.feature_columns])
    return pd.concat(frames, ignore_index=True).to_numpy(dtype=float)


def read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_fold_frame(frame, path)
    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN])
    if LABEL_DATE_COLUMN in result.columns:
        result[LABEL_DATE_COLUMN] = pd.to_datetime(
            result[LABEL_DATE_COLUMN]
        )
    result = result.sort_values(DATE_COLUMN).reset_index(drop=True)
    return result


def validate_fold_frame(frame: pd.DataFrame, path: Path) -> None:
    required = {DATE_COLUMN, TARGET_COLUMN, CLOSE_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} is empty")
    if frame.isna().any().any():
        raise ValueError(f"{path} contains missing values")
    duplicate_dates = frame[DATE_COLUMN].duplicated().sum()
    if duplicate_dates:
        raise ValueError(f"{path} contains duplicate dates")


def validate_temporal_split(train: pd.DataFrame, test: pd.DataFrame, spec: FoldSpec) -> None:
    if train[DATE_COLUMN].max() >= test[DATE_COLUMN].min():
        raise ValueError(f"{spec.fold} has overlapping train and test dates")
    if int(train[DATE_COLUMN].dt.year.min()) != spec.train_start_year:
        raise ValueError(f"{spec.fold} train start year mismatch")
    if int(train[DATE_COLUMN].dt.year.max()) != spec.train_end_year:
        raise ValueError(f"{spec.fold} train end year mismatch")
    if set(test[DATE_COLUMN].dt.year.unique()) != {spec.test_year}:
        raise ValueError(f"{spec.fold} test year mismatch")
    if LABEL_DATE_COLUMN in train.columns:
        label_dates = pd.to_datetime(train[LABEL_DATE_COLUMN], errors="coerce")
        if label_dates.isna().any():
            raise ValueError(f"{spec.fold} contains invalid training label dates")
        if label_dates.max() >= test[DATE_COLUMN].min():
            raise ValueError(
                f"{spec.fold} has training labels observed on or after test start"
            )


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {DATE_COLUMN, LABEL_DATE_COLUMN, TARGET_COLUMN}
    features = [column for column in frame.columns if column not in excluded]
    if not features:
        raise ValueError("No feature columns available")
    return features


def split_xy(fold: FoldData) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    x_train = fold.train.loc[:, fold.feature_columns].copy()
    y_train = fold.train.loc[:, TARGET_COLUMN].copy()
    x_test = fold.test.loc[:, fold.feature_columns].copy()
    y_test = fold.test.loc[:, TARGET_COLUMN].copy()
    return x_train, y_train, x_test, y_test


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)
    if true_values.shape != pred_values.shape:
        raise ValueError("Prediction shape does not match target shape")
    if not np.isfinite(pred_values).all():
        raise ValueError("Predictions contain non-finite values")
    non_zero = true_values != 0
    mape = np.nan
    if non_zero.any():
        mape = float(np.mean(np.abs((true_values[non_zero] - pred_values[non_zero]) / true_values[non_zero])) * 100)
    return {
        "rmse": float(math.sqrt(mean_squared_error(true_values, pred_values))),
        "mae": float(mean_absolute_error(true_values, pred_values)),
        "mape": mape,
        "r2": float(r2_score(true_values, pred_values)),
    }


def binary_direction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_close: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate an Up/Down task with explicit tie and abstention handling.

    True no-change observations are outside the binary estimand. A predicted
    exact no-change is an abstention. Direction coverage uses the number of
    non-tied actual observations as its denominator.
    """

    true_values = np.asarray(y_true, dtype=float)
    pred_values = np.asarray(y_pred, dtype=float)
    close_values = np.asarray(current_close, dtype=float)
    if not (
        true_values.shape == pred_values.shape == close_values.shape
    ):
        raise ValueError("Direction metric inputs must have identical shapes")
    if not (
        np.isfinite(true_values).all()
        and np.isfinite(pred_values).all()
        and np.isfinite(close_values).all()
    ):
        raise ValueError("Direction metric inputs contain non-finite values")

    true_direction = np.sign(true_values - close_values).astype(int)
    pred_direction = np.sign(pred_values - close_values).astype(int)
    actual_binary = true_direction != 0
    prediction_available = pred_direction != 0
    evaluated = actual_binary & prediction_available
    n_actual_binary = int(actual_binary.sum())
    n_evaluated = int(evaluated.sum())
    n_actual_ties = int((~actual_binary).sum())
    n_abstentions = int((actual_binary & ~prediction_available).sum())
    coverage = (
        float(n_evaluated / n_actual_binary)
        if n_actual_binary
        else float("nan")
    )

    if n_evaluated == 0:
        accuracy = float("nan")
        balanced_accuracy = float("nan")
        mcc = float("nan")
        tn = fp = fn = tp = 0
        predicted_up_share = float("nan")
    else:
        evaluated_true = true_direction[evaluated]
        evaluated_pred = pred_direction[evaluated]
        accuracy = float(np.mean(evaluated_true == evaluated_pred))
        balanced_accuracy = float(
            balanced_accuracy_score(evaluated_true, evaluated_pred)
        )
        mcc = float(matthews_corrcoef(evaluated_true, evaluated_pred))
        tn, fp, fn, tp = (
            int(value)
            for value in confusion_matrix(
                evaluated_true,
                evaluated_pred,
                labels=[-1, 1],
            ).ravel()
        )
        predicted_up_share = float(np.mean(evaluated_pred == 1))

    return {
        "direction_accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "direction_coverage": coverage,
        "n_direction_evaluated": n_evaluated,
        "n_actual_ties": n_actual_ties,
        "n_predicted_abstentions": n_abstentions,
        "direction_tn": tn,
        "direction_fp": fp,
        "direction_fn": fn,
        "direction_tp": tp,
        "predicted_up_share": predicted_up_share,
    }


def direction_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_close: np.ndarray,
) -> float:
    return float(
        binary_direction_metrics(y_true, y_pred, current_close)[
            "direction_accuracy"
        ]
    )


def evaluate_predictions(fold: FoldData, y_pred: np.ndarray) -> dict[str, float | str | int]:
    y_true = fold.test[TARGET_COLUMN].to_numpy(dtype=float)
    current_close = fold.test[CLOSE_COLUMN].to_numpy(dtype=float)
    metrics = regression_metrics(y_true, y_pred)
    direction_metrics = binary_direction_metrics(
        y_true,
        y_pred,
        current_close,
    )
    return {
        "fold": fold.spec.fold,
        "train_start_year": fold.spec.train_start_year,
        "train_end_year": fold.spec.train_end_year,
        "test_year": fold.spec.test_year,
        "n_train": int(len(fold.train)),
        "n_test": int(len(fold.test)),
        **metrics,
        **direction_metrics,
    }


def predictions_frame(fold: FoldData, y_pred: np.ndarray) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "Date": fold.test[DATE_COLUMN],
            "Close_D": fold.test[CLOSE_COLUMN],
            "y_true": fold.test[TARGET_COLUMN],
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )
    result["true_direction"] = np.sign(result["y_true"] - result["Close_D"])
    result["pred_direction"] = np.sign(result["y_pred"] - result["Close_D"])
    result["actual_binary_direction"] = result["true_direction"] != 0
    result["prediction_available"] = result["pred_direction"] != 0
    result["direction_evaluated"] = (
        result["actual_binary_direction"] & result["prediction_available"]
    )
    return result


def package_versions(packages: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def save_run_outputs(
    model_name: str,
    fold_metrics: list[dict[str, float | str | int]],
    fold_predictions: dict[str, pd.DataFrame],
    config: dict[str, object],
    packages: list[str],
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(fold_metrics)
    metrics_frame.to_csv(model_dir / "metrics_by_fold.csv", index=False)
    for fold_name, frame in fold_predictions.items():
        frame.to_csv(model_dir / f"predictions_{fold_name}.csv", index=False)
    summary = metrics_frame.select_dtypes(include=[np.number]).mean().to_dict()
    metadata_payload = {
        "model_name": model_name,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
        "packages": package_versions(packages),
        "summary_mean_metrics": summary,
    }
    with (model_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata_payload, file, indent=2)
    return model_dir


PredictFold = Callable[[FoldData], np.ndarray]


def run_model_on_folds(
    model_name: str,
    predict_fold: PredictFold,
    config: dict[str, object],
    packages: list[str],
    data_dir: Path = DATA_FOLDS_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    metrics: list[dict[str, float | str | int]] = []
    predictions: dict[str, pd.DataFrame] = {}
    for spec in tqdm(discover_folds(data_dir), desc=model_name, unit="fold"):
        fold = load_fold(spec)
        y_pred = np.asarray(predict_fold(fold), dtype=float)
        if y_pred.shape != (len(fold.test),):
            raise ValueError(f"{model_name} produced invalid prediction shape for {spec.fold}: {y_pred.shape}")
        metrics.append(evaluate_predictions(fold, y_pred))
        predictions[spec.fold] = predictions_frame(fold, y_pred)
    save_run_outputs(model_name, metrics, predictions, config, packages, output_dir=output_dir)
    return pd.DataFrame(metrics)


def print_metrics(metrics: pd.DataFrame) -> None:
    pd.set_option("display.max_columns", None)
    print(metrics)
    numeric_summary = metrics.select_dtypes(include=[np.number]).mean().to_frame("mean").T
    print(numeric_summary)


def dataclass_dict(instance: object) -> dict[str, object]:
    return asdict(instance)
