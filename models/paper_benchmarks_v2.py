from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.baseline_common import (
    CLOSE_COLUMN,
    PROJECT_ROOT,
    FoldData,
    discover_folds,
    evaluate_predictions,
    load_fold,
    predictions_frame,
    save_run_outputs,
    split_xy,
)
from models.full_ta_feature_pool import (
    FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR,
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "paper_benchmarks_point_in_time_v2"
)
SUMMARY_FILE = OUTPUT_DIR / "paper_benchmark_summary.csv"
METRICS = (
    "rmse",
    "mae",
    "mape",
    "r2",
    "direction_accuracy",
    "balanced_accuracy",
    "mcc",
    "direction_coverage",
    "runtime_seconds",
)


def predict_persistence(fold: FoldData) -> np.ndarray:
    return fold.test[CLOSE_COLUMN].to_numpy(dtype=float)


def _training_move_magnitude(fold: FoldData) -> float:
    moves = (
        fold.train["Target_Next_Close"].to_numpy(dtype=float)
        - fold.train[CLOSE_COLUMN].to_numpy(dtype=float)
    )
    nonzero = np.abs(moves[moves != 0.0])
    return float(np.median(nonzero)) if len(nonzero) else 1.0


def predict_training_majority_direction(fold: FoldData) -> np.ndarray:
    moves = (
        fold.train["Target_Next_Close"].to_numpy(dtype=float)
        - fold.train[CLOSE_COLUMN].to_numpy(dtype=float)
    )
    binary_moves = moves[moves != 0.0]
    if not len(binary_moves):
        raise ValueError("Training fold has no non-tied direction labels")
    direction = 1.0 if np.mean(binary_moves > 0.0) >= 0.5 else -1.0
    return (
        fold.test[CLOSE_COLUMN].to_numpy(dtype=float)
        + direction * _training_move_magnitude(fold)
    )


def predict_previous_day_direction(fold: FoldData) -> np.ndarray:
    lag_column = "Close_D_lag1"
    if lag_column not in fold.test.columns:
        raise ValueError(f"Fold is missing {lag_column}")
    direction = np.sign(
        fold.test[CLOSE_COLUMN].to_numpy(dtype=float)
        - fold.test[lag_column].to_numpy(dtype=float)
    )
    direction[direction == 0.0] = 1.0
    return (
        fold.test[CLOSE_COLUMN].to_numpy(dtype=float)
        + direction * _training_move_magnitude(fold)
    )


def predict_ridge(fold: FoldData, *, alpha: float = 1.0) -> np.ndarray:
    x_train, y_train, x_test, _ = split_xy(fold)
    model = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=float)


Predictor = Callable[[FoldData], np.ndarray]


def run_benchmark(
    model_name: str,
    predictor: Predictor,
    *,
    config: dict[str, object],
    data_dir: Path = FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    metrics: list[dict[str, object]] = []
    predictions: dict[str, pd.DataFrame] = {}
    for spec in discover_folds(data_dir):
        fold = load_fold(spec)
        started = time.perf_counter()
        prediction = np.asarray(predictor(fold), dtype=float)
        runtime_seconds = time.perf_counter() - started
        if prediction.shape != (len(fold.test),):
            raise ValueError(
                f"{model_name} returned {prediction.shape} for {spec.fold}"
            )
        metrics.append(
            {
                "model": model_name,
                **evaluate_predictions(fold, prediction),
                "runtime_seconds": runtime_seconds,
            }
        )
        predictions[spec.fold] = predictions_frame(fold, prediction)

    save_run_outputs(
        model_name,
        metrics,
        predictions,
        {
            "protocol_version": "point_in_time_v2",
            "data_dir": str(data_dir),
            "label_purge_rule": (
                "retain Label_Date < first evaluation Date"
            ),
            "direction_contract": (
                "binary Up/Down; actual ties excluded; predicted no-change "
                "is an abstention"
            ),
            **config,
        },
        ["numpy", "pandas", "scikit-learn"],
        output_dir=output_dir,
    )
    return pd.DataFrame(metrics)


def summarize_benchmarks(metrics: pd.DataFrame) -> pd.DataFrame:
    missing = sorted({"model", "fold", *METRICS}.difference(metrics.columns))
    if missing:
        raise ValueError(f"Benchmark metrics are missing columns: {missing}")
    summary = metrics.groupby("model", sort=False)[list(METRICS)].agg(
        ["mean", "std"]
    )
    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]
    summary = summary.reset_index()
    folds = (
        metrics.groupby("model", sort=False)["fold"]
        .nunique()
        .rename("outer_folds")
        .reset_index()
    )
    return summary.merge(folds, on="model", validate="one_to_one")


def run_all_benchmarks(
    *,
    data_dir: Path = FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    persistence = run_benchmark(
        "persistence",
        predict_persistence,
        config={"forecast": "y_hat_(t+1) = Close_t"},
        data_dir=data_dir,
        output_dir=output_dir,
    )
    ridge = run_benchmark(
        "ridge_alpha_1",
        lambda fold: predict_ridge(fold, alpha=1.0),
        config={
            "estimator": "StandardScaler + Ridge",
            "alpha": 1.0,
            "hyperparameter_status": "fixed comparator; not tuned",
        },
        data_dir=data_dir,
        output_dir=output_dir,
    )
    majority = run_benchmark(
        "training_majority_direction",
        predict_training_majority_direction,
        config={
            "direction_rule": "majority non-tied training direction",
            "move_size": "median absolute nonzero training move",
        },
        data_dir=data_dir,
        output_dir=output_dir,
    )
    previous = run_benchmark(
        "previous_day_direction",
        predict_previous_day_direction,
        config={
            "direction_rule": "sign(Close_t - Close_(t-1))",
            "move_size": "median absolute nonzero training move",
        },
        data_dir=data_dir,
        output_dir=output_dir,
    )
    metrics = pd.concat(
        [persistence, ridge, majority, previous],
        ignore_index=True,
    )
    summary = summarize_benchmarks(metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    summary.to_csv(output_dir / SUMMARY_FILE.name, index=False)
    return metrics, summary


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, summary = run_all_benchmarks()
    print(summary.to_string(index=False))
    return metrics, summary


if __name__ == "__main__":
    main()
