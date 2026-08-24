from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_common import FoldData, FoldSpec
from models.paper_benchmarks_v2 import (
    predict_previous_day_direction,
    predict_training_majority_direction,
    predict_persistence,
    predict_ridge,
    summarize_benchmarks,
)


def _fold() -> FoldData:
    train = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03"]
            ),
            "Label_Date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-06"]
            ),
            "Close_D": [10.0, 11.0, 12.0],
            "Feature_A": [1.0, 2.0, 3.0],
            "Target_Next_Close": [11.0, 12.0, 13.0],
        }
    )
    test = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2021-01-04", "2021-01-05"]),
            "Label_Date": pd.to_datetime(["2021-01-05", "2021-01-06"]),
            "Close_D": [13.0, 14.0],
            "Feature_A": [4.0, 5.0],
            "Target_Next_Close": [14.0, 15.0],
        }
    )
    spec = FoldSpec(
        fold="fold_1",
        train_path=Path("train.csv"),
        test_path=Path("test.csv"),
        train_start_year=2020,
        train_end_year=2020,
        test_year=2021,
    )
    return FoldData(
        spec=spec,
        train=train,
        test=test,
        feature_columns=["Close_D", "Feature_A"],
    )


def test_persistence_predicts_the_current_close() -> None:
    prediction = predict_persistence(_fold())

    assert prediction.tolist() == [13.0, 14.0]


def test_ridge_predictor_returns_one_value_per_test_observation() -> None:
    prediction = predict_ridge(_fold(), alpha=1.0)

    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()


def test_training_majority_direction_forecasts_a_nonzero_move() -> None:
    prediction = predict_training_majority_direction(_fold())

    assert (prediction > _fold().test["Close_D"].to_numpy()).all()


def test_previous_day_direction_uses_lagged_close() -> None:
    fold = _fold()
    fold.test["Close_D_lag1"] = [12.0, 15.0]
    fold.feature_columns.append("Close_D_lag1")

    prediction = predict_previous_day_direction(fold)

    assert prediction[0] > fold.test["Close_D"].iloc[0]
    assert prediction[1] < fold.test["Close_D"].iloc[1]


def test_summarize_benchmarks_keeps_fold_level_mean_and_std() -> None:
    metrics = pd.DataFrame(
        {
            "model": ["ridge", "ridge"],
            "fold": ["fold_1", "fold_2"],
            "rmse": [2.0, 4.0],
            "mae": [1.0, 3.0],
            "mape": [0.1, 0.3],
            "r2": [0.2, 0.4],
            "direction_accuracy": [0.5, 0.7],
            "balanced_accuracy": [0.45, 0.65],
            "mcc": [0.0, 0.2],
            "direction_coverage": [1.0, 1.0],
            "runtime_seconds": [0.01, 0.02],
        }
    )

    summary = summarize_benchmarks(metrics)

    assert summary.loc[0, "rmse_mean"] == 3.0
    assert summary.loc[0, "direction_accuracy_mean"] == 0.6
    assert summary.loc[0, "outer_folds"] == 2
