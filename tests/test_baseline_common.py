from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_common import (
    TARGET_COLUMN,
    direction_accuracy,
    get_feature_columns,
    regression_metrics,
    validate_fold_frame,
)


def test_get_feature_columns_excludes_date_and_target():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2),
            "Close_D": [1.0, 2.0],
            "Open_D": [0.5, 1.5],
            TARGET_COLUMN: [2.0, 3.0],
        }
    )

    assert get_feature_columns(frame) == ["Close_D", "Open_D"]


def test_validate_fold_frame_rejects_missing_target():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2),
            "Close_D": [1.0, 2.0],
        }
    )

    try:
        validate_fold_frame(frame, Path("missing.csv"))
    except ValueError as error:
        assert "Target_Next_Close" in str(error)
    else:
        raise AssertionError("validate_fold_frame should reject missing target")


def test_regression_metrics_has_expected_values():
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([1.0, 5.0, 6.0])

    metrics = regression_metrics(y_true, y_pred)

    assert round(metrics["rmse"], 6) == round(np.sqrt(2 / 3), 6)
    assert round(metrics["mae"], 6) == round(2 / 3, 6)
    assert "r2" in metrics


def test_direction_accuracy_compares_close_to_next_close_moves():
    current_close = np.array([10.0, 10.0, 10.0, 10.0])
    y_true = np.array([11.0, 9.0, 10.0, 12.0])
    y_pred = np.array([12.0, 8.0, 9.0, 11.0])

    assert direction_accuracy(y_true, y_pred, current_close) == 0.75
