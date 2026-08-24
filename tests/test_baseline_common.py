from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.baseline_common import (
    CONTEXT_FILE_NAME,
    FoldData,
    FoldSpec,
    TARGET_COLUMN,
    binary_direction_metrics,
    direction_accuracy,
    get_feature_columns,
    load_fold,
    regression_metrics,
    run_model_on_folds,
    validate_fold_frame,
)


def test_get_feature_columns_excludes_date_and_target():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2),
            "Label_Date": pd.date_range("2024-01-02", periods=2),
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


def test_direction_accuracy_excludes_true_no_change_from_binary_task():
    current_close = np.array([10.0, 10.0, 10.0, 10.0])
    y_true = np.array([11.0, 9.0, 10.0, 12.0])
    y_pred = np.array([12.0, 8.0, 9.0, 11.0])

    assert direction_accuracy(y_true, y_pred, current_close) == 1.0


def test_binary_direction_metrics_exclude_ties_and_abstentions():
    metrics = binary_direction_metrics(
        y_true=np.array([101.0, 100.0, 99.0, 101.0]),
        y_pred=np.array([102.0, 99.0, 98.0, 100.0]),
        current_close=np.array([100.0, 100.0, 100.0, 100.0]),
    )

    assert metrics["direction_accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)
    assert metrics["direction_coverage"] == pytest.approx(2 / 3)
    assert metrics["n_direction_evaluated"] == 2
    assert metrics["n_actual_ties"] == 1
    assert metrics["n_predicted_abstentions"] == 1
    assert metrics["direction_tn"] == 1
    assert metrics["direction_tp"] == 1


def test_binary_direction_metrics_report_zero_coverage_for_no_change_forecast():
    metrics = binary_direction_metrics(
        y_true=np.array([101.0, 99.0]),
        y_pred=np.array([100.0, 100.0]),
        current_close=np.array([100.0, 100.0]),
    )

    assert np.isnan(metrics["direction_accuracy"])
    assert np.isnan(metrics["balanced_accuracy"])
    assert np.isnan(metrics["mcc"])
    assert metrics["direction_coverage"] == pytest.approx(0.0)
    assert metrics["n_predicted_abstentions"] == 2


def test_load_fold_keeps_boundary_features_as_unsupervised_test_context(
    tmp_path,
):
    fold_dir = tmp_path / "fold_1"
    fold_dir.mkdir()
    train = pd.DataFrame(
        {
            "Date": ["2020-12-29"],
            "Label_Date": ["2020-12-30"],
            "Close_D": [98.0],
            TARGET_COLUMN: [99.0],
        }
    )
    context = pd.DataFrame(
        {
            "Date": ["2020-12-30"],
            "Label_Date": ["2021-01-04"],
            "Close_D": [99.0],
            TARGET_COLUMN: [100.0],
        }
    )
    test = pd.DataFrame(
        {
            "Date": ["2021-01-04"],
            "Label_Date": ["2021-01-05"],
            "Close_D": [100.0],
            TARGET_COLUMN: [101.0],
        }
    )
    train_path = fold_dir / "train_2020_2020.csv"
    test_path = fold_dir / "test_2021.csv"
    train.to_csv(train_path, index=False)
    context.to_csv(fold_dir / CONTEXT_FILE_NAME, index=False)
    test.to_csv(test_path, index=False)
    spec = FoldSpec(
        "fold_1",
        train_path,
        test_path,
        2020,
        2020,
        2021,
    )

    fold = load_fold(spec)

    assert fold.context is not None
    assert fold.context["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-12-30"
    ]
    assert fold.train["Date"].max() < fold.context["Date"].min()
    assert fold.context["Date"].max() < fold.test["Date"].min()


def test_run_model_on_folds_reports_progress(tmp_path, monkeypatch):
    data_dir = tmp_path / "folds"
    fold_dir = data_dir / "fold_1"
    fold_dir.mkdir(parents=True)
    train = pd.DataFrame(
        {
            "Date": ["2022-01-03", "2022-01-04"],
            "Close_D": [10.0, 11.0],
            TARGET_COLUMN: [11.0, 12.0],
        }
    )
    test = pd.DataFrame(
        {
            "Date": ["2023-01-03"],
            "Close_D": [12.0],
            TARGET_COLUMN: [13.0],
        }
    )
    train.to_csv(fold_dir / "train_2022_2022.csv", index=False)
    test.to_csv(fold_dir / "test_2023.csv", index=False)
    progress_calls = []

    def fake_progress(items, **kwargs):
        progress_calls.append(kwargs)
        return items

    monkeypatch.setattr("models.baseline_common.tqdm", fake_progress)

    run_model_on_folds(
        "progress_test",
        lambda fold: np.array([13.0]),
        {},
        [],
        data_dir=data_dir,
        output_dir=tmp_path / "outputs",
    )

    assert progress_calls == [{"desc": "progress_test", "unit": "fold"}]
