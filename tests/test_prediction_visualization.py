from pathlib import Path

import pandas as pd

from models.prediction_visualization import (
    discover_experiments,
    load_predictions,
    regression_metrics,
)


def _write_predictions(path: Path, year: int) -> None:
    pd.DataFrame(
        {
            "Date": [f"{year}-01-03", f"{year}-01-02"],
            "Close_D": [100.0, 99.0],
            "y_true": [101.0, 100.0],
            "y_pred": [100.5, 100.5],
            "true_direction": [1.0, 1.0],
            "pred_direction": [1.0, 1.0],
        }
    ).to_csv(path, index=False)


def test_discover_experiments_matches_model_family(tmp_path: Path) -> None:
    experiment = tmp_path / "outputs" / "baseline_naive" / "xgboost"
    experiment.mkdir(parents=True)
    _write_predictions(experiment / "predictions_fold_1.csv", 2022)
    unrelated = tmp_path / "outputs" / "baseline_naive" / "lightgbm"
    unrelated.mkdir(parents=True)
    _write_predictions(unrelated / "predictions_fold_1.csv", 2022)

    found = discover_experiments(tmp_path, ("xgboost",))

    assert found == [experiment]


def test_load_predictions_combines_folds_in_date_order(tmp_path: Path) -> None:
    experiment = tmp_path / "model"
    experiment.mkdir()
    _write_predictions(experiment / "predictions_fold_2.csv", 2023)
    _write_predictions(experiment / "predictions_fold_1.csv", 2022)

    predictions = load_predictions(experiment)

    assert predictions["fold"].tolist() == [1, 1, 2, 2]
    assert predictions["Date"].is_monotonic_increasing


def test_regression_metrics_returns_expected_values() -> None:
    predictions = pd.DataFrame(
        {
            "y_true": [100.0, 102.0],
            "y_pred": [101.0, 101.0],
            "true_direction": [1.0, -1.0],
            "pred_direction": [1.0, 1.0],
        }
    )

    metrics = regression_metrics(predictions)

    assert metrics["rmse"] == 1.0
    assert metrics["mae"] == 1.0
    assert metrics["direction_accuracy"] == 0.5
