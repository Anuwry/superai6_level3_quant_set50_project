from __future__ import annotations

import pandas as pd
import pytest

from models.point_in_time_data import (
    CONTEXT_FILE_NAME,
    LABEL_DATE_COLUMN,
    attach_label_dates,
    create_point_in_time_market_folds,
    purge_cross_boundary_training_labels,
)


def _frame(dates: list[str], label_dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            LABEL_DATE_COLUMN: pd.to_datetime(label_dates),
            "Close_D": [100.0 + index for index in range(len(dates))],
            "Target_Next_Close": [
                101.0 + index for index in range(len(dates))
            ],
        }
    )


def test_purge_cross_boundary_training_labels_removes_test_day_label() -> None:
    train = _frame(
        ["2021-12-28", "2021-12-29", "2021-12-30"],
        ["2021-12-29", "2021-12-30", "2022-01-04"],
    )
    test = _frame(
        ["2022-01-04", "2022-01-05"],
        ["2022-01-05", "2022-01-06"],
    )

    purged, audit = purge_cross_boundary_training_labels(train, test)

    assert purged["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2021-12-28",
        "2021-12-29",
    ]
    assert audit.removed_rows == 1
    assert audit.test_start == "2022-01-04"
    assert audit.maximum_retained_label_date == "2021-12-30"


def test_purge_cross_boundary_training_labels_requires_label_dates() -> None:
    train = _frame(
        ["2021-12-29", "2021-12-30"],
        ["2021-12-30", "2022-01-04"],
    ).drop(columns=[LABEL_DATE_COLUMN])
    test = _frame(["2022-01-04"], ["2022-01-05"])

    with pytest.raises(ValueError, match="Label_Date"):
        purge_cross_boundary_training_labels(train, test)


def test_purge_cross_boundary_training_labels_rejects_empty_result() -> None:
    train = _frame(["2021-12-30"], ["2022-01-04"])
    test = _frame(["2022-01-04"], ["2022-01-05"])

    with pytest.raises(ValueError, match="removed every training row"):
        purge_cross_boundary_training_labels(train, test)


def test_attach_label_dates_uses_next_observed_market_day() -> None:
    aligned = pd.DataFrame(
        {
            "Date": ["2021-12-30", "2022-01-04"],
            "Close_D": [100.0, 101.0],
            "Target_Next_Close": [101.0, 103.0],
        }
    )
    daily = pd.DataFrame(
        {
            "Date": ["2021-12-30", "2022-01-04", "2022-01-05"],
            "Close": [100.0, 101.0, 103.0],
        }
    )

    result = attach_label_dates(aligned, daily)

    assert result[LABEL_DATE_COLUMN].dt.strftime("%Y-%m-%d").tolist() == [
        "2022-01-04",
        "2022-01-05",
    ]


def test_attach_label_dates_rejects_target_that_does_not_match_label_day() -> None:
    aligned = pd.DataFrame(
        {
            "Date": ["2021-12-30"],
            "Close_D": [100.0],
            "Target_Next_Close": [999.0],
        }
    )
    daily = pd.DataFrame(
        {
            "Date": ["2021-12-30", "2022-01-04"],
            "Close": [100.0, 101.0],
        }
    )

    with pytest.raises(ValueError, match="does not equal"):
        attach_label_dates(aligned, daily)


def test_create_point_in_time_market_folds_writes_purge_audit(tmp_path) -> None:
    aligned = pd.DataFrame(
        {
            "Date": [
                "2020-12-29",
                "2020-12-30",
                "2021-01-04",
                "2021-01-05",
            ],
            "Close_D": [98.0, 99.0, 100.0, 101.0],
            "Target_Next_Close": [99.0, 100.0, 101.0, 102.0],
        }
    )
    daily = pd.DataFrame(
        {
            "Date": [
                "2020-12-29",
                "2020-12-30",
                "2021-01-04",
                "2021-01-05",
                "2021-01-06",
            ],
            "Close": [98.0, 99.0, 100.0, 101.0, 102.0],
        }
    )
    aligned_path = tmp_path / "aligned.csv"
    daily_path = tmp_path / "daily.csv"
    output_dir = tmp_path / "folds"
    aligned.to_csv(aligned_path, index=False)
    daily.to_csv(daily_path, index=False)

    create_point_in_time_market_folds(
        aligned_path=aligned_path,
        daily_path=daily_path,
        output_dir=output_dir,
        test_years=(2021,),
    )

    train = pd.read_csv(output_dir / "fold_1" / "train_2020_2020.csv")
    context = pd.read_csv(output_dir / "fold_1" / CONTEXT_FILE_NAME)
    master = pd.read_csv(output_dir / "market_master_with_label_dates.csv")
    audit = pd.read_json(
        output_dir / "fold_1" / "point_in_time_contract.json",
        typ="series",
    )
    assert train["Date"].tolist() == ["2020-12-29"]
    assert context["Date"].tolist() == ["2020-12-30"]
    assert context[LABEL_DATE_COLUMN].tolist() == ["2021-01-04"]
    assert "2020-12-30" in master["Date"].tolist()
    assert audit["removed_rows"] == 1
    assert audit["test_start"] == "2021-01-04"
