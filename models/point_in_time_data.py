from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATE_COLUMN = "Date"
LABEL_DATE_COLUMN = "Label_Date"
TARGET_COLUMN = "Target_Next_Close"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIGNED_DATA_PATH = PROJECT_ROOT / "data-prepared" / "SET50_aligned.csv"
DAILY_DATA_PATH = PROJECT_ROOT / "data-prepared" / "SET50_days.csv"
POINT_IN_TIME_DATA_FOLDS_DIR = PROJECT_ROOT / "data-folds-point-in-time-v2"
POINT_IN_TIME_CONTRACT_NAME = "point_in_time_contract.json"
MARKET_MASTER_NAME = "market_master_with_label_dates.csv"
CONTEXT_FILE_NAME = "context_before_test.csv"


@dataclass(frozen=True)
class PurgeAudit:
    original_rows: int
    retained_rows: int
    removed_rows: int
    test_start: str
    maximum_original_label_date: str
    maximum_retained_label_date: str


def _validated_dates(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> pd.DataFrame:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required date columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} is empty")

    result = frame.copy()
    for column in columns:
        result[column] = pd.to_datetime(result[column], errors="coerce")
        if result[column].isna().any():
            raise ValueError(f"{name} contains invalid {column} values")
    return result


def purge_cross_boundary_training_labels(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> tuple[pd.DataFrame, PurgeAudit]:
    """Keep only training rows whose target was observable before evaluation.

    For a one-day-ahead target, the feature date alone is insufficient to
    establish a clean temporal split. A training observation is admissible only
    when its label timestamp is strictly earlier than the first evaluation
    feature timestamp.
    """

    train_dates = _validated_dates(
        train,
        (DATE_COLUMN, LABEL_DATE_COLUMN),
        "training frame",
    )
    evaluation_dates = _validated_dates(
        evaluation,
        (DATE_COLUMN,),
        "evaluation frame",
    )
    test_start = evaluation_dates[DATE_COLUMN].min()
    retained = train_dates.loc[
        train_dates[LABEL_DATE_COLUMN] < test_start
    ].copy()
    if retained.empty:
        raise ValueError("Point-in-time purge removed every training row")

    retained = retained.sort_values(DATE_COLUMN).reset_index(drop=True)
    audit = PurgeAudit(
        original_rows=int(len(train_dates)),
        retained_rows=int(len(retained)),
        removed_rows=int(len(train_dates) - len(retained)),
        test_start=test_start.strftime("%Y-%m-%d"),
        maximum_original_label_date=train_dates[LABEL_DATE_COLUMN]
        .max()
        .strftime("%Y-%m-%d"),
        maximum_retained_label_date=retained[LABEL_DATE_COLUMN]
        .max()
        .strftime("%Y-%m-%d"),
    )
    return retained, audit


def attach_label_dates(
    aligned: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Attach and verify the timestamp at which each next-day label is known."""

    aligned_dates = _validated_dates(
        aligned,
        (DATE_COLUMN,),
        "aligned market frame",
    )
    daily_dates = _validated_dates(
        daily,
        (DATE_COLUMN,),
        "daily market frame",
    )
    required_aligned = {"Close_D", TARGET_COLUMN}
    required_daily = {"Close"}
    missing_aligned = sorted(required_aligned.difference(aligned_dates.columns))
    missing_daily = sorted(required_daily.difference(daily_dates.columns))
    if missing_aligned:
        raise ValueError(
            f"aligned market frame is missing columns: {missing_aligned}"
        )
    if missing_daily:
        raise ValueError(
            f"daily market frame is missing columns: {missing_daily}"
        )
    if aligned_dates[DATE_COLUMN].duplicated().any():
        raise ValueError("aligned market frame contains duplicate dates")
    if daily_dates[DATE_COLUMN].duplicated().any():
        raise ValueError("daily market frame contains duplicate dates")

    calendar = daily_dates.sort_values(DATE_COLUMN).reset_index(drop=True)
    calendar[LABEL_DATE_COLUMN] = calendar[DATE_COLUMN].shift(-1)
    calendar["_Label_Close"] = pd.to_numeric(
        calendar["Close"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).shift(-1)
    lookup = calendar.loc[
        :,
        [DATE_COLUMN, LABEL_DATE_COLUMN, "_Label_Close"],
    ]
    result = aligned_dates.merge(
        lookup,
        on=DATE_COLUMN,
        how="left",
        validate="one_to_one",
    )
    if result[[LABEL_DATE_COLUMN, "_Label_Close"]].isna().any().any():
        raise ValueError(
            "At least one aligned row has no next observed market day"
        )

    target = pd.to_numeric(
        result[TARGET_COLUMN]
        .astype(str)
        .str.replace(",", "", regex=False),
        errors="coerce",
    ).to_numpy()
    label_close = result["_Label_Close"].to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("aligned market frame contains invalid targets")
    if not np.allclose(target, label_close, rtol=0.0, atol=1e-8):
        mismatch_count = int(
            (~np.isclose(target, label_close, rtol=0.0, atol=1e-8)).sum()
        )
        raise ValueError(
            f"{mismatch_count} target value(s) does not equal the close on "
            "its attached Label_Date"
        )

    result = result.drop(columns=["_Label_Close"])
    ordered = [
        DATE_COLUMN,
        LABEL_DATE_COLUMN,
        *[
            column
            for column in result.columns
            if column not in {DATE_COLUMN, LABEL_DATE_COLUMN}
        ],
    ]
    return result.loc[:, ordered].sort_values(DATE_COLUMN).reset_index(drop=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_test_years(test_years: Iterable[int]) -> tuple[int, ...]:
    years = tuple(test_years)
    if not years:
        raise ValueError("At least one test year is required")
    if any(isinstance(year, bool) or not isinstance(year, int) for year in years):
        raise ValueError("Test years must be integers")
    if years != tuple(sorted(set(years))):
        raise ValueError("Test years must be unique and increasing")
    return years


def create_point_in_time_market_folds(
    *,
    aligned_path: Path = ALIGNED_DATA_PATH,
    daily_path: Path = DAILY_DATA_PATH,
    output_dir: Path = POINT_IN_TIME_DATA_FOLDS_DIR,
    test_years: Iterable[int] = (2022, 2023, 2024, 2025),
) -> Path:
    """Create expanding-window folds with strict label-time purging."""

    years = _validated_test_years(test_years)
    aligned = pd.read_csv(aligned_path)
    daily = pd.read_csv(daily_path)
    master = attach_label_dates(aligned, daily)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_master = master.copy()
    saved_master[DATE_COLUMN] = saved_master[DATE_COLUMN].dt.strftime(
        "%Y-%m-%d"
    )
    saved_master[LABEL_DATE_COLUMN] = saved_master[
        LABEL_DATE_COLUMN
    ].dt.strftime("%Y-%m-%d")
    master_path = output_dir / MARKET_MASTER_NAME
    saved_master.to_csv(master_path, index=False)
    fold_metadata: list[dict[str, object]] = []

    for fold_index, test_year in enumerate(years, start=1):
        train = master.loc[master[DATE_COLUMN].dt.year < test_year].copy()
        test = master.loc[master[DATE_COLUMN].dt.year == test_year].copy()
        if train.empty or test.empty:
            raise ValueError(f"Cannot create market fold for {test_year}")
        purged_train, audit = purge_cross_boundary_training_labels(train, test)
        removed = train.loc[
            train[LABEL_DATE_COLUMN]
            >= pd.to_datetime(audit.test_start)
        ]

        train_start_year = int(purged_train[DATE_COLUMN].dt.year.min())
        train_end_year = int(purged_train[DATE_COLUMN].dt.year.max())
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_path = (
            fold_dir / f"train_{train_start_year}_{train_end_year}.csv"
        )
        test_path = fold_dir / f"test_{test_year}.csv"
        saved_train = purged_train.copy()
        saved_context = removed.copy()
        saved_test = test.copy()
        for frame in (saved_train, saved_context, saved_test):
            frame[DATE_COLUMN] = frame[DATE_COLUMN].dt.strftime("%Y-%m-%d")
            frame[LABEL_DATE_COLUMN] = frame[LABEL_DATE_COLUMN].dt.strftime(
                "%Y-%m-%d"
            )
        saved_train.to_csv(train_path, index=False)
        context_path = fold_dir / CONTEXT_FILE_NAME
        saved_context.to_csv(context_path, index=False)
        saved_test.to_csv(test_path, index=False)

        contract = {
            "protocol_version": "point_in_time_v2",
            **asdict(audit),
            "purge_rule": "retain Label_Date < first evaluation Date",
            "removed_feature_dates": removed[DATE_COLUMN]
            .dt.strftime("%Y-%m-%d")
            .tolist(),
            "removed_label_dates": removed[LABEL_DATE_COLUMN]
            .dt.strftime("%Y-%m-%d")
            .tolist(),
            "train_file": train_path.name,
            "context_file": context_path.name,
            "test_file": test_path.name,
            "train_sha256": _sha256(train_path),
            "context_sha256": _sha256(context_path),
            "test_sha256": _sha256(test_path),
        }
        contract_path = fold_dir / POINT_IN_TIME_CONTRACT_NAME
        contract_path.write_text(
            json.dumps(contract, indent=2),
            encoding="utf-8",
        )
        fold_metadata.append(
            {
                "fold": fold_dir.name,
                "test_year": test_year,
                **contract,
            }
        )

    run_metadata = {
        "protocol_version": "point_in_time_v2",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "aligned_source": str(aligned_path),
        "daily_calendar_source": str(daily_path),
        "aligned_source_sha256": _sha256(aligned_path),
        "daily_calendar_source_sha256": _sha256(daily_path),
        "market_master": str(master_path),
        "market_master_sha256": _sha256(master_path),
        "label_definition": "Target_Next_Close observed at Label_Date",
        "purge_rule": "retain Label_Date < first evaluation Date",
        "test_years": list(years),
        "folds": fold_metadata,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )
    return output_dir


def main() -> Path:
    output_dir = create_point_in_time_market_folds()
    print(f"Created point-in-time folds at {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
