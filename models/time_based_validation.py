from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models.baseline_common import DATE_COLUMN, PROJECT_ROOT, discover_folds
from models.full_ta_feature_pool import (
    FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR,
    SCALER_METADATA_NAME,
    create_full_ta_folds,
)
from models.neural_network_folds import numeric_columns
from models.point_in_time_data import (
    LABEL_DATE_COLUMN,
    purge_cross_boundary_training_labels,
)

LEGACY_FULL_TA_TUNING_DIR = PROJECT_ROOT / "data-folds-full-ta-validation"
LEGACY_FULL_TA_TUNING_NN_DIR = (
    PROJECT_ROOT / "data-folds-full-ta-validation-nn"
)
FULL_TA_TUNING_DIR = (
    PROJECT_ROOT / "data-folds-full-ta-validation-point-in-time-v2"
)
FULL_TA_TUNING_NN_DIR = (
    PROJECT_ROOT / "data-folds-full-ta-validation-point-in-time-v2-nn"
)


@dataclass(frozen=True)
class ValidationFoldSpec:
    fold: str
    train_path: Path
    validation_path: Path
    test_path: Path
    train_start_year: int
    train_end_year: int
    validation_year: int
    test_year: int


def discover_validation_folds(data_dir: Path) -> list[ValidationFoldSpec]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing validation data folder: {data_dir}")
    specs = []
    for fold_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        train_files = sorted(fold_dir.glob("train_*.csv"))
        validation_files = sorted(fold_dir.glob("validation_*.csv"))
        test_files = sorted(fold_dir.glob("test_*.csv"))
        if len(train_files) != 1 or len(validation_files) != 1 or len(test_files) != 1:
            raise ValueError(f"Expected one train, validation, and test file in {fold_dir}")
        train_years = train_files[0].stem.removeprefix("train_").split("_")
        specs.append(
            ValidationFoldSpec(
                fold=fold_dir.name,
                train_path=train_files[0],
                validation_path=validation_files[0],
                test_path=test_files[0],
                train_start_year=int(train_years[0]),
                train_end_year=int(train_years[1]),
                validation_year=int(validation_files[0].stem.removeprefix("validation_")),
                test_year=int(test_files[0].stem.removeprefix("test_")),
            )
        )
    if not specs:
        raise ValueError(f"No validation folds found in {data_dir}")
    return specs


def create_full_ta_validation_folds(
    source_dir: Path = FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR,
    output_dir: Path = FULL_TA_TUNING_DIR,
) -> Path:
    if not source_dir.exists():
        create_full_ta_folds(output_dir=source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for outer_spec in discover_folds(source_dir):
        outer_train = pd.read_csv(
            outer_spec.train_path,
            parse_dates=[DATE_COLUMN, LABEL_DATE_COLUMN],
        )
        outer_test = pd.read_csv(outer_spec.test_path)
        validation_year = outer_spec.train_end_year
        inner_train = outer_train[outer_train[DATE_COLUMN].dt.year < validation_year].copy()
        validation = outer_train[outer_train[DATE_COLUMN].dt.year == validation_year].copy()
        if inner_train.empty or validation.empty:
            raise ValueError(f"Cannot create validation split for {outer_spec.fold}")
        inner_train, _ = purge_cross_boundary_training_labels(
            inner_train,
            validation,
        )
        inner_train[DATE_COLUMN] = inner_train[DATE_COLUMN].dt.strftime("%Y-%m-%d")
        inner_train[LABEL_DATE_COLUMN] = inner_train[
            LABEL_DATE_COLUMN
        ].dt.strftime("%Y-%m-%d")
        validation[DATE_COLUMN] = validation[DATE_COLUMN].dt.strftime("%Y-%m-%d")
        validation[LABEL_DATE_COLUMN] = validation[
            LABEL_DATE_COLUMN
        ].dt.strftime("%Y-%m-%d")
        fold_dir = output_dir / outer_spec.fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        inner_train.to_csv(
            fold_dir / f"train_{outer_spec.train_start_year}_{validation_year - 1}.csv",
            index=False,
        )
        validation.to_csv(fold_dir / f"validation_{validation_year}.csv", index=False)
        outer_test.to_csv(fold_dir / outer_spec.test_path.name, index=False)
    return output_dir


def create_scaled_full_ta_validation_folds(
    source_dir: Path = FULL_TA_TUNING_DIR,
    output_dir: Path = FULL_TA_TUNING_NN_DIR,
) -> Path:
    if not source_dir.exists():
        create_full_ta_validation_folds(output_dir=source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in discover_validation_folds(source_dir):
        train = pd.read_csv(spec.train_path)
        validation = pd.read_csv(spec.validation_path)
        test = pd.read_csv(spec.test_path)
        columns = numeric_columns(train)
        scaler = MinMaxScaler()
        scaled_train = train.copy()
        scaled_validation = validation.copy()
        scaled_test = test.copy()
        scaled_train.loc[:, columns] = scaler.fit_transform(train.loc[:, columns])
        scaled_validation.loc[:, columns] = scaler.transform(validation.loc[:, columns])
        scaled_test.loc[:, columns] = scaler.transform(test.loc[:, columns])
        fold_dir = output_dir / spec.fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        scaled_train.to_csv(fold_dir / spec.train_path.name, index=False)
        scaled_validation.to_csv(fold_dir / spec.validation_path.name, index=False)
        scaled_test.to_csv(fold_dir / spec.test_path.name, index=False)
        metadata = {
            "scaler": "MinMaxScaler",
            "feature_range": [0, 1],
            "fit_scope": "inner_train_only",
            "source_dir": str(source_dir),
            "validation_year": spec.validation_year,
            "test_year": spec.test_year,
            "columns": columns,
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
            "data_range": scaler.data_range_.tolist(),
            "scale": scaler.scale_.tolist(),
            "min": scaler.min_.tolist(),
        }
        with (fold_dir / SCALER_METADATA_NAME).open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
    return output_dir


def main() -> tuple[Path, Path]:
    tabular_dir = create_full_ta_validation_folds()
    neural_network_dir = create_scaled_full_ta_validation_folds()
    print(f"Created Full TA validation folds at {tabular_dir}")
    print(f"Created scaled Full TA validation folds at {neural_network_dir}")
    return tabular_dir, neural_network_dir


if __name__ == "__main__":
    main()
