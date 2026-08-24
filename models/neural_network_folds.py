from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]

from models.baseline_common import (
    DATA_FOLDS_DIR,
    DATE_COLUMN,
    TARGET_COLUMN,
    discover_folds,
)
from models.point_in_time_data import LABEL_DATE_COLUMN

NN_DATA_FOLDS_DIR = PROJECT_ROOT / "data-folds-nn"
SCALER_METADATA_NAME = "minmax_scaler.json"


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    metadata_columns = {DATE_COLUMN, LABEL_DATE_COLUMN}
    return [
        column
        for column in frame.columns
        if column not in metadata_columns
    ]


def scale_train_test_frames(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    columns = numeric_columns(train)
    if columns != numeric_columns(test):
        raise ValueError("Train and test numeric columns differ")
    scaler = MinMaxScaler()
    train_values = scaler.fit_transform(train.loc[:, columns])
    test_values = scaler.transform(test.loc[:, columns])
    scaled_train = train.copy()
    scaled_test = test.copy()
    for index, column in enumerate(columns):
        scaled_train[column] = train_values[:, index]
        scaled_test[column] = test_values[:, index]
    metadata = {
        "scaler": "MinMaxScaler",
        "feature_range": [0, 1],
        "fit_scope": "train_only",
        "columns": columns,
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "data_range": scaler.data_range_.tolist(),
        "scale": scaler.scale_.tolist(),
        "min": scaler.min_.tolist(),
        "target_column": TARGET_COLUMN,
    }
    return scaled_train, scaled_test, metadata


def inverse_scaled_target(
    values: np.ndarray, metadata: dict[str, object]
) -> np.ndarray:
    columns = list(metadata["columns"])
    target_index = columns.index(TARGET_COLUMN)
    scale = float(list(metadata["scale"])[target_index])
    min_value = float(list(metadata["min"])[target_index])
    return (np.asarray(values, dtype=float) - min_value) / scale


def create_neural_network_folds(
    source_dir: Path = DATA_FOLDS_DIR,
    output_dir: Path = NN_DATA_FOLDS_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in discover_folds(source_dir):
        fold_output = output_dir / spec.fold
        fold_output.mkdir(parents=True, exist_ok=True)
        train = pd.read_csv(spec.train_path)
        test = pd.read_csv(spec.test_path)
        scaled_train, scaled_test, metadata = scale_train_test_frames(train, test)
        scaled_train.to_csv(fold_output / spec.train_path.name, index=False)
        scaled_test.to_csv(fold_output / spec.test_path.name, index=False)
        with (fold_output / SCALER_METADATA_NAME).open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
    return output_dir


def load_scaler_metadata(
    fold_name: str, data_dir: Path = NN_DATA_FOLDS_DIR
) -> dict[str, object]:
    path = data_dir / fold_name / SCALER_METADATA_NAME
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> Path:
    output_dir = create_neural_network_folds()
    print(f"Created neural network folds at {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
