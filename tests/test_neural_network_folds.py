import numpy as np
import pandas as pd

from models.baseline_common import TARGET_COLUMN, discover_folds
from models.point_in_time_data import LABEL_DATE_COLUMN
from models.neural_network_folds import (
    NN_DATA_FOLDS_DIR,
    SCALER_METADATA_NAME,
    inverse_scaled_target,
    scale_train_test_frames,
)


def test_scale_train_test_frames_fits_train_only():
    train = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=3),
            "Close_D": [10.0, 15.0, 20.0],
            "Open_D": [1.0, 2.0, 3.0],
            TARGET_COLUMN: [11.0, 16.0, 21.0],
        }
    )
    test = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=1),
            "Close_D": [30.0],
            "Open_D": [5.0],
            TARGET_COLUMN: [31.0],
        }
    )

    scaled_train, scaled_test, metadata = scale_train_test_frames(train, test)

    assert scaled_train["Close_D"].tolist() == [0.0, 0.5, 1.0]
    assert scaled_test["Close_D"].iloc[0] == 2.0
    assert metadata["columns"] == ["Close_D", "Open_D", TARGET_COLUMN]


def test_scale_train_test_frames_preserves_label_date_as_metadata():
    train = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            LABEL_DATE_COLUMN: ["2020-01-02", "2020-01-03"],
            "Close_D": [10.0, 20.0],
            TARGET_COLUMN: [11.0, 21.0],
        }
    )
    test = pd.DataFrame(
        {
            "Date": ["2021-01-01"],
            LABEL_DATE_COLUMN: ["2021-01-04"],
            "Close_D": [30.0],
            TARGET_COLUMN: [31.0],
        }
    )

    scaled_train, scaled_test, metadata = scale_train_test_frames(train, test)

    assert scaled_train[LABEL_DATE_COLUMN].tolist() == [
        "2020-01-02",
        "2020-01-03",
    ]
    assert scaled_test[LABEL_DATE_COLUMN].tolist() == ["2021-01-04"]
    assert LABEL_DATE_COLUMN not in metadata["columns"]


def test_inverse_scaled_target_restores_original_units():
    train = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=2),
            "Close_D": [10.0, 20.0],
            TARGET_COLUMN: [100.0, 200.0],
        }
    )
    test = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=1),
            "Close_D": [15.0],
            TARGET_COLUMN: [150.0],
        }
    )

    _, _, metadata = scale_train_test_frames(train, test)
    restored = inverse_scaled_target(np.array([0.25, 0.75]), metadata)

    assert restored.tolist() == [125.0, 175.0]


def test_scale_train_test_frames_accepts_integer_count_features():
    train = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=3),
            "Close_D": [10.0, 15.0, 20.0],
            "article_count": pd.Series([0, 2, 4], dtype="int64"),
            TARGET_COLUMN: [11.0, 16.0, 21.0],
        }
    )
    test = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=1),
            "Close_D": [30.0],
            "article_count": pd.Series([1], dtype="int64"),
            TARGET_COLUMN: [31.0],
        }
    )

    scaled_train, scaled_test, _ = scale_train_test_frames(train, test)

    assert scaled_train["article_count"].tolist() == [0.0, 0.5, 1.0]
    assert scaled_test["article_count"].tolist() == [0.25]
    assert scaled_train["article_count"].dtype.kind == "f"


def test_generated_neural_network_folds_are_ready():
    specs = discover_folds(NN_DATA_FOLDS_DIR)

    assert len(specs) == 4
    for spec in specs:
        metadata_path = spec.train_path.parent / SCALER_METADATA_NAME
        train = pd.read_csv(spec.train_path)
        numeric = train.drop(columns=["Date"])

        assert metadata_path.exists()
        assert numeric.min().min() >= -1e-12
        assert numeric.max().max() <= 1.0 + 1e-12
        assert not train.isna().any().any()
