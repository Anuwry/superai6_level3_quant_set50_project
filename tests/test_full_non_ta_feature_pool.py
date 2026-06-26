import pandas as pd

from models.baseline_common import TARGET_COLUMN, discover_folds
from models.full_non_ta_feature_pool import (
    FULL_NON_TA_DATA_FOLDS_DIR,
    FULL_NON_TA_NN_DATA_FOLDS_DIR,
    FULL_NON_TA_FEATURES,
    build_full_non_ta_features,
)


def test_build_full_non_ta_features_uses_past_values_for_lags():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=70),
            "Open_D": range(70),
            "High_D": range(1, 71),
            "Low_D": range(70),
            "Close_D": range(10, 80),
            "Volume_D": range(100, 170),
            "Change_pct_D": [1.0] * 70,
            "Open_W": range(70),
            "High_W": range(70),
            "Low_W": range(70),
            "Close_W": range(20, 90),
            "Volume_W": range(200, 270),
            "Change_pct_W": [2.0] * 70,
            "Open_M": range(70),
            "High_M": range(70),
            "Low_M": range(70),
            "Close_M": range(30, 100),
            "Volume_M": range(300, 370),
            "Change_pct_M": [3.0] * 70,
            TARGET_COLUMN: range(11, 81),
        }
    )

    features = build_full_non_ta_features(frame)

    assert features.loc[65, "Close_D_lag1"] == frame.loc[64, "Close_D"]
    assert features.loc[65, "Close_D_lag60"] == frame.loc[5, "Close_D"]
    assert features.loc[65, "Return_5D"] == frame.loc[65, "Close_D"] / frame.loc[60, "Close_D"] - 1


def test_full_non_ta_feature_list_excludes_target_leakage_columns():
    forbidden = {"Target_Direction", "Next_Day_Return", "Future_Return"}

    assert TARGET_COLUMN not in FULL_NON_TA_FEATURES
    assert not forbidden.intersection(FULL_NON_TA_FEATURES)


def test_generated_full_non_ta_folds_are_ready():
    specs = discover_folds(FULL_NON_TA_DATA_FOLDS_DIR)

    assert len(specs) == 4
    for spec in specs:
        train = pd.read_csv(spec.train_path)
        test = pd.read_csv(spec.test_path)

        assert list(train.columns) == list(test.columns)
        assert TARGET_COLUMN in train.columns
        assert all(feature in train.columns for feature in FULL_NON_TA_FEATURES)
        assert not train.isna().any().any()
        assert not test.isna().any().any()


def test_generated_full_non_ta_nn_folds_are_scaled_train_only():
    specs = discover_folds(FULL_NON_TA_NN_DATA_FOLDS_DIR)

    assert len(specs) == 4
    for spec in specs:
        train = pd.read_csv(spec.train_path)
        numeric_train = train.drop(columns=["Date"])
        metadata_path = spec.train_path.parent / "minmax_scaler.json"

        assert metadata_path.exists()
        assert numeric_train.min().min() >= -1e-12
        assert numeric_train.max().max() <= 1.0 + 1e-12
