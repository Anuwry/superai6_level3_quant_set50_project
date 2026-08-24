import numpy as np
import pandas as pd

from models.baseline_common import TARGET_COLUMN, discover_folds
from models.point_in_time_data import LABEL_DATE_COLUMN
from models.full_ta_feature_pool import (
    FULL_TA_DATA_FOLDS_DIR,
    FULL_TA_FEATURES,
    FULL_TA_NN_DATA_FOLDS_DIR,
    PAPER_ALIGNED_TA_FEATURES,
    build_full_ta_features,
    prepare_full_ta_fold_frames,
)


def make_market_frame(rows: int = 140) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100.0 + index + np.sin(index / 3.0)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=rows, freq="B"),
            "Open_D": close - 0.5,
            "High_D": close + 1.0,
            "Low_D": close - 1.0,
            "Close_D": close,
            "Volume_D": 1000.0 + index * 10.0,
            "Change_pct_D": pd.Series(close).pct_change().fillna(0.0) * 100.0,
            "Open_W": close - 1.0,
            "High_W": close + 2.0,
            "Low_W": close - 2.0,
            "Close_W": close - 0.5,
            "Volume_W": 5000.0 + index * 20.0,
            "Change_pct_W": pd.Series(close).pct_change(5).fillna(0.0) * 100.0,
            "Open_M": close - 2.0,
            "High_M": close + 3.0,
            "Low_M": close - 3.0,
            "Close_M": close - 1.0,
            "Volume_M": 20000.0 + index * 30.0,
            "Change_pct_M": pd.Series(close).pct_change(20).fillna(0.0) * 100.0,
            TARGET_COLUMN: np.roll(close, -1),
        }
    )


def test_full_ta_features_include_paper_indicators_without_duplicates():
    expected = {
        "WMA_5",
        "WMA_10",
        "WMA_20",
        "WMA_60",
        "StochK_14",
        "StochD_3",
        "RSI_14",
        "MACD_12_26",
        "MACD_Signal_9",
        "MACD_Histogram",
        "WilliamsR_14",
        "CCI_20",
        "ADX_14",
        "PlusDI_14",
        "MinusDI_14",
    }

    assert set(PAPER_ALIGNED_TA_FEATURES) == expected
    assert len(FULL_TA_FEATURES) == len(set(FULL_TA_FEATURES))
    assert TARGET_COLUMN not in FULL_TA_FEATURES


def test_build_full_ta_features_produces_finite_causal_indicators():
    frame = make_market_frame()
    features = build_full_ta_features(frame)
    valid = features.loc[:, PAPER_ALIGNED_TA_FEATURES].dropna()

    assert not valid.empty
    assert np.isfinite(valid.to_numpy(dtype=float)).all()
    assert valid["RSI_14"].between(0.0, 100.0).all()
    assert valid["StochK_14"].between(0.0, 100.0).all()
    assert valid["WilliamsR_14"].between(-100.0, 0.0).all()

    changed_future = frame.copy()
    changed_future.loc[120:, ["Open_D", "High_D", "Low_D", "Close_D", "Volume_D"]] *= 10.0
    changed_features = build_full_ta_features(changed_future)
    pd.testing.assert_series_equal(features.loc[:119, "RSI_14"], changed_features.loc[:119, "RSI_14"])


def test_build_full_ta_features_preserves_label_date_metadata():
    frame = make_market_frame()
    frame.insert(
        1,
        LABEL_DATE_COLUMN,
        frame["Date"] + pd.offsets.BDay(1),
    )
    features = build_full_ta_features(frame)

    assert LABEL_DATE_COLUMN in features.columns
    pd.testing.assert_series_equal(
        pd.to_datetime(features[LABEL_DATE_COLUMN]),
        pd.to_datetime(frame[LABEL_DATE_COLUMN]),
        check_names=False,
    )


def test_fold_preparation_keeps_purged_boundary_row_as_feature_context():
    context = make_market_frame(rows=80)
    train = context.iloc[:78].copy()
    test = context.iloc[[79]].copy()

    _, test_features = prepare_full_ta_fold_frames(
        train,
        test,
        context=context,
    )

    assert test_features["Close_D_lag1"].iloc[0] == context[
        "Close_D"
    ].iloc[78]


def test_generated_full_ta_folds_are_ready():
    specs = discover_folds(FULL_TA_DATA_FOLDS_DIR)

    assert len(specs) == 4
    for spec in specs:
        train = pd.read_csv(spec.train_path)
        test = pd.read_csv(spec.test_path)

        assert list(train.columns) == list(test.columns)
        assert all(feature in train.columns for feature in FULL_TA_FEATURES)
        assert not train.isna().any().any()
        assert not test.isna().any().any()


def test_generated_full_ta_nn_folds_are_scaled_train_only():
    specs = discover_folds(FULL_TA_NN_DATA_FOLDS_DIR)

    assert len(specs) == 4
    for spec in specs:
        train = pd.read_csv(spec.train_path)
        numeric_train = train.drop(columns=["Date"])
        metadata_path = spec.train_path.parent / "minmax_scaler.json"

        assert metadata_path.exists()
        assert numeric_train.min().min() >= -1e-12
        assert numeric_train.max().max() <= 1.0 + 1e-12
