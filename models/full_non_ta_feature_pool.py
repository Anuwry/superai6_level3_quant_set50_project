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

from models.baseline_common import DATA_FOLDS_DIR, DATE_COLUMN, TARGET_COLUMN, discover_folds

FULL_NON_TA_DATA_FOLDS_DIR = PROJECT_ROOT / "data-folds-full-non-ta"
FULL_NON_TA_NN_DATA_FOLDS_DIR = PROJECT_ROOT / "data-folds-full-non-ta-nn"
SCALER_METADATA_NAME = "minmax_scaler.json"

RAW_FEATURES = [
    "Open_D",
    "High_D",
    "Low_D",
    "Close_D",
    "Volume_D",
    "Change_pct_D",
    "Open_W",
    "High_W",
    "Low_W",
    "Close_W",
    "Volume_W",
    "Change_pct_W",
    "Open_M",
    "High_M",
    "Low_M",
    "Close_M",
    "Volume_M",
    "Change_pct_M",
]

CLOSE_D_LAGS = [1, 2, 3, 5, 10, 20, 60]
CHANGE_PCT_D_LAGS = [1, 2, 3, 5, 10, 20]
VOLUME_D_LAGS = [1, 2, 3, 5, 10, 20]
RETURN_WINDOWS = [1, 3, 5, 10, 20, 60]
ROLLING_WINDOWS = [5, 10, 20, 60]
MOMENTUM_WINDOWS = [5, 10, 20, 60]
DIRECTION_WINDOWS = [5, 10, 20]

FULL_NON_TA_FEATURES = [
    *RAW_FEATURES,
    *[f"Close_D_lag{lag}" for lag in CLOSE_D_LAGS],
    *[f"Change_pct_D_lag{lag}" for lag in CHANGE_PCT_D_LAGS],
    *[f"Volume_D_lag{lag}" for lag in VOLUME_D_LAGS],
    "Close_W_lag1",
    "Close_W_lag2",
    "Close_W_lag4",
    "Close_M_lag1",
    "Close_M_lag3",
    *[f"Return_{window}D" for window in RETURN_WINDOWS],
    *[f"SMA_{window}" for window in ROLLING_WINDOWS],
    *[f"Close_to_SMA_{window}" for window in ROLLING_WINDOWS],
    *[f"Volatility_{window}" for window in ROLLING_WINDOWS],
    *[f"Momentum_{window}" for window in MOMENTUM_WINDOWS],
    *[f"ROC_{window}" for window in MOMENTUM_WINDOWS],
    "Ratio_Close_D_to_W",
    "Ratio_Close_D_to_M",
    "Ratio_Volume_D_to_W",
    "Spread_D",
    "Body_D",
    "Body_abs_D",
    "Upper_Shadow_D",
    "Lower_Shadow_D",
    "Body_to_Range_D",
    "Close_Position_D",
    "Spread_W",
    "Body_W",
    "Close_Position_W",
    "Spread_M",
    "Body_M",
    "Close_Position_M",
    "Volume_MA_5",
    "Volume_MA_20",
    "Volume_Ratio_5",
    "Volume_Ratio_20",
    "Volume_Change_1D",
    "Volume_Change_5D",
    "Volume_Change_20D",
    "Direction_lag1",
    "Direction_lag2",
    "Direction_lag3",
    "Direction_lag5",
    *[f"Up_Ratio_{window}" for window in DIRECTION_WINDOWS],
    *[f"Down_Ratio_{window}" for window in DIRECTION_WINDOWS],
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def build_full_non_ta_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values(DATE_COLUMN).reset_index(drop=True).copy()
    for lag in CLOSE_D_LAGS:
        data[f"Close_D_lag{lag}"] = data["Close_D"].shift(lag)
    for lag in CHANGE_PCT_D_LAGS:
        data[f"Change_pct_D_lag{lag}"] = data["Change_pct_D"].shift(lag)
    for lag in VOLUME_D_LAGS:
        data[f"Volume_D_lag{lag}"] = data["Volume_D"].shift(lag)
    for lag in [1, 2, 4]:
        data[f"Close_W_lag{lag}"] = data["Close_W"].shift(lag)
    for lag in [1, 3]:
        data[f"Close_M_lag{lag}"] = data["Close_M"].shift(lag)
    for window in RETURN_WINDOWS:
        data[f"Return_{window}D"] = data["Close_D"].pct_change(window)
    for window in ROLLING_WINDOWS:
        sma = data["Close_D"].rolling(window=window, min_periods=window).mean()
        data[f"SMA_{window}"] = sma
        data[f"Close_to_SMA_{window}"] = safe_divide(data["Close_D"], sma) - 1
        data[f"Volatility_{window}"] = data["Close_D"].pct_change().rolling(window=window, min_periods=window).std()
    for window in MOMENTUM_WINDOWS:
        shifted_close = data["Close_D"].shift(window)
        data[f"Momentum_{window}"] = data["Close_D"] - shifted_close
        data[f"ROC_{window}"] = safe_divide(data["Close_D"] - shifted_close, shifted_close)
    data["Ratio_Close_D_to_W"] = safe_divide(data["Close_D"], data["Close_W"])
    data["Ratio_Close_D_to_M"] = safe_divide(data["Close_D"], data["Close_M"])
    data["Ratio_Volume_D_to_W"] = safe_divide(data["Volume_D"], data["Volume_W"])
    add_price_action_features(data, "D")
    add_price_action_features(data, "W")
    add_price_action_features(data, "M")
    for window in [5, 20]:
        volume_ma = data["Volume_D"].rolling(window=window, min_periods=window).mean()
        data[f"Volume_MA_{window}"] = volume_ma
        data[f"Volume_Ratio_{window}"] = safe_divide(data["Volume_D"], volume_ma)
    data["Volume_Change_1D"] = data["Volume_D"].pct_change(1)
    data["Volume_Change_5D"] = data["Volume_D"].pct_change(5)
    data["Volume_Change_20D"] = data["Volume_D"].pct_change(20)
    direction = np.sign(data["Close_D"].diff())
    for lag in [1, 2, 3, 5]:
        data[f"Direction_lag{lag}"] = direction.shift(lag)
    up = (direction > 0).astype(float)
    down = (direction < 0).astype(float)
    for window in DIRECTION_WINDOWS:
        data[f"Up_Ratio_{window}"] = up.rolling(window=window, min_periods=window).mean()
        data[f"Down_Ratio_{window}"] = down.rolling(window=window, min_periods=window).mean()
    selected_columns = [DATE_COLUMN, *FULL_NON_TA_FEATURES, TARGET_COLUMN]
    return data.loc[:, selected_columns]


def add_price_action_features(data: pd.DataFrame, suffix: str) -> None:
    open_col = data[f"Open_{suffix}"]
    high_col = data[f"High_{suffix}"]
    low_col = data[f"Low_{suffix}"]
    close_col = data[f"Close_{suffix}"]
    spread = high_col - low_col
    body = close_col - open_col
    data[f"Spread_{suffix}"] = spread
    data[f"Body_{suffix}"] = body
    if suffix == "D":
        data[f"Body_abs_{suffix}"] = body.abs()
        data[f"Upper_Shadow_{suffix}"] = high_col - pd.concat([open_col, close_col], axis=1).max(axis=1)
        data[f"Lower_Shadow_{suffix}"] = pd.concat([open_col, close_col], axis=1).min(axis=1) - low_col
        data[f"Body_to_Range_{suffix}"] = safe_divide(body.abs(), spread)
    data[f"Close_Position_{suffix}"] = safe_divide(close_col - low_col, spread)


def prepare_fold_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dates = set(pd.to_datetime(train[DATE_COLUMN]))
    test_dates = set(pd.to_datetime(test[DATE_COLUMN]))
    combined = pd.concat([train, test], ignore_index=True)
    combined_features = build_full_non_ta_features(combined)
    combined_features[DATE_COLUMN] = pd.to_datetime(combined_features[DATE_COLUMN])
    cleaned = combined_features.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    train_features = cleaned[cleaned[DATE_COLUMN].isin(train_dates)].copy()
    test_features = cleaned[cleaned[DATE_COLUMN].isin(test_dates)].copy()
    train_features[DATE_COLUMN] = train_features[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    test_features[DATE_COLUMN] = test_features[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    return train_features, test_features


def create_full_non_ta_folds(
    source_dir: Path = DATA_FOLDS_DIR,
    output_dir: Path = FULL_NON_TA_DATA_FOLDS_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in discover_folds(source_dir):
        train = pd.read_csv(spec.train_path)
        test = pd.read_csv(spec.test_path)
        train_features, test_features = prepare_fold_frames(train, test)
        fold_dir = output_dir / spec.fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_features.to_csv(fold_dir / spec.train_path.name, index=False)
        test_features.to_csv(fold_dir / spec.test_path.name, index=False)
    return output_dir


def create_scaled_full_non_ta_nn_folds(
    source_dir: Path = FULL_NON_TA_DATA_FOLDS_DIR,
    output_dir: Path = FULL_NON_TA_NN_DATA_FOLDS_DIR,
) -> Path:
    if not source_dir.exists():
        create_full_non_ta_folds(output_dir=source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in discover_folds(source_dir):
        train = pd.read_csv(spec.train_path)
        test = pd.read_csv(spec.test_path)
        columns = [column for column in train.columns if column != DATE_COLUMN]
        scaler = MinMaxScaler()
        scaled_train = train.copy()
        scaled_test = test.copy()
        scaled_train.loc[:, columns] = scaler.fit_transform(train.loc[:, columns])
        scaled_test.loc[:, columns] = scaler.transform(test.loc[:, columns])
        fold_dir = output_dir / spec.fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        scaled_train.to_csv(fold_dir / spec.train_path.name, index=False)
        scaled_test.to_csv(fold_dir / spec.test_path.name, index=False)
        metadata = {
            "scaler": "MinMaxScaler",
            "feature_range": [0, 1],
            "fit_scope": "train_only",
            "source_dir": str(source_dir),
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
    full_non_ta_dir = create_full_non_ta_folds()
    nn_dir = create_scaled_full_non_ta_nn_folds()
    print(f"Created full non-TA folds at {full_non_ta_dir}")
    print(f"Created scaled full non-TA NN folds at {nn_dir}")
    return full_non_ta_dir, nn_dir


if __name__ == "__main__":
    main()
