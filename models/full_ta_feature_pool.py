from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models.baseline_common import DATA_FOLDS_DIR, DATE_COLUMN, PROJECT_ROOT, TARGET_COLUMN, discover_folds
from models.full_non_ta_feature_pool import (
    FULL_NON_TA_FEATURES,
    SCALER_METADATA_NAME,
    build_full_non_ta_features,
    safe_divide,
)
from models.neural_network_folds import numeric_columns
from models.point_in_time_data import (
    CONTEXT_FILE_NAME,
    LABEL_DATE_COLUMN,
    MARKET_MASTER_NAME,
    POINT_IN_TIME_DATA_FOLDS_DIR,
)

FULL_TA_DATA_FOLDS_DIR = PROJECT_ROOT / "data-folds-full-ta"
FULL_TA_NN_DATA_FOLDS_DIR = PROJECT_ROOT / "data-folds-full-ta-nn"
FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR = (
    PROJECT_ROOT / "data-folds-full-ta-point-in-time-v2"
)
FULL_TA_POINT_IN_TIME_NN_DATA_FOLDS_DIR = (
    PROJECT_ROOT / "data-folds-full-ta-point-in-time-v2-nn"
)
WMA_WINDOWS = [5, 10, 20, 60]
PAPER_ALIGNED_TA_FEATURES = [
    *[f"WMA_{window}" for window in WMA_WINDOWS],
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
]
FULL_TA_FEATURES = [*FULL_NON_TA_FEATURES, *PAPER_ALIGNED_TA_FEATURES]


def weighted_moving_average(values: np.ndarray) -> float:
    weights = np.arange(1, len(values) + 1, dtype=float)
    return float(np.dot(values, weights) / weights.sum())


def wilder_average(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def build_full_ta_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = build_full_non_ta_features(frame).copy()
    close = data["Close_D"]
    high = data["High_D"]
    low = data["Low_D"]

    for window in WMA_WINDOWS:
        data[f"WMA_{window}"] = close.rolling(window, min_periods=window).apply(
            weighted_moving_average,
            raw=True,
        )

    rolling_high = high.rolling(14, min_periods=14).max()
    rolling_low = low.rolling(14, min_periods=14).min()
    stoch_k = safe_divide(close - rolling_low, rolling_high - rolling_low) * 100.0
    data["StochK_14"] = stoch_k
    data["StochD_3"] = stoch_k.rolling(3, min_periods=3).mean()

    close_change = close.diff()
    average_gain = wilder_average(close_change.clip(lower=0.0), 14)
    average_loss = wilder_average(-close_change.clip(upper=0.0), 14)
    relative_strength = safe_divide(average_gain, average_loss)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    rsi = rsi.mask((average_loss == 0.0) & (average_gain > 0.0), 100.0)
    rsi = rsi.mask((average_gain == 0.0) & (average_loss > 0.0), 0.0)
    data["RSI_14"] = rsi.mask((average_gain == 0.0) & (average_loss == 0.0), 50.0)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    data["MACD_12_26"] = macd
    data["MACD_Signal_9"] = macd_signal
    data["MACD_Histogram"] = macd - macd_signal

    data["WilliamsR_14"] = -100.0 * safe_divide(rolling_high - close, rolling_high - rolling_low)

    typical_price = (high + low + close) / 3.0
    typical_mean = typical_price.rolling(20, min_periods=20).mean()
    mean_deviation = typical_price.rolling(20, min_periods=20).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))),
        raw=True,
    )
    data["CCI_20"] = safe_divide(typical_price - typical_mean, 0.015 * mean_deviation)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    upward_move = high.diff()
    downward_move = -low.diff()
    plus_dm = upward_move.where((upward_move > downward_move) & (upward_move > 0.0), 0.0)
    minus_dm = downward_move.where((downward_move > upward_move) & (downward_move > 0.0), 0.0)
    average_true_range = wilder_average(true_range, 14)
    plus_di = 100.0 * safe_divide(wilder_average(plus_dm, 14), average_true_range)
    minus_di = 100.0 * safe_divide(wilder_average(minus_dm, 14), average_true_range)
    directional_index = 100.0 * safe_divide((plus_di - minus_di).abs(), plus_di + minus_di)
    data["ADX_14"] = wilder_average(directional_index, 14)
    data["PlusDI_14"] = plus_di
    data["MinusDI_14"] = minus_di

    metadata_columns = (
        [LABEL_DATE_COLUMN]
        if LABEL_DATE_COLUMN in data.columns
        else []
    )
    return data.loc[
        :,
        [
            DATE_COLUMN,
            *metadata_columns,
            *FULL_TA_FEATURES,
            TARGET_COLUMN,
        ],
    ]


def prepare_full_ta_fold_artifacts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_dates = set(pd.to_datetime(train[DATE_COLUMN]))
    test_dates = set(pd.to_datetime(test[DATE_COLUMN]))
    feature_source = (
        pd.concat([train, test], ignore_index=True)
        if context is None
        else context.copy()
    )
    context_dates = set(pd.to_datetime(feature_source[DATE_COLUMN]))
    missing_context = sorted((train_dates | test_dates).difference(context_dates))
    if missing_context:
        raise ValueError(
            "Feature context is missing split dates: "
            f"{missing_context[:5]}"
        )
    features = build_full_ta_features(feature_source)
    features[DATE_COLUMN] = pd.to_datetime(features[DATE_COLUMN])
    cleaned = features.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    train_features = cleaned[cleaned[DATE_COLUMN].isin(train_dates)].copy()
    test_features = cleaned[cleaned[DATE_COLUMN].isin(test_dates)].copy()
    train_end = pd.to_datetime(train[DATE_COLUMN]).max()
    test_start = pd.to_datetime(test[DATE_COLUMN]).min()
    context_features = cleaned.loc[
        cleaned[DATE_COLUMN].gt(train_end)
        & cleaned[DATE_COLUMN].lt(test_start)
    ].copy()
    if context is not None and context_features.empty:
        raw_dates = pd.to_datetime(context[DATE_COLUMN])
        has_boundary_context = raw_dates.gt(train_end).lt(test_start).any()
        if has_boundary_context:
            raise ValueError(
                "Full TA preprocessing removed every boundary context row"
            )
    for frame in (train_features, context_features, test_features):
        frame[DATE_COLUMN] = frame[DATE_COLUMN].dt.strftime("%Y-%m-%d")
        if LABEL_DATE_COLUMN in frame.columns:
            frame[LABEL_DATE_COLUMN] = pd.to_datetime(
                frame[LABEL_DATE_COLUMN]
            ).dt.strftime("%Y-%m-%d")
    return train_features, context_features, test_features


def prepare_full_ta_fold_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features, _, test_features = prepare_full_ta_fold_artifacts(
        train,
        test,
        context=context,
    )
    return train_features, test_features


def create_full_ta_folds(
    source_dir: Path = DATA_FOLDS_DIR,
    output_dir: Path = FULL_TA_DATA_FOLDS_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = source_dir / MARKET_MASTER_NAME
    master = pd.read_csv(master_path) if master_path.exists() else None
    for spec in discover_folds(source_dir):
        train = pd.read_csv(spec.train_path)
        test = pd.read_csv(spec.test_path)
        fold_context = None
        if master is not None:
            context_dates = pd.to_datetime(master[DATE_COLUMN])
            test_end = pd.to_datetime(test[DATE_COLUMN]).max()
            fold_context = master.loc[context_dates <= test_end].copy()
        (
            train_features,
            context_features,
            test_features,
        ) = prepare_full_ta_fold_artifacts(
            train,
            test,
            context=fold_context,
        )
        fold_dir = output_dir / spec.fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_features.to_csv(fold_dir / spec.train_path.name, index=False)
        if not context_features.empty:
            context_features.to_csv(
                fold_dir / CONTEXT_FILE_NAME,
                index=False,
            )
        test_features.to_csv(fold_dir / spec.test_path.name, index=False)
    return output_dir


def create_scaled_full_ta_nn_folds(
    source_dir: Path = FULL_TA_DATA_FOLDS_DIR,
    output_dir: Path = FULL_TA_NN_DATA_FOLDS_DIR,
) -> Path:
    if not source_dir.exists():
        create_full_ta_folds(output_dir=source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in discover_folds(source_dir):
        train = pd.read_csv(spec.train_path)
        test = pd.read_csv(spec.test_path)
        context_path = spec.train_path.parent / CONTEXT_FILE_NAME
        context = (
            pd.read_csv(context_path)
            if context_path.is_file()
            else None
        )
        columns = numeric_columns(train)
        scaler = MinMaxScaler()
        scaled_train = train.copy()
        scaled_test = test.copy()
        scaled_train.loc[:, columns] = scaler.fit_transform(train.loc[:, columns])
        scaled_test.loc[:, columns] = scaler.transform(test.loc[:, columns])
        scaled_context = None
        if context is not None:
            scaled_context = context.copy()
            scaled_context.loc[:, columns] = scaler.transform(
                context.loc[:, columns]
            )
        fold_dir = output_dir / spec.fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        scaled_train.to_csv(fold_dir / spec.train_path.name, index=False)
        if scaled_context is not None:
            scaled_context.to_csv(
                fold_dir / CONTEXT_FILE_NAME,
                index=False,
            )
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
    full_ta_dir = create_full_ta_folds(
        source_dir=POINT_IN_TIME_DATA_FOLDS_DIR,
        output_dir=FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR,
    )
    nn_dir = create_scaled_full_ta_nn_folds(
        source_dir=FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR,
        output_dir=FULL_TA_POINT_IN_TIME_NN_DATA_FOLDS_DIR,
    )
    print(f"Created full TA folds at {full_ta_dir}")
    print(f"Created scaled full TA NN folds at {nn_dir}")
    return full_ta_dir, nn_dir


if __name__ == "__main__":
    main()
