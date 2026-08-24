from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.pit_fcg_controls import (
    ControlMatches,
    coverage_descriptors,
    match_external_controls,
    match_training_controls,
)
from models.track_b_data import DAILY_FEATURE_COLUMNS

PROTOCOL_ID = "pit-fcg-lstm-inner-development-v1"
NEWS_FEATURES = tuple(DAILY_FEATURE_COLUMNS)
CONTEXT_FEATURES = (
    "prob_bull",
    "prob_sideway",
    "prob_bear",
    "routing_entropy",
)
DATE_COLUMN = "Date"
LABEL_DATE_COLUMN = "Label_Date"
CLOSE_COLUMN = "Close_D"
TARGET_COLUMN = "Target_Next_Close"
REGIME_COLUMN = "routing_regime"
FIRST_NEWS_YEAR = 2019
DEFAULT_WINDOW = 5


@dataclass(frozen=True)
class SequenceDataset:
    numeric: np.ndarray
    news: np.ndarray
    raw_news: np.ndarray
    context: np.ndarray
    labels: np.ndarray
    endpoint_dates: pd.DatetimeIndex
    regimes: np.ndarray
    current_close: np.ndarray
    next_close: np.ndarray


@dataclass(frozen=True)
class PreparedInnerFold:
    name: str
    validation_year: int
    train: SequenceDataset
    validation: SequenceDataset
    matched_train_controls: ControlMatches
    random_train_controls: ControlMatches
    validation_controls: ControlMatches
    random_validation_controls: ControlMatches
    numeric_scaler: StandardScaler
    news_scaler: StandardScaler


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_inputs(
    project_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("PIT-FCG freeze protocol id is incorrect")
    hashes = manifest.get("input_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("PIT-FCG freeze has no input hashes")
    checked: dict[str, str] = {}
    for relative, expected in hashes.items():
        path = (root / str(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("Frozen input path escapes project root") from error
        if not path.is_file():
            raise FileNotFoundError(f"Frozen input is missing: {relative}")
        actual = _sha256(path)
        if actual != str(expected):
            raise ValueError(f"Frozen input hash mismatch: {relative}")
        checked[str(relative)] = actual
    return {
        "passed": True,
        "protocol_id": PROTOCOL_ID,
        "checked_files": len(checked),
        "input_sha256": checked,
    }


def _validated_feature_names(
    names: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    values = tuple(str(value) for value in names)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be non-empty and unique")
    return values


def _validate_development_frame(
    frame: pd.DataFrame,
    *,
    numeric_features: tuple[str, ...],
) -> pd.DataFrame:
    required = {
        DATE_COLUMN,
        LABEL_DATE_COLUMN,
        CLOSE_COLUMN,
        TARGET_COLUMN,
        REGIME_COLUMN,
        *numeric_features,
        *NEWS_FEATURES,
        *CONTEXT_FEATURES,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Development frame is missing columns: {missing}")
    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], errors="raise")
    result[LABEL_DATE_COLUMN] = pd.to_datetime(
        result[LABEL_DATE_COLUMN],
        errors="raise",
    )
    dates = pd.DatetimeIndex(result[DATE_COLUMN])
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Development dates must be unique and increasing")
    if not (result[LABEL_DATE_COLUMN] > result[DATE_COLUMN]).all():
        raise ValueError("Every label date must strictly follow its endpoint date")
    numeric_columns = [
        *numeric_features,
        *NEWS_FEATURES,
        *CONTEXT_FEATURES,
        CLOSE_COLUMN,
        TARGET_COLUMN,
    ]
    values = result.loc[:, numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Development inputs contain non-finite values")
    if (result[DATE_COLUMN].dt.year < FIRST_NEWS_YEAR).any():
        raise ValueError("PIT-FCG development starts in 2019")
    if result[REGIME_COLUMN].astype(str).eq("").any():
        raise ValueError("Development regimes must be non-empty")
    return result.reset_index(drop=True)


def _scaled_frames(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    numeric_features: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, StandardScaler]:
    numeric_scaler = StandardScaler().fit(train.loc[:, numeric_features])
    news_scaler = StandardScaler().fit(train.loc[:, NEWS_FEATURES])
    scaled: list[pd.DataFrame] = []
    for source in (train, validation):
        result = source.astype(
            {column: np.float64 for column in (*numeric_features, *NEWS_FEATURES)}
        ).copy()
        result.loc[:, numeric_features] = numeric_scaler.transform(
            source.loc[:, numeric_features]
        )
        result.loc[:, NEWS_FEATURES] = news_scaler.transform(
            source.loc[:, NEWS_FEATURES]
        )
        scaled.append(result)
    return scaled[0], scaled[1], numeric_scaler, news_scaler


def _windowed(values: np.ndarray, *, window: int) -> np.ndarray:
    if len(values) < window:
        raise ValueError("Not enough rows for the registered sequence window")
    return np.stack(
        [values[index - window + 1 : index + 1] for index in range(window - 1, len(values))]
    ).astype(np.float32)


def _sequence_dataset(
    scaled: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    numeric_features: tuple[str, ...],
    window: int,
) -> SequenceDataset:
    if len(scaled) != len(raw) or not scaled[DATE_COLUMN].equals(raw[DATE_COLUMN]):
        raise ValueError("Scaled and raw frames do not align")
    endpoint_rows = np.arange(window - 1, len(scaled), dtype=int)
    numeric = _windowed(
        scaled.loc[:, numeric_features].to_numpy(dtype=np.float64),
        window=window,
    )
    news = _windowed(
        scaled.loc[:, NEWS_FEATURES].to_numpy(dtype=np.float64),
        window=window,
    )
    raw_news = _windowed(
        raw.loc[:, NEWS_FEATURES].to_numpy(dtype=np.float64),
        window=window,
    )
    close = raw.loc[endpoint_rows, CLOSE_COLUMN].to_numpy(dtype=np.float64)
    target = raw.loc[endpoint_rows, TARGET_COLUMN].to_numpy(dtype=np.float64)
    return SequenceDataset(
        numeric=numeric,
        news=news,
        raw_news=raw_news,
        context=raw.loc[endpoint_rows, CONTEXT_FEATURES]
        .to_numpy(dtype=np.float32),
        labels=(target > close).astype(np.float32),
        endpoint_dates=pd.DatetimeIndex(raw.loc[endpoint_rows, DATE_COLUMN]),
        regimes=raw.loc[endpoint_rows, REGIME_COLUMN]
        .astype(str)
        .to_numpy(dtype=object),
        current_close=close,
        next_close=target,
    )


def prepare_inner_fold(
    frame: pd.DataFrame,
    *,
    numeric_features: Sequence[str],
    validation_year: int,
    seed: int,
    window: int = DEFAULT_WINDOW,
) -> PreparedInnerFold:
    if isinstance(validation_year, bool) or validation_year <= FIRST_NEWS_YEAR:
        raise ValueError("validation_year must be after 2019")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    numeric = _validated_feature_names(numeric_features, name="numeric_features")
    validated = _validate_development_frame(frame, numeric_features=numeric)
    years = validated[DATE_COLUMN].dt.year
    train_raw = validated.loc[
        years.ge(FIRST_NEWS_YEAR) & years.lt(validation_year)
    ].reset_index(drop=True)
    validation_raw = validated.loc[years.eq(validation_year)].reset_index(drop=True)
    if len(train_raw) < window or validation_raw.empty:
        raise ValueError("Inner fold does not contain enough train/validation rows")
    if train_raw[DATE_COLUMN].max() >= validation_raw[DATE_COLUMN].min():
        raise ValueError("Inner train and validation periods overlap")

    train_scaled, validation_scaled, numeric_scaler, news_scaler = _scaled_frames(
        train_raw,
        validation_raw,
        numeric_features=numeric,
    )
    train_sequences = _sequence_dataset(
        train_scaled,
        train_raw,
        numeric_features=numeric,
        window=window,
    )
    context_raw = pd.concat(
        [train_raw.tail(window - 1), validation_raw],
        ignore_index=True,
    )
    context_scaled = pd.concat(
        [train_scaled.tail(window - 1), validation_scaled],
        ignore_index=True,
    )
    validation_sequences = _sequence_dataset(
        context_scaled,
        context_raw,
        numeric_features=numeric,
        window=window,
    )
    if len(validation_sequences.labels) != len(validation_raw):
        raise ValueError("Validation sequence coverage is incomplete")

    train_descriptors = coverage_descriptors(train_sequences.raw_news)
    validation_descriptors = coverage_descriptors(validation_sequences.raw_news)
    matched = match_training_controls(
        train_sequences.endpoint_dates,
        train_sequences.regimes,
        train_descriptors,
        seed=seed,
        temporal_gap=window,
        strategy="matched",
    )
    random_control = match_training_controls(
        train_sequences.endpoint_dates,
        train_sequences.regimes,
        train_descriptors,
        seed=seed,
        temporal_gap=window,
        strategy="random",
        required_anchor_indices=matched.anchor_indices,
    )
    validation_control = match_external_controls(
        validation_sequences.endpoint_dates,
        validation_sequences.regimes,
        validation_descriptors,
        source_dates=train_sequences.endpoint_dates,
        source_regimes=train_sequences.regimes,
        source_descriptors=train_descriptors,
        seed=seed,
    )
    random_validation_control = match_external_controls(
        validation_sequences.endpoint_dates,
        validation_sequences.regimes,
        validation_descriptors,
        source_dates=train_sequences.endpoint_dates,
        source_regimes=train_sequences.regimes,
        source_descriptors=train_descriptors,
        seed=seed,
        strategy="random",
    )
    return PreparedInnerFold(
        name=f"inner_{validation_year}",
        validation_year=int(validation_year),
        train=train_sequences,
        validation=validation_sequences,
        matched_train_controls=matched,
        random_train_controls=random_control,
        validation_controls=validation_control,
        random_validation_controls=random_validation_control,
        numeric_scaler=numeric_scaler,
        news_scaler=news_scaler,
    )
