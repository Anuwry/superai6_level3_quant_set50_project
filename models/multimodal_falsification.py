from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from models.baseline_common import DATE_COLUMN, FoldData
from models.track_b_data import DAILY_FEATURE_COLUMNS

PROTOCOL_ID = "multimodal-falsification-v1"
NEWS_ONLY_ARM = "News-Only"
CONTROL_TRANSFORMS = (
    "shuffled_news",
    "lagged_news",
    "random_features",
)
CONTROL_ARMS = (
    "Global-Numeric-Shuffled-News",
    "Global-Numeric-Lagged-News",
    "Global-Numeric-Random-Features",
)
CONTROL_TO_ARM = dict(zip(CONTROL_TRANSFORMS, CONTROL_ARMS, strict=True))
DEFAULT_LAG = 5
MARKET_ONLY_ARM = "Market-Only"
OBSERVED_NEWS_ARM = "Observed-News"
ANALYSIS_ARMS = (
    MARKET_ONLY_ARM,
    OBSERVED_NEWS_ARM,
    *CONTROL_ARMS,
    NEWS_ONLY_ARM,
)
CONTROL_CONTRASTS: dict[str, tuple[str, str]] = {
    "observed_news_effect": (OBSERVED_NEWS_ARM, MARKET_ONLY_ARM),
    "observed_vs_shuffled": (
        OBSERVED_NEWS_ARM,
        "Global-Numeric-Shuffled-News",
    ),
    "observed_vs_lagged": (
        OBSERVED_NEWS_ARM,
        "Global-Numeric-Lagged-News",
    ),
    "observed_vs_random": (
        OBSERVED_NEWS_ARM,
        "Global-Numeric-Random-Features",
    ),
    "news_only_vs_market": (NEWS_ONLY_ARM, MARKET_ONLY_ARM),
    "shuffled_vs_market": (
        "Global-Numeric-Shuffled-News",
        MARKET_ONLY_ARM,
    ),
}
METRIC_COLUMNS = (
    "balanced_accuracy",
    "direction_accuracy",
    "mcc",
    "rmse",
    "mae",
)


def _validate_fold(fold: FoldData) -> None:
    missing_features = sorted(
        set(DAILY_FEATURE_COLUMNS).difference(fold.feature_columns)
    )
    if missing_features:
        raise ValueError(f"Fold is missing news features: {missing_features}")
    previous_max: pd.Timestamp | None = None
    for split in ("train", "context", "test"):
        frame = getattr(fold, split)
        if frame is None:
            continue
        missing = sorted(
            {DATE_COLUMN, *DAILY_FEATURE_COLUMNS}.difference(frame.columns)
        )
        if missing:
            raise ValueError(f"{split} is missing columns: {missing}")
        dates = pd.to_datetime(frame[DATE_COLUMN], errors="raise")
        if dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError(f"{split} dates must be unique and increasing")
        if previous_max is not None and dates.min() <= previous_max:
            raise ValueError("Fold split dates overlap or are out of order")
        previous_max = dates.max()
        values = frame.loc[:, DAILY_FEATURE_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{split} news features contain non-finite values")


def _replace_news(frame: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    if values.shape != (len(frame), len(DAILY_FEATURE_COLUMNS)):
        raise ValueError("Replacement news matrix has an invalid shape")
    result = frame.astype(
        {feature: float for feature in DAILY_FEATURE_COLUMNS},
    ).copy()
    replacement = pd.DataFrame(
        values,
        index=result.index,
        columns=DAILY_FEATURE_COLUMNS,
    )
    result[DAILY_FEATURE_COLUMNS] = replacement
    return result


def _map_splits(
    fold: FoldData,
    transformed: dict[str, pd.DataFrame | None],
) -> FoldData:
    return replace(
        fold,
        train=transformed["train"],
        context=transformed["context"],
        test=transformed["test"],
    )


def _shuffled_fold(fold: FoldData, *, seed: int) -> FoldData:
    rng = np.random.default_rng(seed)
    transformed: dict[str, pd.DataFrame | None] = {}
    for split in ("train", "context", "test"):
        frame = getattr(fold, split)
        if frame is None:
            transformed[split] = None
            continue
        values = frame.loc[:, DAILY_FEATURE_COLUMNS].to_numpy(dtype=float)
        transformed[split] = _replace_news(frame, values[rng.permutation(len(frame))])
    return _map_splits(fold, transformed)


def _lagged_fold(fold: FoldData, *, lag: int) -> FoldData:
    if isinstance(lag, bool) or lag < 1:
        raise ValueError("lag must be a positive integer")
    present_splits = [
        split
        for split in ("train", "context", "test")
        if getattr(fold, split) is not None
    ]
    lengths = {split: len(getattr(fold, split)) for split in present_splits}
    combined = pd.concat(
        [getattr(fold, split) for split in present_splits],
        ignore_index=True,
    )
    lagged_values = (
        combined.loc[:, DAILY_FEATURE_COLUMNS].shift(lag).fillna(0.0)
    )
    transformed: dict[str, pd.DataFrame | None] = {
        "train": None,
        "context": None,
        "test": None,
    }
    start = 0
    for split in present_splits:
        stop = start + lengths[split]
        frame = getattr(fold, split)
        transformed[split] = _replace_news(
            frame,
            lagged_values.iloc[start:stop].to_numpy(dtype=float),
        )
        start = stop
    return _map_splits(fold, transformed)


def _random_fold(fold: FoldData, *, seed: int) -> FoldData:
    rng = np.random.default_rng(seed)
    transformed: dict[str, pd.DataFrame | None] = {}
    for split in ("train", "context", "test"):
        frame = getattr(fold, split)
        if frame is None:
            transformed[split] = None
            continue
        values = rng.standard_normal((len(frame), len(DAILY_FEATURE_COLUMNS)))
        transformed[split] = _replace_news(frame, values)
    return _map_splits(fold, transformed)


def transform_news_fold(
    fold: FoldData,
    control: str,
    *,
    seed: int = 20_260_804,
    lag: int = DEFAULT_LAG,
) -> FoldData:
    """Apply a label-independent news falsification transform to a fold."""

    _validate_fold(fold)
    if control == "shuffled_news":
        return _shuffled_fold(fold, seed=int(seed))
    if control == "lagged_news":
        return _lagged_fold(fold, lag=lag)
    if control == "random_features":
        return _random_fold(fold, seed=int(seed))
    raise ValueError(f"Unknown news control: {control}")


def control_feature_sets(
    full_feature_pool: list[str] | tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    features = tuple(str(feature) for feature in full_feature_pool)
    if not features or len(set(features)) != len(features):
        raise ValueError("full_feature_pool must be non-empty and unique")
    missing = sorted(set(DAILY_FEATURE_COLUMNS).difference(features))
    if missing:
        raise ValueError(f"Feature pool is missing news features: {missing}")
    return {
        NEWS_ONLY_ARM: tuple(DAILY_FEATURE_COLUMNS),
        **{arm: features for arm in CONTROL_ARMS},
    }


def build_control_fold_contrasts(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "fold",
        "test_year",
        "arm",
        *METRIC_COLUMNS,
    }
    missing = sorted(required.difference(fold_metrics.columns))
    if missing:
        raise ValueError(f"Fold metrics are missing columns: {missing}")
    available = set(fold_metrics["arm"].astype(str))
    if available != set(ANALYSIS_ARMS):
        raise ValueError("Fold metrics do not contain exactly the analysis arms")
    keys = ["model", "fold", "test_year"]
    rows: list[pd.DataFrame] = []
    for contrast, (treatment_arm, control_arm) in CONTROL_CONTRASTS.items():
        treatment = fold_metrics.loc[
            fold_metrics["arm"].eq(treatment_arm), [*keys, *METRIC_COLUMNS]
        ]
        control = fold_metrics.loc[
            fold_metrics["arm"].eq(control_arm), [*keys, *METRIC_COLUMNS]
        ]
        paired = treatment.merge(
            control,
            on=keys,
            how="inner",
            validate="one_to_one",
            suffixes=("_treatment", "_control"),
        )
        if len(paired) != len(treatment) or len(paired) != len(control):
            raise ValueError(f"Incomplete fold pairing for {contrast}")
        paired.insert(3, "contrast", contrast)
        paired.insert(4, "treatment_arm", treatment_arm)
        paired.insert(5, "control_arm", control_arm)
        paired["balanced_accuracy_delta_pp"] = (
            paired["balanced_accuracy_treatment"]
            - paired["balanced_accuracy_control"]
        ) * 100.0
        paired["direction_accuracy_delta_pp"] = (
            paired["direction_accuracy_treatment"]
            - paired["direction_accuracy_control"]
        ) * 100.0
        for metric in ("mcc", "rmse", "mae"):
            paired[f"{metric}_delta"] = (
                paired[f"{metric}_treatment"]
                - paired[f"{metric}_control"]
            )
        rows.append(paired)
    return pd.concat(rows, ignore_index=True)
