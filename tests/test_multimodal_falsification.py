from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.baseline_common import FoldData, FoldSpec
from models.multimodal_falsification import (
    ANALYSIS_ARMS,
    CONTROL_ARMS,
    CONTROL_CONTRASTS,
    NEWS_ONLY_ARM,
    build_control_fold_contrasts,
    control_feature_sets,
    transform_news_fold,
)
from models.track_b_data import DAILY_FEATURE_COLUMNS


def _frame(start: str, rows: int, offset: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=rows, freq="D")
    values = np.arange(offset, offset + rows, dtype=float)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Label_Date": dates + pd.Timedelta(days=1),
            "Close_D": 100.0 + values,
            "technical": values / 10.0,
            "Target_Next_Close": 101.0 + values,
        }
    )
    for index, feature in enumerate(DAILY_FEATURE_COLUMNS):
        frame[feature] = values + (index * 100.0)
    frame["article_count"] = frame["article_count"].astype(int)
    frame["ticker_mention_count"] = frame["ticker_mention_count"].astype(int)
    frame["news_available"] = frame["news_available"].astype(int)
    return frame.loc[
        :,
        [
            "Date",
            "Label_Date",
            "Close_D",
            "technical",
            *DAILY_FEATURE_COLUMNS,
            "Target_Next_Close",
        ],
    ]


def _fold() -> FoldData:
    spec = FoldSpec(
        fold="fold_1",
        train_path=Path("train.csv"),
        test_path=Path("test.csv"),
        train_start_year=2019,
        train_end_year=2021,
        test_year=2022,
    )
    return FoldData(
        spec=spec,
        train=_frame("2020-01-01", 8, 0),
        context=_frame("2020-01-09", 3, 8),
        test=_frame("2020-01-12", 4, 11),
        feature_columns=["Close_D", "technical", *DAILY_FEATURE_COLUMNS],
    )


def test_shuffled_news_is_deterministic_and_split_local() -> None:
    original = _fold()
    first = transform_news_fold(original, "shuffled_news", seed=20260804)
    second = transform_news_fold(original, "shuffled_news", seed=20260804)

    for split in ("train", "context", "test"):
        source = getattr(original, split)
        transformed = getattr(first, split)
        repeated = getattr(second, split)
        assert source is not None and transformed is not None and repeated is not None
        pd.testing.assert_frame_equal(transformed, repeated)
        pd.testing.assert_frame_equal(
            transformed.drop(columns=DAILY_FEATURE_COLUMNS),
            source.drop(columns=DAILY_FEATURE_COLUMNS),
        )
        for feature in DAILY_FEATURE_COLUMNS:
            assert sorted(transformed[feature]) == sorted(source[feature])
    assert not first.train[DAILY_FEATURE_COLUMNS].equals(
        original.train[DAILY_FEATURE_COLUMNS]
    )


def test_lagged_news_uses_only_prior_rows_across_fold_boundaries() -> None:
    original = _fold()
    lagged = transform_news_fold(original, "lagged_news", lag=2)
    combined_original = pd.concat(
        [original.train, original.context, original.test], ignore_index=True
    )
    combined_lagged = pd.concat(
        [lagged.train, lagged.context, lagged.test], ignore_index=True
    )

    expected = combined_original[DAILY_FEATURE_COLUMNS].shift(2).fillna(0.0)
    np.testing.assert_allclose(
        combined_lagged[DAILY_FEATURE_COLUMNS].to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
    )
    pd.testing.assert_series_equal(
        combined_lagged["Target_Next_Close"],
        combined_original["Target_Next_Close"],
    )


def test_random_features_are_deterministic_and_do_not_touch_market_columns() -> None:
    original = _fold()
    first = transform_news_fold(original, "random_features", seed=77)
    second = transform_news_fold(original, "random_features", seed=77)

    pd.testing.assert_frame_equal(first.train, second.train)
    assert np.isfinite(first.train[DAILY_FEATURE_COLUMNS]).all().all()
    pd.testing.assert_frame_equal(
        first.train.drop(columns=DAILY_FEATURE_COLUMNS),
        original.train.drop(columns=DAILY_FEATURE_COLUMNS),
    )
    assert not first.train[DAILY_FEATURE_COLUMNS].equals(
        original.train[DAILY_FEATURE_COLUMNS]
    )


def test_control_feature_sets_lock_news_only_and_eight_feature_placebos() -> None:
    original = _fold()
    feature_sets = control_feature_sets(original.feature_columns)

    assert tuple(feature_sets) == (NEWS_ONLY_ARM, *CONTROL_ARMS)
    assert feature_sets[NEWS_ONLY_ARM] == tuple(DAILY_FEATURE_COLUMNS)
    for arm in CONTROL_ARMS:
        assert feature_sets[arm] == tuple(original.feature_columns)


def test_unknown_control_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown news control"):
        transform_news_fold(_fold(), "future_news")


def test_control_fold_contrasts_are_paired_with_registered_signs() -> None:
    rows = []
    for fold_index in range(1, 5):
        for arm_index, arm in enumerate(ANALYSIS_ARMS):
            rows.append(
                {
                    "model": "lstm",
                    "fold": f"fold_{fold_index}",
                    "test_year": 2021 + fold_index,
                    "arm": arm,
                    "balanced_accuracy": 0.50 + arm_index / 100.0,
                    "direction_accuracy": 0.51 + arm_index / 100.0,
                    "mcc": arm_index / 100.0,
                    "rmse": 10.0 - arm_index / 10.0,
                    "mae": 8.0 - arm_index / 10.0,
                }
            )
    paired = build_control_fold_contrasts(pd.DataFrame(rows))

    assert len(paired) == 4 * len(CONTROL_CONTRASTS)
    assert set(paired["contrast"]) == set(CONTROL_CONTRASTS)
    primary = paired.loc[paired["contrast"].eq("observed_vs_shuffled")]
    np.testing.assert_allclose(primary["balanced_accuracy_delta_pp"], -1.0)
    np.testing.assert_allclose(primary["rmse_delta"], 0.1)
