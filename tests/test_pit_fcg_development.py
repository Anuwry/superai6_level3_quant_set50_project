from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.pit_fcg_development import (
    NEWS_FEATURES,
    prepare_inner_fold,
    verify_frozen_inputs,
)


def _synthetic_frame() -> tuple[pd.DataFrame, tuple[str, ...]]:
    dates = pd.bdate_range("2019-01-02", periods=36)
    dates = dates.append(pd.bdate_range("2020-01-02", periods=12))
    values = np.arange(len(dates), dtype=float)
    numeric = ("numeric_1", "numeric_2", "numeric_3")
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Label_Date": dates + pd.offsets.BDay(1),
            "Close_D": 100.0 + values,
            "Target_Next_Close": 100.0 + values + np.where(values % 2, -1.0, 1.0),
            "numeric_1": values,
            "numeric_2": values**2,
            "numeric_3": np.sin(values),
            "prob_bull": np.where(values % 3 == 0, 0.8, 0.1),
            "prob_sideway": np.where(values % 3 == 1, 0.8, 0.1),
            "prob_bear": np.where(values % 3 == 2, 0.8, 0.1),
            "routing_entropy": 0.2 + (values % 4) / 10.0,
            "routing_regime": np.asarray(["bull", "sideway", "bear"] * 16),
        }
    )
    for index, name in enumerate(NEWS_FEATURES):
        frame[name] = (values + index) / 10.0
    frame["article_count"] = (values % 5).astype(int)
    frame["ticker_mention_count"] = (values % 7).astype(int)
    frame["news_available"] = (values % 2).astype(int)
    return frame, numeric


def test_prepare_inner_fold_is_walk_forward_scaled_and_control_aligned() -> None:
    frame, numeric = _synthetic_frame()

    prepared = prepare_inner_fold(
        frame,
        numeric_features=numeric,
        validation_year=2020,
        seed=42,
        window=5,
    )

    assert prepared.name == "inner_2020"
    assert prepared.train.numeric.shape[1:] == (5, 3)
    assert prepared.validation.numeric.shape[1:] == (5, 3)
    assert prepared.validation.news.shape[1:] == (5, 8)
    assert prepared.validation.context.shape[1] == 4
    assert prepared.validation.endpoint_dates.year.unique().tolist() == [2020]
    assert prepared.train.endpoint_dates.max() < prepared.validation.endpoint_dates.min()
    np.testing.assert_array_equal(
        prepared.random_train_controls.anchor_indices,
        prepared.matched_train_controls.anchor_indices,
    )
    assert np.all(
        prepared.matched_train_controls.source_indices
        <= prepared.matched_train_controls.anchor_indices - 5
    )
    assert len(prepared.validation_controls.anchor_indices) == len(
        prepared.validation.labels
    )
    assert np.all(
        prepared.train.endpoint_dates[
            prepared.validation_controls.source_indices
        ].to_numpy()
        < prepared.validation.endpoint_dates.to_numpy()
    )


def test_validation_extreme_does_not_change_train_only_scaler() -> None:
    frame, numeric = _synthetic_frame()
    baseline = prepare_inner_fold(
        frame,
        numeric_features=numeric,
        validation_year=2020,
        seed=42,
    )
    changed = frame.copy()
    changed.loc[changed["Date"].dt.year.eq(2020), "numeric_1"] = 1_000_000.0

    repeated = prepare_inner_fold(
        changed,
        numeric_features=numeric,
        validation_year=2020,
        seed=42,
    )

    np.testing.assert_allclose(
        baseline.numeric_scaler.mean_,
        repeated.numeric_scaler.mean_,
    )
    assert repeated.validation.numeric[:, :, 0].max() > 10_000.0


def test_prepare_inner_fold_rejects_nonchronological_or_overlapping_data() -> None:
    frame, numeric = _synthetic_frame()
    swapped = frame.iloc[[1, 0, *range(2, len(frame))]].reset_index(drop=True)

    with pytest.raises(ValueError, match="unique and increasing"):
        prepare_inner_fold(
            swapped,
            numeric_features=numeric,
            validation_year=2020,
            seed=42,
        )


def test_verify_frozen_inputs_detects_hash_change(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("a\n1\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "freeze.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol_id": "pit-fcg-lstm-inner-development-v1",
                "input_sha256": {"source.csv": digest},
            }
        ),
        encoding="utf-8",
    )

    audit = verify_frozen_inputs(tmp_path, manifest)
    source.write_text("a\n2\n", encoding="utf-8")

    assert audit["passed"] is True
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_frozen_inputs(tmp_path, manifest)
