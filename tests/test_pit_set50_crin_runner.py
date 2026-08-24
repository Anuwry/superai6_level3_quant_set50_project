from __future__ import annotations

import numpy as np
import pandas as pd

from models.pit_set50_crin_runner import _standardize_top, _symbol_features


def test_symbol_features_do_not_forward_fill_missing_prices():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"])
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Close": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Volume": [1000, 1200, 1100],
        }
    )
    calendar = pd.DatetimeIndex(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    )
    features, available, _, next_valid = _symbol_features(frame, calendar)
    assert features.loc[pd.Timestamp("2024-01-04")].eq(0.0).all()
    assert not bool(available.loc[pd.Timestamp("2024-01-04")])
    assert not bool(next_valid.loc[pd.Timestamp("2024-01-03")])


def test_top_standardization_fits_only_supplied_training_rows():
    scores = np.array([[1.0, 2.0], [3.0, 4.0], [1000.0, 2000.0]])
    scaled, metadata = _standardize_top(scores, np.array([0, 1]))
    assert metadata["mean"]
    assert np.allclose(scaled[:2].mean(axis=0), 0.0)
    assert np.all(scaled[2] == 10.0)
