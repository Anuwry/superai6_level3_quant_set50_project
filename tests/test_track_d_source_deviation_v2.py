from __future__ import annotations

import pytest

from models.track_d_source_deviation_v2 import (
    normalize_investing_history_v2,
)


def test_v2_retains_and_audits_preexisting_ohlc_anomaly():
    payload = {
        "data": [
            {
                "rowDateTimestamp": "2025-12-11T00:00:00Z",
                "last_open": "840.26",
                "last_max": "839.88",
                "last_min": "825.54",
                "last_close": "828.42",
                "volumeRaw": 1_200_242_944,
            },
            {
                "rowDateTimestamp": "2025-12-12T00:00:00Z",
                "last_open": "827.00",
                "last_max": "830.00",
                "last_min": "826.00",
                "last_close": "829.00",
                "volumeRaw": 1_000_000_000,
            },
        ]
    }

    result = normalize_investing_history_v2(payload)

    assert len(result) == 2
    assert result.loc[0, "Open"] == pytest.approx(840.26)
    assert result.loc[0, "High"] == pytest.approx(839.88)
    assert result.attrs["ohlc_containment_anomaly_rows"] == 1
    assert result.attrs["maximum_high_shortfall"] == pytest.approx(0.38)
