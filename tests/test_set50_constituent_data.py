from __future__ import annotations

import pandas as pd
import pytest

from models.set50_constituent_data import parse_yahoo_chart, validate_membership, yahoo_ticker


def _membership() -> pd.DataFrame:
    rows = []
    for version, start, end in (
        ("h1", "2024-01-01", "2024-06-30"),
        ("h2", "2024-07-01", "2024-12-31"),
    ):
        for index in range(50):
            rows.append(
                {
                    "effective_from": start,
                    "effective_to": end,
                    "symbol": f"S{index:02d}",
                    "membership_version": version,
                    "source_document": "official.pdf",
                    "source_url": "https://example.test/official.pdf",
                    "source_sha256": "a" * 64,
                }
            )
    return pd.DataFrame(rows)


def test_membership_requires_fifty_unique_symbols_and_no_overlap():
    valid = validate_membership(_membership())
    assert valid.groupby("membership_version")["symbol"].nunique().tolist() == [50, 50]

    overlapping = _membership()
    overlapping.loc[overlapping["membership_version"].eq("h1"), "effective_to"] = "2024-07-01"
    with pytest.raises(ValueError, match="overlap"):
        validate_membership(overlapping)


def test_yahoo_ticker_rejects_unsafe_symbols():
    assert yahoo_ticker("advanc") == "ADVANC.BK"
    with pytest.raises(ValueError, match="Unsafe"):
        yahoo_ticker("ADVANC/../../secret")


def test_parse_yahoo_chart_uses_bangkok_session_date_and_drops_missing_close():
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "timestamp": [1704173400, 1704259800],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, None],
                                "high": [101.0, None],
                                "low": [99.0, None],
                                "close": [100.5, None],
                                "volume": [1000, None],
                            }
                        ],
                        "adjclose": [{"adjclose": [98.0, None]}],
                    },
                }
            ],
        }
    }
    frame = parse_yahoo_chart(payload, symbol="ADVANC", ticker="ADVANC.BK")
    assert frame["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02"]
    assert frame["Close"].tolist() == [100.5]
