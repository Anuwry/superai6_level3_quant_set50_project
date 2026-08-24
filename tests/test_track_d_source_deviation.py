from __future__ import annotations

import json

import pytest

from models.track_d_source_deviation import (
    normalize_investing_history,
    source_deviation_freeze,
)


def test_investing_history_normalizes_registered_ohlcv_fields():
    payload = {
        "data": [
            {
                "rowDate": "Dec 23, 2025",
                "last_open": "839.20",
                "last_max": "845.50",
                "last_min": "837.10",
                "last_close": "843.25",
                "volume": "1.25B",
            },
            {
                "rowDate": "Dec 22, 2025",
                "last_open": "834.20",
                "last_max": "840.16",
                "last_min": "831.71",
                "last_close": "839.24",
                "volume": "1.27B",
            },
        ]
    }

    result = normalize_investing_history(payload)

    assert result["Date"].tolist() == ["2025-12-22", "2025-12-23"]
    assert result["Close"].tolist() == pytest.approx([839.24, 843.25])
    assert result["Volume"].tolist() == pytest.approx(
        [1_270_000_000, 1_250_000_000]
    )


def test_deviation_freeze_records_that_alternative_series_is_unseen(tmp_path):
    parent = tmp_path / "parent.json"
    parent.write_text("{}", encoding="utf-8")
    evidence = tmp_path / "yahoo.csv"
    evidence.write_text("Date,Close\n2026-07-31,1085.54\n", encoding="utf-8")
    protocol = tmp_path / "deviation.md"
    protocol.write_text("source deviation", encoding="utf-8")
    adapter = tmp_path / "adapter.py"
    adapter.write_text("VALUE = 1\n", encoding="utf-8")

    payload = source_deviation_freeze(
        output_path=tmp_path / "deviation-freeze.json",
        parent_freeze_path=parent,
        protocol_path=protocol,
        implementation_paths=(adapter,),
        failed_source_evidence_paths=(evidence,),
        alternative_series_accessed=False,
    )

    saved = json.loads(
        (tmp_path / "deviation-freeze.json").read_text(encoding="utf-8")
    )
    assert saved == payload
    assert saved["alternative_full_series_accessed_before_deviation_freeze"] is False
    assert saved["failed_registered_source_observed_before_deviation"] is True
