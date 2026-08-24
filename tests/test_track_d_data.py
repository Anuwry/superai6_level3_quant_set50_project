from __future__ import annotations

import pandas as pd
import pytest

from models.track_d_data import (
    build_causal_multiscale_frame,
    merge_validated_history,
    normalize_yahoo_chart,
    parse_local_investing_daily,
    prepare_forward_2026_data,
    validate_overlap,
    yahoo_chart_request,
)


def test_yahoo_chart_normalization_rejects_missing_ohlc():
    payload = {"chart": {"result": [{"timestamp": [1], "indicators": {}}]}}

    with pytest.raises(ValueError, match="quote"):
        normalize_yahoo_chart(payload)


def test_yahoo_normalization_preserves_raw_volume_coverage_audit():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1766019600, 1766106000],
                    "indicators": {
                        "quote": [
                            {
                                "open": [800.0, 801.0],
                                "high": [802.0, 803.0],
                                "low": [799.0, 800.0],
                                "close": [801.0, 802.0],
                                "volume": [1000.0, None],
                            }
                        ]
                    },
                }
            ]
        }
    }

    normalized = normalize_yahoo_chart(payload)

    assert len(normalized) == 1
    assert normalized.attrs["raw_quote_rows"] == 2
    assert normalized.attrs["raw_positive_volume_share"] == pytest.approx(0.5)


def test_forward_data_preparation_requires_freeze_before_network(tmp_path):
    with pytest.raises(FileNotFoundError, match="Freeze"):
        prepare_forward_2026_data(
            freeze_manifest_path=tmp_path / "missing-freeze.json",
            output_root=tmp_path / "track-d-data",
            local_raw_path=tmp_path / "local.csv",
            retrieval_end="2026-08-01",
        )


def test_overlap_validation_fails_on_material_close_difference():
    local = pd.DataFrame(
        {"Date": ["2025-12-18", "2025-12-19"], "Close": [828.64, 828.50]}
    )
    fetched = pd.DataFrame(
        {"Date": ["2025-12-18", "2025-12-19"], "Close": [828.64, 830.00]}
    )

    with pytest.raises(ValueError, match="overlap"):
        validate_overlap(local, fetched, maximum_close_difference=0.50)


def test_multiscale_features_only_use_completed_week_and_month():
    dates = pd.date_range("2025-11-03", periods=50, freq="B")
    daily = pd.DataFrame(
        {
            "Date": dates,
            "Open": range(100, 150),
            "High": range(101, 151),
            "Low": range(99, 149),
            "Close": range(100, 150),
            "Volume": [1000.0] * 50,
        }
    )

    aligned = build_causal_multiscale_frame(daily)
    january = aligned[pd.to_datetime(aligned["Date"]).dt.month.eq(1)]
    december_close = daily.loc[daily["Date"].dt.month.eq(12), "Close"].iloc[-1]
    first_date = pd.Timestamp(aligned["Date"].iloc[0])
    first_source_index = daily.index[daily["Date"].eq(first_date)][0]

    assert not january.empty
    assert (january["Close_M"] == december_close).all()
    assert aligned["Target_Next_Close"].iloc[0] == pytest.approx(
        daily.loc[first_source_index + 1, "Close"]
    )


def test_local_investing_parser_normalizes_thai_columns_and_volume(tmp_path):
    path = tmp_path / "daily.csv"
    path.write_text(
        '"วันเดือนปี","ล่าสุด","ราคาเปิด","สูงสุด","ต่ำสุด","ปริมาณ","% เปลี่ยน"\n'
        '"12/22/2025","839.24","834.20","840.16","831.71","1.27B","1.30%"\n'
        '"12/19/2025","828.50","831.34","832.63","826.38","1.54M","-0.02%"\n',
        encoding="utf-8",
    )

    parsed = parse_local_investing_daily(path)

    assert parsed["Date"].tolist() == ["2025-12-19", "2025-12-22"]
    assert parsed["Volume"].tolist() == pytest.approx([1_540_000, 1_270_000_000])


def test_yahoo_request_is_registered_symbol_and_overlap_start():
    url, params = yahoo_chart_request(
        start="2025-10-01",
        end="2026-08-01",
    )

    assert url.endswith("/%5ESET50.BK")
    assert params["interval"] == "1d"
    assert params["period1"] < params["period2"]
    assert params["events"] == "history"


def test_merge_history_extends_only_after_local_end_and_audits_source():
    local_dates = pd.date_range("2025-10-01", periods=25, freq="B")
    local = pd.DataFrame(
        {
            "Date": local_dates,
            "Open": 800.0,
            "High": 802.0,
            "Low": 798.0,
            "Close": range(800, 825),
            "Volume": 1_000.0,
        }
    )
    fetched = local.copy()
    fetched = pd.concat(
        [
            fetched,
            pd.DataFrame(
                {
                    "Date": ["2026-01-02", "2026-01-05"],
                    "Open": [825.0, 826.0],
                    "High": [827.0, 828.0],
                    "Low": [824.0, 825.0],
                    "Close": [826.0, 827.0],
                    "Volume": [2_000.0, 2_100.0],
                }
            ),
        ],
        ignore_index=True,
    )

    merged, audit = merge_validated_history(
        local,
        fetched,
        maximum_close_difference=0.50,
        minimum_overlap_rows=20,
        required_extension_year=2026,
    )

    assert len(merged) == len(local) + 2
    assert merged["Date"].iloc[-1] == "2026-01-05"
    assert audit["appended_rows"] == 2
    assert audit["overlap_rows"] == 25


def test_merge_history_fails_closed_on_unusable_volume():
    dates = pd.date_range("2025-10-01", periods=22, freq="B")
    local = pd.DataFrame(
        {
            "Date": dates,
            "Open": 800.0,
            "High": 802.0,
            "Low": 798.0,
            "Close": 800.0,
            "Volume": 1_000.0,
        }
    )
    fetched = pd.concat(
        [
            local.assign(Volume=0.0),
            pd.DataFrame(
                {
                    "Date": ["2026-01-02"],
                    "Open": [800.0],
                    "High": [802.0],
                    "Low": [798.0],
                    "Close": [800.0],
                    "Volume": [0.0],
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="volume"):
        merge_validated_history(
            local,
            fetched,
            maximum_close_difference=0.50,
            minimum_overlap_rows=20,
            required_extension_year=2026,
        )
