from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import models.track_d_source_deviation as source_v1
from models.track_d_data import REQUIRED_MARKET_COLUMNS

_LAST_SOURCE_QUALITY_AUDIT: dict[str, float | int] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value(row: Mapping[str, object], names: tuple[str, ...]) -> object:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _number(value: object) -> float:
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"none", "nan", "-"}:
        return float("nan")
    multiplier = 1.0
    if text[-1].upper() in {"K", "M", "B"}:
        suffix = text[-1].upper()
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9}[suffix]
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return float("nan")


def normalize_investing_history_v2(
    payload: Mapping[str, object],
) -> pd.DataFrame:
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Investing history response lacks data rows")
    values: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Investing history row is not an object")
        date = pd.to_datetime(
            _value(
                row,
                ("rowDateTimestamp", "rowDate", "date", "tradeDate"),
            ),
            errors="coerce",
            utc=True,
        )
        if pd.isna(date) and row.get("rowDateRaw") is not None:
            date = pd.to_datetime(
                row["rowDateRaw"], unit="s", errors="coerce", utc=True
            )
        values.append(
            {
                "Date": date,
                "Open": _number(_value(row, ("last_open", "open"))),
                "High": _number(_value(row, ("last_max", "high"))),
                "Low": _number(_value(row, ("last_min", "low"))),
                "Close": _number(_value(row, ("last_close", "close"))),
                "Volume": _number(
                    _value(row, ("volumeRaw", "volume", "vol"))
                ),
            }
        )
    frame = pd.DataFrame(values, columns=REQUIRED_MARKET_COLUMNS)
    if frame.isna().any().any():
        raise ValueError("Investing history contains incomplete OHLCV rows")
    if frame["Date"].duplicated().any():
        raise ValueError("Investing history contains duplicate dates")
    if not bool((frame["Volume"] > 0.0).all()):
        raise ValueError("Investing history contains non-positive volume")
    high_shortfall = (
        frame[["Open", "Close"]].max(axis=1) - frame["High"]
    ).clip(lower=0.0)
    low_excess = (
        frame["Low"] - frame[["Open", "Close"]].min(axis=1)
    ).clip(lower=0.0)
    anomaly = (high_shortfall > 0.0) | (low_excess > 0.0)
    frame = frame.sort_values("Date").reset_index(drop=True)
    frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
    frame.attrs.update(
        {
            "raw_positive_volume_share": 1.0,
            "raw_complete_ohlcv_share": 1.0,
            "ohlc_containment_anomaly_rows": int(anomaly.sum()),
            "maximum_high_shortfall": float(high_shortfall.max()),
            "maximum_low_excess": float(low_excess.max()),
        }
    )
    return frame


def fetch_investing_history_v2(
    *,
    start: str,
    end: str,
    timeout_seconds: float = 60.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    from curl_cffi import requests

    response = requests.get(
        source_v1.INVESTING_ENDPOINT,
        params={
            "start-date": start,
            "end-date": end,
            "time-frame": "Daily",
            "add-missing-rows": "false",
        },
        headers={"domain-id": "www", "Accept": "application/json"},
        impersonate="chrome",
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Investing history response is not a JSON object")
    frame = normalize_investing_history_v2(payload)
    global _LAST_SOURCE_QUALITY_AUDIT
    _LAST_SOURCE_QUALITY_AUDIT = {
        "ohlc_containment_anomaly_rows": int(
            frame.attrs["ohlc_containment_anomaly_rows"]
        ),
        "maximum_high_shortfall": float(
            frame.attrs["maximum_high_shortfall"]
        ),
        "maximum_low_excess": float(frame.attrs["maximum_low_excess"]),
        "raw_rows_retained_without_repair": len(frame),
    }
    return frame, payload


def source_amendment_freeze(
    *,
    output_path: Path,
    parent_deviation_freeze_path: Path,
    protocol_path: Path,
    implementation_paths: Iterable[Path],
    evidence_paths: Iterable[Path],
) -> dict[str, object]:
    groups = {
        "parent_freeze_sha256": (parent_deviation_freeze_path,),
        "protocol_sha256": (protocol_path,),
        "implementation_sha256": tuple(implementation_paths),
        "failed_source_evidence_sha256": tuple(evidence_paths),
    }
    missing = [
        str(path)
        for paths in groups.values()
        for path in paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Amendment freeze input not found: {missing}")
    payload: dict[str, object] = {
        "protocol_version": "track-d-source-deviation-amendment-v2",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "failed_registered_source_observed_before_deviation": True,
        "alternative_full_series_accessed_before_deviation_freeze": True,
        "parser_fix_frozen_before_accepted_snapshot": True,
    }
    for name, paths in groups.items():
        payload[name] = {str(path): _sha256(path) for path in paths}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def prepare_deviation_forward_data_v2(
    *,
    parent_freeze_path: Path,
    amendment_freeze_path: Path,
    output_root: Path,
    local_raw_path: Path,
    retrieval_end: str,
) -> dict[str, object]:
    source_v1.verify_source_deviation_freeze(amendment_freeze_path)
    source_v1.fetch_investing_history = fetch_investing_history_v2
    metadata = source_v1.prepare_deviation_forward_data(
        parent_freeze_path=parent_freeze_path,
        deviation_freeze_path=amendment_freeze_path,
        output_root=output_root,
        local_raw_path=local_raw_path,
        retrieval_end=retrieval_end,
    )
    metadata["source_quality_audit"] = dict(_LAST_SOURCE_QUALITY_AUDIT)
    (output_root / "forward_data_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
