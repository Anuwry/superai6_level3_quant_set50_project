from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from models.track_d_data import (
    REQUIRED_MARKET_COLUMNS,
    build_causal_multiscale_frame,
    merge_validated_history,
    parse_local_investing_daily,
)

INVESTING_INSTRUMENT_ID = 41_049
INVESTING_PAGE = "https://www.investing.com/indices/set-50-historical-data"
INVESTING_ENDPOINT = (
    "https://api.investing.com/api/financialdata/historical/41049"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: object) -> float:
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"none", "nan", "-"}:
        return float("nan")
    multiplier = 1.0
    suffix = text[-1].upper()
    if suffix in {"K", "M", "B"}:
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9}[suffix]
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return float("nan")


def _first(row: Mapping[str, object], names: tuple[str, ...]) -> object:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def normalize_investing_history(payload: Mapping[str, object]) -> pd.DataFrame:
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Investing history response lacks data rows")
    normalized: list[dict[str, object]] = []
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise TypeError("Investing history row is not an object")
        date_value = _first(
            raw_row,
            ("rowDateTimestamp", "rowDate", "date", "tradeDate"),
        )
        date = pd.to_datetime(date_value, errors="coerce", utc=True)
        if pd.isna(date) and raw_row.get("rowDateRaw") is not None:
            date = pd.to_datetime(
                raw_row["rowDateRaw"], unit="s", errors="coerce", utc=True
            )
        normalized.append(
            {
                "Date": date,
                "Open": _number(
                    _first(raw_row, ("last_open", "open", "openPrice"))
                ),
                "High": _number(
                    _first(raw_row, ("last_max", "high", "highPrice"))
                ),
                "Low": _number(
                    _first(raw_row, ("last_min", "low", "lowPrice"))
                ),
                "Close": _number(
                    _first(raw_row, ("last_close", "close", "price"))
                ),
                "Volume": _number(
                    _first(raw_row, ("volumeRaw", "volume", "vol"))
                ),
            }
        )
    frame = pd.DataFrame(normalized, columns=REQUIRED_MARKET_COLUMNS)
    if frame.isna().any().any():
        bad_rows = int(frame.isna().any(axis=1).sum())
        raise ValueError(
            f"Investing history contains {bad_rows} incomplete OHLCV row(s)"
        )
    if frame["Date"].duplicated().any():
        raise ValueError("Investing history contains duplicate dates")
    high_gate = frame["High"] >= frame[["Open", "Close"]].max(axis=1)
    low_gate = frame["Low"] <= frame[["Open", "Close"]].min(axis=1)
    if not bool((high_gate & low_gate).all()):
        raise ValueError("Investing history violates OHLC containment")
    if not bool((frame["Volume"] > 0.0).all()):
        raise ValueError("Investing history contains non-positive volume")
    frame = frame.sort_values("Date").reset_index(drop=True)
    frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
    frame.attrs["raw_positive_volume_share"] = 1.0
    frame.attrs["raw_complete_ohlcv_share"] = 1.0
    return frame


def fetch_investing_history(
    *,
    start: str,
    end: str,
    timeout_seconds: float = 60.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    from curl_cffi import requests

    params = {
        "start-date": start,
        "end-date": end,
        "time-frame": "Daily",
        "add-missing-rows": "false",
    }
    response = requests.get(
        INVESTING_ENDPOINT,
        params=params,
        headers={
            "domain-id": "www",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        },
        impersonate="chrome",
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Investing history response is not a JSON object")
    return normalize_investing_history(payload), payload


def source_deviation_freeze(
    *,
    output_path: Path,
    parent_freeze_path: Path,
    protocol_path: Path,
    implementation_paths: Iterable[Path],
    failed_source_evidence_paths: Iterable[Path],
    alternative_series_accessed: bool,
) -> dict[str, object]:
    if alternative_series_accessed:
        raise ValueError(
            "Alternative full series must remain unseen before deviation freeze"
        )
    groups = {
        "parent_freeze_sha256": (parent_freeze_path,),
        "protocol_sha256": (protocol_path,),
        "implementation_sha256": tuple(implementation_paths),
        "failed_source_evidence_sha256": tuple(
            failed_source_evidence_paths
        ),
    }
    missing = [
        str(path)
        for paths in groups.values()
        for path in paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Deviation freeze input not found: {missing}")
    payload: dict[str, object] = {
        "protocol_version": "track-d-source-deviation-2026-07-31-v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "failed_registered_source_observed_before_deviation": True,
        "alternative_full_series_accessed_before_deviation_freeze": False,
        "alternative_source": {
            "page": INVESTING_PAGE,
            "endpoint": INVESTING_ENDPOINT,
            "instrument_id": INVESTING_INSTRUMENT_ID,
        },
    }
    for name, paths in groups.items():
        payload[name] = {str(path): _sha256(path) for path in paths}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify_source_deviation_freeze(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Source deviation freeze not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field in (
        "parent_freeze_sha256",
        "protocol_sha256",
        "implementation_sha256",
        "failed_source_evidence_sha256",
    ):
        artifacts = payload.get(field)
        if not isinstance(artifacts, dict):
            raise TypeError(f"Deviation freeze field {field} is invalid")
        for raw_path, expected in artifacts.items():
            artifact = Path(raw_path)
            if not artifact.is_file() or _sha256(artifact) != expected:
                raise ValueError(
                    f"Deviation frozen file hash mismatch: {artifact}"
                )
    return payload


def prepare_deviation_forward_data(
    *,
    parent_freeze_path: Path,
    deviation_freeze_path: Path,
    output_root: Path,
    local_raw_path: Path,
    retrieval_end: str,
) -> dict[str, object]:
    from models.baseline_common import discover_folds, load_fold
    from models.point_in_time_data import create_point_in_time_market_folds
    from models.track_d_protocol import verify_freeze_manifest
    from models.vmd_feature_pool import (
        create_full_ta_vmd_folds,
        create_scaled_full_ta_vmd_nn_folds,
    )

    verify_freeze_manifest(parent_freeze_path)
    deviation_freeze = verify_source_deviation_freeze(deviation_freeze_path)
    source_dir = output_root / "source_deviation_investing"
    snapshot_path = source_dir / "investing_set50_daily.csv"
    response_path = source_dir / "investing_history_response.json"
    access_ledger_path = source_dir / "data_access_ledger.json"
    merged_path = source_dir / "set50_daily_extended.csv"
    aligned_path = source_dir / "set50_aligned_causal.csv"
    point_in_time_dir = output_root / "point_in_time_2026"
    forward_dir = output_root / "forward_2026"
    scaled_dir = output_root / "forward_2026_nn"
    manifest_path = output_root / "forward_data_manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    local = parse_local_investing_daily(local_raw_path)
    if snapshot_path.is_file():
        if not access_ledger_path.is_file():
            raise ValueError("Investing snapshot lacks its access ledger")
        fetched = pd.read_csv(snapshot_path)
        ledger = json.loads(access_ledger_path.read_text(encoding="utf-8"))
    else:
        fetched, raw_payload = fetch_investing_history(
            start="2025-10-01",
            end=retrieval_end,
        )
        source_dir.mkdir(parents=True, exist_ok=True)
        response_path.write_text(
            json.dumps(raw_payload, ensure_ascii=False), encoding="utf-8"
        )
        fetched.to_csv(snapshot_path, index=False)
        import curl_cffi

        ledger = {
            "accessed_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "source_deviation_freeze": str(deviation_freeze_path),
            "source_deviation_freeze_sha256": _sha256(
                deviation_freeze_path
            ),
            "endpoint": INVESTING_ENDPOINT,
            "instrument_id": INVESTING_INSTRUMENT_ID,
            "start_date": "2025-10-01",
            "end_date_inclusive": retrieval_end,
            "curl_cffi_version": curl_cffi.__version__,
            "source_snapshot": str(snapshot_path),
            "source_snapshot_sha256": _sha256(snapshot_path),
        }
        access_ledger_path.write_text(
            json.dumps(ledger, indent=2), encoding="utf-8"
        )
    merged, overlap_audit = merge_validated_history(
        local,
        fetched,
        maximum_close_difference=0.50,
        minimum_overlap_rows=20,
        required_extension_year=2026,
    )
    merged.to_csv(merged_path, index=False)
    aligned = build_causal_multiscale_frame(merged)
    aligned.to_csv(aligned_path, index=False)
    create_point_in_time_market_folds(
        aligned_path=aligned_path,
        daily_path=merged_path,
        output_dir=point_in_time_dir,
        test_years=(2026,),
    )
    create_full_ta_vmd_folds(
        source_dir=point_in_time_dir,
        output_dir=forward_dir,
    )
    create_scaled_full_ta_vmd_nn_folds(
        source_dir=forward_dir,
        output_dir=scaled_dir,
    )
    original_specs = discover_folds(forward_dir)
    scaled_specs = discover_folds(scaled_dir)
    if len(original_specs) != 1 or len(scaled_specs) != 1:
        raise ValueError("Deviation preprocessing did not create one fold")
    original = load_fold(original_specs[0])
    scaled = load_fold(scaled_specs[0])
    if original.feature_columns != scaled.feature_columns:
        raise ValueError("Deviation original and scaled features differ")
    if len(original.feature_columns) != 122:
        raise ValueError("Deviation forward fold must contain 122 features")
    dates = pd.to_datetime(original.test["Date"])
    if set(dates.dt.year.astype(int)) != {2026}:
        raise ValueError("Deviation forward fold includes a non-2026 date")
    artifacts = (
        snapshot_path,
        access_ledger_path,
        merged_path,
        aligned_path,
        original_specs[0].train_path,
        original_specs[0].test_path,
        scaled_specs[0].train_path,
        scaled_specs[0].test_path,
    )
    metadata: dict[str, object] = {
        "protocol_version": "track-d-direction-forward-v1",
        "source_protocol_status": "dated_source_deviation",
        "source_deviation_protocol_version": deviation_freeze[
            "protocol_version"
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "access_ledger": ledger,
        "overlap_audit": overlap_audit,
        "local_source": str(local_raw_path),
        "local_source_sha256": _sha256(local_raw_path),
        "extended_daily_rows": len(merged),
        "aligned_rows": len(aligned),
        "forward_train_rows": len(original.train),
        "forward_test_rows": len(original.test),
        "forward_test_start": dates.min().strftime("%Y-%m-%d"),
        "forward_test_end": dates.max().strftime("%Y-%m-%d"),
        "features": len(original.feature_columns),
        "artifact_sha256": {
            str(artifact): _sha256(artifact) for artifact in artifacts
        },
    }
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
