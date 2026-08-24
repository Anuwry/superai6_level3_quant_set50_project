from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REQUIRED_MARKET_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")
REGISTERED_YAHOO_SYMBOL = "^SET50.BK"
REGISTERED_OVERLAP_START = "2025-10-01"


def _parse_volume(value: object) -> float:
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "-"}:
        return float("nan")
    multiplier = 1.0
    suffix = text[-1].upper()
    if suffix in {"K", "M", "B"}:
        multiplier = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}[
            suffix
        ]
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return float("nan")


def parse_local_investing_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    aliases = {
        "วันเดือนปี": "Date",
        "ล่าสุด": "Close",
        "ราคาเปิด": "Open",
        "สูงสุด": "High",
        "ต่ำสุด": "Low",
        "ปริมาณ": "Volume",
    }
    missing = sorted(set(aliases).difference(frame.columns))
    if missing:
        raise ValueError(f"Local Investing daily file lacks columns: {missing}")
    result = frame.rename(columns=aliases).loc[:, REQUIRED_MARKET_COLUMNS].copy()
    result["Date"] = pd.to_datetime(
        result["Date"],
        format="%m/%d/%Y",
        errors="coerce",
    )
    for column in ("Open", "High", "Low", "Close"):
        result[column] = pd.to_numeric(
            result[column].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    result["Volume"] = result["Volume"].map(_parse_volume)
    if result.isna().any().any():
        raise ValueError("Local Investing daily file contains invalid OHLCV rows")
    result = result.sort_values("Date").reset_index(drop=True)
    if result["Date"].duplicated().any():
        raise ValueError("Local Investing daily file contains duplicate dates")
    result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")
    return result


def yahoo_chart_request(
    *,
    start: str = REGISTERED_OVERLAP_START,
    end: str,
) -> tuple[str, dict[str, object]]:
    start_time = pd.Timestamp(start, tz="Asia/Bangkok")
    end_time = pd.Timestamp(end, tz="Asia/Bangkok")
    if end_time <= start_time:
        raise ValueError("Yahoo request end must be after start")
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quote(REGISTERED_YAHOO_SYMBOL, safe='')}"
    )
    return url, {
        "period1": int(start_time.timestamp()),
        "period2": int(end_time.timestamp()),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }


def fetch_yahoo_daily(
    *,
    end: str,
    start: str = REGISTERED_OVERLAP_START,
    timeout_seconds: float = 30.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    import requests

    url, params = yahoo_chart_request(start=start, end=end)
    response = requests.get(
        url,
        params=params,
        timeout=timeout_seconds,
        headers={"User-Agent": "SET50-direction-research/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Yahoo chart response is not a JSON object")
    chart = payload.get("chart")
    if isinstance(chart, dict) and chart.get("error") is not None:
        raise ValueError(f"Yahoo chart API returned an error: {chart['error']}")
    return normalize_yahoo_chart(payload), payload


def normalize_yahoo_chart(payload: Mapping[str, object]) -> pd.DataFrame:
    try:
        chart = payload["chart"]
        result = chart["result"][0]  # type: ignore[index]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Yahoo chart response lacks a valid quote") from error
    required = ("open", "high", "low", "close", "volume")
    if any(name not in quote for name in required):
        raise ValueError("Yahoo chart response lacks required quote fields")
    lengths = {len(timestamps), *(len(quote[name]) for name in required)}
    if len(lengths) != 1:
        raise ValueError("Yahoo quote field lengths differ")
    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(
        "Asia/Bangkok"
    )
    frame = pd.DataFrame(
        {
            "Date": dates.date,
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "Volume": quote["volume"],
        }
    )
    for column in REQUIRED_MARKET_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.attrs["raw_quote_rows"] = len(frame)
    frame.attrs["raw_complete_ohlcv_share"] = float(
        frame.loc[:, REQUIRED_MARKET_COLUMNS[1:]].notna().all(axis=1).mean()
    )
    frame.attrs["raw_positive_volume_share"] = float(
        np.mean(frame["Volume"].fillna(0.0).to_numpy(dtype=float) > 0.0)
    )
    frame = frame.dropna().drop_duplicates("Date").sort_values("Date")
    if frame.empty:
        raise ValueError("Yahoo chart response has no complete OHLCV rows")
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
    return frame.reset_index(drop=True)


def validate_overlap(
    local: pd.DataFrame,
    fetched: pd.DataFrame,
    *,
    maximum_close_difference: float,
    minimum_overlap_rows: int = 2,
) -> dict[str, float | int]:
    left = local.copy()
    right = fetched.copy()
    left["Date"] = pd.to_datetime(left["Date"])
    right["Date"] = pd.to_datetime(right["Date"])
    overlap = left[["Date", "Close"]].merge(
        right[["Date", "Close"]],
        on="Date",
        suffixes=("_local", "_fetched"),
        validate="one_to_one",
    )
    if len(overlap) < minimum_overlap_rows:
        raise ValueError("Market-source overlap has too few rows")
    difference = np.abs(
        overlap["Close_local"].to_numpy(dtype=float)
        - overlap["Close_fetched"].to_numpy(dtype=float)
    )
    if not np.isfinite(difference).all() or float(difference.max()) > float(
        maximum_close_difference
    ):
        raise ValueError("Market-source overlap close difference is material")
    return {
        "overlap_rows": len(overlap),
        "maximum_close_difference": float(difference.max()),
        "median_close_difference": float(np.median(difference)),
    }


def merge_validated_history(
    local: pd.DataFrame,
    fetched: pd.DataFrame,
    *,
    maximum_close_difference: float,
    minimum_overlap_rows: int,
    required_extension_year: int,
    minimum_positive_volume_share: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not 0.0 < minimum_positive_volume_share <= 1.0:
        raise ValueError("minimum_positive_volume_share must be in (0, 1]")
    local_values = _validated_daily(local)
    fetched_values = _validated_daily(fetched)
    overlap_audit = validate_overlap(
        local_values,
        fetched_values,
        maximum_close_difference=maximum_close_difference,
        minimum_overlap_rows=minimum_overlap_rows,
    )
    positive_volume_share = float(
        fetched.attrs.get(
            "raw_positive_volume_share",
            np.mean(fetched_values["Volume"].to_numpy(dtype=float) > 0.0),
        )
    )
    if positive_volume_share < minimum_positive_volume_share:
        raise ValueError(
            "Fetched market volume fails the positive-volume coverage gate"
        )
    local_end = local_values["Date"].max()
    extension = fetched_values.loc[fetched_values["Date"] > local_end].copy()
    if extension.empty:
        raise ValueError("Fetched market source does not extend local history")
    extension_years = set(extension["Date"].dt.year.astype(int))
    if required_extension_year not in extension_years:
        raise ValueError(
            f"Fetched market source has no {required_extension_year} extension rows"
        )
    merged = pd.concat([local_values, extension], ignore_index=True)
    merged = _validated_daily(merged)
    audit: dict[str, object] = {
        **overlap_audit,
        "local_rows": len(local_values),
        "fetched_rows": len(fetched_values),
        "appended_rows": len(extension),
        "local_end": local_end.strftime("%Y-%m-%d"),
        "extension_start": extension["Date"].min().strftime("%Y-%m-%d"),
        "extension_end": extension["Date"].max().strftime("%Y-%m-%d"),
        "fetched_positive_volume_share": positive_volume_share,
        "fetched_raw_complete_ohlcv_share": float(
            fetched.attrs.get("raw_complete_ohlcv_share", 1.0)
        ),
    }
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
    return merged, audit


def _validated_daily(daily: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_MARKET_COLUMNS).difference(daily.columns))
    if missing:
        raise ValueError(f"Daily market data lacks columns: {missing}")
    result = daily.loc[:, REQUIRED_MARKET_COLUMNS].copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
    for column in REQUIRED_MARKET_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result.isna().any().any():
        raise ValueError("Daily market data contains invalid values")
    result = result.sort_values("Date").reset_index(drop=True)
    if result["Date"].duplicated().any():
        raise ValueError("Daily market data contains duplicate dates")
    return result


def _periodic_aggregate(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    values = daily.copy()
    periods = values["Date"].dt.to_period(period)
    values["Period_Date"] = periods.dt.start_time
    aggregate = (
        values.groupby("Period_Date", sort=True)
        .agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        )
        .reset_index()
        .rename(columns={"Period_Date": "Date"})
    )
    aggregate["Change_pct"] = aggregate["Close"].pct_change() * 100.0
    return aggregate


def _lagged_period_features(
    daily_dates: pd.DataFrame,
    periodic: pd.DataFrame,
    suffix: str,
) -> pd.DataFrame:
    feature_columns = ["Open", "High", "Low", "Close", "Volume", "Change_pct"]
    lagged = pd.concat(
        [
            periodic[["Date"]].reset_index(drop=True),
            periodic[feature_columns]
            .astype(float)
            .shift(1)
            .reset_index(drop=True),
        ],
        axis=1,
    )
    lagged = lagged.dropna().rename(
        columns={column: f"{column}_{suffix}" for column in feature_columns}
    )
    return pd.merge_asof(
        daily_dates.sort_values("Date"),
        lagged.sort_values("Date"),
        on="Date",
        direction="backward",
    )


def build_causal_multiscale_frame(daily: pd.DataFrame) -> pd.DataFrame:
    values = _validated_daily(daily)
    values["Change_pct"] = values["Close"].pct_change() * 100.0
    daily_frame = values.rename(
        columns={
            column: f"{column}_D"
            for column in ("Open", "High", "Low", "Close", "Volume", "Change_pct")
        }
    )
    dates = daily_frame[["Date"]]
    weekly = _periodic_aggregate(values, "W-SAT")
    monthly = _periodic_aggregate(values, "M")
    weekly_aligned = _lagged_period_features(dates, weekly, "W")
    monthly_aligned = _lagged_period_features(dates, monthly, "M")
    result = daily_frame.merge(
        weekly_aligned,
        on="Date",
        validate="one_to_one",
    ).merge(
        monthly_aligned,
        on="Date",
        validate="one_to_one",
    )
    result["Target_Next_Close"] = result["Close_D"].shift(-1)
    result = result.replace([np.inf, -np.inf], np.nan).dropna().reset_index(
        drop=True
    )
    result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")
    ordered = [
        "Date",
        "Close_D",
        "Open_D",
        "High_D",
        "Low_D",
        "Volume_D",
        "Change_pct_D",
        "Close_W",
        "Open_W",
        "High_W",
        "Low_W",
        "Volume_W",
        "Change_pct_W",
        "Close_M",
        "Open_M",
        "High_M",
        "Low_M",
        "Volume_M",
        "Change_pct_M",
        "Target_Next_Close",
    ]
    return result.loc[:, ordered]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_forward_2026_data(
    *,
    freeze_manifest_path: Path,
    output_root: Path,
    local_raw_path: Path,
    retrieval_end: str | None = None,
) -> dict[str, object]:
    from models.baseline_common import discover_folds, load_fold
    from models.point_in_time_data import create_point_in_time_market_folds
    from models.track_d_protocol import verify_freeze_manifest
    from models.vmd_feature_pool import (
        create_full_ta_vmd_folds,
        create_scaled_full_ta_vmd_nn_folds,
    )

    if not freeze_manifest_path.is_file():
        raise FileNotFoundError(
            f"Freeze manifest is required before network access: "
            f"{freeze_manifest_path}"
        )
    verify_freeze_manifest(freeze_manifest_path)
    if not local_raw_path.is_file():
        raise FileNotFoundError(f"Local daily source not found: {local_raw_path}")
    now = datetime.now(ZoneInfo("Asia/Bangkok"))
    request_end = retrieval_end or (now.date() + timedelta(days=1)).isoformat()
    source_dir = output_root / "source"
    source_snapshot_path = source_dir / "yahoo_set50_daily.csv"
    response_path = source_dir / "yahoo_chart_response.json"
    access_ledger_path = source_dir / "data_access_ledger.json"
    merged_path = source_dir / "set50_daily_extended.csv"
    aligned_path = source_dir / "set50_aligned_causal.csv"
    point_in_time_dir = output_root / "point_in_time_2026"
    forward_dir = output_root / "forward_2026"
    scaled_dir = output_root / "forward_2026_nn"
    manifest_path = output_root / "forward_data_manifest.json"
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        registered_hashes = saved.get("artifact_sha256", {})
        for raw_path, expected in registered_hashes.items():
            path = Path(raw_path)
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"Forward data artifact hash mismatch: {path}")
        return saved

    local = parse_local_investing_daily(local_raw_path)
    if source_snapshot_path.is_file():
        if not access_ledger_path.is_file():
            raise ValueError("Existing Yahoo snapshot lacks its access ledger")
        fetched = pd.read_csv(source_snapshot_path)
        access_ledger = json.loads(access_ledger_path.read_text(encoding="utf-8"))
    else:
        fetched, raw_payload = fetch_yahoo_daily(end=request_end)
        source_dir.mkdir(parents=True, exist_ok=True)
        response_path.write_text(
            json.dumps(raw_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        fetched.to_csv(source_snapshot_path, index=False)
        request_url, request_params = yahoo_chart_request(end=request_end)
        access_ledger = {
            "accessed_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(
                timespec="seconds"
            ),
            "freeze_manifest": str(freeze_manifest_path),
            "freeze_manifest_sha256": _sha256(freeze_manifest_path),
            "request_url": request_url,
            "request_params": request_params,
            "retrieval_end_exclusive": request_end,
            "source_snapshot": str(source_snapshot_path),
            "source_snapshot_sha256": _sha256(source_snapshot_path),
        }
        access_ledger_path.write_text(
            json.dumps(access_ledger, indent=2),
            encoding="utf-8",
        )

    merged, overlap_audit = merge_validated_history(
        local,
        fetched,
        maximum_close_difference=0.50,
        minimum_overlap_rows=20,
        required_extension_year=2026,
    )
    source_dir.mkdir(parents=True, exist_ok=True)
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
        raise ValueError("Forward preprocessing did not create exactly one fold")
    original_fold = load_fold(original_specs[0])
    scaled_fold = load_fold(scaled_specs[0])
    if original_fold.feature_columns != scaled_fold.feature_columns:
        raise ValueError("Forward original and scaled feature pools differ")
    if len(original_fold.feature_columns) != 122:
        raise ValueError(
            "Forward feature contract expected 122 columns; found "
            f"{len(original_fold.feature_columns)}"
        )
    test_dates = pd.to_datetime(original_fold.test["Date"])
    if set(test_dates.dt.year.astype(int)) != {2026}:
        raise ValueError("Forward holdout includes dates outside 2026")
    artifact_paths = (
        source_snapshot_path,
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
        "created_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(
            timespec="seconds"
        ),
        "access_ledger": access_ledger,
        "overlap_audit": overlap_audit,
        "local_source": str(local_raw_path),
        "local_source_sha256": _sha256(local_raw_path),
        "extended_daily_rows": len(merged),
        "aligned_rows": len(aligned),
        "forward_train_rows": len(original_fold.train),
        "forward_test_rows": len(original_fold.test),
        "forward_test_start": test_dates.min().strftime("%Y-%m-%d"),
        "forward_test_end": test_dates.max().strftime("%Y-%m-%d"),
        "features": len(original_fold.feature_columns),
        "artifact_sha256": {
            str(path): _sha256(path) for path in artifact_paths
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata
