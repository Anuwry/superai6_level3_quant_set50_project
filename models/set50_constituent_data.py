from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from models.baseline_common import PROJECT_ROOT

MEMBERSHIP_FILE = (
    PROJECT_ROOT
    / "data-raw"
    / "track_b"
    / "SET50_membership_2024_2025"
    / "set50_membership_intervals.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data-raw"
    / "track_a"
    / "SET50_constituents_yahoo_2012_2025"
)
SOURCE_NAME = "Yahoo Finance chart endpoint (provisional internal research source)"
SOURCE_URL_TEMPLATE = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
START_DATE = "2012-01-01"
END_DATE_EXCLUSIVE = "2026-01-01"
EXPECTED_MEMBERS = 50
REQUIRED_MEMBERSHIP_COLUMNS = {
    "effective_from",
    "effective_to",
    "symbol",
    "membership_version",
    "source_document",
    "source_url",
    "source_sha256",
}
PRICE_COLUMNS = (
    "Date",
    "Symbol",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "AdjClose",
    "Volume",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_membership(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_MEMBERSHIP_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Membership file is missing columns: {missing}")
    result = frame.copy()
    result["effective_from"] = pd.to_datetime(result["effective_from"], errors="raise")
    result["effective_to"] = pd.to_datetime(result["effective_to"], errors="raise")
    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
    if result["symbol"].eq("").any():
        raise ValueError("Membership contains a blank symbol")
    if (result["effective_from"] > result["effective_to"]).any():
        raise ValueError("Membership contains an inverted effective interval")
    counts = result.groupby("membership_version")["symbol"].nunique()
    invalid = counts[counts.ne(EXPECTED_MEMBERS)]
    if not invalid.empty:
        raise ValueError(f"Membership versions must contain 50 symbols: {invalid.to_dict()}")
    if result.duplicated(["membership_version", "symbol"]).any():
        raise ValueError("Membership contains duplicate version-symbol rows")
    intervals = (
        result[["membership_version", "effective_from", "effective_to"]]
        .drop_duplicates()
        .sort_values("effective_from")
        .reset_index(drop=True)
    )
    if intervals["membership_version"].duplicated().any():
        raise ValueError("A membership version has multiple effective intervals")
    previous_end: pd.Timestamp | None = None
    for row in intervals.itertuples(index=False):
        if previous_end is not None and row.effective_from <= previous_end:
            raise ValueError("Membership effective intervals overlap")
        previous_end = row.effective_to
    return result.sort_values(["effective_from", "symbol"]).reset_index(drop=True)


def yahoo_ticker(symbol: str) -> str:
    value = str(symbol).strip().upper()
    if not value or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for character in value):
        raise ValueError(f"Unsafe SET symbol: {symbol!r}")
    return f"{value}.BK"


def _epoch_seconds(date: str) -> int:
    value = pd.Timestamp(date, tz="UTC")
    return int(value.timestamp())


def parse_yahoo_chart(payload: dict[str, Any], *, symbol: str, ticker: str) -> pd.DataFrame:
    chart = payload.get("chart")
    if not isinstance(chart, dict) or chart.get("error") is not None:
        raise ValueError(f"Yahoo chart error for {ticker}: {chart.get('error') if isinstance(chart, dict) else chart}")
    results = chart.get("result")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"Yahoo chart returned no unique result for {ticker}")
    result = results[0]
    timestamps = result.get("timestamp")
    indicators = result.get("indicators", {})
    quotes = indicators.get("quote")
    adjusted = indicators.get("adjclose")
    if not timestamps or not isinstance(quotes, list) or len(quotes) != 1:
        raise ValueError(f"Yahoo chart result is incomplete for {ticker}")
    quote = quotes[0]
    adjusted_values = (
        adjusted[0].get("adjclose")
        if isinstance(adjusted, list) and len(adjusted) == 1
        else [np.nan] * len(timestamps)
    )
    dates = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert("Asia/Bangkok")
        .normalize()
        .tz_localize(None)
    )
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Symbol": symbol,
            "Ticker": ticker,
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "AdjClose": adjusted_values,
            "Volume": quote.get("volume"),
        }
    )
    if len(frame) != len(timestamps):
        raise ValueError(f"Yahoo chart arrays are not aligned for {ticker}")
    frame = frame.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    if frame.empty or frame["Date"].duplicated().any():
        raise ValueError(f"Yahoo chart has empty or duplicate sessions for {ticker}")
    numeric = ["Open", "High", "Low", "Close", "AdjClose", "Volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(frame["Close"]).all() or frame["Close"].le(0).any():
        raise ValueError(f"Yahoo chart contains invalid close prices for {ticker}")
    return frame.loc[:, PRICE_COLUMNS]


def fetch_symbol(
    session: requests.Session,
    *,
    symbol: str,
    start_date: str = START_DATE,
    end_date_exclusive: str = END_DATE_EXCLUSIVE,
    timeout_seconds: float = 30.0,
) -> tuple[pd.DataFrame, str]:
    ticker = yahoo_ticker(symbol)
    url = SOURCE_URL_TEMPLATE.format(ticker=ticker)
    response = session.get(
        url,
        params={
            "period1": _epoch_seconds(start_date),
            "period2": _epoch_seconds(end_date_exclusive),
            "interval": "1d",
            "events": "div,splits",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return parse_yahoo_chart(response.json(), symbol=symbol, ticker=ticker), response.url


def download_constituents(
    *,
    membership_path: Path = MEMBERSHIP_FILE,
    output_dir: Path = OUTPUT_DIR,
    pause_seconds: float = 0.15,
) -> dict[str, Any]:
    membership = validate_membership(pd.read_csv(membership_path))
    symbols = sorted(membership["symbol"].unique())
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 SET50 academic reliability audit"})
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    for index, symbol in enumerate(symbols, start=1):
        ticker = yahoo_ticker(symbol)
        try:
            frame, request_url = fetch_symbol(session, symbol=symbol)
            path = output_dir / f"{symbol}.csv"
            frame.assign(Date=frame["Date"].dt.strftime("%Y-%m-%d")).to_csv(path, index=False)
            records.append(
                {
                    "symbol": symbol,
                    "ticker": ticker,
                    "rows": int(len(frame)),
                    "first_date": frame["Date"].min().strftime("%Y-%m-%d"),
                    "last_date": frame["Date"].max().strftime("%Y-%m-%d"),
                    "missing_close_rows": int(frame["Close"].isna().sum()),
                    "sha256": sha256_file(path),
                    "request_url": request_url,
                    "status": "downloaded",
                }
            )
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            failures.append({"symbol": symbol, "ticker": ticker, "error": str(error)})
        print(f"[{index}/{len(symbols)}] {symbol}: {'ok' if records and records[-1]['symbol'] == symbol else 'failed'}", flush=True)
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": SOURCE_NAME,
        "source_url_template": SOURCE_URL_TEMPLATE,
        "source_terms_note": (
            "Provisional internal research input only. Do not redistribute raw rows. "
            "Replace with institution-authorized data before treating this extension as paper-ready."
        ),
        "timezone_convention": "Yahoo timestamps converted from UTC to Asia/Bangkok and normalized to local session date",
        "adjustment_convention": (
            "Raw Close retained for SET50 price-index alignment; AdjClose retained for sensitivity checks. "
            "No forward filling of close prices."
        ),
        "requested_start": START_DATE,
        "requested_end_exclusive": END_DATE_EXCLUSIVE,
        "membership_path": str(membership_path.relative_to(PROJECT_ROOT)),
        "membership_sha256": sha256_file(membership_path),
        "symbols_requested": len(symbols),
        "symbols_downloaded": len(records),
        "symbols_failed": len(failures),
        "runtime_seconds": time.perf_counter() - started,
        "files": records,
        "failures": failures,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(records).to_csv(output_dir / "coverage.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download provisional SET50 constituent price histories")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    args = parser.parse_args()
    manifest = download_constituents(output_dir=args.output_dir, pause_seconds=args.pause_seconds)
    print(json.dumps({key: manifest[key] for key in ("symbols_requested", "symbols_downloaded", "symbols_failed", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
