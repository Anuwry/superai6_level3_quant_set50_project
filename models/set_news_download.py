from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import requests

SET_NEWS_PAGE_URL = "https://www.set.or.th/th/market/news-and-alert/news"
SET_NEWS_API_URL = "https://www.set.or.th/api/cms/v1/news/set"
USER_AGENT = "SET50-direction-research/0.1 (academic metadata collection)"
DEFAULT_PAGE_SIZE = 5_000
DEFAULT_DELAY_SECONDS = 0.25
DEFAULT_TIMEOUT_SECONDS = 90
MAX_RETRIES = 4

CSV_FIELDS = (
    "set_news_id",
    "published_at_bangkok",
    "publication_year",
    "symbol",
    "source",
    "headline",
    "url",
    "product",
    "language",
    "tag",
    "market_alert_type_id",
    "view_clarification",
    "percent_price_change",
    "view_count",
    "is_today_news",
    "full_text_collected",
    "text_scope",
)


def build_query_params(
    start_date: date,
    end_date: date,
    *,
    page: int,
    per_page: int,
    language: str,
) -> dict[str, str | int]:
    """Build the public SET news-page query contract."""
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    if page < 0:
        raise ValueError("page must be non-negative")
    if per_page <= 0:
        raise ValueError("per_page must be positive")
    if language not in {"th", "en"}:
        raise ValueError("language must be 'th' or 'en'")
    return {
        "sourceId": "company",
        "securityTypeIds": "S",
        "fromDate": start_date.strftime("%d/%m/%Y"),
        "toDate": end_date.strftime("%d/%m/%Y"),
        "page": page,
        "perPage": per_page,
        "orderBy": "date",
        "lang": language,
    }


def parse_news_page(
    payload: Mapping[str, object],
) -> tuple[int, list[dict[str, object]]]:
    """Validate one SET response and return its total count and records."""
    paginate_news = payload.get("paginateNews")
    if not isinstance(paginate_news, Mapping):
        raise ValueError("SET response is missing a paginateNews object")

    total_count = paginate_news.get("totalCount")
    news_info_list = paginate_news.get("newsInfoList")
    if not isinstance(total_count, int) or total_count < 0:
        raise ValueError("SET response has an invalid totalCount")
    if not isinstance(news_info_list, list):
        raise ValueError("SET response has an invalid newsInfoList")
    if not all(isinstance(record, dict) for record in news_info_list):
        raise ValueError("SET response contains a non-object news record")
    return total_count, list(news_info_list)


def normalize_records(
    records: Iterable[Mapping[str, object]],
    *,
    expected_year: int,
    language: str,
) -> list[dict[str, object]]:
    """Normalize, validate, and exactly deduplicate SET headline metadata."""
    normalized_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for record in records:
        news_id = str(record.get("id") or "").strip()
        published_raw = str(record.get("datetime") or "").strip()
        symbol = str(record.get("symbol") or "").strip()
        headline = str(record.get("headline") or "").strip()
        if not news_id or not published_raw or not headline:
            raise ValueError("SET news record is missing id, datetime, or headline")

        try:
            published_at = datetime.fromisoformat(published_raw)
        except ValueError as error:
            raise ValueError(
                f"SET news record {news_id} has an invalid datetime"
            ) from error
        if published_at.year != expected_year:
            raise ValueError(
                f"SET news record {news_id} is outside requested year {expected_year}"
            )

        normalized = {
            "set_news_id": news_id,
            "published_at_bangkok": published_at.isoformat(),
            "publication_year": published_at.year,
            "symbol": symbol,
            "source": str(record.get("source") or "").strip(),
            "headline": headline,
            "url": str(record.get("url") or "").strip(),
            "product": str(record.get("product") or "").strip(),
            "language": language,
            "tag": str(record.get("tag") or "").strip(),
            "market_alert_type_id": record.get("marketAlertTypeId"),
            "view_clarification": record.get("viewClarification"),
            "percent_price_change": record.get("percentPriceChange"),
            "view_count": record.get("view"),
            "is_today_news": bool(record.get("isTodayNews", False)),
            "full_text_collected": False,
            "text_scope": "headline_and_metadata_only",
        }
        deduplication_key = (news_id, symbol, headline)
        normalized_by_key[deduplication_key] = normalized

    return sorted(
        normalized_by_key.values(),
        key=lambda row: (
            str(row["published_at_bangkok"]),
            str(row["set_news_id"]),
            str(row["symbol"]),
        ),
    )


def _request_json(
    session: requests.Session,
    *,
    params: Mapping[str, str | int],
) -> dict[str, object]:
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                SET_NEWS_API_URL,
                params=params,
                timeout=DEFAULT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("SET API returned a non-object JSON payload")
            return payload
        except (requests.RequestException, ValueError):
            if attempt + 1 == MAX_RETRIES:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable retry state")


def _load_or_download_page(
    session: requests.Session,
    *,
    raw_page_path: Path,
    params: Mapping[str, str | int],
    delay_seconds: float,
) -> tuple[dict[str, object], bool]:
    if raw_page_path.exists():
        payload = json.loads(raw_page_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Cached SET page is invalid: {raw_page_path}")
        return payload, True

    payload = _request_json(session, params=params)
    raw_page_path.parent.mkdir(parents=True, exist_ok=True)
    raw_page_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    return payload, False


def collect_year(
    session: requests.Session,
    *,
    year: int,
    language: str,
    per_page: int,
    output_dir: Path,
    delay_seconds: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Download or resume one complete calendar year of SET company headlines."""
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)
    raw_dir = output_dir / "raw-pages" / str(year) / language / f"per-page-{per_page}"
    started_at = time.perf_counter()
    all_records: list[dict[str, object]] = []
    cached_page_count = 0

    first_params = build_query_params(
        start_date,
        end_date,
        page=0,
        per_page=per_page,
        language=language,
    )
    first_payload, was_cached = _load_or_download_page(
        session,
        raw_page_path=raw_dir / "page-00000.json",
        params=first_params,
        delay_seconds=delay_seconds,
    )
    cached_page_count += int(was_cached)
    total_count, first_records = parse_news_page(first_payload)
    all_records.extend(first_records)
    page_count = math.ceil(total_count / per_page) if total_count else 1

    for page in range(1, page_count):
        params = build_query_params(
            start_date,
            end_date,
            page=page,
            per_page=per_page,
            language=language,
        )
        payload, was_cached = _load_or_download_page(
            session,
            raw_page_path=raw_dir / f"page-{page:05d}.json",
            params=params,
            delay_seconds=delay_seconds,
        )
        cached_page_count += int(was_cached)
        page_total_count, page_records = parse_news_page(payload)
        if page_total_count != total_count:
            raise ValueError(
                f"SET totalCount changed during download: "
                f"{total_count} -> {page_total_count}"
            )
        all_records.extend(page_records)

    if len(all_records) != total_count:
        raise ValueError(
            f"Expected {total_count} raw records for {year}, "
            f"received {len(all_records)}"
        )

    normalized = normalize_records(
        all_records,
        expected_year=year,
        language=language,
    )
    metadata: dict[str, object] = {
        "year": year,
        "language": language,
        "query_start_date": start_date.isoformat(),
        "query_end_date": end_date.isoformat(),
        "source_id": "company",
        "security_type_ids": "S",
        "order_by": "date",
        "per_page": per_page,
        "page_count": page_count,
        "cached_page_count": cached_page_count,
        "raw_record_count": len(all_records),
        "normalized_record_count": len(normalized),
        "exact_duplicate_count": len(all_records) - len(normalized),
        "minimum_datetime": (
            normalized[0]["published_at_bangkok"] if normalized else None
        ),
        "maximum_datetime": (
            normalized[-1]["published_at_bangkok"] if normalized else None
        ),
        "runtime_seconds": time.perf_counter() - started_at,
        "full_text_collected": False,
        "text_scope": "headline_and_metadata_only",
    }
    return normalized, metadata


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in CSV_FIELDS} for row in rows
        )


def _write_jsonl(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def download_years(
    years: Sequence[int],
    *,
    language: str,
    per_page: int,
    output_dir: Path,
    delay_seconds: float,
) -> dict[str, object]:
    """Download selected years and write reproducibility-ready data artifacts."""
    if not years:
        raise ValueError("At least one year is required")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be non-negative")

    run_started_at = datetime.now().astimezone()
    run_timer = time.perf_counter()
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Referer": SET_NEWS_PAGE_URL,
            "x-channel": "WEB_SET",
        }
    )
    warmup_response = session.get(
        SET_NEWS_PAGE_URL,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    warmup_response.raise_for_status()

    year_manifests: list[dict[str, object]] = []
    for year in sorted(set(years)):
        rows, year_metadata = collect_year(
            session,
            year=year,
            language=language,
            per_page=per_page,
            output_dir=output_dir,
            delay_seconds=delay_seconds,
        )
        stem = f"set_company_news_{year}_{language}"
        csv_path = output_dir / f"{stem}.csv"
        jsonl_path = output_dir / f"{stem}.jsonl"
        _write_csv(rows, csv_path)
        _write_jsonl(rows, jsonl_path)
        year_manifests.append(
            year_metadata
            | {
                "csv_path": csv_path.as_posix(),
                "csv_sha256": _sha256(csv_path),
                "jsonl_path": jsonl_path.as_posix(),
                "jsonl_sha256": _sha256(jsonl_path),
            }
        )

    manifest: dict[str, object] = {
        "source": "Stock Exchange of Thailand public news search",
        "source_page": SET_NEWS_PAGE_URL,
        "api_endpoint": SET_NEWS_API_URL,
        "retrieved_at": run_started_at.isoformat(),
        "years": year_manifests,
        "total_raw_records": sum(
            int(item["raw_record_count"]) for item in year_manifests
        ),
        "total_normalized_records": sum(
            int(item["normalized_record_count"]) for item in year_manifests
        ),
        "runtime_seconds": time.perf_counter() - run_timer,
        "usage_note": (
            "Official SET company-news headline metadata. It is not a "
            "pre-labelled sentiment dataset and is not yet filtered by "
            "point-in-time SET50 membership."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download reproducible SET company-news headline metadata."
    )
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--language", choices=("th", "en"), default="th")
    parser.add_argument(
        "--per-page",
        type=int,
        default=DEFAULT_PAGE_SIZE,
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = download_years(
        args.years,
        language=args.language,
        per_page=args.per_page,
        output_dir=args.output_dir,
        delay_seconds=args.delay_seconds,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
