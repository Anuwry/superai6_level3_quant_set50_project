from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

DATE_SCAN_CHARACTERS = 4_000
ORIGINAL_DATASET = "airesearch/CMDF_VISTEC"
DATE_EXTRACTION_METHOD = "first_exact_thai_date_line_in_first_4000_chars"

THAI_MONTHS = {
    "ม.ค.": 1,
    "มกราคม": 1,
    "ก.พ.": 2,
    "กุมภาพันธ์": 2,
    "มี.ค.": 3,
    "มีนาคม": 3,
    "เม.ย.": 4,
    "เมษายน": 4,
    "พ.ค.": 5,
    "พฤษภาคม": 5,
    "มิ.ย.": 6,
    "มิถุนายน": 6,
    "ก.ค.": 7,
    "กรกฎาคม": 7,
    "ส.ค.": 8,
    "สิงหาคม": 8,
    "ก.ย.": 9,
    "กันยายน": 9,
    "ต.ค.": 10,
    "ตุลาคม": 10,
    "พ.ย.": 11,
    "พฤศจิกายน": 11,
    "ธ.ค.": 12,
    "ธันวาคม": 12,
}
MONTH_PATTERN = "|".join(
    re.escape(month_name) for month_name in sorted(THAI_MONTHS, key=len, reverse=True)
)
PUBLICATION_DATE_PATTERN = re.compile(
    rf"(?m)^\s*(?P<day>\d{{1,2}})\s+"
    rf"(?P<month>{MONTH_PATTERN})\s+"
    rf"(?P<year>\d{{4}})\s*$"
)

METADATA_FIELDS = (
    "cmdf_id",
    "published_date",
    "publication_year",
    "source",
    "language",
    "text_scope",
    "original_dataset",
    "date_extraction_method",
)


def extract_thai_publication_date(text: str) -> date | None:
    """Parse the first standalone Thai date near the start of an article."""
    match = PUBLICATION_DATE_PATTERN.search(text[:DATE_SCAN_CHARACTERS])
    if match is None:
        return None

    day = int(match.group("day"))
    month = THAI_MONTHS[match.group("month")]
    raw_year = int(match.group("year"))
    year = raw_year - 543 if raw_year >= 2400 else raw_year
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _normalize_record(
    record: Mapping[str, object],
    *,
    published_date: date,
    source: str,
) -> dict[str, object]:
    return {
        "cmdf_id": str(record.get("id") or "").strip(),
        "published_date": published_date.isoformat(),
        "publication_year": published_date.year,
        "source": source,
        "language": "th",
        "text": str(record.get("text") or ""),
        "text_scope": "full_article_text_from_source_dataset",
        "original_dataset": ORIGINAL_DATASET,
        "date_extraction_method": DATE_EXTRACTION_METHOD,
    }


def select_dated_records(
    records: Iterable[Mapping[str, object]],
    *,
    start_date: date,
    end_date: date,
    source: str,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Select records for tests or small in-memory collections."""
    selected: list[dict[str, object]] = []
    rows_scanned = 0
    rows_with_parsed_date = 0
    rows_without_parsed_date = 0
    for record in records:
        rows_scanned += 1
        published_date = extract_thai_publication_date(str(record.get("text") or ""))
        if published_date is None:
            rows_without_parsed_date += 1
            continue
        rows_with_parsed_date += 1
        if start_date <= published_date <= end_date:
            selected.append(
                _normalize_record(
                    record,
                    published_date=published_date,
                    source=source,
                )
            )
    stats = {
        "rows_scanned": rows_scanned,
        "rows_with_parsed_date": rows_with_parsed_date,
        "rows_without_parsed_date": rows_without_parsed_date,
        "rows_in_requested_period": len(selected),
    }
    return selected, stats


def _increase_csv_field_limit() -> None:
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            return
        except OverflowError:
            field_limit //= 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def extract_file(
    input_csv: Path,
    output_dir: Path,
    *,
    start_date: date,
    end_date: date,
    source: str,
) -> dict[str, object]:
    """Stream a CMDF CSV and write the selected full-text JSONL plus metadata."""
    _increase_csv_field_limit()
    output_dir.mkdir(parents=True, exist_ok=True)
    period_stem = f"{source}_{start_date.year}_{end_date.year}"
    jsonl_path = output_dir / f"{period_stem}.jsonl"
    metadata_csv_path = output_dir / f"{period_stem}_metadata.csv"

    rows_scanned = 0
    rows_with_parsed_date = 0
    rows_without_parsed_date = 0
    rows_in_requested_period = 0
    selected_by_year: Counter[int] = Counter()
    minimum_date: date | None = None
    maximum_date: date | None = None
    started_at = time.perf_counter()

    with (
        input_csv.open(encoding="utf-8-sig", newline="") as input_file,
        jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl_file,
        metadata_csv_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as metadata_file,
    ):
        reader = csv.DictReader(input_file)
        if reader.fieldnames != ["id", "text"]:
            raise ValueError(
                f"Expected CMDF columns ['id', 'text'], got {reader.fieldnames}"
            )
        metadata_writer = csv.DictWriter(
            metadata_file,
            fieldnames=METADATA_FIELDS,
        )
        metadata_writer.writeheader()

        for record in reader:
            rows_scanned += 1
            published_date = extract_thai_publication_date(record.get("text", ""))
            if published_date is None:
                rows_without_parsed_date += 1
                continue
            rows_with_parsed_date += 1
            if not start_date <= published_date <= end_date:
                continue

            normalized = _normalize_record(
                record,
                published_date=published_date,
                source=source,
            )
            jsonl_file.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            metadata_writer.writerow(
                {field: normalized[field] for field in METADATA_FIELDS}
            )
            rows_in_requested_period += 1
            selected_by_year[published_date.year] += 1
            minimum_date = (
                min(minimum_date, published_date) if minimum_date else published_date
            )
            maximum_date = (
                max(maximum_date, published_date) if maximum_date else published_date
            )

    manifest: dict[str, object] = {
        "source": source,
        "original_dataset": ORIGINAL_DATASET,
        "input_csv": input_csv.as_posix(),
        "input_csv_sha256": _sha256(input_csv),
        "requested_start_date": start_date.isoformat(),
        "requested_end_date": end_date.isoformat(),
        "date_extraction_method": DATE_EXTRACTION_METHOD,
        "rows_scanned": rows_scanned,
        "rows_with_parsed_date": rows_with_parsed_date,
        "rows_without_parsed_date": rows_without_parsed_date,
        "parsed_date_rate": (
            rows_with_parsed_date / rows_scanned if rows_scanned else 0.0
        ),
        "rows_in_requested_period": rows_in_requested_period,
        "selected_by_year": {
            str(year): selected_by_year[year] for year in sorted(selected_by_year)
        },
        "minimum_selected_date": minimum_date.isoformat() if minimum_date else None,
        "maximum_selected_date": maximum_date.isoformat() if maximum_date else None,
        "jsonl_path": jsonl_path.as_posix(),
        "jsonl_sha256": _sha256(jsonl_path),
        "metadata_csv_path": metadata_csv_path.as_posix(),
        "metadata_csv_sha256": _sha256(metadata_csv_path),
        "runtime_seconds": time.perf_counter() - started_at,
        "sentiment_labels_present": False,
        "ticker_labels_present": False,
        "usage_note": (
            "Dates are heuristically parsed from the first standalone Thai "
            "date near the start of each article and require a manual QA sample."
        ),
    }
    (output_dir / f"{period_stem}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a dated period from a CMDF-VISTEC id,text CSV."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--source", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.start_date > args.end_date:
        raise ValueError("start-date must be on or before end-date")
    manifest = extract_file(
        args.input_csv,
        args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        source=args.source,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
