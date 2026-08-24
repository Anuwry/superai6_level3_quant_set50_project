from __future__ import annotations

import argparse
import csv
import html
import json
import re
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

GDELT_BASE_URL = "http://data.gdeltproject.org/gdeltv2"
GKG_FIELD_COUNT = 27
BANGKOK_TIMEZONE = timezone(timedelta(hours=7))
DEFAULT_MIN_RELEVANCE_SCORE = 5
DEFAULT_STEP_MINUTES = 60
MAX_PUBLICATION_LAG_DAYS = 7
USER_AGENT = "SET50-direction-research/0.1 (academic metadata collection)"

PAGE_TAG_TEMPLATE = r"<{tag}>(.*?)</{tag}>"
THAILAND_MARKERS = ("#thailand#th", "#thai#th")
MARKET_TITLE_PATTERNS: tuple[tuple[str, str, int], ...] = (
    ("set50", "title:set50", 6),
    ("set 50", "title:set_50", 6),
    ("set index", "title:set_index", 5),
    ("thai stock market", "title:thai_stock_market", 5),
    ("thailand stock market", "title:thailand_stock_market", 5),
    ("ตลาดหุ้นไทย", "title:thai_stock_market_th", 5),
    ("หุ้นไทย", "title:thai_stocks_th", 4),
)

CSV_FIELDS = (
    "gdelt_record_id",
    "gdelt_batch_at_utc",
    "published_at_utc",
    "published_at_bangkok",
    "publication_timestamp_source",
    "publication_lag_days",
    "is_publication_date_consistent",
    "source",
    "url",
    "title",
    "gdelt_tone",
    "positive_score",
    "negative_score",
    "polarity",
    "activity_reference_density",
    "self_group_reference_density",
    "word_count",
    "is_translated",
    "relevance_score",
    "relevance_reasons",
    "full_text_collected",
    "text_scope",
    "archive_name",
)


def extract_xml_tag(extras: str, tag: str) -> str:
    """Extract one XML-like metadata tag from GDELT's Extras field."""
    pattern = PAGE_TAG_TEMPLATE.format(tag=re.escape(tag))
    match = re.search(pattern, extras or "", flags=re.IGNORECASE | re.DOTALL)
    return html.unescape(match.group(1).strip()) if match else ""


def parse_tone(raw_tone: str) -> dict[str, float | int | None]:
    """Parse GDELT V1.5TONE into a small, explicit data contract."""
    names = (
        "gdelt_tone",
        "positive_score",
        "negative_score",
        "polarity",
        "activity_reference_density",
        "self_group_reference_density",
        "word_count",
    )
    empty = dict.fromkeys(names)
    if not raw_tone:
        return empty

    values = raw_tone.split(",")
    if len(values) < len(names):
        return empty

    parsed: dict[str, float | int | None] = {}
    for name, value in zip(names[:-1], values[:-1], strict=True):
        try:
            parsed[name] = float(value)
        except ValueError:
            parsed[name] = None
    try:
        parsed["word_count"] = int(float(values[-1]))
    except ValueError:
        parsed["word_count"] = None
    return parsed


def score_relevance(fields: Sequence[str]) -> tuple[int, tuple[str, ...]]:
    """Score SET50 relevance without using article body text."""
    if len(fields) != GKG_FIELD_COUNT:
        raise ValueError(f"GDELT GKG row must contain {GKG_FIELD_COUNT} fields")

    title = extract_xml_tag(fields[26], "PAGE_TITLE").casefold()
    themes = fields[7].casefold()
    locations = fields[9].casefold()
    organizations = fields[13].casefold()
    source = fields[3].casefold()
    reasons: list[str] = []
    score = 0

    for phrase, reason, weight in MARKET_TITLE_PATTERNS:
        if phrase in title:
            reasons.append(reason)
            score += weight
            break

    if "stock exchange of thailand" in organizations:
        reasons.append("organization:stock_exchange_of_thailand")
        score += 3
    if "econ_stockmarket" in themes:
        reasons.append("theme:econ_stockmarket")
        score += 1
    if any(marker in locations for marker in THAILAND_MARKERS):
        reasons.append("location:thailand")
        score += 1
    if source.endswith(".th") or "thailand" in source:
        reasons.append("source:thailand")
        score += 1

    return score, tuple(reasons)


def _parse_utc_timestamp(raw_timestamp: str) -> datetime:
    return datetime.strptime(raw_timestamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def build_article(
    fields: Sequence[str],
    *,
    archive_name: str,
) -> dict[str, object]:
    """Convert one validated GKG row to the metadata-only research schema."""
    if len(fields) != GKG_FIELD_COUNT:
        raise ValueError(f"GDELT GKG row must contain {GKG_FIELD_COUNT} fields")

    batch_at = _parse_utc_timestamp(fields[1])
    precise_timestamp = extract_xml_tag(fields[26], "PAGE_PRECISEPUBTIMESTAMP")
    if precise_timestamp and re.fullmatch(r"\d{14}", precise_timestamp):
        published_at = _parse_utc_timestamp(precise_timestamp)
        timestamp_source = "page_precise_timestamp"
    else:
        published_at = batch_at
        timestamp_source = "gdelt_batch_timestamp"

    relevance_score, relevance_reasons = score_relevance(fields)
    publication_lag_days = abs((batch_at - published_at).total_seconds()) / 86400
    article: dict[str, object] = {
        "gdelt_record_id": fields[0],
        "gdelt_batch_at_utc": batch_at.isoformat(),
        "published_at_utc": published_at.isoformat(),
        "published_at_bangkok": published_at.astimezone(BANGKOK_TIMEZONE).isoformat(),
        "publication_timestamp_source": timestamp_source,
        "publication_lag_days": publication_lag_days,
        "is_publication_date_consistent": (
            publication_lag_days <= MAX_PUBLICATION_LAG_DAYS
        ),
        "source": fields[3],
        "url": fields[4],
        "title": extract_xml_tag(fields[26], "PAGE_TITLE"),
        "is_translated": "-T" in fields[0],
        "relevance_score": relevance_score,
        "relevance_reasons": ";".join(relevance_reasons),
        "full_text_collected": False,
        "text_scope": "gdelt_metadata_and_title_only",
        "archive_name": archive_name,
    }
    return article | parse_tone(fields[15])


def iter_snapshot_stamps(
    query_date: date,
    *,
    step_minutes: int = DEFAULT_STEP_MINUTES,
) -> Iterator[str]:
    if step_minutes <= 0 or 1440 % step_minutes != 0:
        raise ValueError("step_minutes must be positive and divide 1440 exactly")

    current = datetime.combine(query_date, datetime.min.time())
    next_day = current + timedelta(days=1)
    while current < next_day:
        yield current.strftime("%Y%m%d%H%M%S")
        current += timedelta(minutes=step_minutes)


def download_archives(
    query_date: date,
    archive_dir: Path,
    *,
    step_minutes: int = DEFAULT_STEP_MINUTES,
) -> list[Path]:
    """Download immutable GDELT snapshots; existing valid ZIPs are reused."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for stamp in iter_snapshot_stamps(query_date, step_minutes=step_minutes):
        archive_path = archive_dir / f"{stamp}.gkg.csv.zip"
        if not archive_path.exists() or not zipfile.is_zipfile(archive_path):
            request = urllib.request.Request(
                f"{GDELT_BASE_URL}/{archive_path.name}",
                headers={"User-Agent": USER_AGENT},
            )
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                archive_path.open("wb") as output_file,
            ):
                output_file.write(response.read())
            if not zipfile.is_zipfile(archive_path):
                raise ValueError(f"Downloaded file is not a ZIP: {archive_path}")
        paths.append(archive_path)
    return paths


def iter_gkg_fields(archive_path: Path) -> Iterator[list[str]]:
    """Stream rows from a GKG ZIP without expanding the large archive."""
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".gkg.csv")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one .gkg.csv member in {archive_path}")
        with archive.open(members[0]) as raw_file:
            for raw_line in raw_file:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                fields = line.split("\t")
                if len(fields) == GKG_FIELD_COUNT:
                    yield fields


def collect_candidates(
    archive_paths: Iterable[Path],
    *,
    min_candidate_score: int = 2,
) -> list[dict[str, object]]:
    """Collect and URL-deduplicate candidates, preserving the best score."""
    by_url: dict[str, dict[str, object]] = {}
    for archive_path in archive_paths:
        for fields in iter_gkg_fields(archive_path):
            score, _ = score_relevance(fields)
            if score < min_candidate_score:
                continue
            article = build_article(fields, archive_name=archive_path.name)
            url = str(article["url"])
            previous = by_url.get(url)
            if previous is None or int(article["relevance_score"]) > int(
                previous["relevance_score"]
            ):
                by_url[url] = article
    return sorted(
        by_url.values(),
        key=lambda row: (
            str(row["published_at_utc"]),
            -int(row["relevance_score"]),
            str(row["url"]),
        ),
    )


def write_csv(rows: Iterable[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in CSV_FIELDS} for row in rows
        )


def write_outputs(
    candidates: list[dict[str, object]],
    output_dir: Path,
    *,
    query_date: date,
    step_minutes: int,
    min_relevance_score: int,
    archive_count: int,
) -> dict[str, object]:
    relevant = [
        row
        for row in candidates
        if int(row["relevance_score"]) >= min_relevance_score
        and bool(row["is_publication_date_consistent"])
    ]
    candidates_path = output_dir / "gdelt_candidates.csv"
    relevant_path = output_dir / "gdelt_relevant.csv"
    metadata_path = output_dir / "run_metadata.json"
    write_csv(candidates, candidates_path)
    write_csv(relevant, relevant_path)

    metadata: dict[str, object] = {
        "source": "GDELT 2.1 GKG",
        "query_date": query_date.isoformat(),
        "snapshot_step_minutes": step_minutes,
        "expected_full_day_snapshots": 96,
        "archives_scanned": archive_count,
        "temporal_sampling_fraction": archive_count / 96,
        "candidate_rule": "metadata relevance score >= 2",
        "strict_relevance_threshold": min_relevance_score,
        "maximum_publication_lag_days": MAX_PUBLICATION_LAG_DAYS,
        "candidate_count": len(candidates),
        "strict_relevant_count": len(relevant),
        "full_text_collected": False,
        "usage_note": (
            "Pilot metadata audit only; not a complete historical-news dataset "
            "and not ready for model benchmarking."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/scan a reproducible GDELT GKG metadata sample."
    )
    parser.add_argument("--date", required=True, help="UTC date in YYYY-MM-DD")
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--step-minutes",
        type=int,
        default=DEFAULT_STEP_MINUTES,
        help="15 for complete GDELT coverage; 60 for a 25%% pilot sample.",
    )
    parser.add_argument(
        "--min-relevance-score",
        type=int,
        default=DEFAULT_MIN_RELEVANCE_SCORE,
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing snapshots before scanning.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    query_date = date.fromisoformat(args.date)
    if args.download:
        archive_paths = download_archives(
            query_date,
            args.archive_dir,
            step_minutes=args.step_minutes,
        )
    else:
        archive_paths = sorted(args.archive_dir.glob(f"{query_date:%Y%m%d}*.zip"))
    if not archive_paths:
        raise FileNotFoundError(
            f"No GDELT ZIP archives found in {args.archive_dir}; use --download"
        )

    candidates = collect_candidates(archive_paths)
    metadata = write_outputs(
        candidates,
        args.output_dir,
        query_date=query_date,
        step_minutes=args.step_minutes,
        min_relevance_score=args.min_relevance_score,
        archive_count=len(archive_paths),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
