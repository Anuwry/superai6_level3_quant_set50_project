from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_common import DATE_COLUMN, PROJECT_ROOT, discover_folds
from models.track_b_baseline import (
    LocalRelevanceClassifier,
    LocalSentimentClassifier,
    classification_metrics,
    make_relevance_labels,
)
from models.track_b_data import (
    POLARITY_LABELS,
    aggregate_daily_sentiment,
    filter_polarity_pairs,
    load_stocktbsa_pairs,
)
from models.track_b_experiment import DATASET_PATH
from models.vmd_feature_pool import FULL_TA_VMD_DATA_FOLDS_DIR

RANDOM_SEED = 42
FROZEN_TRAINING_END_YEAR = 2023
RELEVANCE_THRESHOLD = 0.50
HEADLINE_FALLBACK_TOKENS = 24
FORWARD_YEARS = (2024, 2025)
HISTORICAL_INFERENCE_YEARS = (2019, 2020, 2021, 2022, 2023)

NEWS_ROOT = (
    PROJECT_ROOT / "data-raw" / "track_b" / "SET_company_news_2024_2025"
)
MEMBERSHIP_ROOT = (
    PROJECT_ROOT / "data-raw" / "track_b" / "SET50_membership_2024_2025"
)
MEMBERSHIP_FILE = MEMBERSHIP_ROOT / "set50_membership_intervals.csv"
MEMBERSHIP_MANIFEST_FILE = MEMBERSHIP_ROOT / "manifest.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_b" / "forward_news"
ALL_PREDICTIONS_FILE = OUTPUT_DIR / "headline_predictions_all_2019_2025.csv"
SELECTED_PREDICTIONS_FILE = (
    OUTPUT_DIR / "headline_predictions_selected_2019_2025.csv"
)
DAILY_NEWS_FILE = OUTPUT_DIR / "daily_news_features_2019_2025.csv"
INTRINSIC_METRICS_FILE = OUTPUT_DIR / "headline_proxy_intrinsic_metrics.csv"
RUNTIME_FILE = OUTPUT_DIR / "runtime_by_fit.csv"
DOMAIN_SHIFT_FILE = OUTPUT_DIR / "domain_shift_audit.csv"
RUN_METADATA_FILE = OUTPUT_DIR / "run_metadata.json"

GULF_TRANSITION_URL = (
    "https://www.set.or.th/en/market/news-and-alert/"
    "newsdetails?id=94908600&symbol=GULFI"
)
GULF_INCLUSION_URL = (
    "https://www.set.or.th/en/market/news-and-alert/"
    "newsdetails?id=95454300&symbol=SET"
)
VGI_INCLUSION_URL = (
    "https://www.set.or.th/en/market/news-and-alert/"
    "newsdetails?id=95454400&symbol=VGI"
)
INTUCH_EXCLUSION_URL = (
    "https://www.set.or.th/en/market/news-and-alert/"
    "newsdetails?id=95455900&symbol=SET"
)


@dataclass(frozen=True)
class MembershipDocument:
    key: str
    filename: str
    url: str


MEMBERSHIP_DOCUMENTS = (
    MembershipDocument(
        "2024_h1",
        "SET50_100_H1_2024.pdf",
        "https://media.set.or.th/set/Documents/2023/Dec/"
        "SET50_100_H1_2024.pdf",
    ),
    MembershipDocument(
        "2024_h2",
        "SET50_100_H2_2024.pdf",
        "https://media.set.or.th/set/Documents/2024/Jun/"
        "SET50_100_H2_2024.pdf",
    ),
    MembershipDocument(
        "2025_h1",
        "SET50_100_H1_2025.pdf",
        "https://media.set.or.th/set/Documents/2024/Dec/"
        "SET50_100_H1_2025.pdf",
    ),
    MembershipDocument(
        "2025_h1_revised",
        "SET50_100_H1_2025_revise.pdf",
        "https://media.set.or.th/set/Documents/2025/Feb/"
        "SET50_100_H1_2025_revise.pdf",
    ),
    MembershipDocument(
        "2025_h2",
        "SET50_100_H2_2025.pdf",
        "https://media.set.or.th/set/Documents/2025/Jun/"
        "SET50_100_H2_2025.pdf",
    ),
)

MEMBERSHIP_COLUMNS = [
    "effective_from",
    "effective_to",
    "symbol",
    "membership_version",
    "source_document",
    "source_url",
    "source_sha256",
    "change_source_url",
]

PREDICTION_KEYS = ["article_id", "date", "year", "ticker"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    context: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing columns: {missing}")


def _normalize_bangkok_timestamp(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Publication timestamps contain invalid values")
    return timestamps.dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)


def _download_document(document: MembershipDocument, *, force: bool) -> Path:
    MEMBERSHIP_ROOT.mkdir(parents=True, exist_ok=True)
    destination = MEMBERSHIP_ROOT / document.filename
    if destination.is_file() and not force:
        return destination
    request = urllib.request.Request(
        document.url,
        headers={"User-Agent": "SET50-direction-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
        content_type = response.headers.get("Content-Type", "")
    if not payload.startswith(b"%PDF") or "pdf" not in content_type.lower():
        raise ValueError(f"SET membership source is not a PDF: {document.url}")
    destination.write_bytes(payload)
    return destination


def extract_set50_symbols_from_pdf(path: str | Path) -> list[str]:
    from pypdf import PdfReader

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"SET50 constituent PDF not found: {source}")
    reader = PdfReader(source)
    if len(reader.pages) < 2:
        raise ValueError(f"SET50 constituent PDF has fewer than two pages: {source}")
    text = "\n".join((page.extract_text() or "") for page in reader.pages[:2])
    matches = re.findall(r"(?m)^\s*(\d{1,2})\s+([A-Z][A-Z0-9]*)\s+", text)

    symbols: list[str] = []
    next_rank = 1
    for raw_rank, symbol in matches:
        rank = int(raw_rank)
        if rank != next_rank:
            continue
        symbols.append(symbol.upper())
        next_rank += 1
        if next_rank == 51:
            break
    if len(symbols) != 50 or len(set(symbols)) != 50:
        raise ValueError(
            f"Expected 50 unique SET50 symbols in {source}, found {len(symbols)}"
        )
    return symbols


def _membership_rows(
    symbols: Sequence[str],
    *,
    effective_from: str,
    effective_to: str,
    membership_version: str,
    document: MembershipDocument,
    sha256: str,
    change_source_url: str = "",
) -> list[dict[str, object]]:
    return [
        {
            "effective_from": effective_from,
            "effective_to": effective_to,
            "symbol": symbol,
            "membership_version": membership_version,
            "source_document": document.filename,
            "source_url": document.url,
            "source_sha256": sha256,
            "change_source_url": change_source_url,
        }
        for symbol in symbols
    ]


def validate_membership_intervals(
    frame: pd.DataFrame,
    *,
    expected_size: int = 50,
) -> pd.DataFrame:
    _require_columns(
        frame,
        set(MEMBERSHIP_COLUMNS).difference({"change_source_url"}),
        context="membership intervals",
    )
    if expected_size <= 0:
        raise ValueError("Expected membership size must be positive")
    result = frame.copy()
    if "change_source_url" not in result:
        result["change_source_url"] = ""
    result["effective_from"] = pd.to_datetime(
        result["effective_from"], errors="coerce"
    ).dt.normalize()
    result["effective_to"] = pd.to_datetime(
        result["effective_to"], errors="coerce"
    ).dt.normalize()
    if result[["effective_from", "effective_to"]].isna().any().any():
        raise ValueError("Membership intervals contain invalid dates")
    if (result["effective_from"] > result["effective_to"]).any():
        raise ValueError("Membership interval starts after it ends")
    for column in (
        "symbol",
        "membership_version",
        "source_document",
        "source_url",
        "source_sha256",
    ):
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq("").any():
            raise ValueError(f"Membership intervals contain empty {column}")
    result["symbol"] = result["symbol"].str.upper()
    if not result["source_sha256"].str.fullmatch(r"[A-Fa-f0-9]{64}").all():
        raise ValueError("Membership intervals contain invalid SHA-256 values")

    versions: list[dict[str, object]] = []
    for version, group in result.groupby("membership_version", sort=False):
        if len(group) != expected_size or group["symbol"].nunique() != expected_size:
            raise ValueError(
                f"Membership version {version} must contain "
                f"{expected_size} unique symbols"
            )
        if group["effective_from"].nunique() != 1 or group["effective_to"].nunique() != 1:
            raise ValueError(
                f"Membership version {version} contains inconsistent dates"
            )
        versions.append(
            {
                "membership_version": version,
                "effective_from": group["effective_from"].iloc[0],
                "effective_to": group["effective_to"].iloc[0],
            }
        )
    version_frame = pd.DataFrame(versions).sort_values("effective_from")
    previous_end: pd.Timestamp | None = None
    for row in version_frame.itertuples(index=False):
        if previous_end is not None and row.effective_from <= previous_end:
            raise ValueError("Membership versions overlap")
        previous_end = row.effective_to

    if result.duplicated(["membership_version", "symbol"]).any():
        raise ValueError("Membership intervals contain duplicate version-symbol rows")
    return result.loc[:, MEMBERSHIP_COLUMNS].sort_values(
        ["effective_from", "symbol"]
    ).reset_index(drop=True)


def build_membership_artifacts(*, force_download: bool = False) -> pd.DataFrame:
    retrieved_at = _utc_now()
    documents = {document.key: document for document in MEMBERSHIP_DOCUMENTS}
    paths = {
        key: _download_document(document, force=force_download)
        for key, document in documents.items()
    }
    hashes = {key: _sha256(path) for key, path in paths.items()}
    symbols = {
        key: extract_set50_symbols_from_pdf(path) for key, path in paths.items()
    }

    original_h1_2025 = symbols["2025_h1"]
    if "GULF" not in original_h1_2025 or "INTUCH" not in original_h1_2025:
        raise ValueError("Original 2025 H1 membership lacks GULF or INTUCH")
    transition_h1_2025 = [
        "GULFI" if symbol == "GULF" else symbol for symbol in original_h1_2025
    ]
    revised_h1_2025 = symbols["2025_h1_revised"]
    if not {"GULF", "VGI"}.issubset(revised_h1_2025):
        raise ValueError("Revised 2025 H1 membership lacks GULF or VGI")
    if {"GULFI", "INTUCH"}.intersection(revised_h1_2025):
        raise ValueError("Revised 2025 H1 membership retains GULFI or INTUCH")

    rows: list[dict[str, object]] = []
    rows.extend(
        _membership_rows(
            symbols["2024_h1"],
            effective_from="2024-01-02",
            effective_to="2024-06-30",
            membership_version="2024_h1",
            document=documents["2024_h1"],
            sha256=hashes["2024_h1"],
        )
    )
    rows.extend(
        _membership_rows(
            symbols["2024_h2"],
            effective_from="2024-07-01",
            effective_to="2024-12-31",
            membership_version="2024_h2",
            document=documents["2024_h2"],
            sha256=hashes["2024_h2"],
        )
    )
    rows.extend(
        _membership_rows(
            original_h1_2025,
            effective_from="2025-01-01",
            effective_to="2025-03-20",
            membership_version="2025_h1_pre_symbol_change",
            document=documents["2025_h1"],
            sha256=hashes["2025_h1"],
        )
    )
    rows.extend(
        _membership_rows(
            transition_h1_2025,
            effective_from="2025-03-21",
            effective_to="2025-04-01",
            membership_version="2025_h1_gulfi_transition",
            document=documents["2025_h1"],
            sha256=hashes["2025_h1"],
            change_source_url=GULF_TRANSITION_URL,
        )
    )
    rows.extend(
        _membership_rows(
            revised_h1_2025,
            effective_from="2025-04-02",
            effective_to="2025-06-30",
            membership_version="2025_h1_post_amalgamation",
            document=documents["2025_h1_revised"],
            sha256=hashes["2025_h1_revised"],
            change_source_url=(
                f"{GULF_INCLUSION_URL};{VGI_INCLUSION_URL};"
                f"{INTUCH_EXCLUSION_URL}"
            ),
        )
    )
    rows.extend(
        _membership_rows(
            symbols["2025_h2"],
            effective_from="2025-07-01",
            effective_to="2025-12-31",
            membership_version="2025_h2",
            document=documents["2025_h2"],
            sha256=hashes["2025_h2"],
        )
    )
    membership = validate_membership_intervals(pd.DataFrame(rows))
    MEMBERSHIP_ROOT.mkdir(parents=True, exist_ok=True)
    membership.to_csv(MEMBERSHIP_FILE, index=False, date_format="%Y-%m-%d")
    manifest = {
        "created_at": retrieved_at,
        "population": "point-in-time SET50 constituents",
        "interval_file": str(MEMBERSHIP_FILE.relative_to(PROJECT_ROOT)),
        "interval_sha256": _sha256(MEMBERSHIP_FILE),
        "interval_rows": len(membership),
        "membership_versions": int(membership["membership_version"].nunique()),
        "members_per_version": 50,
        "documents": [
            {
                **asdict(document),
                "path": str(paths[document.key].relative_to(PROJECT_ROOT)),
                "sha256": hashes[document.key],
                "set50_symbol_count": len(symbols[document.key]),
            }
            for document in MEMBERSHIP_DOCUMENTS
        ],
        "midcycle_sources": {
            "gulf_to_gulfi_effective_2025_03_21": GULF_TRANSITION_URL,
            "gulf_inclusion_effective_2025_04_02": GULF_INCLUSION_URL,
            "vgi_inclusion_effective_2025_04_02": VGI_INCLUSION_URL,
            "intuch_exclusion_effective_2025_04_02": INTUCH_EXCLUSION_URL,
        },
    }
    _write_json(MEMBERSHIP_MANIFEST_FILE, manifest)
    return membership


def load_membership_intervals(
    path: Path = MEMBERSHIP_FILE,
) -> pd.DataFrame:
    if not path.is_file():
        return build_membership_artifacts()
    return validate_membership_intervals(pd.read_csv(path))


def filter_point_in_time_membership(
    news: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        news,
        {"set_news_id", "published_at_bangkok", "symbol", "headline"},
        context="official SET news",
    )
    _require_columns(
        membership,
        {
            "effective_from",
            "effective_to",
            "symbol",
            "membership_version",
        },
        context="membership intervals",
    )
    items = news.copy().reset_index(drop=True)
    items["_news_row"] = np.arange(len(items))
    items["symbol"] = items["symbol"].astype(str).str.strip().str.upper()
    items["publication_date"] = _normalize_bangkok_timestamp(
        items["published_at_bangkok"]
    ).dt.normalize()

    intervals = membership.copy()
    intervals["symbol"] = intervals["symbol"].astype(str).str.strip().str.upper()
    intervals["effective_from"] = pd.to_datetime(
        intervals["effective_from"]
    ).dt.normalize()
    intervals["effective_to"] = pd.to_datetime(
        intervals["effective_to"]
    ).dt.normalize()
    matched = items.merge(
        intervals,
        on="symbol",
        how="inner",
        suffixes=("", "_membership"),
    )
    matched = matched.loc[
        matched["publication_date"].between(
            matched["effective_from"],
            matched["effective_to"],
            inclusive="both",
        )
    ]
    if matched["_news_row"].duplicated().any():
        raise ValueError("A news item matched overlapping membership intervals")
    return matched.sort_values("_news_row").drop(columns="_news_row").reset_index(
        drop=True
    )


def headline_proxy(
    text: object,
    *,
    fallback_tokens: int = HEADLINE_FALLBACK_TOKENS,
    minimum_repeat_tokens: int = 5,
) -> str:
    normalized = " ".join(str(text).split())
    if not normalized:
        raise ValueError("Article text must be non-empty")
    if fallback_tokens <= 0 or minimum_repeat_tokens <= 0:
        raise ValueError("Headline token limits must be positive")
    tokens = normalized.split()
    upper = min(len(tokens) // 2, 100)
    for size in range(minimum_repeat_tokens, upper + 1):
        if tokens[:size] == tokens[size : 2 * size]:
            return " ".join(tokens[:size])
    return " ".join(tokens[:fallback_tokens])


def prepare_training_headlines(pairs: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        pairs,
        {"article_id", "source", "date", "year", "text", "ticker", "label"},
        context="StockTBSA pairs",
    )
    result = pairs.copy()
    result["text"] = result["text"].map(headline_proxy)
    result["text_contract"] = "stocktbsa_repeated_prefix_or_24_tokens"
    return result


def prepare_official_headlines(news: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        news,
        {
            "set_news_id",
            "published_at_bangkok",
            "symbol",
            "source",
            "headline",
            "membership_version",
        },
        context="eligible official SET news",
    )
    timestamps = _normalize_bangkok_timestamp(news["published_at_bangkok"])
    dates = timestamps.dt.normalize()
    text = news["headline"].map(lambda value: " ".join(str(value).split()))
    ticker = news["symbol"].astype(str).str.strip().str.upper()
    if text.eq("").any() or ticker.eq("").any():
        raise ValueError("Official SET news contains empty headline or symbol")
    result = pd.DataFrame(
        {
            "article_id": news["set_news_id"].astype(str).str.strip(),
            "source": news["source"].astype(str).str.strip(),
            "date": dates,
            "year": dates.dt.year.astype(int),
            "text": text,
            "ticker": ticker,
            "published_at_bangkok": timestamps,
            "membership_version": news["membership_version"].astype(str),
            "text_contract": "official_set_headline",
        }
    )
    if result["article_id"].eq("").any():
        raise ValueError("Official SET news contains empty news IDs")
    if result.duplicated(["article_id", "ticker"]).any():
        raise ValueError("Official SET news contains duplicate news-symbol pairs")
    return result.reset_index(drop=True)


def inference_training_years(
    inference_year: int,
    available_years: Iterable[int],
    *,
    frozen_end_year: int = FROZEN_TRAINING_END_YEAR,
) -> tuple[int, ...]:
    year = int(inference_year)
    cutoff = min(year - 1, int(frozen_end_year))
    selected = tuple(
        sorted({int(value) for value in available_years if int(value) <= cutoff})
    )
    if not selected:
        raise ValueError(f"No labelled training years precede {inference_year}")
    if max(selected) >= year and year <= frozen_end_year:
        raise ValueError("Inference training years leak the inference year")
    return selected


def _merge_all_predictions(
    relevance: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        relevance,
        {
            *PREDICTION_KEYS,
            "relevant_probability",
            "predicted_relevant",
        },
        context="relevance predictions",
    )
    _require_columns(
        sentiment,
        {
            *PREDICTION_KEYS,
            "positive_probability",
            "neutral_probability",
            "negative_probability",
            "predicted_label",
            "sentiment_score",
            "confidence",
        },
        context="sentiment predictions",
    )
    sentiment_columns = [
        *PREDICTION_KEYS,
        *(
            ["label"]
            if "label" in sentiment.columns and "label" not in relevance.columns
            else []
        ),
        "positive_probability",
        "neutral_probability",
        "negative_probability",
        "predicted_label",
        "sentiment_score",
        "confidence",
    ]
    merged = relevance.merge(
        sentiment.loc[:, sentiment_columns],
        on=PREDICTION_KEYS,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(relevance) or len(merged) != len(sentiment):
        raise ValueError("Relevance and sentiment predictions do not align")
    expected_relevant = merged["relevant_probability"].ge(RELEVANCE_THRESHOLD).astype(
        int
    )
    if not expected_relevant.eq(merged["predicted_relevant"].astype(int)).all():
        raise ValueError("Relevance predictions violate the locked 0.50 threshold")
    return merged


def select_relevant_sentiment_predictions(
    relevance: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> pd.DataFrame:
    merged = _merge_all_predictions(relevance, sentiment)
    return merged.loc[merged["predicted_relevant"].eq(1)].reset_index(drop=True)


def _prediction_metrics(
    all_predictions: pd.DataFrame,
    *,
    inference_year: int,
    train_years: Sequence[int],
) -> list[dict[str, object]]:
    if "label" not in all_predictions:
        return []
    metrics: list[dict[str, object]] = []
    relevance_rows = all_predictions.loc[
        ~all_predictions["label"].eq("ambiguous")
    ].copy()
    expected_relevance = make_relevance_labels(relevance_rows["label"]).astype(int)
    relevance_rows["expected_relevance"] = np.where(
        expected_relevance.eq(1), "relevant", "irrelevant"
    )
    relevance_rows["predicted_relevance"] = np.where(
        relevance_rows["predicted_relevant"].eq(1), "relevant", "irrelevant"
    )
    metrics.append(
        {
            "task": "relevance",
            "inference_year": inference_year,
            "train_start_year": min(train_years),
            "train_end_year": max(train_years),
            "train_years": ",".join(map(str, train_years)),
            "test_pairs": len(relevance_rows),
            **classification_metrics(
                relevance_rows["expected_relevance"],
                relevance_rows["predicted_relevance"],
                labels=("relevant", "irrelevant"),
            ),
        }
    )
    sentiment_rows = all_predictions.loc[
        all_predictions["label"].isin(POLARITY_LABELS)
    ]
    metrics.append(
        {
            "task": "sentiment",
            "inference_year": inference_year,
            "train_start_year": min(train_years),
            "train_end_year": max(train_years),
            "train_years": ",".join(map(str, train_years)),
            "test_pairs": len(sentiment_rows),
            **classification_metrics(
                sentiment_rows["label"],
                sentiment_rows["predicted_label"],
                labels=("positive", "neutral", "negative"),
            ),
        }
    )
    return metrics


def _fit_score(
    training_pairs: pd.DataFrame,
    inference_pairs: pd.DataFrame,
    *,
    train_years: Sequence[int],
    inference_label: str,
) -> tuple[
    pd.DataFrame,
    list[dict[str, object]],
    tuple[LocalRelevanceClassifier, LocalSentimentClassifier],
]:
    train = training_pairs.loc[training_pairs["year"].isin(train_years)].reset_index(
        drop=True
    )
    if train.empty:
        raise ValueError(f"Training data is empty for {inference_label}")
    relevance = LocalRelevanceClassifier(
        random_seed=RANDOM_SEED,
        threshold=RELEVANCE_THRESHOLD,
    ).fit(train)
    sentiment = LocalSentimentClassifier(random_seed=RANDOM_SEED).fit(
        filter_polarity_pairs(train)
    )
    relevance_predictions = relevance.predict(inference_pairs)
    sentiment_predictions = sentiment.predict(inference_pairs)
    all_predictions = _merge_all_predictions(
        relevance_predictions,
        sentiment_predictions,
    )
    all_predictions["included_in_daily"] = all_predictions[
        "predicted_relevant"
    ].astype(int)
    runtime_rows = [
        {
            "inference_label": inference_label,
            "task": "relevance",
            "train_start_year": min(train_years),
            "train_end_year": max(train_years),
            "train_pairs": len(train),
            "inference_pairs": len(inference_pairs),
            "fit_seconds": relevance.runtime.fit_seconds,
            "predict_seconds": relevance.runtime.predict_seconds,
        },
        {
            "inference_label": inference_label,
            "task": "sentiment",
            "train_start_year": min(train_years),
            "train_end_year": max(train_years),
            "train_pairs": len(filter_polarity_pairs(train)),
            "inference_pairs": len(inference_pairs),
            "fit_seconds": sentiment.runtime.fit_seconds,
            "predict_seconds": sentiment.runtime.predict_seconds,
        },
    ]
    return all_predictions, runtime_rows, (relevance, sentiment)


def _load_forward_news(news_root: Path = NEWS_ROOT) -> pd.DataFrame:
    paths = [news_root / f"set_company_news_{year}_th.csv" for year in FORWARD_YEARS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Forward SET news files not found: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if len(frame) != len(frame.drop_duplicates()):
        raise ValueError("Forward SET news contains exact duplicate rows")
    return frame


def _market_dates_2019_2025(
    source_dir: Path = FULL_TA_VMD_DATA_FOLDS_DIR,
) -> pd.DatetimeIndex:
    values: list[pd.Series] = []
    for spec in discover_folds(source_dir):
        values.extend(
            [
                pd.read_csv(spec.train_path, usecols=[DATE_COLUMN])[DATE_COLUMN],
                pd.read_csv(spec.test_path, usecols=[DATE_COLUMN])[DATE_COLUMN],
            ]
        )
    if not values:
        raise ValueError("No market dates found for forward Track B news")
    dates = pd.DatetimeIndex(
        pd.to_datetime(pd.concat(values, ignore_index=True)).unique()
    ).normalize()
    dates = dates[(dates.year >= 2019) & (dates.year <= 2025)].sort_values()
    if dates.empty or dates.year.min() != 2019 or dates.year.max() != 2025:
        raise ValueError("Market calendar does not cover 2019-2025")
    return dates


def _compact_metrics(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "task",
        "inference_year",
        "train_start_year",
        "train_end_year",
        "train_years",
        "test_pairs",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
    ]
    return pd.DataFrame([{key: row[key] for key in columns} for row in rows])


def _vectorizer_coverage(
    model: LocalRelevanceClassifier | LocalSentimentClassifier,
    frame: pd.DataFrame,
) -> tuple[float, float]:
    matrix = model.pipeline.named_steps["tfidf"].transform(model._inputs(frame))
    nnz = np.diff(matrix.indptr)
    return float(np.mean(nnz > 0)), float(np.mean(nnz))


def _domain_shift_audit(
    historical: pd.DataFrame,
    forward: pd.DataFrame,
    predictions: pd.DataFrame,
    frozen_models: tuple[LocalRelevanceClassifier, LocalSentimentClassifier],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year, group in historical.groupby("year"):
        rows.append(
            {
                "year": int(year),
                "source_contract": "stocktbsa_headline_proxy",
                "pairs_before_membership": len(group),
                "pairs_after_membership": len(group),
                "headline_chars_mean": float(group["text"].str.len().mean()),
                "headline_chars_median": float(group["text"].str.len().median()),
            }
        )
    for year, group in forward.groupby("year"):
        rows.append(
            {
                "year": int(year),
                "source_contract": "official_set_headline_point_in_time_set50",
                "pairs_before_membership": np.nan,
                "pairs_after_membership": len(group),
                "headline_chars_mean": float(group["text"].str.len().mean()),
                "headline_chars_median": float(group["text"].str.len().median()),
            }
        )
    result = pd.DataFrame(rows)
    prediction_summary = (
        predictions.groupby("year")
        .agg(
            predicted_relevant_rate=("predicted_relevant", "mean"),
            relevance_probability_mean=("relevant_probability", "mean"),
            sentiment_confidence_mean=("confidence", "mean"),
            selected_pairs=("included_in_daily", "sum"),
        )
        .reset_index()
    )
    class_ratios = (
        pd.crosstab(
            predictions["year"],
            predictions["predicted_label"],
            normalize="index",
        )
        .reindex(columns=["positive", "neutral", "negative"], fill_value=0.0)
        .rename(
            columns={
                "positive": "predicted_positive_ratio",
                "neutral": "predicted_neutral_ratio",
                "negative": "predicted_negative_ratio",
            }
        )
        .reset_index()
    )
    coverage_rows: list[dict[str, object]] = []
    relevance_model, sentiment_model = frozen_models
    for year, group in predictions.groupby("year"):
        relevance_rate, relevance_nnz = _vectorizer_coverage(
            relevance_model, group
        )
        sentiment_rate, sentiment_nnz = _vectorizer_coverage(
            sentiment_model, group
        )
        coverage_rows.append(
            {
                "year": int(year),
                "relevance_nonzero_vector_rate": relevance_rate,
                "relevance_char_ngram_nnz_mean": relevance_nnz,
                "sentiment_nonzero_vector_rate": sentiment_rate,
                "sentiment_char_ngram_nnz_mean": sentiment_nnz,
            }
        )
    return (
        result.merge(prediction_summary, on="year", how="left")
        .merge(class_ratios, on="year", how="left")
        .merge(pd.DataFrame(coverage_rows), on="year", how="left")
    )


def run_forward_news_pipeline(
    *,
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = OUTPUT_DIR,
    force_membership_download: bool = False,
) -> dict[str, object]:
    started_at = _utc_now()
    started = time.perf_counter()
    membership = build_membership_artifacts(
        force_download=force_membership_download
    )
    raw_forward = _load_forward_news()
    eligible_forward = filter_point_in_time_membership(raw_forward, membership)
    forward_pairs = prepare_official_headlines(eligible_forward)
    training_pairs = prepare_training_headlines(load_stocktbsa_pairs(dataset_path))
    available_years = sorted(training_pairs["year"].unique())

    all_runs: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for year in HISTORICAL_INFERENCE_YEARS:
        inference = training_pairs.loc[training_pairs["year"].eq(year)].reset_index(
            drop=True
        )
        train_years = inference_training_years(year, available_years)
        predictions, runtimes, _ = _fit_score(
            training_pairs,
            inference,
            train_years=train_years,
            inference_label=str(year),
        )
        predictions["inference_protocol"] = "expanding_labelled"
        all_runs.append(predictions)
        runtime_rows.extend(runtimes)
        metric_rows.extend(
            _prediction_metrics(
                predictions,
                inference_year=year,
                train_years=train_years,
            )
        )

    frozen_train_years = inference_training_years(2024, available_years)
    forward_predictions, forward_runtimes, frozen_models = _fit_score(
        training_pairs,
        forward_pairs,
        train_years=frozen_train_years,
        inference_label="2024_2025_frozen",
    )
    forward_predictions["inference_protocol"] = "frozen_through_2023"
    all_runs.append(forward_predictions)
    runtime_rows.extend(forward_runtimes)

    predictions = pd.concat(all_runs, ignore_index=True)
    metadata_columns = [
        *PREDICTION_KEYS,
        "source",
        "text",
        "text_contract",
        "membership_version",
    ]
    historical_metadata = training_pairs.assign(membership_version="")[
        metadata_columns
    ]
    forward_metadata = forward_pairs[metadata_columns]
    item_metadata = pd.concat(
        [historical_metadata, forward_metadata],
        ignore_index=True,
    )
    predictions = predictions.merge(
        item_metadata,
        on=PREDICTION_KEYS,
        how="left",
        validate="one_to_one",
    )
    if predictions[["source", "text", "text_contract"]].isna().any().any():
        raise ValueError("Prediction metadata did not align with inference items")
    predictions = predictions.sort_values(
        ["date", "article_id", "ticker"]
    ).reset_index(drop=True)
    selected = predictions.loc[predictions["included_in_daily"].eq(1)].copy()
    daily = aggregate_daily_sentiment(selected, _market_dates_2019_2025())

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / ALL_PREDICTIONS_FILE.name, index=False)
    selected.to_csv(output_dir / SELECTED_PREDICTIONS_FILE.name, index=False)
    daily.to_csv(output_dir / DAILY_NEWS_FILE.name, index=False)
    intrinsic = _compact_metrics(metric_rows)
    intrinsic.to_csv(output_dir / INTRINSIC_METRICS_FILE.name, index=False)
    runtime = pd.DataFrame(runtime_rows)
    runtime.to_csv(output_dir / RUNTIME_FILE.name, index=False)
    domain_shift = _domain_shift_audit(
        training_pairs.loc[
            training_pairs["year"].isin(HISTORICAL_INFERENCE_YEARS)
        ],
        forward_pairs,
        predictions,
        frozen_models,
    )
    before_counts = (
        raw_forward.assign(
            year=_normalize_bangkok_timestamp(
                raw_forward["published_at_bangkok"]
            ).dt.year
        )
        .groupby("year")
        .size()
    )
    domain_shift["pairs_before_membership"] = domain_shift.apply(
        lambda row: (
            int(before_counts.get(row["year"], 0))
            if row["year"] in FORWARD_YEARS
            else row["pairs_before_membership"]
        ),
        axis=1,
    )
    domain_shift.to_csv(output_dir / DOMAIN_SHIFT_FILE.name, index=False)

    elapsed = time.perf_counter() - started
    metadata = {
        "started_at": started_at,
        "completed_at": _utc_now(),
        "total_runtime_seconds": elapsed,
        "random_seed": RANDOM_SEED,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "headline_fallback_tokens": HEADLINE_FALLBACK_TOKENS,
        "frozen_training_end_year": FROZEN_TRAINING_END_YEAR,
        "forward_training_years": list(frozen_train_years),
        "pseudo_label_retraining": False,
        "optuna_tuning": False,
        "temporal_assignment": "strictly_next_trading_date",
        "raw_forward_rows": len(raw_forward),
        "point_in_time_set50_rows": len(forward_pairs),
        "selected_forward_rows": int(
            forward_predictions["included_in_daily"].sum()
        ),
        "prediction_rows": len(predictions),
        "selected_prediction_rows": len(selected),
        "daily_rows": len(daily),
        "daily_start": daily["date"].min().date().isoformat(),
        "daily_end": daily["date"].max().date().isoformat(),
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_sha256": _sha256(dataset_path),
        "membership_path": str(MEMBERSHIP_FILE.relative_to(PROJECT_ROOT)),
        "membership_sha256": _sha256(MEMBERSHIP_FILE),
        "news_manifest_sha256": _sha256(NEWS_ROOT / "manifest.json"),
        "news_inputs": [
            {
                "year": year,
                "path": str(
                    (
                        NEWS_ROOT / f"set_company_news_{year}_th.csv"
                    ).relative_to(PROJECT_ROOT)
                ),
                "sha256": _sha256(
                    NEWS_ROOT / f"set_company_news_{year}_th.csv"
                ),
            }
            for year in FORWARD_YEARS
        ],
        "outputs": {
            "all_predictions": str(
                (output_dir / ALL_PREDICTIONS_FILE.name).relative_to(PROJECT_ROOT)
            ),
            "selected_predictions": str(
                (
                    output_dir / SELECTED_PREDICTIONS_FILE.name
                ).relative_to(PROJECT_ROOT)
            ),
            "daily_news": str(
                (output_dir / DAILY_NEWS_FILE.name).relative_to(PROJECT_ROOT)
            ),
            "intrinsic_metrics": str(
                (
                    output_dir / INTRINSIC_METRICS_FILE.name
                ).relative_to(PROJECT_ROOT)
            ),
            "runtime": str((output_dir / RUNTIME_FILE.name).relative_to(PROJECT_ROOT)),
            "domain_shift": str(
                (output_dir / DOMAIN_SHIFT_FILE.name).relative_to(PROJECT_ROOT)
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "scipy": importlib.metadata.version("scipy"),
            "pypdf": importlib.metadata.version("pypdf"),
        },
    }
    _write_json(output_dir / RUN_METADATA_FILE.name, metadata)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build point-in-time SET50 forward news features for Track B."
    )
    parser.add_argument(
        "--membership-only",
        action="store_true",
        help="Download, parse, and validate only the official membership artifacts.",
    )
    parser.add_argument(
        "--force-membership-download",
        action="store_true",
        help="Redownload official SET constituent PDFs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.membership_only:
        membership = build_membership_artifacts(
            force_download=args.force_membership_download
        )
        print(
            f"Wrote {len(membership)} membership rows across "
            f"{membership['membership_version'].nunique()} versions"
        )
        return 0
    metadata = run_forward_news_pipeline(
        force_membership_download=args.force_membership_download
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
