from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

POLARITY_LABELS = frozenset({"positive", "neutral", "negative"})
IRRELEVANT_LABELS = frozenset({"exclude", "not stock"})
AMBIGUOUS_LABEL = "ambiguous"

PAIR_COLUMNS = [
    "article_id",
    "source",
    "date",
    "year",
    "text",
    "ticker",
    "label",
]
DAILY_FEATURE_COLUMNS = [
    "news_sentiment_mean",
    "news_sentiment_std",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "article_count",
    "ticker_mention_count",
    "news_available",
]


def _require_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _required_text(mapping: dict[str, object], key: str, context: str) -> str:
    value = mapping.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{context} has an empty {key}")
    return str(value).strip()


def _article_date(value: object) -> pd.Timestamp:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Article Date must be Unix epoch milliseconds")
    timestamp = pd.to_datetime(value, unit="ms", utc=True)
    return timestamp.tz_convert("Asia/Bangkok").tz_localize(None).normalize()


def _article_pairs(article: object, index: int) -> list[dict[str, object]]:
    payload = _require_mapping(article, f"article[{index}]")
    article_id = _required_text(payload, "Article_ID", f"article[{index}]")
    source = _required_text(payload, "Data-Source", f"article[{index}]")
    text = _required_text(payload, "Text", f"article[{index}]")
    date = _article_date(payload.get("Date"))
    year = int(payload.get("Year"))
    if year != date.year:
        raise ValueError(f"article[{index}] Year does not match Date")
    sentiments = payload.get("Ticker_sentiments")
    if not isinstance(sentiments, list) or not sentiments:
        raise ValueError(f"article[{index}] has no ticker sentiments")

    rows: list[dict[str, object]] = []
    for sentiment_index, raw_sentiment in enumerate(sentiments):
        context = f"article[{index}].Ticker_sentiments[{sentiment_index}]"
        sentiment = _require_mapping(raw_sentiment, context)
        rows.append(
            {
                "article_id": article_id,
                "source": source,
                "date": date,
                "year": year,
                "text": text,
                "ticker": _required_text(sentiment, "ticker", context).upper(),
                "label": _required_text(sentiment, "sentiment", context).lower(),
            }
        )
    return rows


def load_stocktbsa_pairs(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"StockTBSA dataset not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("StockTBSA root must be a non-empty JSON array")
    rows = [
        row
        for article_index, article in enumerate(payload)
        for row in _article_pairs(article, article_index)
    ]
    frame = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    duplicated = frame.duplicated(["article_id", "ticker"], keep=False)
    if duplicated.any():
        raise ValueError("Duplicate article-ticker pairs found in StockTBSA")
    return frame.sort_values(["date", "article_id", "ticker"]).reset_index(drop=True)


def filter_polarity_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, {"label"})
    return frame.loc[frame["label"].isin(POLARITY_LABELS)].reset_index(drop=True)


def target_aware_text(ticker: object, text: object) -> str:
    ticker_value = str(ticker).strip().upper()
    text_value = " ".join(str(text).split())
    if not ticker_value or not text_value:
        raise ValueError("Ticker and article text must be non-empty")
    return f"[TARGET_TICKER] {ticker_value}\n[ARTICLE]\n{text_value}"


def split_by_year(
    frame: pd.DataFrame,
    *,
    train_years: Iterable[int],
    test_years: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(frame, {"year"})
    train_set = frozenset(int(year) for year in train_years)
    test_set = frozenset(int(year) for year in test_years)
    if not train_set or not test_set:
        raise ValueError("Train years and test years must be non-empty")
    if train_set.intersection(test_set):
        raise ValueError("Train and test years overlap")
    if max(train_set) >= min(test_set):
        raise ValueError("Train years must strictly precede test years")
    train = frame.loc[frame["year"].isin(train_set)].reset_index(drop=True)
    test = frame.loc[frame["year"].isin(test_set)].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty train or test set")
    return train, test


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Data frame is missing columns: {missing}")


def _trading_index(values: Iterable[object]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(list(values))).normalize()
    if index.hasnans:
        raise ValueError("Trading dates contain invalid timestamps")
    index = pd.DatetimeIndex(index.unique()).sort_values()
    if index.empty:
        raise ValueError("At least one trading date is required")
    return index


def _assign_next_trading_date(
    news_dates: pd.Series,
    trading_dates: pd.DatetimeIndex,
) -> pd.Series:
    normalized = pd.to_datetime(news_dates).dt.normalize()
    positions = trading_dates.searchsorted(normalized, side="right")
    assigned = pd.Series(pd.NaT, index=news_dates.index, dtype="datetime64[ns]")
    valid = positions < len(trading_dates)
    assigned.loc[valid] = trading_dates.take(positions[valid]).to_numpy()
    return assigned


def _empty_daily_features(trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    daily = pd.DataFrame({"date": trading_dates})
    count_columns = {"article_count", "ticker_mention_count", "news_available"}
    for column in DAILY_FEATURE_COLUMNS:
        daily[column] = 0 if column in count_columns else 0.0
    return daily


def aggregate_daily_sentiment(
    predictions: pd.DataFrame,
    trading_dates: Iterable[object],
) -> pd.DataFrame:
    required = {
        "article_id",
        "date",
        "ticker",
        "sentiment_score",
        "predicted_label",
    }
    _require_columns(predictions, required)
    market_dates = _trading_index(trading_dates)
    daily = _empty_daily_features(market_dates)
    if predictions.empty:
        return daily

    news = predictions.copy()
    news["assigned_date"] = _assign_next_trading_date(news["date"], market_dates)
    news = news.dropna(subset=["assigned_date"])
    if news.empty:
        return daily
    if not news["predicted_label"].isin(POLARITY_LABELS).all():
        raise ValueError("Predicted labels must be positive, neutral, or negative")
    if not np.isfinite(news["sentiment_score"].to_numpy(dtype=float)).all():
        raise ValueError("Sentiment scores must be finite")

    grouped = news.groupby("assigned_date", sort=True)
    aggregated = grouped.agg(
        news_sentiment_mean=("sentiment_score", "mean"),
        news_sentiment_std=("sentiment_score", lambda values: values.std(ddof=0)),
        article_count=("article_id", "nunique"),
        ticker_mention_count=("ticker", "size"),
    )
    label_counts = (
        news.assign(count=1)
        .pivot_table(
            index="assigned_date",
            columns="predicted_label",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=["positive", "negative", "neutral"], fill_value=0)
    )
    denominator = label_counts.sum(axis=1).replace(0, np.nan)
    aggregated["positive_ratio"] = label_counts["positive"] / denominator
    aggregated["negative_ratio"] = label_counts["negative"] / denominator
    aggregated["neutral_ratio"] = label_counts["neutral"] / denominator
    aggregated["news_available"] = 1

    result = daily.set_index("date")
    result.update(aggregated)
    result = result.reset_index()
    result[DAILY_FEATURE_COLUMNS] = result[DAILY_FEATURE_COLUMNS].fillna(0)
    for column in ("article_count", "ticker_mention_count", "news_available"):
        result[column] = result[column].astype(int)
    return result[["date", *DAILY_FEATURE_COLUMNS]]
