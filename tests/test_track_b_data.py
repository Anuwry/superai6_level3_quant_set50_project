import json

import numpy as np
import pandas as pd
import pytest

from models.track_b_data import (
    POLARITY_LABELS,
    aggregate_daily_sentiment,
    filter_polarity_pairs,
    load_stocktbsa_pairs,
    split_by_year,
    target_aware_text,
)


def _write_fixture(path):
    articles = [
        {
            "Article_ID": "a1",
            "Data-Source": "source-a",
            "Date": 1672963200000,
            "Year": 2023,
            "Text": "บริษัทมีกำไรเพิ่มขึ้น",
            "Ticker_sentiments": [
                {"ticker": "AAA", "sentiment": "positive"},
                {"ticker": "BBB", "sentiment": "exclude"},
            ],
        },
        {
            "Article_ID": "a2",
            "Data-Source": "source-b",
            "Date": 1673222400000,
            "Year": 2023,
            "Text": "บริษัทชี้แจงข่าว",
            "Ticker_sentiments": [
                {"ticker": "AAA", "sentiment": "neutral"},
            ],
        },
    ]
    path.write_text(json.dumps(articles, ensure_ascii=False), encoding="utf-8")


def test_load_stocktbsa_pairs_expands_article_ticker_labels(tmp_path):
    source = tmp_path / "stocktbsa.json"
    _write_fixture(source)

    pairs = load_stocktbsa_pairs(source)

    assert pairs.columns.tolist() == [
        "article_id",
        "source",
        "date",
        "year",
        "text",
        "ticker",
        "label",
    ]
    assert len(pairs) == 3
    assert pairs["article_id"].tolist() == ["a1", "a1", "a2"]
    assert pairs["ticker"].tolist() == ["AAA", "BBB", "AAA"]
    assert pairs["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2023-01-06",
        "2023-01-06",
        "2023-01-09",
    ]


def test_filter_polarity_pairs_does_not_map_excluded_labels_to_neutral(tmp_path):
    source = tmp_path / "stocktbsa.json"
    _write_fixture(source)

    polarity = filter_polarity_pairs(load_stocktbsa_pairs(source))

    assert set(polarity["label"]) <= POLARITY_LABELS
    assert polarity["label"].tolist() == ["positive", "neutral"]


def test_target_aware_text_prepends_ticker_without_using_gold_label():
    text = target_aware_text("PTT", "กำไรสุทธิเพิ่มขึ้น")

    assert text.startswith("[TARGET_TICKER] PTT")
    assert "กำไรสุทธิเพิ่มขึ้น" in text


def test_split_by_year_is_temporal_and_rejects_overlap(tmp_path):
    source = tmp_path / "stocktbsa.json"
    _write_fixture(source)
    pairs = load_stocktbsa_pairs(source)
    older = pairs.assign(year=[2021, 2021, 2022])

    train, test = split_by_year(older, train_years=(2021,), test_years=(2022,))

    assert set(train["year"]) == {2021}
    assert set(test["year"]) == {2022}
    with pytest.raises(ValueError, match="overlap"):
        split_by_year(older, train_years=(2021, 2022), test_years=(2022,))


def test_aggregate_daily_sentiment_assigns_news_to_strictly_next_trading_day():
    predictions = pd.DataFrame(
        {
            "article_id": ["a1", "a1", "a2"],
            "date": pd.to_datetime(["2023-01-06", "2023-01-06", "2023-01-09"]),
            "ticker": ["AAA", "BBB", "AAA"],
            "sentiment_score": [1.0, -1.0, 1.0],
            "predicted_label": ["positive", "negative", "positive"],
        }
    )
    trading_dates = pd.to_datetime(
        ["2023-01-06", "2023-01-09", "2023-01-10", "2023-01-11"]
    )

    daily = aggregate_daily_sentiment(predictions, trading_dates)

    friday = daily.loc[daily["date"].eq(pd.Timestamp("2023-01-06"))].iloc[0]
    monday = daily.loc[daily["date"].eq(pd.Timestamp("2023-01-09"))].iloc[0]
    tuesday = daily.loc[daily["date"].eq(pd.Timestamp("2023-01-10"))].iloc[0]
    assert friday["news_available"] == 0
    assert monday["article_count"] == 1
    assert monday["ticker_mention_count"] == 2
    assert monday["news_sentiment_mean"] == pytest.approx(0.0)
    assert monday["news_sentiment_std"] == pytest.approx(1.0)
    assert monday["positive_ratio"] == pytest.approx(0.5)
    assert monday["negative_ratio"] == pytest.approx(0.5)
    assert tuesday["news_sentiment_mean"] == pytest.approx(1.0)
    assert np.isfinite(daily.drop(columns="date").to_numpy(dtype=float)).all()
