from __future__ import annotations

from datetime import date

import pytest

from models.gdelt_news_sample import (
    build_article,
    extract_xml_tag,
    iter_snapshot_stamps,
    parse_tone,
    score_relevance,
)


def _gkg_fields(
    *,
    title: str = "Thai Stock Market Navigates Global Headwinds",
    themes: str = "ECON_STOCKMARKET;EPU_ECONOMY;",
    locations: str = "1#Thailand#TH#TH#15#100#TH",
    organizations: str = "stock exchange of thailand",
    extras: str = (
        "<PAGE_PRECISEPUBTIMESTAMP>20251219004300</PAGE_PRECISEPUBTIMESTAMP>"
        "<PAGE_TITLE>Thai Stock Market Navigates Global Headwinds</PAGE_TITLE>"
    ),
) -> list[str]:
    fields = [""] * 27
    fields[0] = "20251219040000-509"
    fields[1] = "20251219040000"
    fields[2] = "1"
    fields[3] = "thailand-business-news.com"
    fields[4] = "https://example.com/thai-stock-market"
    fields[7] = themes
    fields[9] = locations
    fields[13] = organizations
    fields[15] = "-1.5,4.0,5.5,9.5,20.0,0.1,900"
    fields[26] = extras.replace("Thai Stock Market Navigates Global Headwinds", title)
    return fields


def test_extract_xml_tag_returns_page_metadata():
    extras = "<PAGE_TITLE>SET rises &amp; baht firms</PAGE_TITLE>"

    assert extract_xml_tag(extras, "PAGE_TITLE") == "SET rises & baht firms"
    assert extract_xml_tag(extras, "PAGE_AUTHORS") == ""


def test_parse_tone_returns_named_values():
    tone = parse_tone("-1.5,4.0,5.5,9.5,20.0,0.1,900")

    assert tone["gdelt_tone"] == pytest.approx(-1.5)
    assert tone["positive_score"] == pytest.approx(4.0)
    assert tone["negative_score"] == pytest.approx(5.5)
    assert tone["word_count"] == 900


def test_relevance_score_requires_market_and_thailand_context():
    relevant_score, relevant_reasons = score_relevance(_gkg_fields())
    unrelated_score, unrelated_reasons = score_relevance(
        _gkg_fields(
            title="Last-minute Christmas decorating advice",
            organizations="royal academy of art",
        )
    )

    assert relevant_score >= 5
    assert "title:thai_stock_market" in relevant_reasons
    assert unrelated_score < 5
    assert "title:thai_stock_market" not in unrelated_reasons


def test_build_article_prefers_precise_timestamp_and_converts_to_bangkok():
    article = build_article(
        _gkg_fields(),
        archive_name="20251219040000.gkg.csv.zip",
    )

    assert article["published_at_utc"] == "2025-12-19T00:43:00+00:00"
    assert article["published_at_bangkok"] == "2025-12-19T07:43:00+07:00"
    assert article["publication_timestamp_source"] == "page_precise_timestamp"
    assert article["publication_lag_days"] == pytest.approx((3 * 60 + 17) / 1440)
    assert article["is_publication_date_consistent"] is True
    assert article["title"] == "Thai Stock Market Navigates Global Headwinds"
    assert article["full_text_collected"] is False


def test_build_article_marks_old_pages_recrawled_in_a_new_batch():
    fields = _gkg_fields(
        extras=(
            "<PAGE_PRECISEPUBTIMESTAMP>20221221082600</PAGE_PRECISEPUBTIMESTAMP>"
            "<PAGE_TITLE>Old page recrawled later</PAGE_TITLE>"
        )
    )

    article = build_article(fields, archive_name="20251219040000.gkg.csv.zip")

    assert article["publication_lag_days"] > 1000
    assert article["is_publication_date_consistent"] is False


def test_build_article_rejects_malformed_gkg_rows():
    with pytest.raises(ValueError, match="27 fields"):
        build_article(["too", "short"], archive_name="broken.zip")


def test_hourly_snapshot_stamps_cover_one_day_without_future_dates():
    stamps = list(iter_snapshot_stamps(date(2025, 12, 19), step_minutes=60))

    assert len(stamps) == 24
    assert stamps[0] == "20251219000000"
    assert stamps[-1] == "20251219230000"


def test_snapshot_interval_must_divide_one_day():
    with pytest.raises(ValueError, match="divide 1440"):
        list(iter_snapshot_stamps(date(2025, 12, 19), step_minutes=17))
