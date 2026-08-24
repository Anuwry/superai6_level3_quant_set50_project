from __future__ import annotations

from datetime import date

import pytest

from models.set_news_download import (
    build_query_params,
    normalize_records,
    parse_news_page,
)


def _sample_payload() -> dict[str, object]:
    return {
        "newsGroups": [],
        "paginateNews": {
            "group": "",
            "totalCount": 2,
            "newsInfoList": [
                {
                    "id": "100",
                    "datetime": "2024-01-31T21:29:00+07:00",
                    "symbol": "AAA",
                    "source": "AAA",
                    "url": "https://www.set.or.th/th/example/100",
                    "headline": "ข่าวตัวอย่าง",
                    "product": "S",
                },
                {
                    "id": "101",
                    "datetime": "2024-01-30T08:00:00+07:00",
                    "symbol": "BBB",
                    "source": "BBB",
                    "url": "https://www.set.or.th/th/example/101",
                    "headline": "ข่าวอีกชิ้น",
                    "product": "S",
                },
            ],
        },
    }


def test_build_query_params_uses_set_date_format() -> None:
    params = build_query_params(
        date(2024, 1, 1),
        date(2024, 12, 31),
        page=3,
        per_page=5_000,
        language="th",
    )

    assert params == {
        "sourceId": "company",
        "securityTypeIds": "S",
        "fromDate": "01/01/2024",
        "toDate": "31/12/2024",
        "page": 3,
        "perPage": 5_000,
        "orderBy": "date",
        "lang": "th",
    }


def test_parse_news_page_returns_total_and_records() -> None:
    total_count, records = parse_news_page(_sample_payload())

    assert total_count == 2
    assert [record["id"] for record in records] == ["100", "101"]


def test_parse_news_page_rejects_invalid_contract() -> None:
    with pytest.raises(ValueError, match="paginateNews"):
        parse_news_page({"unexpected": "payload"})


def test_normalize_records_deduplicates_and_adds_provenance() -> None:
    _, source_records = parse_news_page(_sample_payload())
    normalized = normalize_records(
        [*source_records, source_records[0]],
        expected_year=2024,
        language="th",
    )

    assert len(normalized) == 2
    assert normalized[0]["set_news_id"] == "101"
    assert normalized[0]["publication_year"] == 2024
    assert normalized[0]["language"] == "th"
    assert normalized[0]["text_scope"] == "headline_and_metadata_only"
    assert normalized[0]["full_text_collected"] is False
    assert normalized[1]["set_news_id"] == "100"


def test_normalize_records_rejects_out_of_range_year() -> None:
    _, source_records = parse_news_page(_sample_payload())

    with pytest.raises(ValueError, match="outside requested year"):
        normalize_records(source_records, expected_year=2025, language="th")
