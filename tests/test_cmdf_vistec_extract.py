from __future__ import annotations

from datetime import date

from models.cmdf_vistec_extract import (
    extract_thai_publication_date,
    select_dated_records,
)


def test_extract_thai_publication_date_from_buddhist_year() -> None:
    text = "หัวข้อข่าว\nบล. ตัวอย่าง ระบุว่า\n25 มี.ค. 2558\nเนื้อข่าว"

    assert extract_thai_publication_date(text) == date(2015, 3, 25)


def test_extract_thai_publication_date_supports_full_month_name() -> None:
    text = "หัวข้อข่าว\n2 มกราคม 2560\nเนื้อข่าว"

    assert extract_thai_publication_date(text) == date(2017, 1, 2)


def test_extract_thai_publication_date_returns_none_without_exact_date_line() -> None:
    text = "บริษัทตั้งเป้าปี 2560 เติบโต 20%\nแต่ไม่มีบรรทัดวันที่เผยแพร่"

    assert extract_thai_publication_date(text) is None


def test_select_dated_records_filters_years_and_reports_undated_rows() -> None:
    records = [
        {"id": "1", "text": "ข่าวหนึ่ง\n31 ธ.ค. 2559\nเนื้อหา"},
        {"id": "2", "text": "ข่าวสอง\n1 ม.ค. 2560\nเนื้อหา"},
        {"id": "3", "text": "ข่าวสาม\n2 ม.ค. 2561\nเนื้อหา"},
        {"id": "4", "text": "ไม่มีวันที่"},
    ]

    selected, stats = select_dated_records(
        records,
        start_date=date(2016, 1, 1),
        end_date=date(2017, 12, 31),
        source="kaohoon",
    )

    assert [row["cmdf_id"] for row in selected] == ["1", "2"]
    assert [row["published_date"] for row in selected] == [
        "2016-12-31",
        "2017-01-01",
    ]
    assert stats == {
        "rows_scanned": 4,
        "rows_with_parsed_date": 3,
        "rows_without_parsed_date": 1,
        "rows_in_requested_period": 2,
    }
