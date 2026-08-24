from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from models import track_b_forward_news as forward
from models.track_b_forward_news import (
    filter_point_in_time_membership,
    headline_proxy,
    inference_training_years,
    prepare_official_headlines,
    select_relevant_sentiment_predictions,
    validate_membership_intervals,
)


def _membership() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "effective_from": "2025-01-01",
                "effective_to": "2025-04-01",
                "symbol": symbol,
                "membership_version": "2025_h1_pre_amalgamation",
                "source_document": "h1.pdf",
                "source_url": "https://example.test/h1.pdf",
                "source_sha256": "a" * 64,
            }
            for symbol in ("GULFI", "INTUCH")
        ]
        + [
            {
                "effective_from": "2025-04-02",
                "effective_to": "2025-06-30",
                "symbol": symbol,
                "membership_version": "2025_h1_post_amalgamation",
                "source_document": "h1_revised.pdf",
                "source_url": "https://example.test/h1_revised.pdf",
                "source_sha256": "b" * 64,
            }
            for symbol in ("GULF", "VGI")
        ]
    )


def test_membership_validation_and_filter_respect_midcycle_boundary():
    membership = validate_membership_intervals(_membership(), expected_size=2)
    news = pd.DataFrame(
        {
            "set_news_id": ["old-valid", "old-invalid", "new-valid", "new-invalid"],
            "published_at_bangkok": [
                "2025-04-01T17:00:00+07:00",
                "2025-04-01T17:00:00+07:00",
                "2025-04-02T08:00:00+07:00",
                "2025-04-02T08:00:00+07:00",
            ],
            "symbol": ["GULFI", "GULF", "GULF", "INTUCH"],
            "headline": ["a", "b", "c", "d"],
        }
    )

    selected = filter_point_in_time_membership(news, membership)

    assert selected["set_news_id"].tolist() == ["old-valid", "new-valid"]
    assert selected["membership_version"].tolist() == [
        "2025_h1_pre_amalgamation",
        "2025_h1_post_amalgamation",
    ]


def test_membership_validation_rejects_overlapping_versions():
    membership = _membership()
    membership.loc[
        membership["membership_version"].eq("2025_h1_pre_amalgamation"),
        "effective_to",
    ] = "2025-04-02"

    with pytest.raises(ValueError, match="overlap"):
        validate_membership_intervals(membership, expected_size=2)


def test_headline_proxy_uses_repeated_leading_stocktbsa_title():
    text = (
        "หุ้น ตัวอย่าง กำไร โต แรง รับ ยอดขาย ฟื้น "
        "หุ้น ตัวอย่าง กำไร โต แรง รับ ยอดขาย ฟื้น "
        "รายละเอียดข่าวและบทวิเคราะห์ส่วนที่เหลือ"
    )

    assert headline_proxy(text) == "หุ้น ตัวอย่าง กำไร โต แรง รับ ยอดขาย ฟื้น"


def test_headline_proxy_has_deterministic_fallback_for_non_repeated_text():
    text = "หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด เก้า สิบ สิบเอ็ด สิบสอง"

    assert headline_proxy(text, fallback_tokens=8) == (
        "หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด"
    )


def test_prepare_official_headlines_creates_unlabelled_target_pairs():
    news = pd.DataFrame(
        {
            "set_news_id": ["95454301"],
            "published_at_bangkok": ["2025-04-02T08:15:00+07:00"],
            "symbol": [" gulf "],
            "source": ["SET"],
            "headline": ["  ข่าว   สำคัญ  "],
            "membership_version": ["2025_h1_post_amalgamation"],
        }
    )

    pairs = prepare_official_headlines(news)

    assert pairs.loc[0, "article_id"] == "95454301"
    assert pairs.loc[0, "date"] == pd.Timestamp("2025-04-02")
    assert pairs.loc[0, "year"] == 2025
    assert pairs.loc[0, "ticker"] == "GULF"
    assert pairs.loc[0, "text"] == "ข่าว สำคัญ"
    assert "label" not in pairs.columns


def test_inference_training_years_freeze_after_2023():
    available = [2018, 2019, 2020, 2021, 2022, 2023]

    assert inference_training_years(2022, available) == (2018, 2019, 2020, 2021)
    assert inference_training_years(2024, available) == tuple(available)
    assert inference_training_years(2025, available) == tuple(available)


def test_select_relevant_predictions_aligns_pairs_and_applies_locked_threshold():
    keys = {
        "article_id": ["a", "b"],
        "date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
        "year": [2025, 2025],
        "ticker": ["AAA", "BBB"],
    }
    relevance = pd.DataFrame(
        {
            **keys,
            "relevant_probability": [0.5, 0.4999],
            "predicted_relevant": [1, 0],
        }
    )
    sentiment = pd.DataFrame(
        {
            **keys,
            "positive_probability": [0.8, 0.1],
            "neutral_probability": [0.1, 0.2],
            "negative_probability": [0.1, 0.7],
            "predicted_label": ["positive", "negative"],
            "sentiment_score": [0.7, -0.6],
            "confidence": [0.8, 0.7],
        }
    )

    selected = select_relevant_sentiment_predictions(relevance, sentiment)

    assert selected["article_id"].tolist() == ["a"]
    assert selected.loc[0, "relevant_probability"] == pytest.approx(0.5)


def test_official_membership_pdf_parser_finds_ranked_set50_constituents():
    source = (
        forward.PROJECT_ROOT
        / "data-raw"
        / "track_b"
        / "SET50_membership_2024_2025"
        / "SET50_100_H1_2024.pdf"
    )

    symbols = forward.extract_set50_symbols_from_pdf(source)

    assert len(symbols) == 50
    assert len(set(symbols)) == 50
    assert "ADVANC" in symbols


def test_build_membership_artifacts_creates_six_point_in_time_versions(
    monkeypatch,
    tmp_path,
):
    membership_root = tmp_path / "membership"
    membership_file = membership_root / "set50_membership_intervals.csv"
    manifest_file = membership_root / "manifest.json"
    monkeypatch.setattr(forward, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(forward, "MEMBERSHIP_ROOT", membership_root)
    monkeypatch.setattr(forward, "MEMBERSHIP_FILE", membership_file)
    monkeypatch.setattr(forward, "MEMBERSHIP_MANIFEST_FILE", manifest_file)

    for document in forward.MEMBERSHIP_DOCUMENTS:
        membership_root.mkdir(parents=True, exist_ok=True)
        (membership_root / document.filename).write_bytes(b"%PDF-test")

    base = [f"S{index:02d}" for index in range(1, 49)]
    original = [*base, "GULF", "INTUCH"]
    revised = [*base, "GULF", "VGI"]

    def fake_extract(path: str | Path) -> list[str]:
        return (
            revised
            if Path(path).name == "SET50_100_H1_2025_revise.pdf"
            else original
        )

    monkeypatch.setattr(forward, "extract_set50_symbols_from_pdf", fake_extract)

    membership = forward.build_membership_artifacts()

    assert len(membership) == 300
    assert membership["membership_version"].nunique() == 6
    transition = membership.loc[
        membership["membership_version"].eq("2025_h1_gulfi_transition"),
        "symbol",
    ]
    post = membership.loc[
        membership["membership_version"].eq("2025_h1_post_amalgamation"),
        "symbol",
    ]
    assert {"GULFI", "INTUCH"}.issubset(set(transition))
    assert {"GULF", "VGI"}.issubset(set(post))
    assert {"GULFI", "INTUCH"}.isdisjoint(set(post))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["interval_rows"] == 300
    assert manifest["membership_versions"] == 6


def test_fit_score_and_domain_shift_audit_use_frozen_local_models():
    rows = []
    labels = ("positive", "negative", "neutral", "exclude", "positive", "negative")
    for index, label in enumerate(labels):
        rows.append(
            {
                "article_id": f"train-{index}",
                "source": "StockTBSA",
                "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=index),
                "year": 2023,
                "text": f"common market headline class {label} repeated words",
                "ticker": f"S{index:02d}",
                "label": label,
            }
        )
    training = pd.DataFrame(rows)
    inference = training.iloc[:3].copy()

    predictions, runtimes, models = forward._fit_score(
        training,
        inference,
        train_years=(2023,),
        inference_label="test",
    )
    predictions = predictions.merge(
        inference[forward.PREDICTION_KEYS + ["text"]],
        on=forward.PREDICTION_KEYS,
        validate="one_to_one",
    )
    audit = forward._domain_shift_audit(
        inference,
        inference.assign(year=2024),
        predictions,
        models,
    )

    assert len(predictions) == 3
    assert {row["task"] for row in runtimes} == {"relevance", "sentiment"}
    assert predictions["included_in_daily"].isin([0, 1]).all()
    assert audit.loc[audit["year"].eq(2023), "relevance_nonzero_vector_rate"].iloc[
        0
    ] == pytest.approx(1.0)


def test_forward_pipeline_writes_reproducible_artifacts_with_stubbed_fits(
    monkeypatch,
    tmp_path,
):
    project_root = tmp_path
    output_dir = project_root / "outputs"
    news_root = project_root / "news"
    membership_file = project_root / "membership.csv"
    dataset_path = project_root / "dataset.json"
    news_root.mkdir()
    dataset_path.write_text("{}", encoding="utf-8")
    membership_file.write_text("membership", encoding="utf-8")
    (news_root / "manifest.json").write_text("{}", encoding="utf-8")
    for year in forward.FORWARD_YEARS:
        (news_root / f"set_company_news_{year}_th.csv").write_text(
            "placeholder\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(forward, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(forward, "NEWS_ROOT", news_root)
    monkeypatch.setattr(forward, "MEMBERSHIP_FILE", membership_file)

    membership = pd.DataFrame(
        [
            {
                "effective_from": f"{year}-01-01",
                "effective_to": f"{year}-12-31",
                "symbol": "AAA",
                "membership_version": f"{year}_test",
                "source_document": "test.pdf",
                "source_url": "https://example.test/test.pdf",
                "source_sha256": "a" * 64,
                "change_source_url": "",
            }
            for year in forward.FORWARD_YEARS
        ]
    )
    raw_forward = pd.DataFrame(
        {
            "set_news_id": [f"news-{year}" for year in forward.FORWARD_YEARS],
            "published_at_bangkok": [
                f"{year}-01-01T08:00:00+07:00" for year in forward.FORWARD_YEARS
            ],
            "symbol": ["AAA", "AAA"],
            "source": ["SET", "SET"],
            "headline": ["positive market headline", "negative market headline"],
        }
    )
    training = pd.DataFrame(
        [
            {
                "article_id": f"train-{year}",
                "source": "StockTBSA",
                "date": pd.Timestamp(f"{year}-01-01"),
                "year": year,
                "text": "common market headline positive",
                "ticker": "AAA",
                "label": "positive",
            }
            for year in range(2018, 2024)
        ]
    )

    monkeypatch.setattr(
        forward,
        "build_membership_artifacts",
        lambda **_: membership,
    )
    monkeypatch.setattr(forward, "_load_forward_news", lambda: raw_forward)
    monkeypatch.setattr(forward, "load_stocktbsa_pairs", lambda _: training)
    monkeypatch.setattr(
        forward,
        "_market_dates_2019_2025",
        lambda: pd.DatetimeIndex(
            [pd.Timestamp(f"{year}-01-02") for year in range(2019, 2026)]
        ),
    )

    def fake_fit_score(
        _training,
        inference,
        *,
        train_years,
        inference_label,
    ):
        result = inference[forward.PREDICTION_KEYS].copy()
        if "label" in inference:
            result["label"] = inference["label"].to_numpy()
        result["relevant_probability"] = 0.8
        result["predicted_relevant"] = 1
        result["positive_probability"] = 0.7
        result["neutral_probability"] = 0.2
        result["negative_probability"] = 0.1
        result["predicted_label"] = "positive"
        result["sentiment_score"] = 0.6
        result["confidence"] = 0.7
        result["included_in_daily"] = 1
        runtime = [
            {
                "inference_label": inference_label,
                "task": task,
                "train_start_year": min(train_years),
                "train_end_year": max(train_years),
                "train_pairs": len(_training),
                "inference_pairs": len(inference),
                "fit_seconds": 0.1,
                "predict_seconds": 0.01,
            }
            for task in ("relevance", "sentiment")
        ]
        return result, runtime, (object(), object())

    monkeypatch.setattr(forward, "_fit_score", fake_fit_score)
    monkeypatch.setattr(
        forward,
        "_domain_shift_audit",
        lambda *_: pd.DataFrame(
            {
                "year": list(range(2019, 2026)),
                "pairs_before_membership": [1] * 7,
            }
        ),
    )

    metadata = forward.run_forward_news_pipeline(
        dataset_path=dataset_path,
        output_dir=output_dir,
    )

    assert metadata["prediction_rows"] == 7
    assert metadata["selected_forward_rows"] == 2
    assert metadata["daily_rows"] == 7
    assert (output_dir / forward.DAILY_NEWS_FILE.name).is_file()
    saved = json.loads(
        (output_dir / forward.RUN_METADATA_FILE.name).read_text(encoding="utf-8")
    )
    assert saved["pseudo_label_retraining"] is False
    assert saved["temporal_assignment"] == "strictly_next_trading_date"
