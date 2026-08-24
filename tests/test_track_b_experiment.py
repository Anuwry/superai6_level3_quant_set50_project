from __future__ import annotations

import json

import pandas as pd

from models.track_b_experiment import (
    load_completed_item_ids,
    select_stratified_pairs,
    serialize_debate_result,
)
from models.track_b_llm import (
    BenchmarkItem,
    DebateResult,
    LLMCall,
    SentimentVerdict,
    TokenUsage,
    WorkerArgument,
)


def _call(role, parsed):
    return LLMCall(
        role=role,
        response_id=f"response-{role}",
        parsed=parsed,
        usage=TokenUsage(input_tokens=100, output_tokens=20),
        runtime_seconds=0.1,
        cost_usd=0.00055,
    )


def _result():
    item = BenchmarkItem(
        item_id="a1::PTT",
        article_id="a1",
        ticker="PTT",
        date="2023-01-03",
        text="เนื้อข่าวที่ต้องไม่ถูกบันทึก",
        text_sha256="hash",
    )
    verdict = SentimentVerdict(
        relevant=True,
        positive_probability=0.8,
        neutral_probability=0.1,
        negative_probability=0.1,
        predicted_label="positive",
        sentiment_score=0.7,
        confidence=0.8,
        rationale="ผลกระทบเป็นบวก",
    )
    bull = WorkerArgument(
        stance="bullish",
        relevance_probability=0.9,
        claim_strength=80,
        evidence=["กำไรเพิ่ม"],
        counterevidence="ไม่มี",
    )
    bear = WorkerArgument(
        stance="bearish",
        relevance_probability=0.9,
        claim_strength=20,
        evidence=["ต้นทุนเพิ่ม"],
        counterevidence="กำไรยังเพิ่ม",
    )
    return DebateResult(
        item=item,
        single=_call("single", verdict),
        bull=_call("bull", bull),
        bear=_call("bear", bear),
        leader=_call("leader", verdict),
    )


def test_serialize_debate_result_excludes_api_key_and_raw_article_text():
    payload = serialize_debate_result(_result(), gold_label="positive")
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["item_id"] == "a1::PTT"
    assert payload["gold_label"] == "positive"
    assert "เนื้อข่าวที่ต้องไม่ถูกบันทึก" not in serialized
    assert "api_key" not in serialized.lower()
    assert payload["leader"]["parsed"]["predicted_label"] == "positive"


def test_load_completed_item_ids_ignores_failed_rows(tmp_path):
    checkpoint = tmp_path / "predictions.jsonl"
    checkpoint.write_text(
        "\n".join(
            [
                json.dumps({"item_id": "a1::AAA", "status": "completed"}),
                json.dumps({"item_id": "a2::BBB", "status": "failed"}),
            ]
        ),
        encoding="utf-8",
    )

    assert load_completed_item_ids(checkpoint) == {"a1::AAA"}


def test_select_stratified_pairs_is_reproducible_and_covers_classes():
    frame = pd.DataFrame(
        {
            "article_id": [f"a{index}" for index in range(30)],
            "ticker": ["AAA"] * 30,
            "label": ["positive"] * 10 + ["neutral"] * 10 + ["negative"] * 10,
        }
    )

    first = select_stratified_pairs(frame, sample_size=9, random_seed=42)
    second = select_stratified_pairs(frame, sample_size=9, random_seed=42)

    assert first[["article_id", "ticker"]].equals(second[["article_id", "ticker"]])
    assert set(first["label"]) == {"positive", "neutral", "negative"}
