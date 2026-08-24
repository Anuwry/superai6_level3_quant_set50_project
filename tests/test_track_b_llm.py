from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.track_b_llm import (
    BenchmarkItem,
    BudgetExceededError,
    BudgetLedger,
    DebateOrchestrator,
    LLMCall,
    Pricing,
    SentimentVerdict,
    TokenUsage,
    WorkerArgument,
    build_single_messages,
    calculate_cost_usd,
    extract_api_key,
)


def _verdict(label: str = "positive") -> SentimentVerdict:
    probabilities = {
        "positive": (0.8, 0.15, 0.05, 0.75),
        "neutral": (0.1, 0.8, 0.1, 0.0),
        "negative": (0.05, 0.15, 0.8, -0.75),
    }
    positive, neutral, negative, score = probabilities[label]
    return SentimentVerdict(
        relevant=True,
        positive_probability=positive,
        neutral_probability=neutral,
        negative_probability=negative,
        predicted_label=label,
        sentiment_score=score,
        confidence=max(positive, neutral, negative),
        rationale="สรุปจากผลกระทบต่อบริษัทเป้าหมาย",
    )


def _argument(stance: str) -> WorkerArgument:
    return WorkerArgument(
        stance=stance,
        relevance_probability=0.9,
        claim_strength=70,
        evidence=["ข้อความหลักฐาน"],
        counterevidence="มีความไม่แน่นอนบางส่วน",
    )


class FakeStructuredClient:
    def __init__(self):
        self.roles = []

    def generate(self, *, role, messages, response_type, max_output_tokens):
        self.roles.append(role)
        if role == "bull":
            parsed = _argument("bullish")
        elif role == "bear":
            parsed = _argument("bearish")
        else:
            parsed = _verdict()
        return LLMCall(
            role=role,
            response_id=f"response-{role}",
            parsed=parsed,
            usage=TokenUsage(input_tokens=100, output_tokens=20),
            runtime_seconds=0.01,
            cost_usd=0.00055,
        )


def _item() -> BenchmarkItem:
    return BenchmarkItem(
        item_id="a1::PTT",
        article_id="a1",
        ticker="PTT",
        date="2023-01-03",
        text="PTT รายงานกำไรสุทธิเพิ่มขึ้น",
        text_sha256="abc123",
    )


def test_single_prompt_contains_target_and_article_but_has_no_gold_label_field():
    messages = build_single_messages(_item())
    serialized = json.dumps(messages, ensure_ascii=False)

    assert "PTT" in serialized
    assert "กำไรสุทธิเพิ่มขึ้น" in serialized
    assert "gold_label" not in serialized
    assert "true_label" not in serialized


def test_calculate_cost_uses_actual_input_and_output_tokens():
    usage = TokenUsage(input_tokens=1_000, output_tokens=200)
    pricing = Pricing(input_usd_per_million=2.5, output_usd_per_million=15.0)

    assert calculate_cost_usd(usage, pricing) == pytest.approx(0.0055)


def test_budget_ledger_reserves_capacity_and_fails_closed():
    ledger = BudgetLedger(limit_usd=0.01)
    reservation = ledger.reserve(0.006)

    with pytest.raises(BudgetExceededError):
        ledger.reserve(0.005)

    ledger.commit(reservation, actual_cost_usd=0.004)
    assert ledger.spent_usd == pytest.approx(0.004)
    assert ledger.reserved_usd == pytest.approx(0.0)


def test_extract_api_key_accepts_markdown_wrapper_without_exposing_other_text(
    tmp_path: Path,
):
    key_file = tmp_path / "key.md"
    fake_key = "sk-test_" + "abcdefghijklmnopqrstuvwxyz1234"
    key_file.write_text(f"API key: `{fake_key}`", encoding="utf-8")

    key = extract_api_key(key_file)

    assert key == fake_key


def test_debate_orchestrator_runs_single_bull_bear_then_leader():
    client = FakeStructuredClient()
    orchestrator = DebateOrchestrator(client=client)

    result = orchestrator.process(_item(), include_single=True)

    assert set(client.roles[:3]) == {"single", "bull", "bear"}
    assert client.roles[-1] == "leader"
    assert result.single is not None
    assert result.bull.parsed.stance == "bullish"
    assert result.bear.parsed.stance == "bearish"
    assert result.leader.parsed.predicted_label == "positive"
    assert result.total_cost_usd == pytest.approx(4 * 0.00055)
