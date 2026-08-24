from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from models.track_b_compute_matched import (
    PROTOCOL_ID,
    apply_accuracy_holm,
    assemble_control_records,
    consensus_verdict,
    metrics_by_arm,
    paired_control_comparisons,
    validate_checkpoint_design,
    verify_freeze_manifest,
)
from models.track_b_compute_matched_runner import (
    checkpoint_payload,
    checkpoint_runtime_audit,
    pending_call_keys,
)
from models.track_b_llm import LLMCall, SentimentVerdict, TokenUsage


def _parsed(
    label: str,
    probabilities: tuple[float, float, float],
) -> dict[str, object]:
    negative, neutral, positive = probabilities
    return {
        "relevant": True,
        "positive_probability": positive,
        "neutral_probability": neutral,
        "negative_probability": negative,
        "predicted_label": label,
        "sentiment_score": positive - negative,
        "confidence": max(probabilities),
        "rationale": "test",
    }


def _existing_record(
    item_id: str,
    gold: str,
    single: dict[str, object],
    leader: dict[str, object],
) -> dict[str, object]:
    return {
        "status": "completed",
        "item_id": item_id,
        "article_id": item_id.split("::")[0],
        "gold_label": gold,
        "single": {"parsed": single, "cost_usd": 0.004},
        "leader": {"parsed": leader, "cost_usd": 0.015},
        "bull": {"cost_usd": 0.007},
        "bear": {"cost_usd": 0.007},
    }


def _repeat(
    item_id: str,
    replicate: int,
    parsed: dict[str, object],
) -> dict[str, object]:
    return {
        "status": "completed",
        "item_id": item_id,
        "role": f"single_rep_{replicate}",
        "response_id": f"response-{item_id}-{replicate}",
        "parsed": parsed,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
        },
        "runtime_seconds": 0.2,
        "cost_usd": 0.00055,
    }


def test_verify_freeze_manifest_rejects_changed_input(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("frozen", encoding="utf-8")
    manifest = tmp_path / "freeze.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "status": "frozen_before_any_repeated_single_control_call",
                "new_results_seen_before_freeze": False,
                "model": "gpt-5.6-terra",
                "pairs": 1333,
                "unique_articles": 738,
                "new_replicates": [2, 3, 4],
                "primary_metric": "accuracy",
                "incremental_cost_guard_usd": 18.0,
                "frozen_inputs": [
                    {
                        "path": "source.txt",
                        "bytes": source.stat().st_size,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert verify_freeze_manifest(tmp_path, manifest)["all_inputs_match"] is True
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen input mismatch"):
        verify_freeze_manifest(tmp_path, manifest)


def test_consensus_verdict_uses_mean_probabilities_and_fixed_class_order() -> None:
    verdicts = [
        _parsed("positive", (0.10, 0.10, 0.80)),
        _parsed("negative", (0.60, 0.20, 0.20)),
        _parsed("positive", (0.20, 0.10, 0.70)),
    ]

    result = consensus_verdict(verdicts)

    assert result["predicted_label"] == "positive"
    assert result["positive_probability"] == pytest.approx(0.5666666667)
    assert sum(result[f"{label}_probability"] for label in ("negative", "neutral", "positive")) == pytest.approx(1.0)


def test_assemble_records_requires_all_replicates_and_preserves_leader() -> None:
    existing = [
        _existing_record(
            "a::AAA",
            "positive",
            _parsed("positive", (0.1, 0.1, 0.8)),
            _parsed("neutral", (0.1, 0.8, 0.1)),
        )
    ]
    repeats = [
        _repeat("a::AAA", 2, _parsed("positive", (0.1, 0.2, 0.7))),
        _repeat("a::AAA", 3, _parsed("positive", (0.1, 0.1, 0.8))),
        _repeat("a::AAA", 4, _parsed("negative", (0.7, 0.2, 0.1))),
    ]

    records = assemble_control_records(existing, repeats)

    assert records[0]["leader"] == existing[0]["leader"]
    assert records[0]["self_consistency_3"]["parsed"]["predicted_label"] == "positive"
    assert records[0]["self_consistency_4"]["parsed"]["predicted_label"] == "positive"

    with pytest.raises(ValueError, match="checkpoint design"):
        assemble_control_records(existing, repeats[:-1])


def test_checkpoint_design_and_pending_keys_are_exact() -> None:
    item_ids = ("a::AAA", "b::BBB")
    calls = [
        _repeat("a::AAA", 2, _parsed("positive", (0.1, 0.2, 0.7))),
        _repeat("a::AAA", 3, _parsed("neutral", (0.1, 0.8, 0.1))),
        _repeat("a::AAA", 4, _parsed("negative", (0.8, 0.1, 0.1))),
        _repeat("b::BBB", 2, _parsed("positive", (0.1, 0.2, 0.7))),
        _repeat("b::BBB", 3, _parsed("neutral", (0.1, 0.8, 0.1))),
        _repeat("b::BBB", 4, _parsed("negative", (0.8, 0.1, 0.1))),
    ]

    audit = validate_checkpoint_design(item_ids, calls)
    assert audit["expected_calls"] == 6
    assert audit["observed_calls"] == 6
    assert pending_call_keys(item_ids, calls) == []
    assert pending_call_keys(item_ids, calls[:-1]) == [("b::BBB", 4)]


def test_checkpoint_payload_contains_no_article_or_gold_label() -> None:
    call = LLMCall(
        role="single_rep_2",
        response_id="response-id",
        parsed=SentimentVerdict.model_validate(
            _parsed("positive", (0.1, 0.1, 0.8))
        ),
        usage=TokenUsage(input_tokens=100, output_tokens=20),
        runtime_seconds=0.2,
        cost_usd=0.00055,
    )

    payload = checkpoint_payload("a::AAA", "hash", call)

    assert payload["protocol_id"] == PROTOCOL_ID
    assert "text" not in payload
    assert "gold_label" not in payload
    assert payload["text_sha256"] == "hash"


def test_comparison_and_holm_report_registered_two_control_family() -> None:
    existing: list[dict[str, object]] = []
    repeats: list[dict[str, object]] = []
    labels = ("positive", "neutral", "negative", "positive", "neutral", "negative")
    for index, gold in enumerate(labels):
        item_id = f"article-{index // 2}::T{index}"
        wrong = "neutral" if gold != "neutral" else "positive"
        wrong_probabilities = (
            0.8 if wrong == "negative" else 0.1,
            0.8 if wrong == "neutral" else 0.1,
            0.8 if wrong == "positive" else 0.1,
        )
        existing.append(
            _existing_record(
                item_id,
                gold,
                _parsed(wrong, wrong_probabilities),
                _parsed(gold, (
                    0.8 if gold == "negative" else 0.1,
                    0.8 if gold == "neutral" else 0.1,
                    0.8 if gold == "positive" else 0.1,
                )),
            )
        )
        for replicate in (2, 3, 4):
            repeats.append(
                _repeat(item_id, replicate, _parsed(wrong, wrong_probabilities))
            )
    records = assemble_control_records(existing, repeats)

    comparison = paired_control_comparisons(
        records,
        bootstrap_iterations=50,
        random_seed=42,
    )
    adjusted = apply_accuracy_holm(comparison)

    assert set(comparison["comparison_id"]) == {
        "leader_minus_self_consistency_3",
        "leader_minus_self_consistency_4",
    }
    assert np.allclose(comparison["accuracy_delta_pp"], 100.0)
    assert set(adjusted["models_in_family"]) == {2}
    assert adjusted["holm_adjusted_pvalue"].between(0.0, 1.0).all()


def test_metrics_by_arm_aligns_probability_columns_with_sklearn_labels() -> None:
    records: list[dict[str, object]] = []
    for label in ("negative", "neutral", "positive"):
        probabilities = (
            0.98 if label == "negative" else 0.01,
            0.98 if label == "neutral" else 0.01,
            0.98 if label == "positive" else 0.01,
        )
        arm = {"parsed": _parsed(label, probabilities)}
        records.append(
            {
                "gold_label": label,
                "single": arm,
                "self_consistency_3": arm,
                "self_consistency_4": arm,
                "leader": arm,
            }
        )

    metrics = metrics_by_arm(records)

    assert np.all(metrics["log_loss"] < 0.03)
    assert np.all(metrics["multiclass_brier_score"] < 0.001)


def test_checkpoint_runtime_audit_uses_persisted_call_timings() -> None:
    rows = [
        {
            "completed_at_utc": "2026-08-03T10:00:00+00:00",
            "runtime_seconds": 1.0,
        },
        {
            "completed_at_utc": "2026-08-03T10:00:10+00:00",
            "runtime_seconds": 3.0,
        },
    ]

    audit = checkpoint_runtime_audit(rows)

    assert audit["checkpoint_completion_span_seconds"] == pytest.approx(10.0)
    assert audit["new_call_runtime_seconds_total"] == pytest.approx(4.0)
    assert audit["new_call_runtime_seconds_mean"] == pytest.approx(2.0)
    assert audit["new_call_runtime_seconds_median"] == pytest.approx(2.0)
    assert audit["new_call_runtime_seconds_p95"] == pytest.approx(2.9)
