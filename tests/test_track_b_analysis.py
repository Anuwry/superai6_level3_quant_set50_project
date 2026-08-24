import json

import pandas as pd
import pytest

from models.track_b_analysis import (
    audit_llm_checkpoints,
    build_analysis_artifacts,
    build_intrinsic_method_table,
    local_leader_paired_comparison,
    paired_llm_comparison,
)


def _verdict(label: str) -> dict[str, object]:
    probabilities = {
        "positive": (0.8, 0.1, 0.1),
        "neutral": (0.1, 0.8, 0.1),
        "negative": (0.1, 0.1, 0.8),
    }[label]
    return {
        "parsed": {
            "positive_probability": probabilities[0],
            "neutral_probability": probabilities[1],
            "negative_probability": probabilities[2],
            "predicted_label": label,
        }
    }


def _records() -> list[dict[str, object]]:
    return [
        {
            "item_id": "1::AAA",
            "gold_label": "positive",
            "single": _verdict("neutral"),
            "leader": _verdict("positive"),
        },
        {
            "item_id": "2::AAA",
            "gold_label": "neutral",
            "single": _verdict("positive"),
            "leader": _verdict("neutral"),
        },
        {
            "item_id": "3::AAA",
            "gold_label": "negative",
            "single": _verdict("negative"),
            "leader": _verdict("negative"),
        },
        {
            "item_id": "4::AAA",
            "gold_label": "positive",
            "single": _verdict("positive"),
            "leader": _verdict("positive"),
        },
    ]


def test_paired_llm_comparison_uses_exact_mcnemar_and_seeded_bootstrap():
    comparison = paired_llm_comparison(
        _records(),
        bootstrap_iterations=200,
        random_seed=42,
    )

    assert comparison["pairs"] == 4
    assert comparison["single_accuracy"] == pytest.approx(0.5)
    assert comparison["leader_accuracy"] == pytest.approx(1.0)
    assert comparison["accuracy_delta_pp"] == pytest.approx(50.0)
    assert comparison["leader_correct_single_wrong"] == 2
    assert comparison["single_correct_leader_wrong"] == 0
    assert comparison["mcnemar_exact_pvalue"] == pytest.approx(0.5)
    assert comparison["cluster_sign_flip_pvalue"] == pytest.approx(0.5)
    assert comparison["primary_inference_unit"] == "article_id"
    assert comparison["bootstrap_iterations"] == 200
    assert comparison["bootstrap_unit"] == "article_id"
    assert comparison["unique_articles"] == 4
    assert comparison["accuracy_delta_pp_ci95_lower"] <= 50.0
    assert comparison["accuracy_delta_pp_ci95_upper"] >= 50.0


def test_paired_llm_comparison_clusters_multi_ticker_rows_by_article():
    records = _records()
    records[1]["item_id"] = "1::BBB"

    comparison = paired_llm_comparison(
        records,
        bootstrap_iterations=20,
        random_seed=42,
    )

    assert comparison["pairs"] == 4
    assert comparison["unique_articles"] == 3
    assert comparison["bootstrap_unit"] == "article_id"


def test_local_leader_comparison_matches_exact_article_ticker_pairs():
    local = pd.DataFrame(
        {
            "article_id": ["1", "2", "3", "4"],
            "ticker": ["AAA"] * 4,
            "label": ["positive", "neutral", "negative", "positive"],
            "predicted_label": ["positive", "positive", "negative", "positive"],
        }
    )

    comparison = local_leader_paired_comparison(
        _records(),
        local,
        bootstrap_iterations=20,
    )

    assert comparison["pairs"] == 4
    assert comparison["local_accuracy"] == pytest.approx(0.75)
    assert comparison["leader_accuracy"] == pytest.approx(1.0)
    assert comparison["accuracy_delta_pp_leader_minus_local"] == pytest.approx(25.0)
    assert comparison["bootstrap_unit"] == "article_id"
    assert comparison["unique_articles"] == 4


def test_intrinsic_method_table_compares_all_methods_on_same_pairs():
    local = pd.DataFrame(
        {
            "article_id": ["1", "2", "3", "4"],
            "ticker": ["AAA"] * 4,
            "label": ["positive", "neutral", "negative", "positive"],
            "predicted_label": ["positive", "positive", "negative", "positive"],
        }
    )

    table = build_intrinsic_method_table(_records(), local)

    assert table["method"].tolist() == [
        "local_char_tfidf",
        "terra_single",
        "terra_debate_leader",
    ]
    assert table["pairs"].eq(4).all()
    assert table.loc[0, "accuracy"] == pytest.approx(0.75)
    assert table.loc[2, "accuracy"] == pytest.approx(1.0)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audit_llm_checkpoints_confirms_complete_roles_and_recovered_errors(
    tmp_path,
):
    predictions = []
    calls = []
    for index, record in enumerate(_records()[:2], start=1):
        predictions.append({"status": "completed", **record})
        for role in ("single", "bull", "bear", "leader"):
            calls.append(
                {
                    "status": "completed",
                    "item_id": record["item_id"],
                    "role": role,
                    "response_id": f"resp-{index}-{role}",
                    "runtime_seconds": 1.0,
                    "cost_usd": 0.01,
                }
            )
    _write_jsonl(tmp_path / "predictions.jsonl", predictions)
    _write_jsonl(tmp_path / "calls.jsonl", calls)
    _write_jsonl(
        tmp_path / "errors.jsonl",
        [
            {
                "status": "failed",
                "item_id": "1::AAA",
                "error_type": "bull:ValidationError",
            }
        ],
    )

    audit = audit_llm_checkpoints(tmp_path, expected_pairs=2)

    assert audit["valid"] is True
    assert audit["completed_pairs"] == 2
    assert audit["successful_calls"] == 8
    assert audit["historical_errors"] == 1
    assert audit["recovered_error_items"] == 1
    assert audit["unresolved_items"] == []
    assert audit["duplicate_item_roles"] == []
    assert audit["duplicate_response_ids"] == []
    assert audit["probability_violations"] == []


def test_audit_llm_checkpoints_rejects_duplicate_item_role(tmp_path):
    record = {"status": "completed", **_records()[0]}
    _write_jsonl(tmp_path / "predictions.jsonl", [record])
    duplicated_call = {
        "status": "completed",
        "item_id": "1::AAA",
        "role": "single",
        "response_id": "resp-1",
    }
    _write_jsonl(tmp_path / "calls.jsonl", [duplicated_call, duplicated_call])

    audit = audit_llm_checkpoints(tmp_path, expected_pairs=1)

    assert audit["valid"] is False
    assert audit["duplicate_item_roles"] == ["1::AAA::single"]


def test_audit_treats_valid_probability_argmax_disagreement_as_warning(tmp_path):
    record = {"status": "completed", **_records()[0]}
    record["leader"] = {
        "parsed": {
            "positive_probability": 0.38,
            "neutral_probability": 0.25,
            "negative_probability": 0.37,
            "predicted_label": "neutral",
        }
    }
    calls = [
        {
            "status": "completed",
            "item_id": "1::AAA",
            "role": role,
            "response_id": f"resp-{role}",
        }
        for role in ("single", "bull", "bear", "leader")
    ]
    _write_jsonl(tmp_path / "predictions.jsonl", [record])
    _write_jsonl(tmp_path / "calls.jsonl", calls)

    audit = audit_llm_checkpoints(tmp_path, expected_pairs=1)

    assert audit["valid"] is True
    assert audit["probability_violations"] == []
    assert audit["decision_probability_disagreements"] == ["1::AAA::leader"]


def test_build_analysis_artifacts_writes_paper_outputs(tmp_path):
    predictions = [{"status": "completed", **record} for record in _records()]
    calls = [
        {
            "status": "completed",
            "item_id": record["item_id"],
            "role": role,
            "response_id": f"{record['item_id']}-{role}",
        }
        for record in _records()
        for role in ("single", "bull", "bear", "leader")
    ]
    _write_jsonl(tmp_path / "predictions.jsonl", predictions)
    _write_jsonl(tmp_path / "calls.jsonl", calls)

    comparison, audit = build_analysis_artifacts(
        tmp_path,
        expected_pairs=4,
        bootstrap_iterations=20,
    )

    assert comparison["pairs"] == 4
    assert audit["valid"] is True
    assert (tmp_path / "llm_paired_comparison.json").is_file()
    assert (tmp_path / "llm_checkpoint_audit.json").is_file()
    assert (tmp_path / "paper_llm_paired_comparison.csv").is_file()
