from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from models.track_b_baseline import classification_metrics

LABELS = ("positive", "neutral", "negative")
REQUIRED_ROLES = frozenset({"single", "bull", "bear", "leader"})
CLUSTER_SIGN_FLIP_ITERATIONS = 50_000


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _predicted_labels(
    records: list[dict[str, object]],
    method: str,
) -> list[str]:
    return [
        str(record[method]["parsed"]["predicted_label"])  # type: ignore[index]
        for record in records
    ]


def _bootstrap_deltas(
    expected: np.ndarray,
    single: np.ndarray,
    leader: np.ndarray,
    *,
    iterations: int,
    random_seed: int,
    clusters: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")
    generator = np.random.default_rng(random_seed)
    sample_size = len(expected)
    cluster_values = (
        np.arange(sample_size).astype(str)
        if clusters is None
        else np.asarray(clusters).astype(str)
    )
    if cluster_values.shape != (sample_size,):
        raise ValueError("Bootstrap clusters must match paired observations")
    unique_clusters = np.unique(cluster_values)
    cluster_indices = {
        cluster: np.flatnonzero(cluster_values == cluster)
        for cluster in unique_clusters
    }
    accuracy_deltas = np.empty(iterations, dtype=float)
    macro_f1_deltas = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_clusters = generator.choice(
            unique_clusters,
            size=len(unique_clusters),
            replace=True,
        )
        indices = np.concatenate(
            [cluster_indices[cluster] for cluster in sampled_clusters]
        )
        sampled_expected = expected[indices]
        sampled_single = single[indices]
        sampled_leader = leader[indices]
        accuracy_deltas[iteration] = (
            np.mean(sampled_leader == sampled_expected)
            - np.mean(sampled_single == sampled_expected)
        ) * 100.0
        macro_f1_deltas[iteration] = f1_score(
            sampled_expected,
            sampled_leader,
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        ) - f1_score(
            sampled_expected,
            sampled_single,
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        )
    return accuracy_deltas, macro_f1_deltas


def _article_ids(item_ids: list[str]) -> np.ndarray:
    values = np.asarray(
        [item_id.rsplit("::", maxsplit=1)[0] for item_id in item_ids]
    )
    if np.any(np.char.str_len(values.astype(str)) == 0):
        raise ValueError("Every paired item requires a non-empty article ID")
    return values


def _cluster_sign_flip_pvalue(
    expected: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    clusters: np.ndarray,
    *,
    iterations: int = CLUSTER_SIGN_FLIP_ITERATIONS,
    random_seed: int = 42,
) -> tuple[float, str, int]:
    """Two-sided paired randomization test with article-level swaps."""

    cluster_values = np.asarray(clusters).astype(str)
    row_delta = (second == expected).astype(int) - (
        first == expected
    ).astype(int)
    contributions = np.asarray(
        [
            row_delta[cluster_values == cluster].sum()
            for cluster in np.unique(cluster_values)
        ],
        dtype=float,
    )
    contributions = contributions[contributions != 0.0]
    if len(contributions) == 0:
        return 1.0, "exact", 1
    observed = abs(float(contributions.sum()))
    if len(contributions) <= 20:
        statistics = np.asarray(
            [
                abs(
                    float(
                        np.dot(
                            np.asarray(signs, dtype=float),
                            contributions,
                        )
                    )
                )
                for signs in itertools.product(
                    (-1.0, 1.0),
                    repeat=len(contributions),
                )
            ]
        )
        return (
            float(np.mean(statistics >= observed - 1e-12)),
            "exact",
            int(len(statistics)),
        )
    generator = np.random.default_rng(random_seed)
    extreme = 0
    completed = 0
    batch_size = 5_000
    while completed < iterations:
        current = min(batch_size, iterations - completed)
        signs = generator.choice(
            (-1.0, 1.0),
            size=(current, len(contributions)),
        )
        statistics = np.abs(signs @ contributions)
        extreme += int(np.sum(statistics >= observed - 1e-12))
        completed += current
    return (
        float((extreme + 1) / (iterations + 1)),
        "monte_carlo",
        int(iterations),
    )


def paired_llm_comparison(
    records: list[dict[str, object]],
    *,
    bootstrap_iterations: int = 5_000,
    random_seed: int = 42,
) -> dict[str, object]:
    if not records:
        raise ValueError("At least one completed paired record is required")
    item_ids = [str(record["item_id"]) for record in records]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Paired records contain duplicate item IDs")
    if any(
        record.get("single") is None or record.get("leader") is None
        for record in records
    ):
        raise ValueError("Every paired record requires single and leader predictions")

    expected = np.asarray([str(record["gold_label"]) for record in records])
    article_ids = _article_ids(item_ids)
    single = np.asarray(_predicted_labels(records, "single"))
    leader = np.asarray(_predicted_labels(records, "leader"))
    single_metrics = classification_metrics(expected, single, labels=LABELS)
    leader_metrics = classification_metrics(expected, leader, labels=LABELS)
    single_correct = single == expected
    leader_correct = leader == expected
    leader_only = int(np.sum(leader_correct & ~single_correct))
    single_only = int(np.sum(single_correct & ~leader_correct))
    discordant = leader_only + single_only
    mcnemar_pvalue = (
        float(binomtest(min(leader_only, single_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    cluster_pvalue, cluster_test_mode, cluster_iterations = (
        _cluster_sign_flip_pvalue(
            expected,
            single,
            leader,
            article_ids,
            random_seed=random_seed,
        )
    )
    accuracy_bootstrap, macro_f1_bootstrap = _bootstrap_deltas(
        expected,
        single,
        leader,
        iterations=bootstrap_iterations,
        random_seed=random_seed,
        clusters=article_ids,
    )
    return {
        "pairs": len(records),
        "unique_articles": int(len(np.unique(article_ids))),
        "single_accuracy": single_metrics["accuracy"],
        "leader_accuracy": leader_metrics["accuracy"],
        "accuracy_delta_pp": (
            float(leader_metrics["accuracy"]) - float(single_metrics["accuracy"])
        )
        * 100.0,
        "accuracy_delta_pp_ci95_lower": float(np.quantile(accuracy_bootstrap, 0.025)),
        "accuracy_delta_pp_ci95_upper": float(np.quantile(accuracy_bootstrap, 0.975)),
        "single_macro_f1": single_metrics["macro_f1"],
        "leader_macro_f1": leader_metrics["macro_f1"],
        "macro_f1_delta": float(leader_metrics["macro_f1"])
        - float(single_metrics["macro_f1"]),
        "macro_f1_delta_ci95_lower": float(np.quantile(macro_f1_bootstrap, 0.025)),
        "macro_f1_delta_ci95_upper": float(np.quantile(macro_f1_bootstrap, 0.975)),
        "leader_correct_single_wrong": leader_only,
        "single_correct_leader_wrong": single_only,
        "discordant_pairs": discordant,
        "mcnemar_exact_pvalue": mcnemar_pvalue,
        "mcnemar_unit": "article_ticker_pair_supplementary",
        "cluster_sign_flip_pvalue": cluster_pvalue,
        "cluster_sign_flip_mode": cluster_test_mode,
        "cluster_sign_flip_iterations": cluster_iterations,
        "primary_inference_unit": "article_id",
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_random_seed": random_seed,
        "bootstrap_unit": "article_id",
    }


def build_intrinsic_method_table(
    records: list[dict[str, object]],
    local_predictions: pd.DataFrame,
) -> pd.DataFrame:
    required = {"article_id", "ticker", "label", "predicted_label"}
    missing = sorted(required.difference(local_predictions.columns))
    if missing:
        raise ValueError(f"Local predictions are missing columns: {missing}")
    local = local_predictions.copy()
    local["item_id"] = (
        local["article_id"].astype(str) + "::" + local["ticker"].astype(str).str.upper()
    )
    if local["item_id"].duplicated().any():
        raise ValueError("Local predictions contain duplicate article-ticker pairs")
    local_by_id = local.set_index("item_id")
    item_ids = [str(record["item_id"]) for record in records]
    article_ids = _article_ids(item_ids)
    missing_items = sorted(set(item_ids).difference(local_by_id.index))
    if missing_items:
        raise ValueError(
            f"Local predictions are missing {len(missing_items)} LLM pairs"
        )
    matched = local_by_id.loc[item_ids]
    expected = [str(record["gold_label"]) for record in records]
    if expected != matched["label"].astype(str).tolist():
        raise ValueError("Local and LLM gold labels do not match")
    methods = (
        ("local_char_tfidf", matched["predicted_label"].astype(str).tolist()),
        ("terra_single", _predicted_labels(records, "single")),
        ("terra_debate_leader", _predicted_labels(records, "leader")),
    )
    rows: list[dict[str, object]] = []
    for method, predicted in methods:
        metrics = classification_metrics(expected, predicted, labels=LABELS)
        rows.append(
            {
                "method": method,
                "pairs": len(records),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "mcc": metrics["mcc"],
            }
        )
    return pd.DataFrame(rows)


def local_leader_paired_comparison(
    records: list[dict[str, object]],
    local_predictions: pd.DataFrame,
    *,
    bootstrap_iterations: int = 5_000,
    random_seed: int = 42,
) -> dict[str, object]:
    required = {"article_id", "ticker", "label", "predicted_label"}
    missing = sorted(required.difference(local_predictions.columns))
    if missing:
        raise ValueError(f"Local predictions are missing columns: {missing}")
    local = local_predictions.copy()
    local["item_id"] = (
        local["article_id"].astype(str) + "::" + local["ticker"].astype(str).str.upper()
    )
    if local["item_id"].duplicated().any():
        raise ValueError("Local predictions contain duplicate article-ticker pairs")
    local_by_id = local.set_index("item_id")
    item_ids = [str(record["item_id"]) for record in records]
    article_ids = _article_ids(item_ids)
    missing_items = sorted(set(item_ids).difference(local_by_id.index))
    if missing_items:
        raise ValueError(
            f"Local predictions are missing {len(missing_items)} LLM pairs"
        )
    matched = local_by_id.loc[item_ids]
    expected = np.asarray([str(record["gold_label"]) for record in records])
    local_gold = matched["label"].astype(str).to_numpy()
    if not np.array_equal(expected, local_gold):
        raise ValueError("Local and LLM gold labels do not match")
    local_predicted = matched["predicted_label"].astype(str).to_numpy()
    leader_predicted = np.asarray(_predicted_labels(records, "leader"))
    local_metrics = classification_metrics(
        expected,
        local_predicted,
        labels=LABELS,
    )
    leader_metrics = classification_metrics(
        expected,
        leader_predicted,
        labels=LABELS,
    )
    local_correct = local_predicted == expected
    leader_correct = leader_predicted == expected
    leader_only = int(np.sum(leader_correct & ~local_correct))
    local_only = int(np.sum(local_correct & ~leader_correct))
    discordant = leader_only + local_only
    cluster_pvalue, cluster_test_mode, cluster_iterations = (
        _cluster_sign_flip_pvalue(
            expected,
            local_predicted,
            leader_predicted,
            article_ids,
            random_seed=random_seed,
        )
    )
    accuracy_bootstrap, macro_f1_bootstrap = _bootstrap_deltas(
        expected,
        local_predicted,
        leader_predicted,
        iterations=bootstrap_iterations,
        random_seed=random_seed,
        clusters=article_ids,
    )
    return {
        "pairs": len(records),
        "unique_articles": int(len(np.unique(article_ids))),
        "local_accuracy": local_metrics["accuracy"],
        "leader_accuracy": leader_metrics["accuracy"],
        "accuracy_delta_pp_leader_minus_local": (
            float(leader_metrics["accuracy"]) - float(local_metrics["accuracy"])
        )
        * 100.0,
        "accuracy_delta_pp_ci95_lower": float(np.quantile(accuracy_bootstrap, 0.025)),
        "accuracy_delta_pp_ci95_upper": float(np.quantile(accuracy_bootstrap, 0.975)),
        "local_macro_f1": local_metrics["macro_f1"],
        "leader_macro_f1": leader_metrics["macro_f1"],
        "macro_f1_delta_leader_minus_local": float(leader_metrics["macro_f1"])
        - float(local_metrics["macro_f1"]),
        "macro_f1_delta_ci95_lower": float(np.quantile(macro_f1_bootstrap, 0.025)),
        "macro_f1_delta_ci95_upper": float(np.quantile(macro_f1_bootstrap, 0.975)),
        "leader_correct_local_wrong": leader_only,
        "local_correct_leader_wrong": local_only,
        "discordant_pairs": discordant,
        "mcnemar_exact_pvalue": (
            float(binomtest(min(leader_only, local_only), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        ),
        "mcnemar_unit": "article_ticker_pair_supplementary",
        "cluster_sign_flip_pvalue": cluster_pvalue,
        "cluster_sign_flip_mode": cluster_test_mode,
        "cluster_sign_flip_iterations": cluster_iterations,
        "primary_inference_unit": "article_id",
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_random_seed": random_seed,
        "bootstrap_unit": "article_id",
    }


def _probability_violations(
    predictions: list[dict[str, object]],
) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    decision_disagreements: list[str] = []
    for record in predictions:
        item_id = str(record["item_id"])
        for method in ("single", "leader"):
            verdict = record.get(method)
            if not isinstance(verdict, dict) or not isinstance(
                verdict.get("parsed"), dict
            ):
                violations.append(f"{item_id}::{method}::missing")
                continue
            parsed = verdict["parsed"]
            values = np.asarray(
                [float(parsed[f"{label}_probability"]) for label in LABELS]
            )
            predicted = str(parsed["predicted_label"])
            valid = (
                np.isfinite(values).all()
                and (values >= 0.0).all()
                and (values <= 1.0).all()
                and np.isclose(values.sum(), 1.0, atol=1e-6)
                and predicted in LABELS
            )
            if not valid:
                violations.append(f"{item_id}::{method}")
            elif values[LABELS.index(predicted)] < values.max() - 1e-9:
                decision_disagreements.append(f"{item_id}::{method}")
    return violations, decision_disagreements


def audit_llm_checkpoints(
    output_dir: Path,
    *,
    expected_pairs: int,
) -> dict[str, object]:
    if expected_pairs < 1:
        raise ValueError("expected_pairs must be positive")
    predictions = [
        row
        for row in _read_jsonl(output_dir / "predictions.jsonl")
        if row.get("status") == "completed"
    ]
    calls = [
        row
        for row in _read_jsonl(output_dir / "calls.jsonl")
        if row.get("status") == "completed"
    ]
    errors = _read_jsonl(output_dir / "errors.jsonl")
    prediction_counts = Counter(str(row["item_id"]) for row in predictions)
    duplicate_predictions = sorted(
        item_id for item_id, count in prediction_counts.items() if count > 1
    )
    completed_ids = set(prediction_counts)

    item_roles = Counter((str(row["item_id"]), str(row["role"])) for row in calls)
    duplicate_item_roles = sorted(
        f"{item_id}::{role}"
        for (item_id, role), count in item_roles.items()
        if count > 1
    )
    response_counts = Counter(
        str(row["response_id"]) for row in calls if row.get("response_id")
    )
    duplicate_response_ids = sorted(
        response_id for response_id, count in response_counts.items() if count > 1
    )
    roles_by_item: dict[str, set[str]] = {}
    for item_id, role in item_roles:
        roles_by_item.setdefault(item_id, set()).add(role)
    incomplete_roles = sorted(
        f"{item_id}:{','.join(sorted(REQUIRED_ROLES.difference(roles)))}"
        for item_id, roles in roles_by_item.items()
        if roles != REQUIRED_ROLES
    )
    unresolved_items = sorted(
        item_id
        for item_id in completed_ids
        if roles_by_item.get(item_id, set()) != REQUIRED_ROLES
    )
    if len(completed_ids) < expected_pairs:
        unresolved_items.append(
            f"<missing_completed_pairs:{expected_pairs - len(completed_ids)}>"
        )
    orphan_call_items = sorted(set(roles_by_item).difference(completed_ids))
    probability_violations, decision_probability_disagreements = (
        _probability_violations(predictions)
    )
    error_items = {
        str(row["item_id"]) for row in errors if row.get("item_id") is not None
    }
    recovered_error_items = error_items.intersection(completed_ids)
    valid = not any(
        (
            len(completed_ids) != expected_pairs,
            duplicate_predictions,
            duplicate_item_roles,
            duplicate_response_ids,
            incomplete_roles,
            unresolved_items,
            orphan_call_items,
            probability_violations,
        )
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "valid": valid,
        "expected_pairs": expected_pairs,
        "completed_pairs": len(completed_ids),
        "prediction_rows": len(predictions),
        "successful_calls": len(calls),
        "expected_roles_per_pair": sorted(REQUIRED_ROLES),
        "historical_errors": len(errors),
        "historical_error_items": len(error_items),
        "recovered_error_items": len(recovered_error_items),
        "duplicate_predictions": duplicate_predictions,
        "duplicate_item_roles": duplicate_item_roles,
        "duplicate_response_ids": duplicate_response_ids,
        "incomplete_roles": incomplete_roles,
        "unresolved_items": unresolved_items,
        "orphan_call_items": orphan_call_items,
        "probability_violations": probability_violations,
        "decision_probability_disagreements": (decision_probability_disagreements),
    }


def build_analysis_artifacts(
    output_dir: Path,
    *,
    expected_pairs: int,
    bootstrap_iterations: int = 5_000,
    random_seed: int = 42,
    local_predictions_file: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    records = [
        row
        for row in _read_jsonl(output_dir / "predictions.jsonl")
        if row.get("status") == "completed"
    ]
    unique = {str(row["item_id"]): row for row in records}
    comparison = paired_llm_comparison(
        list(unique.values()),
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
    )
    audit = audit_llm_checkpoints(output_dir, expected_pairs=expected_pairs)
    (output_dir / "llm_paired_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    (output_dir / "llm_checkpoint_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([comparison]).to_csv(
        output_dir / "paper_llm_paired_comparison.csv",
        index=False,
    )
    if local_predictions_file is not None:
        local_predictions = pd.read_csv(local_predictions_file)
        local_comparison = local_leader_paired_comparison(
            list(unique.values()),
            local_predictions,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed,
        )
        (output_dir / "local_leader_paired_comparison.json").write_text(
            json.dumps(local_comparison, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame([local_comparison]).to_csv(
            output_dir / "paper_local_leader_paired_comparison.csv",
            index=False,
        )
        build_intrinsic_method_table(
            list(unique.values()),
            local_predictions,
        ).to_csv(
            output_dir / "paper_track_b_intrinsic_comparison.csv",
            index=False,
        )
    return comparison, audit


def main() -> tuple[dict[str, object], dict[str, object]]:
    parser = argparse.ArgumentParser(
        description="Audit and compare paired Track B LLM predictions."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5_000)
    parser.add_argument("--local-predictions", type=Path)
    args = parser.parse_args()
    result = build_analysis_artifacts(
        args.output_dir,
        expected_pairs=args.expected_pairs,
        bootstrap_iterations=args.bootstrap_iterations,
        local_predictions_file=args.local_predictions,
    )
    print(json.dumps({"comparison": result[0], "audit": result[1]}, indent=2))
    return result


if __name__ == "__main__":
    main()
