from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from models.track_b_compute_matched import (
    MODEL_ID,
    NEW_REPLICATES,
    PROTOCOL_ID,
    apply_accuracy_holm,
    assemble_control_records,
    metrics_by_arm,
    paired_control_comparisons,
    sha256_file,
    validate_checkpoint_design,
    verify_freeze_manifest,
)
from models.track_b_data import filter_polarity_pairs, load_stocktbsa_pairs
from models.track_b_llm import (
    BenchmarkItem,
    BudgetLedger,
    LLMCall,
    OpenAIStructuredClient,
    Pricing,
    SentimentVerdict,
    build_single_messages,
    resolve_api_key,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT
    / "data-raw"
    / "track_b"
    / "Bilingual_StockTBSA"
    / "Thai_Financial_TBSA_dataset_Updated.json"
)
EXISTING_DIR = PROJECT_ROOT / "outputs" / "track_b" / "llm" / "locked_test_2023"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_b" / "llm" / "compute_matched_v1"
PRIVATE_DIR_NAME = "private"
FREEZE_MANIFEST = PROJECT_ROOT / "test" / "track_b_compute_matched_freeze_v1.json"
TEST_YEAR = 2023
PAIR_COUNT = 1333
ARTICLE_COUNT = 738
MAX_OUTPUT_TOKENS = 384
INCREMENTAL_COST_GUARD_USD = 18.0
DEFAULT_CONCURRENCY = 4
DEFAULT_BATCH_SIZE = 40
BOOTSTRAP_ITERATIONS = 5_000
RANDOM_SEED = 42
NEAR_COST_TOLERANCE = 0.15


@dataclass(frozen=True)
class CohortRow:
    item: BenchmarkItem
    gold_label: str
    existing: dict[str, object]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def checkpoint_runtime_audit(
    repeated_calls: list[dict[str, object]],
) -> dict[str, object]:
    if not repeated_calls:
        raise ValueError("At least one checkpoint is required for runtime audit")
    timestamps = [
        datetime.fromisoformat(str(row["completed_at_utc"]))
        for row in repeated_calls
    ]
    runtimes = pd.Series(
        [float(row["runtime_seconds"]) for row in repeated_calls],
        dtype=float,
    )
    if runtimes.isna().any() or (runtimes < 0).any():
        raise ValueError("Checkpoint runtimes must be finite and non-negative")
    first = min(timestamps)
    last = max(timestamps)
    return {
        "checkpoint_first_completed_at_utc": first.isoformat(),
        "checkpoint_last_completed_at_utc": last.isoformat(),
        "checkpoint_completion_span_seconds": (last - first).total_seconds(),
        "new_call_runtime_seconds_total": float(runtimes.sum()),
        "new_call_runtime_seconds_mean": float(runtimes.mean()),
        "new_call_runtime_seconds_median": float(runtimes.median()),
        "new_call_runtime_seconds_p95": float(runtimes.quantile(0.95)),
    }


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as destination:
        destination.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )
        destination.write("\n")
        destination.flush()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def checkpoint_payload(
    item_id: str,
    text_sha256: str,
    call: LLMCall,
) -> dict[str, object]:
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "completed",
        "completed_at_utc": _utc_now(),
        "item_id": item_id,
        "text_sha256": text_sha256,
        "role": call.role,
        "response_id": call.response_id,
        "parsed": call.parsed.model_dump(),
        "usage": asdict(call.usage),
        "runtime_seconds": call.runtime_seconds,
        "cost_usd": call.cost_usd,
    }


def _call_key(row: Mapping[str, object]) -> tuple[str, int]:
    item_id = str(row.get("item_id", "")).strip()
    role = str(row.get("role", ""))
    try:
        replicate = int(role.rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError("Checkpoint has an invalid repeated-single role") from error
    if not item_id or replicate not in NEW_REPLICATES:
        raise ValueError("Checkpoint key is outside the registered design")
    return item_id, replicate


def pending_call_keys(
    item_ids: Iterable[str],
    checkpoints: Iterable[Mapping[str, object]],
) -> list[tuple[str, int]]:
    items = tuple(str(item_id) for item_id in item_ids)
    if not items or len(items) != len(set(items)):
        raise ValueError("Item IDs must be non-empty and unique")
    rows = list(checkpoints)
    keys = [_call_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Checkpoint file contains duplicate calls")
    expected = {
        (item_id, replicate)
        for item_id in items
        for replicate in NEW_REPLICATES
    }
    observed = set(keys)
    extras = observed - expected
    if extras:
        raise ValueError(f"Checkpoint file contains extra calls: {sorted(extras)[:3]}")
    return sorted(expected - observed)


def _load_cohort() -> tuple[list[CohortRow], dict[str, object]]:
    freeze = verify_freeze_manifest(PROJECT_ROOT, FREEZE_MANIFEST)
    pairs = filter_polarity_pairs(load_stocktbsa_pairs(DATASET_PATH))
    selected = pairs.loc[pairs["year"].eq(TEST_YEAR)].copy()
    selected["item_id"] = (
        selected["article_id"].astype(str)
        + "::"
        + selected["ticker"].astype(str).str.upper()
    )
    if len(selected) != PAIR_COUNT or selected["item_id"].nunique() != PAIR_COUNT:
        raise ValueError("Locked 2023 cohort cardinality changed")
    existing_rows = _read_jsonl(EXISTING_DIR / "predictions.jsonl")
    existing = {str(row["item_id"]): row for row in existing_rows}
    if len(existing) != PAIR_COUNT or set(existing) != set(selected["item_id"]):
        raise ValueError("Existing LLM prediction IDs do not match the cohort")

    cohort: list[CohortRow] = []
    for row in selected.sort_values("item_id").itertuples(index=False):
        item = BenchmarkItem.from_values(
            article_id=row.article_id,
            ticker=row.ticker,
            date=pd.Timestamp(row.date).strftime("%Y-%m-%d"),
            text=row.text,
        )
        inherited = existing[item.item_id]
        if inherited.get("text_sha256") != item.text_sha256:
            raise ValueError(f"Text hash mismatch: {item.item_id}")
        if str(inherited.get("gold_label")) != str(row.label):
            raise ValueError(f"Gold label mismatch: {item.item_id}")
        cohort.append(
            CohortRow(
                item=item,
                gold_label=str(row.label),
                existing=inherited,
            )
        )
    unique_articles = len({row.item.article_id for row in cohort})
    if unique_articles != ARTICLE_COUNT:
        raise ValueError("Locked article-cluster count changed")
    label_counts = pd.Series([row.gold_label for row in cohort]).value_counts()
    audit = {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": _utc_now(),
        "passed": True,
        "freeze_inputs_match": freeze["all_inputs_match"],
        "pairs": len(cohort),
        "unique_articles": unique_articles,
        "label_counts": {
            label: int(label_counts.get(label, 0))
            for label in ("negative", "neutral", "positive")
        },
        "item_ids_exact": True,
        "text_hashes_exact": True,
        "gold_labels_exact": True,
        "raw_article_written_to_output": False,
    }
    return cohort, audit


def _generate_call(
    client: OpenAIStructuredClient,
    row: CohortRow,
    replicate: int,
) -> tuple[CohortRow, LLMCall]:
    call = client.generate(
        role=f"single_rep_{replicate}",
        messages=build_single_messages(row.item),
        response_type=SentimentVerdict,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return row, call


def _call_totals(calls: Iterable[Mapping[str, object]]) -> dict[str, float | int]:
    rows = list(calls)
    return {
        "calls": len(rows),
        "input_tokens": sum(int(row["usage"]["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["usage"]["output_tokens"]) for row in rows),
        "reasoning_tokens": sum(
            int(row["usage"].get("reasoning_tokens", 0)) for row in rows
        ),
        "cost_usd": float(sum(float(row["cost_usd"]) for row in rows)),
        "runtime_seconds_total": float(
            sum(float(row["runtime_seconds"]) for row in rows)
        ),
        "runtime_seconds_mean": float(
            sum(float(row["runtime_seconds"]) for row in rows) / len(rows)
        ),
    }


def _runtime_cost_summary(
    existing: list[dict[str, object]],
    repeats: list[dict[str, object]],
) -> pd.DataFrame:
    first = [dict(row["single"]) for row in existing]
    debate = [
        dict(row[role])
        for row in existing
        for role in ("bull", "bear", "leader")
    ]
    repeat_by_number = {
        replicate: [row for row in repeats if _call_key(row)[1] == replicate]
        for replicate in NEW_REPLICATES
    }
    arms = {
        "single": first,
        "self_consistency_3": [*first, *repeat_by_number[2], *repeat_by_number[3]],
        "self_consistency_4": [
            *first,
            *repeat_by_number[2],
            *repeat_by_number[3],
            *repeat_by_number[4],
        ],
        "leader_system": debate,
    }
    return pd.DataFrame(
        [{"arm": arm, **_call_totals(calls)} for arm, calls in arms.items()]
    )


def _write_public_results(
    output_dir: Path,
    cohort: list[CohortRow],
    repeated_calls: list[dict[str, object]],
    cohort_audit: dict[str, object],
) -> dict[str, object]:
    aggregation_started = time.perf_counter()
    existing = [row.existing for row in cohort]
    assembled = assemble_control_records(existing, repeated_calls)
    metrics = metrics_by_arm(assembled)
    comparisons = paired_control_comparisons(
        assembled,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        random_seed=RANDOM_SEED,
    )
    adjusted = apply_accuracy_holm(comparisons)
    runtime = _runtime_cost_summary(existing, repeated_calls)
    costs = runtime.set_index("arm")["cost_usd"]
    near_cost_ratio = float(costs["self_consistency_4"] / costs["leader_system"])
    near_cost_passed = abs(near_cost_ratio - 1.0) <= NEAR_COST_TOLERANCE
    paper = adjusted.merge(
        metrics.add_prefix("control_").rename(
            columns={"control_arm": "control_arm"}
        ),
        on="control_arm",
        how="left",
        validate="one_to_one",
    )
    leader_metrics = metrics.loc[metrics["arm"].eq("leader")].iloc[0]
    for column in (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "log_loss",
        "multiclass_brier_score",
    ):
        paper[f"leader_{column}"] = leader_metrics[column]

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "metrics_by_arm.csv": metrics,
        "paired_comparisons.csv": comparisons,
        "paired_comparisons_holm.csv": adjusted,
        "runtime_cost_summary.csv": runtime,
        "paper_compute_matched_table.csv": paper,
    }
    public_paths: list[Path] = []
    for name, frame in outputs.items():
        path = output_dir / name
        frame.to_csv(path, index=False)
        public_paths.append(path)
    cohort_path = output_dir / "cohort_integrity_audit.json"
    _write_json(cohort_path, cohort_audit)
    public_paths.append(cohort_path)
    checkpoint_audit = {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": _utc_now(),
        "passed": True,
        **validate_checkpoint_design(
            [row.item.item_id for row in cohort], repeated_calls
        ),
        "duplicate_response_ids": len(
            {str(row["response_id"]) for row in repeated_calls}
        )
        != len(repeated_calls),
        "non_finite_metric_values": False,
    }
    if checkpoint_audit["duplicate_response_ids"]:
        raise ValueError("Repeated-single checkpoints duplicate response IDs")
    checkpoint_path = output_dir / "checkpoint_audit.json"
    _write_json(checkpoint_path, checkpoint_audit)
    public_paths.append(checkpoint_path)
    runtime_audit = checkpoint_runtime_audit(repeated_calls)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "completed_at_utc": _utc_now(),
        "status": "complete",
        "model": MODEL_ID,
        "reasoning_effort": "low",
        "prompt_version": "track-b-terra-v1",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "new_calls": len(repeated_calls),
        "incremental_cost_usd": float(
            sum(float(row["cost_usd"]) for row in repeated_calls)
        ),
        "incremental_cost_guard_usd": INCREMENTAL_COST_GUARD_USD,
        **runtime_audit,
        "aggregation_compute_seconds": time.perf_counter() - aggregation_started,
        "near_cost_ratio_sc4_to_leader": near_cost_ratio,
        "near_cost_gate_passed": near_cost_passed,
        "near_cost_tolerance": NEAR_COST_TOLERANCE,
        "primary_comparison": "leader_minus_self_consistency_3",
        "packages": _package_versions(
            ["numpy", "pandas", "scikit-learn", "openai", "pydantic"]
        ),
        "python": sys.version,
        "platform": platform.platform(),
    }
    metadata_path = output_dir / "run_metadata.json"
    _write_json(metadata_path, metadata)
    public_paths.append(metadata_path)
    manifest_path = output_dir / "output_manifest.json"
    _write_json(
        manifest_path,
        {
            "protocol_id": PROTOCOL_ID,
            "created_at_utc": _utc_now(),
            "private_checkpoints_excluded": True,
            "files": [
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(public_paths)
            ],
        },
    )
    primary = adjusted.loc[
        adjusted["comparison_id"].eq("leader_minus_self_consistency_3")
    ].iloc[0]
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "pairs": len(cohort),
        "new_calls": len(repeated_calls),
        "incremental_cost_usd": metadata["incremental_cost_usd"],
        "leader_minus_sc3_accuracy_delta_pp": primary["accuracy_delta_pp"],
        "leader_minus_sc3_holm_pvalue": primary["holm_adjusted_pvalue"],
        "near_cost_gate_passed": near_cost_passed,
    }


def run_compute_matched(
    *,
    output_dir: Path = OUTPUT_DIR,
    key_file: Path = PROJECT_ROOT / "key.md",
    max_concurrent_calls: int = DEFAULT_CONCURRENCY,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, object]:
    if max_concurrent_calls < 1 or batch_size < 1:
        raise ValueError("Concurrency and batch size must be positive")
    cohort, cohort_audit = _load_cohort()
    private_dir = output_dir / PRIVATE_DIR_NAME
    calls_path = private_dir / "calls.jsonl"
    errors_path = private_dir / "errors.jsonl"
    repeated_calls = _read_jsonl(calls_path)
    item_ids = [row.item.item_id for row in cohort]
    pending = pending_call_keys(item_ids, repeated_calls)
    existing_cost = float(sum(float(row["cost_usd"]) for row in repeated_calls))
    if existing_cost > INCREMENTAL_COST_GUARD_USD + 1e-12:
        raise ValueError("Existing checkpoint cost exceeds the frozen guard")
    if pending:
        remaining_budget = INCREMENTAL_COST_GUARD_USD - existing_cost
        client = OpenAIStructuredClient(
            api_key=resolve_api_key(key_file),
            budget=BudgetLedger(remaining_budget),
            model=MODEL_ID,
            pricing=Pricing(),
            reasoning_effort="low",
        )
        by_id = {row.item.item_id: row for row in cohort}
        completed = len(repeated_calls)
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            with ThreadPoolExecutor(max_workers=max_concurrent_calls) as executor:
                futures = {
                    executor.submit(
                        _generate_call,
                        client,
                        by_id[item_id],
                        replicate,
                    ): (item_id, replicate)
                    for item_id, replicate in batch
                }
                for future in as_completed(futures):
                    item_id, replicate = futures[future]
                    try:
                        row, call = future.result()
                        payload = checkpoint_payload(
                            item_id,
                            row.item.text_sha256,
                            call,
                        )
                        _append_jsonl(calls_path, payload)
                        repeated_calls.append(payload)
                        completed += 1
                    except Exception as error:  # noqa: BLE001 - resumable batch
                        _append_jsonl(
                            errors_path,
                            {
                                "protocol_id": PROTOCOL_ID,
                                "status": "failed",
                                "failed_at_utc": _utc_now(),
                                "item_id": item_id,
                                "replicate": replicate,
                                "error_type": type(error).__name__,
                            },
                        )
            print(
                f"Compute-matched LLM progress: {completed}/{PAIR_COUNT * 3} "
                f"calls, tracked incremental cost "
                f"${sum(float(row['cost_usd']) for row in repeated_calls):.4f}",
                flush=True,
            )
    remaining = pending_call_keys(item_ids, repeated_calls)
    if remaining:
        return {
            "protocol_id": PROTOCOL_ID,
            "status": "incomplete",
            "completed_calls": len(repeated_calls),
            "remaining_calls": len(remaining),
        }
    return _write_public_results(
        output_dir,
        cohort,
        repeated_calls,
        cohort_audit,
    )


def aggregate_existing(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    cohort, cohort_audit = _load_cohort()
    repeated_calls = _read_jsonl(output_dir / PRIVATE_DIR_NAME / "calls.jsonl")
    validate_checkpoint_design(
        [row.item.item_id for row in cohort], repeated_calls
    )
    return _write_public_results(
        output_dir,
        cohort,
        repeated_calls,
        cohort_audit,
    )


def main(argv: list[str] | None = None) -> object:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    run.add_argument("--key-file", type=Path, default=PROJECT_ROOT / "key.md")
    run.add_argument("--max-concurrent-calls", type=int, default=DEFAULT_CONCURRENCY)
    run.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_compute_matched(
            output_dir=args.output_dir,
            key_file=args.key_file,
            max_concurrent_calls=args.max_concurrent_calls,
            batch_size=args.batch_size,
        )
    else:
        result = aggregate_existing(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
