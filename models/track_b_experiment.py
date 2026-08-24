from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import log_loss

from models.track_b_baseline import (
    LocalRelevanceClassifier,
    LocalSentimentClassifier,
    classification_metrics,
    make_relevance_labels,
)
from models.track_b_data import filter_polarity_pairs, load_stocktbsa_pairs
from models.track_b_llm import (
    MODEL_ID,
    PROMPT_VERSION,
    BenchmarkItem,
    BudgetExceededError,
    BudgetLedger,
    DebateResult,
    LLMCall,
    OpenAIStructuredClient,
    Pricing,
    SentimentVerdict,
    TokenUsage,
    WorkerArgument,
    build_leader_messages,
    build_single_messages,
    build_worker_messages,
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
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "track_b"
LOCAL_OUTPUT_DIR = OUTPUT_ROOT / "local_baseline"
LLM_OUTPUT_DIR = OUTPUT_ROOT / "llm"
RANDOM_SEED = 42
TEST_YEARS = (2019, 2020, 2021, 2022, 2023)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        file.write("\n")
        file.flush()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from error
            if not isinstance(payload, dict):
                raise TypeError(f"JSONL row {line_number} is not an object")
            rows.append(payload)
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _packages(names: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def select_stratified_pairs(
    frame: pd.DataFrame,
    *,
    sample_size: int,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("Sample size must be positive")
    if "label" not in frame:
        raise ValueError("Frame is missing label")
    if sample_size >= len(frame):
        return frame.sort_values(["article_id", "ticker"]).reset_index(drop=True)
    labels = sorted(frame["label"].unique())
    base, remainder = divmod(sample_size, len(labels))
    samples: list[pd.DataFrame] = []
    for index, label in enumerate(labels):
        group = frame.loc[frame["label"].eq(label)]
        requested = base + (1 if index < remainder else 0)
        if requested > len(group):
            raise ValueError(f"Class {label} is too small for stratified sample")
        samples.append(group.sample(n=requested, random_state=random_seed + index))
    return (
        pd.concat(samples).sort_values(["article_id", "ticker"]).reset_index(drop=True)
    )


def _expanding_train(frame: pd.DataFrame, test_year: int) -> pd.DataFrame:
    train = frame.loc[frame["year"].lt(test_year)].reset_index(drop=True)
    if train.empty:
        raise ValueError(f"No training data precedes {test_year}")
    if int(train["year"].max()) >= test_year:
        raise ValueError("Expanding training split leaks test-year data")
    return train


def _prediction_metrics(
    frame: pd.DataFrame,
    *,
    labels: tuple[str, ...],
) -> dict[str, object]:
    return classification_metrics(
        frame["label"],
        frame["predicted_label"],
        labels=labels,
    )


def _run_sentiment_year(
    polarity: pd.DataFrame,
    test_year: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    train = _expanding_train(polarity, test_year)
    test = polarity.loc[polarity["year"].eq(test_year)].reset_index(drop=True)
    model = LocalSentimentClassifier()
    model.fit(train)
    prediction = model.predict(test)
    runtime = model.runtime
    metrics = {
        "task": "sentiment",
        "test_year": test_year,
        "train_start_year": int(train["year"].min()),
        "train_end_year": int(train["year"].max()),
        "train_pairs": len(train),
        "test_pairs": len(test),
        **_prediction_metrics(
            prediction,
            labels=("positive", "neutral", "negative"),
        ),
        "fit_seconds": runtime.fit_seconds,
        "predict_seconds": runtime.predict_seconds,
    }
    return prediction.assign(test_year=test_year), metrics


def _run_relevance_year(
    pairs: pd.DataFrame,
    test_year: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    eligible = pairs.loc[~pairs["label"].eq("ambiguous")].reset_index(drop=True)
    train = _expanding_train(eligible, test_year)
    test = eligible.loc[eligible["year"].eq(test_year)].reset_index(drop=True)
    model = LocalRelevanceClassifier()
    model.fit(train)
    prediction = model.predict(test)
    expected = make_relevance_labels(prediction["label"]).astype(int)
    scored = prediction.assign(
        expected_label=np.where(expected.eq(1), "relevant", "irrelevant"),
        predicted_label=np.where(
            prediction["predicted_relevant"].eq(1),
            "relevant",
            "irrelevant",
        ),
        test_year=test_year,
    )
    runtime = model.runtime
    metrics = {
        "task": "relevance",
        "test_year": test_year,
        "train_start_year": int(train["year"].min()),
        "train_end_year": int(train["year"].max()),
        "train_pairs": len(train),
        "test_pairs": len(test),
        **classification_metrics(
            scored["expected_label"],
            scored["predicted_label"],
            labels=("relevant", "irrelevant"),
        ),
        "fit_seconds": runtime.fit_seconds,
        "predict_seconds": runtime.predict_seconds,
    }
    return scored, metrics


def run_local_baseline(
    *,
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = LOCAL_OUTPUT_DIR,
) -> dict[str, object]:
    started = time.perf_counter()
    pairs = load_stocktbsa_pairs(dataset_path)
    polarity = filter_polarity_pairs(pairs)
    sentiment_runs = [_run_sentiment_year(polarity, year) for year in TEST_YEARS]
    relevance_runs = [_run_relevance_year(pairs, year) for year in TEST_YEARS]
    sentiment_predictions = pd.concat(
        [prediction for prediction, _ in sentiment_runs],
        ignore_index=True,
    )
    relevance_predictions = pd.concat(
        [prediction for prediction, _ in relevance_runs],
        ignore_index=True,
    )
    metrics = [metrics for _, metrics in [*sentiment_runs, *relevance_runs]]
    output_dir.mkdir(parents=True, exist_ok=True)
    sentiment_predictions.to_csv(
        output_dir / "sentiment_predictions_expanding.csv",
        index=False,
    )
    relevance_predictions.to_csv(
        output_dir / "relevance_predictions_expanding.csv",
        index=False,
    )
    paper_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "task",
                "test_year",
                "train_pairs",
                "test_pairs",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "mcc",
                "fit_seconds",
                "predict_seconds",
            }
        }
        for row in metrics
    ]
    pd.DataFrame(paper_rows).to_csv(
        output_dir / "paper_local_intrinsic_table.csv",
        index=False,
    )
    metadata = {
        "created_at": _utc_now(),
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_sha256": _file_sha256(dataset_path),
        "article_ticker_pairs": len(pairs),
        "polarity_pairs": len(polarity),
        "split_contract": "expanding year; all train years strictly precede test year",
        "test_years": list(TEST_YEARS),
        "model": "target-aware character TF-IDF + class-weighted logistic regression",
        "total_runtime_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _packages(["numpy", "pandas", "scikit-learn", "scipy"]),
        },
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    return metadata


def _call_payload(call: LLMCall) -> dict[str, object]:
    return {
        "role": call.role,
        "response_id": call.response_id,
        "parsed": call.parsed.model_dump(),
        "usage": {
            "input_tokens": call.usage.input_tokens,
            "output_tokens": call.usage.output_tokens,
            "cached_input_tokens": call.usage.cached_input_tokens,
            "reasoning_tokens": call.usage.reasoning_tokens,
        },
        "runtime_seconds": call.runtime_seconds,
        "cost_usd": call.cost_usd,
    }


def serialize_debate_result(
    result: DebateResult,
    *,
    gold_label: str,
) -> dict[str, object]:
    payload = {
        "status": "completed",
        "completed_at": _utc_now(),
        "item_id": result.item.item_id,
        "article_id": result.item.article_id,
        "ticker": result.item.ticker,
        "date": result.item.date,
        "text_sha256": result.item.text_sha256,
        "gold_label": gold_label,
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "single": _call_payload(result.single) if result.single else None,
        "bull": _call_payload(result.bull),
        "bear": _call_payload(result.bear),
        "leader": _call_payload(result.leader),
        "total_cost_usd": result.total_cost_usd,
        "critical_path_runtime_seconds": result.total_runtime_seconds,
    }
    return payload


def load_completed_item_ids(path: Path) -> set[str]:
    return {
        str(row["item_id"])
        for row in _read_jsonl(path)
        if row.get("status") == "completed" and row.get("item_id")
    }


def _deserialize_call(payload: dict[str, object]) -> LLMCall:
    role = str(payload["role"])
    parsed_payload = payload["parsed"]
    parsed: BaseModel
    if role in {"bull", "bear"}:
        parsed = WorkerArgument.model_validate(parsed_payload)
    else:
        parsed = SentimentVerdict.model_validate(parsed_payload)
    usage = TokenUsage(**payload["usage"])
    return LLMCall(
        role=role,
        response_id=str(payload["response_id"]),
        parsed=parsed,
        usage=usage,
        runtime_seconds=float(payload["runtime_seconds"]),
        cost_usd=float(payload["cost_usd"]),
    )


def _call_checkpoint_payload(item_id: str, call: LLMCall) -> dict[str, object]:
    return {
        "status": "completed",
        "completed_at": _utc_now(),
        "item_id": item_id,
        **_call_payload(call),
    }


def _load_call_checkpoints(
    path: Path,
) -> tuple[dict[tuple[str, str], LLMCall], float]:
    calls: dict[tuple[str, str], LLMCall] = {}
    billed_cost = 0.0
    for row in _read_jsonl(path):
        if row.get("status") != "completed":
            continue
        billed_cost += float(row.get("cost_usd", 0.0))
        call = _deserialize_call(row)
        calls[(str(row["item_id"]), call.role)] = call
    return calls, billed_cost


def _benchmark_item(row: pd.Series) -> BenchmarkItem:
    date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    return BenchmarkItem.from_values(
        article_id=row["article_id"],
        ticker=row["ticker"],
        date=date,
        text=row["text"],
    )


@dataclass(frozen=True)
class ItemRun:
    item: BenchmarkItem
    calls: tuple[LLMCall, ...]
    result: DebateResult | None
    errors: tuple[str, ...]
    budget_exhausted: bool


def _first_stage_call(client, item: BenchmarkItem, role: str) -> LLMCall:
    if role == "single":
        return client.generate(
            role=role,
            messages=build_single_messages(item),
            response_type=SentimentVerdict,
            max_output_tokens=384,
        )
    stance = "bullish" if role == "bull" else "bearish"
    return client.generate(
        role=role,
        messages=build_worker_messages(item, stance=stance),
        response_type=WorkerArgument,
        max_output_tokens=384,
    )


def _run_item(
    client,
    item: BenchmarkItem,
    existing: dict[str, LLMCall],
    *,
    include_single: bool,
) -> ItemRun:
    calls = dict(existing)
    new_calls: list[LLMCall] = []
    errors: list[str] = []
    budget_exhausted = False
    required_first = ["bull", "bear"] + (["single"] if include_single else [])
    missing = [role for role in required_first if role not in calls]
    with ThreadPoolExecutor(max_workers=max(1, len(missing))) as executor:
        futures = {
            executor.submit(_first_stage_call, client, item, role): role
            for role in missing
        }
        for future, role in [(future, futures[future]) for future in futures]:
            try:
                call = future.result()
                calls[role] = call
                new_calls.append(call)
            except BudgetExceededError:
                budget_exhausted = True
                errors.append(f"{role}:BudgetExceededError")
            except Exception as error:  # noqa: BLE001 - batch must checkpoint successes
                errors.append(f"{role}:{type(error).__name__}")
    if "bull" in calls and "bear" in calls and "leader" not in calls:
        try:
            leader = client.generate(
                role="leader",
                messages=build_leader_messages(
                    item,
                    calls["bull"].parsed,
                    calls["bear"].parsed,
                ),
                response_type=SentimentVerdict,
                max_output_tokens=512,
            )
            calls["leader"] = leader
            new_calls.append(leader)
        except BudgetExceededError:
            budget_exhausted = True
            errors.append("leader:BudgetExceededError")
        except Exception as error:  # noqa: BLE001 - batch must checkpoint successes
            errors.append(f"leader:{type(error).__name__}")
    required = {"bull", "bear", "leader"} | ({"single"} if include_single else set())
    result = None
    if required.issubset(calls):
        result = DebateResult(
            item=item,
            single=calls.get("single"),
            bull=calls["bull"],
            bear=calls["bear"],
            leader=calls["leader"],
        )
    return ItemRun(
        item=item,
        calls=tuple(new_calls),
        result=result,
        errors=tuple(errors),
        budget_exhausted=budget_exhausted,
    )


def _save_prompt_manifest(output_dir: Path) -> None:
    item = BenchmarkItem(
        item_id="<ARTICLE_ID>::<TICKER>",
        article_id="<ARTICLE_ID>",
        ticker="<TICKER>",
        date="<YYYY-MM-DD>",
        text="<ARTICLE_TEXT>",
        text_sha256="<SHA256>",
    )
    bull = WorkerArgument(
        stance="bullish",
        relevance_probability=0.5,
        claim_strength=50,
        evidence=["<EVIDENCE>"],
        counterevidence="<COUNTEREVIDENCE>",
    )
    bear = bull.model_copy(update={"stance": "bearish"})
    prompts = {
        "single": build_single_messages(item),
        "bull": build_worker_messages(item, stance="bullish"),
        "bear": build_worker_messages(item, stance="bearish"),
        "leader": build_leader_messages(item, bull, bear),
    }
    serialized = json.dumps(prompts, ensure_ascii=False, sort_keys=True)
    _write_json(
        output_dir / "prompt_manifest.json",
        {
            "prompt_version": PROMPT_VERSION,
            "model": MODEL_ID,
            "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "prompts": prompts,
            "schemas": {
                "sentiment_verdict": SentimentVerdict.model_json_schema(),
                "worker_argument": WorkerArgument.model_json_schema(),
            },
        },
    )


def _write_run_metadata(
    output_dir: Path,
    *,
    year: int,
    selected_pairs: int,
    completed_pairs: int,
    existing_billed_cost: float,
    new_cost: float,
    max_budget_usd: float,
    include_single: bool,
    concurrency: int,
    started: float,
) -> None:
    _write_json(
        output_dir / "run_metadata.json",
        {
            "updated_at": _utc_now(),
            "model": MODEL_ID,
            "prompt_version": PROMPT_VERSION,
            "year": year,
            "selected_pairs": selected_pairs,
            "completed_pairs": completed_pairs,
            "include_single": include_single,
            "reasoning_effort": "low",
            "store_api_responses": False,
            "max_concurrent_items": concurrency,
            "maximum_budget_usd": max_budget_usd,
            "existing_billed_cost_usd": existing_billed_cost,
            "new_cost_usd": new_cost,
            "tracked_total_cost_usd": existing_billed_cost + new_cost,
            "runtime_seconds_this_invocation": time.perf_counter() - started,
            "checkpoint_scope": "one JSONL row per API role; raw article omitted",
            "pricing": {
                "input_usd_per_million": Pricing().input_usd_per_million,
                "output_usd_per_million": Pricing().output_usd_per_million,
                "cached_input_discount_ignored_for_conservative_tracking": True,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": _packages(["openai", "pydantic", "tiktoken"]),
            },
        },
    )


def run_llm_benchmark(
    *,
    year: int,
    output_dir: Path,
    key_file: Path,
    sample_size: int | None,
    max_budget_usd: float,
    max_concurrent_items: int = 2,
    include_single: bool = True,
    dataset_path: Path = DATASET_PATH,
) -> dict[str, object]:
    if max_concurrent_items < 1:
        raise ValueError("max_concurrent_items must be positive")
    started = time.perf_counter()
    pairs = filter_polarity_pairs(load_stocktbsa_pairs(dataset_path))
    selected = pairs.loc[pairs["year"].eq(year)].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"No polarity-labelled pairs found for {year}")
    if sample_size is not None:
        selected = select_stratified_pairs(
            selected,
            sample_size=sample_size,
            random_seed=RANDOM_SEED,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_prompt_manifest(output_dir)
    calls_path = output_dir / "calls.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    errors_path = output_dir / "errors.jsonl"
    call_map, existing_billed_cost = _load_call_checkpoints(calls_path)
    if existing_billed_cost >= max_budget_usd:
        raise BudgetExceededError("Existing checkpoints already reach the budget")
    budget = BudgetLedger(max_budget_usd - existing_billed_cost)
    client = OpenAIStructuredClient(
        api_key=resolve_api_key(key_file),
        budget=budget,
        model=MODEL_ID,
        pricing=Pricing(),
        reasoning_effort="low",
    )
    completed_ids = load_completed_item_ids(predictions_path)
    rows = {
        f"{row.article_id}::{str(row.ticker).upper()}": row
        for row in selected.itertuples(index=False)
    }
    pending_ids = sorted(set(rows).difference(completed_ids))
    stop_for_budget = False
    for offset in range(0, len(pending_ids), max_concurrent_items):
        batch_ids = pending_ids[offset : offset + max_concurrent_items]
        with ThreadPoolExecutor(max_workers=max_concurrent_items) as executor:
            futures = {}
            for item_id in batch_ids:
                row = pd.Series(rows[item_id]._asdict())
                item = _benchmark_item(row)
                existing = {
                    role: call
                    for (checkpoint_item, role), call in call_map.items()
                    if checkpoint_item == item_id
                }
                future = executor.submit(
                    _run_item,
                    client,
                    item,
                    existing,
                    include_single=include_single,
                )
                futures[future] = str(row["label"])
            for future in as_completed(futures):
                gold_label = futures[future]
                item_run = future.result()
                for call in item_run.calls:
                    payload = _call_checkpoint_payload(item_run.item.item_id, call)
                    _append_jsonl(calls_path, payload)
                    call_map[(item_run.item.item_id, call.role)] = call
                for error in item_run.errors:
                    _append_jsonl(
                        errors_path,
                        {
                            "status": "failed",
                            "failed_at": _utc_now(),
                            "item_id": item_run.item.item_id,
                            "error_type": error,
                        },
                    )
                if item_run.result is not None:
                    _append_jsonl(
                        predictions_path,
                        serialize_debate_result(
                            item_run.result,
                            gold_label=gold_label,
                        ),
                    )
                    completed_ids.add(item_run.item.item_id)
                stop_for_budget = stop_for_budget or item_run.budget_exhausted
        _write_run_metadata(
            output_dir,
            year=year,
            selected_pairs=len(selected),
            completed_pairs=len(completed_ids.intersection(rows)),
            existing_billed_cost=existing_billed_cost,
            new_cost=budget.spent_usd,
            max_budget_usd=max_budget_usd,
            include_single=include_single,
            concurrency=max_concurrent_items,
            started=started,
        )
        print(
            f"Track B LLM progress: {len(completed_ids.intersection(rows))}/"
            f"{len(selected)} pairs, tracked cost "
            f"${existing_billed_cost + budget.spent_usd:.4f}"
        )
        if stop_for_budget:
            break
    return build_llm_report(output_dir)


def _probabilistic_metrics(
    records: list[dict[str, object]],
    method: str,
) -> dict[str, float]:
    labels = ["negative", "neutral", "positive"]
    expected = [str(record["gold_label"]) for record in records]
    probabilities = np.asarray(
        [
            [
                float(record[method]["parsed"][f"{label}_probability"])
                for label in labels
            ]
            for record in records
        ]
    )
    probabilities = np.clip(probabilities, 1e-12, None)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    one_hot = np.eye(len(labels))[[labels.index(label) for label in expected]]
    return {
        "log_loss": float(log_loss(expected, probabilities, labels=labels)),
        "multiclass_brier_score": float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        ),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }


def build_llm_report(output_dir: Path) -> dict[str, object]:
    records = [
        row
        for row in _read_jsonl(output_dir / "predictions.jsonl")
        if row.get("status") == "completed"
    ]
    unique = {str(row["item_id"]): row for row in records}
    records = list(unique.values())
    if not records:
        report = {"completed_pairs": 0, "metrics": []}
        _write_json(output_dir / "metrics.json", report)
        return report
    metric_rows: list[dict[str, object]] = []
    methods = ["leader"]
    if all(record.get("single") is not None for record in records):
        methods.insert(0, "single")
    for method in methods:
        expected = [str(record["gold_label"]) for record in records]
        predicted = [
            str(record[method]["parsed"]["predicted_label"]) for record in records
        ]
        metric_rows.append(
            {
                "method": method,
                "pairs": len(records),
                **classification_metrics(
                    expected,
                    predicted,
                    labels=("positive", "neutral", "negative"),
                ),
                **_probabilistic_metrics(records, method),
            }
        )
    calls = [
        row
        for row in _read_jsonl(output_dir / "calls.jsonl")
        if row.get("status") == "completed"
    ]
    runtime_rows = []
    if calls:
        call_frame = pd.DataFrame(calls)
        runtime_rows = (
            call_frame.groupby("role")
            .agg(
                calls=("role", "size"),
                runtime_seconds_mean=("runtime_seconds", "mean"),
                runtime_seconds_median=("runtime_seconds", "median"),
                runtime_seconds_total=("runtime_seconds", "sum"),
                input_tokens_total=(
                    "usage",
                    lambda values: sum(v["input_tokens"] for v in values),
                ),
                output_tokens_total=(
                    "usage",
                    lambda values: sum(v["output_tokens"] for v in values),
                ),
                cost_usd_total=("cost_usd", "sum"),
            )
            .reset_index()
            .to_dict(orient="records")
        )
    disagreement = None
    if methods == ["single", "leader"]:
        disagreement = float(
            np.mean(
                [
                    record["single"]["parsed"]["predicted_label"]
                    != record["leader"]["parsed"]["predicted_label"]
                    for record in records
                ]
            )
        )
    report = {
        "created_at": _utc_now(),
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "completed_pairs": len(records),
        "single_leader_disagreement_rate": disagreement,
        "metrics": metric_rows,
        "runtime_and_cost_by_role": runtime_rows,
        "tracked_cost_usd": float(sum(float(row["cost_usd"]) for row in calls)),
    }
    paper_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "method",
                "pairs",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "mcc",
                "log_loss",
                "multiclass_brier_score",
                "mean_confidence",
            }
        }
        for row in metric_rows
    ]
    pd.DataFrame(paper_rows).to_csv(
        output_dir / "paper_llm_intrinsic_table.csv",
        index=False,
    )
    pd.DataFrame(runtime_rows).to_csv(
        output_dir / "runtime_cost_by_role.csv",
        index=False,
    )
    _write_json(output_dir / "metrics.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible Track B experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    local = subparsers.add_parser("local", help="Run expanding-year local baselines")
    local.add_argument("--output-dir", type=Path, default=LOCAL_OUTPUT_DIR)

    llm = subparsers.add_parser("llm", help="Run resumable Terra benchmark")
    llm.add_argument("--year", type=int, required=True)
    llm.add_argument("--output-dir", type=Path, required=True)
    llm.add_argument("--key-file", type=Path, default=PROJECT_ROOT / "key.md")
    llm.add_argument("--sample-size", type=int, default=None)
    llm.add_argument("--budget-usd", type=float, required=True)
    llm.add_argument("--max-concurrent-items", type=int, default=2)
    llm.add_argument(
        "--include-single",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    report = subparsers.add_parser("report", help="Rebuild an LLM report")
    report.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "local":
        result = run_local_baseline(output_dir=args.output_dir)
    elif args.command == "llm":
        result = run_llm_benchmark(
            year=args.year,
            output_dir=args.output_dir,
            key_file=args.key_file,
            sample_size=args.sample_size,
            max_budget_usd=args.budget_usd,
            max_concurrent_items=args.max_concurrent_items,
            include_single=args.include_single,
        )
    else:
        result = build_llm_report(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
