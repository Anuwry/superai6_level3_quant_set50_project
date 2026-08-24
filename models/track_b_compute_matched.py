from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from models.track_b_analysis import paired_llm_comparison
from models.track_b_baseline import classification_metrics
from models.track_c_inference import holm_adjust

PROTOCOL_ID = "track-b-llm-compute-matched-v1"
MODEL_ID = "gpt-5.6-terra"
NEW_REPLICATES = (2, 3, 4)
PRIMARY_REPLICATES = (1, 2, 3)
SECONDARY_REPLICATES = (1, 2, 3, 4)
LABEL_ORDER = ("negative", "neutral", "positive")
METRIC_LABEL_ORDER = LABEL_ORDER
CONTROL_ARMS = (
    ("self_consistency_3", "leader_minus_self_consistency_3"),
    ("self_consistency_4", "leader_minus_self_consistency_4"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze_manifest(
    project_root: Path,
    manifest_path: Path,
) -> dict[str, object]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_any_repeated_single_control_call",
        "new_results_seen_before_freeze": False,
        "model": MODEL_ID,
        "pairs": 1333,
        "unique_articles": 738,
        "new_replicates": list(NEW_REPLICATES),
        "primary_metric": "accuracy",
        "incremental_cost_guard_usd": 18.0,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"Frozen protocol field changed: {key}")
    checks: list[dict[str, object]] = []
    for item in payload.get("frozen_inputs", []):
        relative = Path(str(item["path"]))
        path = project_root / relative
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_hash = sha256_file(path) if exists else None
        matches = (
            exists
            and actual_size == int(item["bytes"])
            and actual_hash == str(item["sha256"])
        )
        checks.append(
            {
                "path": relative.as_posix(),
                "exists": exists,
                "bytes": actual_size,
                "sha256": actual_hash,
                "matches": bool(matches),
            }
        )
    mismatches = [str(row["path"]) for row in checks if not row["matches"]]
    if mismatches:
        raise ValueError(f"Frozen input mismatch: {mismatches[:5]}")
    return {
        "protocol_id": PROTOCOL_ID,
        "all_inputs_match": True,
        "checked_inputs": checks,
    }


def _probabilities(verdict: Mapping[str, object]) -> np.ndarray:
    values = np.asarray(
        [float(verdict[f"{label}_probability"]) for label in LABEL_ORDER],
        dtype=float,
    )
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Verdict probabilities must be three finite values")
    if np.any(values < 0.0):
        raise ValueError("Verdict probabilities cannot be negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("Verdict probabilities must have positive mass")
    return values / total


def consensus_verdict(
    verdicts: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    values = list(verdicts)
    if not values:
        raise ValueError("Consensus requires at least one verdict")
    mean = np.stack([_probabilities(value) for value in values]).mean(axis=0)
    mean = mean / mean.sum()
    label = LABEL_ORDER[int(np.argmax(mean))]
    negative, neutral, positive = mean.tolist()
    return {
        "relevant": bool(
            np.mean([bool(value.get("relevant", True)) for value in values]) >= 0.5
        ),
        "positive_probability": positive,
        "neutral_probability": neutral,
        "negative_probability": negative,
        "predicted_label": label,
        "sentiment_score": positive - negative,
        "confidence": float(mean.max()),
        "rationale": f"Mean-probability consensus across {len(values)} calls.",
        "replicates_averaged": len(values),
    }


def _checkpoint_key(row: Mapping[str, object]) -> tuple[str, int]:
    item_id = str(row.get("item_id", "")).strip()
    role = str(row.get("role", ""))
    if not item_id or not role.startswith("single_rep_"):
        raise ValueError("Repeated-single checkpoint has an invalid key")
    try:
        replicate = int(role.rsplit("_", maxsplit=1)[1])
    except ValueError as error:
        raise ValueError("Repeated-single checkpoint role is invalid") from error
    if replicate not in NEW_REPLICATES:
        raise ValueError(f"Unexpected repeated-single replicate: {replicate}")
    return item_id, replicate


def validate_checkpoint_design(
    item_ids: Iterable[str],
    checkpoints: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    item_values = tuple(str(item_id) for item_id in item_ids)
    if not item_values or len(set(item_values)) != len(item_values):
        raise ValueError("Registered item IDs must be non-empty and unique")
    expected = {
        (item_id, replicate)
        for item_id in item_values
        for replicate in NEW_REPLICATES
    }
    rows = list(checkpoints)
    keys = [_checkpoint_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Repeated-single checkpoints contain duplicate keys")
    if any(row.get("status") != "completed" for row in rows):
        raise ValueError("Repeated-single checkpoints contain incomplete rows")
    observed = set(keys)
    if observed != expected:
        raise ValueError(
            "Repeated-single checkpoint design is incomplete or contains extras: "
            f"missing={sorted(expected - observed)[:3]}, "
            f"extra={sorted(observed - expected)[:3]}"
        )
    return {
        "expected_calls": len(expected),
        "observed_calls": len(observed),
        "unique_items": len(item_values),
        "replicates": list(NEW_REPLICATES),
    }


def assemble_control_records(
    existing_records: Iterable[Mapping[str, object]],
    repeated_checkpoints: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    existing = [dict(row) for row in existing_records]
    item_ids = [str(row["item_id"]) for row in existing]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Existing records contain duplicate item IDs")
    repeats = [dict(row) for row in repeated_checkpoints]
    validate_checkpoint_design(item_ids, repeats)
    by_key = {_checkpoint_key(row): row for row in repeats}
    result: list[dict[str, object]] = []
    for row in existing:
        item_id = str(row["item_id"])
        if row.get("single") is None or row.get("leader") is None:
            raise ValueError("Existing record is missing single or Leader output")
        first = row["single"]
        if not isinstance(first, Mapping) or not isinstance(
            first.get("parsed"), Mapping
        ):
            raise TypeError("Existing single output has an invalid schema")
        verdicts = {
            1: first["parsed"],
            **{
                replicate: by_key[(item_id, replicate)]["parsed"]
                for replicate in NEW_REPLICATES
            },
        }
        result.append(
            {
                **row,
                "self_consistency_3": {
                    "parsed": consensus_verdict(
                        verdicts[index] for index in PRIMARY_REPLICATES
                    )
                },
                "self_consistency_4": {
                    "parsed": consensus_verdict(
                        verdicts[index] for index in SECONDARY_REPLICATES
                    )
                },
            }
        )
    return result


def _arm_metrics(
    records: list[dict[str, object]],
    arm: str,
) -> dict[str, object]:
    expected = [str(row["gold_label"]) for row in records]
    predicted = [str(row[arm]["parsed"]["predicted_label"]) for row in records]
    probabilities = np.asarray(
        [
            [float(row[arm]["parsed"][f"{label}_probability"]) for label in METRIC_LABEL_ORDER]
            for row in records
        ],
        dtype=float,
    )
    probabilities = np.clip(probabilities, 1e-12, None)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    one_hot = np.eye(len(METRIC_LABEL_ORDER))[
        [METRIC_LABEL_ORDER.index(label) for label in expected]
    ]
    base = classification_metrics(
        expected,
        predicted,
        labels=METRIC_LABEL_ORDER,
    )
    return {
        "arm": arm,
        "pairs": len(records),
        "accuracy": base["accuracy"],
        "macro_f1": base["macro_f1"],
        "weighted_f1": base["weighted_f1"],
        "mcc": base["mcc"],
        "log_loss": float(
            log_loss(expected, probabilities, labels=list(METRIC_LABEL_ORDER))
        ),
        "multiclass_brier_score": float(
            np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        ),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }


def metrics_by_arm(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        raise ValueError("At least one assembled control record is required")
    arms = ("single", "self_consistency_3", "self_consistency_4", "leader")
    return pd.DataFrame([_arm_metrics(records, arm) for arm in arms])


def paired_control_comparisons(
    records: list[dict[str, object]],
    *,
    bootstrap_iterations: int = 5_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for control, comparison_id in CONTROL_ARMS:
        comparison_records = [
            {
                "item_id": row["item_id"],
                "gold_label": row["gold_label"],
                "single": row[control],
                "leader": row["leader"],
            }
            for row in records
        ]
        raw = paired_llm_comparison(
            comparison_records,
            bootstrap_iterations=bootstrap_iterations,
            random_seed=random_seed,
        )
        rows.append(
            {
                "comparison_id": comparison_id,
                "control_arm": control,
                "treatment_arm": "leader",
                **{
                    key.replace("single_", "control_"): value
                    for key, value in raw.items()
                },
            }
        )
    return pd.DataFrame(rows)


def apply_accuracy_holm(comparisons: pd.DataFrame) -> pd.DataFrame:
    required = {"comparison_id", "cluster_sign_flip_pvalue"}
    missing = sorted(required.difference(comparisons.columns))
    if missing:
        raise ValueError(f"Control comparisons are missing columns: {missing}")
    if set(comparisons["comparison_id"]) != {
        comparison_id for _, comparison_id in CONTROL_ARMS
    }:
        raise ValueError("The registered two-control comparison family changed")
    result = comparisons.copy()
    result["holm_adjusted_pvalue"] = holm_adjust(
        result["cluster_sign_flip_pvalue"].to_numpy(dtype=float)
    )
    result["models_in_family"] = len(result)
    return result
