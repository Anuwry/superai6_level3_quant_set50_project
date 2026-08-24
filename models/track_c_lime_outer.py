from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import pandas as pd

from models.track_a_final import FINAL_SEEDS, TRACK_A_MODELS

EXPECTED_OUTER_FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4")


def validate_outer_lime_gate(metadata: Mapping[str, object]) -> None:
    """Fail closed unless the registered five-model outer run is complete."""

    complete = bool(metadata.get("outer_results_generated"))
    models = set(metadata.get("models_completed", ()))
    folds = set(metadata.get("folds_completed", ()))
    seeds = {
        int(seed) for seed in metadata.get("seeds_completed", ())
    }
    if not (
        complete
        and models == set(TRACK_A_MODELS)
        and folds == set(EXPECTED_OUTER_FOLDS)
        and seeds == set(FINAL_SEEDS)
    ):
        raise ValueError(
            "The complete registered outer run is required before LIME audit"
        )


def build_audit_instance_id(
    model: str,
    fold: str,
    row_index: int,
    date: str,
) -> str:
    if model not in TRACK_A_MODELS:
        raise ValueError(f"Unknown model: {model}")
    if fold not in EXPECTED_OUTER_FOLDS:
        raise ValueError(f"Unknown fold: {fold}")
    if isinstance(row_index, bool) or row_index < 0:
        raise ValueError("row_index must be a non-negative integer")
    material = f"{model}|{fold}|{row_index}|{date}"
    suffix = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return f"{model}__{fold}__{row_index:04d}__{suffix}"


def _median_or_nan(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return float("nan")
    return float(np.nanmedian(frame[column].to_numpy(dtype=float)))


def _quantile_or_nan(
    frame: pd.DataFrame,
    column: str,
    quantile: float,
) -> float:
    if frame.empty:
        return float("nan")
    return float(
        np.nanquantile(
            frame[column].to_numpy(dtype=float),
            quantile,
        )
    )


def boolean_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized).difference({"true", "false"}))
    if unknown:
        raise ValueError(f"Invalid boolean values: {unknown}")
    return normalized.eq("true")


def summarize_agreement(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model",
        "regime",
        "fidelity_r2",
        "low_fidelity",
        "spearman_abs",
        "top_k_jaccard",
        "sign_agreement_nonzero",
        "lime_runtime_seconds",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Agreement rows are missing columns: {missing}")
    summaries: list[dict[str, object]] = []
    group_columns = ["model"]
    if "fold" in rows.columns:
        group_columns.append("fold")
    group_columns.append("regime")
    for group_key, group in rows.groupby(
        group_columns,
        sort=False,
    ):
        key_values = (
            group_key if isinstance(group_key, tuple) else (group_key,)
        )
        reliable = group.loc[~boolean_mask(group["low_fidelity"])]
        summary = {
                **dict(zip(group_columns, key_values, strict=True)),
                "audit_repeats": len(group),
                "reliable_repeats": len(reliable),
                "low_fidelity_repeats": int(len(group) - len(reliable)),
                "low_fidelity_fraction": float(
                    (len(group) - len(reliable)) / len(group)
                ),
                "fidelity_r2_median_all": _median_or_nan(
                    group,
                    "fidelity_r2",
                ),
                "fidelity_r2_q1_all": _quantile_or_nan(
                    group,
                    "fidelity_r2",
                    0.25,
                ),
                "fidelity_r2_q3_all": _quantile_or_nan(
                    group,
                    "fidelity_r2",
                    0.75,
                ),
                "spearman_abs_median_reliable": _median_or_nan(
                    reliable,
                    "spearman_abs",
                ),
                "spearman_abs_q1_reliable": _quantile_or_nan(
                    reliable,
                    "spearman_abs",
                    0.25,
                ),
                "spearman_abs_q3_reliable": _quantile_or_nan(
                    reliable,
                    "spearman_abs",
                    0.75,
                ),
                "top_k_jaccard_median_reliable": _median_or_nan(
                    reliable,
                    "top_k_jaccard",
                ),
                "top_k_jaccard_q1_reliable": _quantile_or_nan(
                    reliable,
                    "top_k_jaccard",
                    0.25,
                ),
                "top_k_jaccard_q3_reliable": _quantile_or_nan(
                    reliable,
                    "top_k_jaccard",
                    0.75,
                ),
                "sign_agreement_median_reliable": _median_or_nan(
                    reliable,
                    "sign_agreement_nonzero",
                ),
                "lime_runtime_seconds_total": float(
                    group["lime_runtime_seconds"].sum()
                ),
        }
        summaries.append(summary)
    return pd.DataFrame(summaries)
