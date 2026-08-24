from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_common import PROJECT_ROOT
from models.set100_robustness import (
    FOLDS,
    MODELS,
    PROTOCOL_ID,
    SEEDS,
    TEST_YEARS,
    WINDOWS,
    apply_market_holm,
    average_seed_predictions,
    build_market_fold_deltas,
    evaluate_robustness_predictions,
    fold_level_market_inference,
    sha256_file,
    validate_registered_design,
    verify_freeze_manifest,
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "set100_same_exchange_robustness_v1"
)
FREEZE_MANIFEST = (
    PROJECT_ROOT / "test" / "set100_same_exchange_robustness_freeze_v1.json"
)
SET50_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_a_final_point_in_time_v2"
SUMMARY_METRICS = (
    "balanced_accuracy",
    "direction_accuracy",
    "mcc",
    "roc_auc",
    "direction_coverage",
    "mape",
    "nrmse_percent",
    "nmae_percent",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def summarize_market_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"market", "model", "fold", *SUMMARY_METRICS}
    missing = sorted(required.difference(fold_metrics.columns))
    if missing:
        raise ValueError(f"Fold metrics are missing summary columns: {missing}")
    rows: list[dict[str, object]] = []
    for (market, model), group in fold_metrics.groupby(
        ["market", "model"], sort=False
    ):
        if len(group) != len(FOLDS) or set(group["fold"]) != set(FOLDS):
            raise ValueError(f"{market}/{model} does not contain four folds")
        row: dict[str, object] = {
            "market": market,
            "model": model,
            "sequence_window": WINDOWS[str(model)],
            "outer_folds": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = group[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite summary metric: {market}/{model}/{metric}")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    result = pd.DataFrame(rows)
    expected = {(market, model) for market in ("SET50", "SET100") for model in MODELS}
    observed = set(zip(result["market"], result["model"], strict=True))
    if observed != expected:
        raise ValueError("Market summary does not contain the registered 10 rows")
    return result


def build_paper_table(
    summary: pd.DataFrame,
    inference_holm: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        market_rows = summary.loc[summary["model"].eq(model)].set_index("market")
        if set(market_rows.index) != {"SET50", "SET100"}:
            raise ValueError(f"Paper table market rows are incomplete for {model}")
        bacc = inference_holm.loc[
            inference_holm["model"].eq(model)
            & inference_holm["metric"].eq("balanced_accuracy_delta_pp")
        ]
        if len(bacc) != 1:
            raise ValueError(f"Paper table inference is incomplete for {model}")
        inference = bacc.iloc[0]
        row: dict[str, object] = {
            "model": model,
            "sequence_window": WINDOWS[model],
            "balanced_accuracy_set50_mean": market_rows.loc[
                "SET50", "balanced_accuracy_mean"
            ],
            "balanced_accuracy_set50_std": market_rows.loc[
                "SET50", "balanced_accuracy_std"
            ],
            "balanced_accuracy_set100_mean": market_rows.loc[
                "SET100", "balanced_accuracy_mean"
            ],
            "balanced_accuracy_set100_std": market_rows.loc[
                "SET100", "balanced_accuracy_std"
            ],
            "balanced_accuracy_delta_pp": inference["mean_delta"],
            "balanced_accuracy_delta_ci95_lower": inference["ci95_lower"],
            "balanced_accuracy_delta_ci95_upper": inference["ci95_upper"],
            "balanced_accuracy_sign_flip_pvalue": inference[
                "exact_sign_flip_pvalue"
            ],
            "balanced_accuracy_holm_pvalue": inference[
                "holm_adjusted_pvalue"
            ],
        }
        for metric in ("direction_accuracy", "mcc", "roc_auc", "nrmse_percent"):
            row[f"{metric}_set50_mean"] = market_rows.loc[
                "SET50", f"{metric}_mean"
            ]
            row[f"{metric}_set100_mean"] = market_rows.loc[
                "SET100", f"{metric}_mean"
            ]
        rows.append(row)
    return pd.DataFrame(rows)


def _set100_cell_dir(
    output_dir: Path,
    model: str,
    fold: str,
    seed: int,
) -> Path:
    return output_dir / "private" / "cells" / model / fold / f"seed_{seed}"


def _set50_prediction_path(model: str, fold: str, seed: int) -> Path:
    return (
        SET50_OUTPUT_DIR
        / "outer_test"
        / model
        / "full_ta_vmd"
        / f"window_{WINDOWS[model]}"
        / f"seed_{seed}"
        / f"predictions_{fold}.csv"
    )


def _collect_set100_cell_metrics(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for fold in FOLDS:
            for seed in SEEDS:
                path = _set100_cell_dir(output_dir, model, fold, seed) / "metrics.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing registered cell metric: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("protocol_id") != PROTOCOL_ID:
                    raise ValueError(f"Protocol mismatch in {path}")
                rows.append(payload)
    frame = pd.DataFrame(rows)
    validate_registered_design(frame)
    if set(frame["market"]) != {"SET100"}:
        raise ValueError("SET100 cell metrics contain another market")
    return frame.sort_values(["model", "fold", "seed"]).reset_index(drop=True)


def _load_set50_cell_metrics() -> pd.DataFrame:
    path = SET50_OUTPUT_DIR / "final_metrics_by_seed_fold.csv"
    frame = pd.read_csv(path)
    selected = frame.loc[
        frame["feature_set"].eq("full_ta_vmd")
        & frame["model"].isin(MODELS)
    ].copy()
    selected = selected.loc[
        selected.apply(
            lambda row: int(row["sequence_window"]) == WINDOWS[str(row["model"])],
            axis=1,
        )
    ]
    validate_registered_design(selected)
    selected.insert(0, "market", "SET50")
    selected.insert(0, "protocol_id", "track-a-final-point-in-time-v2")
    return selected.sort_values(["model", "fold", "seed"]).reset_index(drop=True)


def _prediction_frames(
    output_dir: Path,
    market: str,
    model: str,
    fold: str,
) -> dict[int, pd.DataFrame]:
    result: dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        if market == "SET100":
            path = _set100_cell_dir(output_dir, model, fold, seed) / "predictions.csv"
        elif market == "SET50":
            path = _set50_prediction_path(model, fold, seed)
        else:
            raise ValueError(f"Unknown market: {market}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing seed prediction: {path}")
        result[seed] = pd.read_csv(path)
    return result


def _seed_averaged_fold_metrics(output_dir: Path) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, object]] = []
    date_checks = 0
    private_dir = output_dir / "private" / "seed_averaged"
    for model in MODELS:
        for fold, test_year in zip(FOLDS, TEST_YEARS, strict=True):
            averaged: dict[str, pd.DataFrame] = {}
            for market in ("SET50", "SET100"):
                frame = average_seed_predictions(
                    _prediction_frames(output_dir, market, model, fold)
                )
                averaged[market] = frame
                destination = private_dir / market.lower() / model
                destination.mkdir(parents=True, exist_ok=True)
                frame.to_csv(destination / f"{fold}.csv", index=False)
                metrics = evaluate_robustness_predictions(
                    frame["y_true"].to_numpy(dtype=float),
                    frame["y_pred"].to_numpy(dtype=float),
                    frame["Close_D"].to_numpy(dtype=float),
                )
                rows.append(
                    {
                        "protocol_id": PROTOCOL_ID,
                        "market": market,
                        "model": model,
                        "fold": fold,
                        "test_year": test_year,
                        "sequence_window": WINDOWS[model],
                        "seeds_averaged": len(SEEDS),
                        "n_test": len(frame),
                        "first_date": frame["Date"].min().strftime("%Y-%m-%d"),
                        "last_date": frame["Date"].max().strftime("%Y-%m-%d"),
                        **metrics,
                    }
                )
            set50 = averaged["SET50"].reset_index(drop=True)
            set100 = averaged["SET100"].reset_index(drop=True)
            if not set50["Date"].equals(set100["Date"]):
                raise ValueError(f"Cross-index dates differ for {model}/{fold}")
            date_checks += 1
    return pd.DataFrame(rows), date_checks


def _runtime_summary(
    set50_cells: pd.DataFrame,
    set100_cells: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for market, frame, runtime_column in (
        ("SET50", set50_cells, "runtime_seconds"),
        ("SET100", set100_cells, "total_model_seconds"),
    ):
        for model, group in frame.groupby("model", sort=False):
            runtime = group[runtime_column].to_numpy(dtype=float)
            row: dict[str, object] = {
                "market": market,
                "model": model,
                "cells": len(group),
                "runtime_measurement": runtime_column,
                "total_model_seconds": float(runtime.sum()),
                "mean_model_seconds": float(runtime.mean()),
                "std_model_seconds": float(runtime.std(ddof=1)),
                "p95_model_seconds": float(np.percentile(runtime, 95)),
                "mean_fit_seconds": np.nan,
                "mean_inference_seconds": np.nan,
                "trainable_parameters": (
                    int(group["trainable_parameters"].iloc[0])
                    if "trainable_parameters" in group
                    else np.nan
                ),
            }
            if market == "SET100":
                row["mean_fit_seconds"] = float(group["fit_seconds"].mean())
                row["mean_inference_seconds"] = float(
                    group["inference_seconds"].mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _write_manifest(output_dir: Path, public_files: list[Path]) -> Path:
    manifest_path = output_dir / "output_manifest.json"
    payload = {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": _utc_now(),
        "scope": "non-reconstructive aggregate outputs only",
        "private_predictions_excluded": True,
        "files": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(public_files)
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def aggregate_results(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze = verify_freeze_manifest(PROJECT_ROOT, FREEZE_MANIFEST)
    feature_audit_path = output_dir / "feature_integrity_audit.json"
    feature_audit = json.loads(feature_audit_path.read_text(encoding="utf-8"))
    if feature_audit.get("passed") is not True:
        raise ValueError("SET100 feature integrity gate did not pass")

    set100_cells = _collect_set100_cell_metrics(output_dir)
    set50_cells = _load_set50_cell_metrics()
    fold_metrics, paired_date_checks = _seed_averaged_fold_metrics(output_dir)
    if len(fold_metrics) != len(MODELS) * len(FOLDS) * 2:
        raise ValueError("Seed-averaged fold metric cardinality changed")
    summary = summarize_market_metrics(fold_metrics)
    deltas = build_market_fold_deltas(fold_metrics)
    inference = fold_level_market_inference(deltas)
    inference_holm = apply_market_holm(inference)
    runtime = _runtime_summary(set50_cells, set100_cells)
    paper = build_paper_table(summary, inference_holm)
    per_seed = pd.concat([set50_cells, set100_cells], ignore_index=True, sort=False)

    outputs = {
        "per_seed_metrics.csv": per_seed,
        "seed_averaged_fold_metrics.csv": fold_metrics,
        "model_market_summary.csv": summary,
        "market_deltas_by_fold.csv": deltas,
        "market_inference.csv": inference,
        "market_inference_holm.csv": inference_holm,
        "runtime_summary.csv": runtime,
        "paper_table.csv": paper,
    }
    public_files: list[Path] = []
    for name, frame in outputs.items():
        path = output_dir / name
        frame.to_csv(path, index=False)
        public_files.append(path)

    audit = {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": _utc_now(),
        "passed": True,
        "freeze_inputs_match": freeze["all_inputs_match"],
        "feature_integrity_passed": True,
        "registered_set100_cells": validate_registered_design(set100_cells),
        "registered_set50_cells": validate_registered_design(set50_cells),
        "seed_averaged_fold_rows": len(fold_metrics),
        "paired_market_fold_rows": len(deltas),
        "paired_cross_index_date_checks": paired_date_checks,
        "primary_metric": "balanced_accuracy",
        "primary_inference_unit": "seed-averaged outer fold",
        "holm_family": "five models within each metric",
        "set100_news_used": False,
        "set100_raw_rows_in_public_outputs": False,
        "scope_label": "same-exchange robustness; not external-market replication",
    }
    audit_path = output_dir / "integrity_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    public_files.append(audit_path)
    manifest_path = _write_manifest(output_dir, public_files)
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "set100_fit_cells": len(set100_cells),
        "seed_averaged_market_fold_rows": len(fold_metrics),
        "paired_market_fold_rows": len(deltas),
        "public_output_files": len(public_files) + 1,
        "output_manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
    }
