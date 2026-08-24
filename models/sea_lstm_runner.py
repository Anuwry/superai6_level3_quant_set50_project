from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT_EARLY = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "mpl"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from models.baseline_common import (
    CLOSE_COLUMN,
    DATE_COLUMN,
    PROJECT_ROOT,
    binary_direction_metrics,
    package_versions,
    regression_metrics,
)
from models.pit_cdr_lstm import direction_metrics
from models.pit_cdr_lstm_runner import PreparedFold, prepare_candidate_fold
from models.sea_lstm import (
    PROTOCOL_ID,
    SEAConfig,
    build_direction_model,
    compile_direction_model,
    verify_freeze_manifest,
)
from models.track_a_final import FINAL_SEEDS

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sea_lstm_direct_2024_2025_v1"
FREEZE_FILE = PROJECT_ROOT / "test" / "sea_lstm_direct_freeze_v1.json"
FROZEN_PREDICTIONS = (
    PROJECT_ROOT
    / "outputs"
    / "final_five_model_prediction_visuals_v1"
    / "final_arm_prediction_series.csv"
)
TEST_YEARS = (2024, 2025)
SEEDS = tuple(int(seed) for seed in FINAL_SEEDS)
INTERNAL_VARIANTS = (
    "standard_lstm",
    "positive_memory_only",
    "negative_memory_only",
    "sea_lstm",
)
FROZEN_MODEL_KEYS = (
    "lstm",
    "cnn",
    "lstm_cnn",
    "lstm_attention",
    "lstm_cnn_attention",
)
MODEL_LABELS = {
    "lstm": "LSTM",
    "cnn": "CNN",
    "lstm_cnn": "LSTM-CNN",
    "lstm_attention": "LSTM-Attention",
    "lstm_cnn_attention": "LSTM-CNN-Attention",
}
EXPECTED_TEST_ROWS = {2024: 244, 2025: 234}
WINDOW = 5
EPOCHS = 20
BATCH_SIZE = 32


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_tensorflow(seed: int):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    return tf


def fit_variant(
    prepared: PreparedFold,
    *,
    variant: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], Any]:
    if variant not in INTERNAL_VARIANTS or seed not in SEEDS:
        raise ValueError("Unknown SEA-LSTM variant or seed")
    tf = _configure_tensorflow(seed)
    config = SEAConfig(window=WINDOW, feature_count=len(prepared.feature_columns))
    model = build_direction_model(config, variant=variant)
    compile_direction_model(model, config)
    fit_started = time.perf_counter()
    model.fit(
        prepared.train_sequence,
        prepared.train_labels.astype(np.float32),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
    )
    fit_seconds = float(time.perf_counter() - fit_started)
    inference_started = time.perf_counter()
    probability = model.predict(prepared.test_sequence, verbose=0).reshape(-1)
    inference_seconds = float(time.perf_counter() - inference_started)
    if probability.shape != (len(prepared.test_sequence),):
        raise ValueError("SEA-LSTM inference shape is invalid")
    if not np.isfinite(probability).all():
        raise ValueError("SEA-LSTM inference contains non-finite probabilities")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("SEA-LSTM probabilities are outside [0, 1]")
    diagnostics: dict[str, Any] = {
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "inference_parameters": int(model.count_params()),
    }
    if variant != "standard_lstm":
        extractor = tf.keras.Model(
            model.input,
            model.get_layer("signed_evidence_recurrence").output,
        )
        evidence = extractor.predict(prepared.test_sequence, verbose=0)
        up, down = np.split(np.asarray(evidence, dtype=float), 2, axis=1)
        diagnostics.update(
            {
                "mean_up_evidence": float(up.mean()),
                "mean_down_evidence": float(down.mean()),
                "nonzero_up_share": float(np.mean(up > 0.0)),
                "nonzero_down_share": float(np.mean(down > 0.0)),
            }
        )
    return probability, diagnostics, model


def _normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized[DATE_COLUMN] = pd.to_datetime(
        normalized[DATE_COLUMN], errors="raise"
    ).dt.normalize()
    normalized["test_year"] = normalized["test_year"].astype(int)
    return normalized


def validate_candidate_cohort(
    candidate: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    candidate_required = {
        "variant",
        "seed",
        "test_year",
        DATE_COLUMN,
        CLOSE_COLUMN,
        "y_true",
        "probability",
    }
    frozen_required = {
        "model",
        "test_year",
        DATE_COLUMN,
        CLOSE_COLUMN,
        "y_true",
        "y_pred",
    }
    if not candidate_required.issubset(candidate.columns):
        raise ValueError("Candidate predictions are missing cohort columns")
    if not frozen_required.issubset(frozen.columns):
        raise ValueError("Frozen predictions are missing cohort columns")
    candidate_values = _normalize_keys(candidate)
    frozen_values = _normalize_keys(frozen)
    candidate_values = candidate_values.loc[candidate_values["test_year"].isin(TEST_YEARS)]
    frozen_values = frozen_values.loc[frozen_values["test_year"].isin(TEST_YEARS)]
    if set(candidate_values["variant"]) != set(INTERNAL_VARIANTS):
        raise ValueError("Candidate predictions do not contain all registered variants")
    if set(candidate_values["seed"].astype(int)) != {int(seed) for seed in seeds}:
        raise ValueError("Candidate predictions do not contain all requested seeds")
    if set(frozen_values["model"]) != set(FROZEN_MODEL_KEYS):
        raise ValueError("Frozen predictions do not contain the five registered models")
    reference = frozen_values.loc[frozen_values["model"].eq("lstm")].sort_values(
        ["test_year", DATE_COLUMN]
    )
    reference_keys = reference[["test_year", DATE_COLUMN]].reset_index(drop=True)
    for (variant, seed), group in candidate_values.groupby(["variant", "seed"], sort=False):
        ordered = group.sort_values(["test_year", DATE_COLUMN]).reset_index(drop=True)
        if not ordered[["test_year", DATE_COLUMN]].equals(reference_keys):
            raise ValueError(f"Candidate dates do not match frozen cohort: {variant}/{seed}")
        if not np.allclose(ordered[CLOSE_COLUMN], reference[CLOSE_COLUMN], atol=1e-12, rtol=0.0):
            raise ValueError("Candidate current-close values do not match frozen cohort")
        if not np.allclose(ordered["y_true"], reference["y_true"], atol=1e-12, rtol=0.0):
            raise ValueError("Candidate actual values do not match frozen cohort")
    return {
        "passed": True,
        "rows_per_variant_seed": len(reference),
        "candidate_cells": int(candidate_values.groupby(["variant", "seed"]).ngroups),
        "frozen_models": len(FROZEN_MODEL_KEYS),
    }


def _candidate_seed_average(
    candidate: pd.DataFrame,
    *,
    seeds: Sequence[int],
) -> pd.DataFrame:
    values = _normalize_keys(candidate)
    keys = ["variant", "test_year", DATE_COLUMN]
    for column in (CLOSE_COLUMN, "y_true"):
        if values.groupby(keys)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Candidate seeds disagree on {column}")
    averaged = values.groupby(keys, as_index=False, sort=False).agg(
        Close_D=(CLOSE_COLUMN, "first"),
        y_true=("y_true", "first"),
        probability=("probability", "mean"),
        seeds_averaged=("seed", "nunique"),
    )
    if not averaged["seeds_averaged"].eq(len(seeds)).all():
        raise ValueError("Candidate seed averaging is incomplete")
    return averaged


def build_ablation_fold_metrics(
    candidate: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    averaged = _candidate_seed_average(candidate, seeds=seeds)
    rows: list[dict[str, Any]] = []
    for (variant, year), group in averaged.groupby(["variant", "test_year"], sort=False):
        metrics = direction_metrics(
            current_close=group[CLOSE_COLUMN],
            next_close=group["y_true"],
            probability=group["probability"],
        )
        labels = (group["y_true"].to_numpy() > group[CLOSE_COLUMN].to_numpy()).astype(int)
        probabilities = group["probability"].to_numpy(dtype=float)
        rows.append(
            {
                "variant": str(variant),
                "test_year": int(year),
                **metrics,
                "brier_score": float(brier_score_loss(labels, probabilities)),
                "binary_crossentropy": float(
                    log_loss(labels, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "test_year"]).reset_index(drop=True)


def build_ablation_table(
    candidate: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    fold_metrics = build_ablation_fold_metrics(candidate, seeds=seeds)
    summary = fold_metrics.groupby("variant", as_index=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        mcc_mean=("mcc", "mean"),
        brier_score_mean=("brier_score", "mean"),
        binary_crossentropy_mean=("binary_crossentropy", "mean"),
        predicted_up_share_mean=("predicted_up_share", "mean"),
        years=("test_year", "nunique"),
    )
    return summary.sort_values("balanced_accuracy_mean", ascending=False).reset_index(drop=True)


def _frozen_fold_metrics(frozen: pd.DataFrame) -> pd.DataFrame:
    values = _normalize_keys(frozen)
    values = values.loc[values["test_year"].isin(TEST_YEARS)]
    rows: list[dict[str, Any]] = []
    for (model, year), group in values.groupby(["model", "test_year"], sort=False):
        direction = binary_direction_metrics(
            group["y_true"].to_numpy(dtype=float),
            group["y_pred"].to_numpy(dtype=float),
            group[CLOSE_COLUMN].to_numpy(dtype=float),
        )
        level = regression_metrics(
            group["y_true"].to_numpy(dtype=float),
            group["y_pred"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "model_key": str(model),
                "model": MODEL_LABELS[str(model)],
                "test_year": int(year),
                "balanced_accuracy": direction["balanced_accuracy"],
                "direction_accuracy": direction["direction_accuracy"],
                "mcc": direction["mcc"],
                "rmse": level["rmse"],
                "mae": level["mae"],
            }
        )
    return pd.DataFrame(rows)


def build_six_model_fold_metrics(
    candidate: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    validate_candidate_cohort(candidate, frozen, seeds=seeds)
    averaged = _candidate_seed_average(candidate, seeds=seeds)
    sea = averaged.loc[averaged["variant"].eq("sea_lstm")]
    rows = _frozen_fold_metrics(frozen).to_dict(orient="records")
    for year, group in sea.groupby("test_year", sort=False):
        metrics = direction_metrics(
            current_close=group[CLOSE_COLUMN],
            next_close=group["y_true"],
            probability=group["probability"],
        )
        rows.append(
            {
                "model_key": "sea_lstm",
                "model": "SEA-LSTM",
                "test_year": int(year),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "direction_accuracy": metrics["direction_accuracy"],
                "mcc": metrics["mcc"],
                "rmse": np.nan,
                "mae": np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "test_year"]).reset_index(drop=True)


def build_six_model_table(
    candidate: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    seeds: Sequence[int] = SEEDS,
) -> pd.DataFrame:
    folds = build_six_model_fold_metrics(candidate, frozen, seeds=seeds)
    table = folds.groupby(["model_key", "model"], as_index=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        direction_accuracy_std=("direction_accuracy", "std"),
        mcc_mean=("mcc", "mean"),
        rmse_mean=("rmse", "mean"),
        mae_mean=("mae", "mean"),
        test_years=("test_year", "nunique"),
    )
    table = table.sort_values("balanced_accuracy_mean", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def promotion_decision(
    ablation: pd.DataFrame,
    annual: pd.DataFrame,
    frozen_models: pd.DataFrame,
) -> dict[str, Any]:
    scores = ablation.set_index("variant")["balanced_accuracy_mean"]
    if set(scores.index) != set(INTERNAL_VARIANTS):
        raise ValueError("Ablation table does not contain every registered variant")
    sea_score = float(scores["sea_lstm"])
    controls = tuple(value for value in INTERNAL_VARIANTS if value != "sea_lstm")
    conditions = {
        f"beats_{control}_mean": sea_score > float(scores[control])
        for control in controls
    }
    annual_scores = annual.set_index(["variant", "test_year"])["balanced_accuracy"]
    for year in TEST_YEARS:
        conditions[f"beats_standard_lstm_in_{year}"] = float(
            annual_scores.loc[("sea_lstm", year)]
        ) > float(annual_scores.loc[("standard_lstm", year)])
    best_frozen = float(frozen_models["balanced_accuracy_mean"].max())
    conditions["beats_best_frozen_model_mean"] = sea_score > best_frozen
    return {
        "promoted": bool(all(conditions.values())),
        "conditions": {key: bool(value) for key, value in conditions.items()},
        "sea_lstm_balanced_accuracy_mean": sea_score,
        "best_frozen_balanced_accuracy_mean": best_frozen,
        "failure_action": "close_without_tuning_or_second_run",
    }


def _cell_directory(output_dir: Path, year: int, seed: int, variant: str) -> Path:
    return output_dir / "cells" / str(year) / f"seed_{seed}" / variant


def _cell_complete(directory: Path) -> bool:
    required = (
        directory / "predictions.csv",
        directory / "metrics.json",
        directory / "run_metadata.json",
        directory / "inference.weights.h5",
    )
    return all(path.is_file() for path in required)


def run_cell(
    prepared: PreparedFold,
    *,
    seed: int,
    variant: str,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    directory = _cell_directory(output_dir, prepared.test_year, seed, variant)
    if _cell_complete(directory) and not force:
        return {
            "status": "skipped_complete",
            "year": prepared.test_year,
            "seed": seed,
            "variant": variant,
        }
    started = time.perf_counter()
    probability, diagnostics, model = fit_variant(prepared, variant=variant, seed=seed)
    metrics = direction_metrics(
        current_close=prepared.test_close,
        next_close=prepared.test_next_close,
        probability=probability,
    )
    predictions = pd.DataFrame(
        {
            "protocol_id": PROTOCOL_ID,
            "variant": variant,
            "seed": seed,
            "fold": prepared.fold,
            "test_year": prepared.test_year,
            DATE_COLUMN: prepared.test_dates,
            "routing_regime": prepared.test_regimes,
            CLOSE_COLUMN: prepared.test_close,
            "y_true": prepared.test_next_close,
            "probability": probability,
            "pred_direction": (probability > 0.5).astype(np.int8),
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(directory / "predictions.csv", index=False)
    (directory / "metrics.json").write_text(
        json.dumps({**metrics, **diagnostics}, indent=2), encoding="utf-8"
    )
    model.save_weights(directory / "inference.weights.h5")
    config = SEAConfig(window=WINDOW, feature_count=len(prepared.feature_columns))
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "evidence_status": "frozen_retrospective_architecture_screen",
        "created_at": _utc_now(),
        "variant": variant,
        "seed": seed,
        "fold": prepared.fold,
        "test_year": prepared.test_year,
        "train_period": [
            pd.Timestamp(prepared.train_dates.min()).date().isoformat(),
            pd.Timestamp(prepared.train_dates.max()).date().isoformat(),
        ],
        "test_period": [
            pd.Timestamp(prepared.test_dates.min()).date().isoformat(),
            pd.Timestamp(prepared.test_dates.max()).date().isoformat(),
        ],
        "configuration": asdict(config),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "feature_count": len(prepared.feature_columns),
        "mask_counts": dict(prepared.mask_counts),
        "training_rows": len(prepared.train_sequence),
        "test_rows": len(prepared.test_sequence),
        "inference_parameters": int(model.count_params()),
        "runtime_scope": "model build, fit, inference, diagnostics, and artifact write",
        "wall_seconds": float(time.perf_counter() - started),
        "packages": package_versions(["numpy", "pandas", "scikit-learn", "tensorflow"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if not _cell_complete(directory):
        raise RuntimeError("SEA-LSTM cell failed post-write completeness check")
    del model
    gc.collect()
    return {"status": "completed", **metadata}


def _collect_predictions(output_dir: Path) -> pd.DataFrame:
    paths = sorted((output_dir / "cells").glob("*/seed_*/*/predictions.csv"))
    expected = len(TEST_YEARS) * len(SEEDS) * len(INTERNAL_VARIANTS)
    if len(paths) != expected:
        raise ValueError(f"SEA-LSTM cells are incomplete: {len(paths)} of {expected}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _runtime_summary(output_dir: Path) -> pd.DataFrame:
    paths = sorted((output_dir / "cells").glob("*/seed_*/*/run_metadata.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    frame = pd.DataFrame(rows)
    return frame.groupby("variant", as_index=False).agg(
        cells=("wall_seconds", "size"),
        wall_seconds_total=("wall_seconds", "sum"),
        wall_seconds_mean=("wall_seconds", "mean"),
        wall_seconds_std=("wall_seconds", "std"),
        inference_parameters=("inference_parameters", "first"),
    )


def aggregate_experiment(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    candidate = _collect_predictions(output_dir)
    frozen = pd.read_csv(FROZEN_PREDICTIONS)
    cohort = validate_candidate_cohort(candidate, frozen, seeds=SEEDS)
    reference = _normalize_keys(frozen)
    reference = reference.loc[
        reference["model"].eq("lstm") & reference["test_year"].isin(TEST_YEARS)
    ]
    rows_by_year = {int(year): len(group) for year, group in reference.groupby("test_year")}
    if rows_by_year != EXPECTED_TEST_ROWS:
        raise ValueError(f"Frozen cohort row counts changed: {rows_by_year}")
    direction_rows = int(
        np.count_nonzero(
            reference["y_true"].to_numpy(dtype=float)
            - reference[CLOSE_COLUMN].to_numpy(dtype=float)
        )
    )
    if direction_rows != sum(EXPECTED_TEST_ROWS.values()):
        raise ValueError("Frozen cohort unexpectedly contains actual direction ties")
    ablation_by_year = build_ablation_fold_metrics(candidate, seeds=SEEDS)
    ablation = build_ablation_table(candidate, seeds=SEEDS)
    six_by_year = build_six_model_fold_metrics(candidate, frozen, seeds=SEEDS)
    six_model = build_six_model_table(candidate, frozen, seeds=SEEDS)
    frozen_summary = six_model.loc[six_model["model_key"].isin(FROZEN_MODEL_KEYS)]
    decision = promotion_decision(ablation, ablation_by_year, frozen_summary)
    runtime = _runtime_summary(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output_dir / "all_seed_predictions.csv", index=False)
    _candidate_seed_average(candidate, seeds=SEEDS).to_csv(
        output_dir / "predictions_seed_averaged.csv", index=False
    )
    ablation_by_year.to_csv(output_dir / "ablation_by_year_2024_2025.csv", index=False)
    ablation.to_csv(output_dir / "ablation_summary_2024_2025.csv", index=False)
    six_by_year.to_csv(output_dir / "six_model_by_year_2024_2025.csv", index=False)
    six_model.to_csv(output_dir / "six_model_comparison_2024_2025.csv", index=False)
    runtime.to_csv(output_dir / "runtime_summary.csv", index=False)
    (output_dir / "promotion_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    integrity = {
        "passed": True,
        "protocol_id": PROTOCOL_ID,
        "cohort": cohort,
        "expected_test_rows": EXPECTED_TEST_ROWS,
        "observed_direction_evaluable_rows": direction_rows,
        "candidate_cells": len(TEST_YEARS) * len(SEEDS) * len(INTERNAL_VARIANTS),
        "variants": list(INTERNAL_VARIANTS),
        "seeds": list(SEEDS),
        "test_years": list(TEST_YEARS),
        "promoted": decision["promoted"],
    }
    (output_dir / "integrity_audit.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    return {
        "integrity": integrity,
        "decision": decision,
        "ablation_by_year": ablation_by_year,
        "ablation": ablation,
        "six_model_by_year": six_by_year,
        "six_model": six_model,
        "runtime": runtime,
    }


def run_experiment(
    *,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
    variants: Iterable[str] = INTERNAL_VARIANTS,
) -> dict[str, Any]:
    requested = tuple(str(value) for value in variants)
    if set(requested) != set(INTERNAL_VARIANTS):
        raise ValueError("A complete registered run requires every SEA-LSTM variant")
    freeze_audit = verify_freeze_manifest(PROJECT_ROOT, FREEZE_FILE)
    started = time.perf_counter()
    for year in TEST_YEARS:
        prepared = prepare_candidate_fold(year)
        for seed in SEEDS:
            for variant in INTERNAL_VARIANTS:
                print(f"SEA-LSTM: year={year}, seed={seed}, variant={variant}", flush=True)
                run_cell(
                    prepared,
                    seed=seed,
                    variant=variant,
                    output_dir=output_dir,
                    force=force,
                )
    result = aggregate_experiment(output_dir)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "created_at": _utc_now(),
        "freeze_audit": freeze_audit,
        "code_sha256": {
            "models/sea_lstm.py": _sha256(PROJECT_ROOT / "models" / "sea_lstm.py"),
            "models/sea_lstm_runner.py": _sha256(
                PROJECT_ROOT / "models" / "sea_lstm_runner.py"
            ),
        },
        "total_wall_seconds": float(time.perf_counter() - started),
        "tensorflow_devices": [],
        "incremental_api_cost_usd": 0,
    }
    try:
        import tensorflow as tf

        metadata["tensorflow_devices"] = [
            device.name for device in tf.config.list_physical_devices()
        ]
    except (ImportError, RuntimeError):
        pass
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen direct SEA-LSTM evaluation")
    parser.add_argument("command", choices=("run", "aggregate"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run_experiment(output_dir=args.output_dir, force=args.force)
    else:
        result = aggregate_experiment(args.output_dir)
    print(result["six_model"].to_string(index=False), flush=True)
    print(json.dumps(result["decision"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
