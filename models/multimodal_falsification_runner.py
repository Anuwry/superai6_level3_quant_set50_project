from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

_PROJECT_ROOT_EARLY = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "numba"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from models.baseline_common import PROJECT_ROOT, discover_folds, package_versions
from models.integrated_multimodal import (
    FitRequest,
    prepare_integrated_fold,
    subset_aligned_regimes,
    verify_freeze_manifest as verify_integrated_freeze_manifest,
)
from models.integrated_multimodal_runner import (
    _load_regimes,
    _prediction_frame,
    _prediction_metrics,
    _window_map,
    fold_metrics_from_seed_averaged_predictions,
)
from models.multimodal_falsification import (
    ANALYSIS_ARMS,
    CONTROL_ARMS,
    CONTROL_CONTRASTS,
    CONTROL_TO_ARM,
    CONTROL_TRANSFORMS,
    NEWS_ONLY_ARM,
    OBSERVED_NEWS_ARM,
    PROTOCOL_ID,
    build_control_fold_contrasts,
    control_feature_sets,
    transform_news_fold,
)
from models.track_a_final import FINAL_SEEDS, TRACK_A_MODELS
from models.track_b_forward_news import DAILY_NEWS_FILE
from models.track_b_fusion import _scaled_fold
from models.track_c_inference import (
    apply_holm_by_family,
    average_seed_predictions,
    fold_level_inference,
    moving_block_bootstrap,
    paired_daily_effects,
)
from models.track_c_outer_runner import _fit_request
from models.vmd_feature_pool import FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "multimodal_falsification_v1"
FREEZE_MANIFEST = PROJECT_ROOT / "test" / "reliability_extension_freeze_v1.json"
REFERENCE_DIR = PROJECT_ROOT / "outputs" / "integrated_multimodal_posthoc_v1"
FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4")
REFERENCE_ARM_MAP = {
    "Global-Numeric": "Market-Only",
    "Global-Numeric-News": OBSERVED_NEWS_ARM,
}
BOOTSTRAP_BLOCK_LENGTH = 10
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_804


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def control_seed(fold: str, control: str) -> int:
    material = f"{PROTOCOL_ID}|{fold}|{control}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _validate_choice(values: Iterable[str], allowed: Iterable[str], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be non-empty and unique")
    unknown = sorted(set(result).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {name}: {unknown}")
    return result


def _validate_seeds(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError("seeds must be non-empty and unique")
    return result


def cell_directory(output_dir: Path, model: str, fold: str, seed: int) -> Path:
    return output_dir / "cells" / model / fold / f"seed_{int(seed)}"


def _cell_paths(directory: Path) -> tuple[Path, ...]:
    return (
        directory / "metrics.csv",
        directory / "fit_registry.csv",
        directory / "run_metadata.json",
        directory / "integrity_audit.json",
        *(directory / f"predictions_{arm}.csv" for arm in (NEWS_ONLY_ARM, *CONTROL_ARMS)),
    )


def validate_control_cell(
    metrics: pd.DataFrame,
    registry: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
) -> dict[str, object]:
    expected_arms = {NEWS_ONLY_ARM, *CONTROL_ARMS}
    if set(metrics["arm"].astype(str)) != expected_arms or metrics["arm"].duplicated().any():
        raise ValueError("Control metrics do not contain four unique arms")
    if len(registry) != 4 or registry["fit_id"].duplicated().any():
        raise ValueError("Control registry does not contain four unique fits")
    if set(registry["arm"].astype(str)) != expected_arms:
        raise ValueError("Control registry arms are invalid")
    if np.any(registry["training_sequences"].to_numpy(dtype=int) < 200):
        raise ValueError("Control fit contains fewer than 200 training sequences")
    if set(predictions) != expected_arms:
        raise ValueError("Control predictions do not contain four unique arms")
    required = {"Date", "routing_regime", "Close_D", "y_true", "y_pred"}
    reference: pd.DataFrame | None = None
    for arm in (NEWS_ONLY_ARM, *CONTROL_ARMS):
        frame = predictions[arm].reset_index(drop=True).copy()
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{arm} predictions are missing columns: {missing}")
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
        numeric = frame[["Close_D", "y_true", "y_pred"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or frame["Date"].duplicated().any():
            raise ValueError(f"{arm} predictions are invalid")
        if reference is None:
            reference = frame
            continue
        if not frame["Date"].equals(reference["Date"]):
            raise ValueError("Control prediction dates differ")
        for column in ("Close_D", "y_true"):
            np.testing.assert_allclose(frame[column], reference[column], rtol=0.0, atol=1e-12)
        if not frame["routing_regime"].astype(str).equals(reference["routing_regime"].astype(str)):
            raise ValueError("Control prediction regimes differ")
    return {
        "passed": True,
        "arms": len(expected_arms),
        "fits": len(registry),
        "test_rows": 0 if reference is None else len(reference),
        "minimum_training_sequences": int(registry["training_sequences"].min()),
    }


def verify_freeze_manifest() -> dict[str, object]:
    payload = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("status") != "frozen_before_control_results":
        raise ValueError("Falsification freeze manifest is invalid")
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Falsification freeze manifest has no inputs")
    for entry in inputs:
        path = PROJECT_ROOT / str(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Frozen input is missing: {path}")
        if path.stat().st_size != int(entry["bytes"]) or _sha256(path) != str(entry["sha256"]):
            raise ValueError(f"Frozen input changed: {path}")
    inherited = verify_integrated_freeze_manifest(
        PROJECT_ROOT,
        PROJECT_ROOT / "test" / "integrated_multimodal_freeze_v1.json",
    )
    return {
        "passed": True,
        "files_checked": len(inputs),
        "inherited_integrated_manifest": inherited,
    }


def cell_complete(output_dir: Path, model: str, fold: str, seed: int) -> bool:
    directory = cell_directory(output_dir, model, fold, seed)
    try:
        if not all(path.is_file() for path in _cell_paths(directory)):
            return False
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("protocol_id") != PROTOCOL_ID:
            return False
        metrics = pd.read_csv(directory / "metrics.csv")
        registry = pd.read_csv(directory / "fit_registry.csv")
        predictions = {
            arm: pd.read_csv(directory / f"predictions_{arm}.csv")
            for arm in (NEWS_ONLY_ARM, *CONTROL_ARMS)
        }
        validate_control_cell(metrics, registry, predictions)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, AssertionError):
        return False
    return True


def build_cell_commands(
    *,
    python_executable: Path,
    output_dir: Path,
    models: Iterable[str] = TRACK_A_MODELS,
    folds: Iterable[str] = FOLDS,
    seeds: Iterable[int] = FINAL_SEEDS,
    force: bool = False,
) -> list[list[str]]:
    model_names = _validate_choice(models, TRACK_A_MODELS, "models")
    fold_names = _validate_choice(folds, FOLDS, "folds")
    seed_values = _validate_seeds(seeds)
    commands: list[list[str]] = []
    for model in model_names:
        for fold in fold_names:
            for seed in seed_values:
                command = [
                    str(python_executable),
                    "-m",
                    "models.multimodal_falsification_runner",
                    "cell",
                    "--model",
                    model,
                    "--fold",
                    fold,
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(output_dir),
                ]
                if force:
                    command.append("--force")
                commands.append(command)
    return commands


def _fit_execution_id(model: str, fold: str, arm: str, seed: int, features: Sequence[str]) -> str:
    material = "|".join([PROTOCOL_ID, model, fold, arm, str(seed), *features])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def run_cell(
    *,
    model: str,
    fold: str,
    seed: int,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict[str, object]:
    _validate_choice((model,), TRACK_A_MODELS, "models")
    _validate_choice((fold,), FOLDS, "folds")
    base_seed = _validate_seeds((seed,))[0]
    if cell_complete(output_dir, model, fold, base_seed) and not force:
        return {"status": "skipped_complete", "model": model, "fold": fold, "seed": base_seed}
    freeze_audit = verify_freeze_manifest()
    specs = {spec.fold: spec for spec in discover_folds(FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR)}
    spec = specs[fold]
    daily_news = pd.read_csv(DAILY_NEWS_FILE)
    prepared = prepare_integrated_fold(spec, daily_news)
    observed_fold = prepared["Global-Numeric-News"]
    feature_sets = control_feature_sets(observed_fold.feature_columns)
    transformed_folds = {NEWS_ONLY_ARM: observed_fold}
    for control in CONTROL_TRANSFORMS:
        transformed_folds[CONTROL_TO_ARM[control]] = transform_news_fold(
            observed_fold,
            control,
            seed=control_seed(fold, control),
        )
    train_regime_frame, test_regime_frame = _load_regimes(fold)
    train_regimes = subset_aligned_regimes(observed_fold.train, train_regime_frame, split="train")
    test_regimes = subset_aligned_regimes(observed_fold.test, test_regime_frame, split="test")
    window = _window_map()[model]
    metrics_rows: list[dict[str, object]] = []
    registry_rows: list[dict[str, object]] = []
    predictions: dict[str, pd.DataFrame] = {}
    started = time.perf_counter()
    for arm in (NEWS_ONLY_ARM, *CONTROL_ARMS):
        raw_fold = transformed_folds[arm]
        scaled_fold, scaler = _scaled_fold(raw_fold)
        features = feature_sets[arm]
        request = FitRequest("global", "global", features, base_seed)
        result = _fit_request(
            model,
            request,
            scaled_fold=scaled_fold,
            train_regimes=train_regimes,
            window=window,
            scaler=scaler,
        )
        predictions[arm] = _prediction_frame(observed_fold, result.prediction, test_regimes)
        registry_rows.append(
            {
                "fit_id": _fit_execution_id(model, fold, arm, base_seed, features),
                "model": model,
                "fold": fold,
                "test_year": spec.test_year,
                "base_seed": base_seed,
                "arm": arm,
                "features": len(features),
                "feature_hash": hashlib.sha256("|".join(features).encode("utf-8")).hexdigest(),
                "training_sequences": result.training_sequences,
                "fit_seconds": result.fit_seconds,
                "inference_seconds": result.inference_seconds,
                "trainable_parameters": result.trainable_parameters,
            }
        )
        metrics_rows.append(
            {
                "model": model,
                "fold": fold,
                "test_year": spec.test_year,
                "base_seed": base_seed,
                "arm": arm,
                "window": window,
                "n_test": len(result.prediction),
                "fit_seconds": result.fit_seconds,
                "inference_seconds": result.inference_seconds,
                **_prediction_metrics(observed_fold, result.prediction),
            }
        )
    metrics = pd.DataFrame(metrics_rows)
    registry = pd.DataFrame(registry_rows)
    audit = validate_control_cell(metrics, registry, predictions)
    audit.update({"protocol_id": PROTOCOL_ID, "input_manifest": freeze_audit})
    directory = cell_directory(output_dir, model, fold, base_seed)
    directory.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(directory / "metrics.csv", index=False)
    registry.to_csv(directory / "fit_registry.csv", index=False)
    for arm, frame in predictions.items():
        frame.to_csv(directory / f"predictions_{arm}.csv", index=False)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "evidence_status": "pre_frozen_retrospective_falsification",
        "created_at_utc": _utc_now(),
        "model": model,
        "fold": fold,
        "test_year": spec.test_year,
        "base_seed": base_seed,
        "window": window,
        "arms": [NEWS_ONLY_ARM, *CONTROL_ARMS],
        "control_seeds": {control: control_seed(fold, control) for control in CONTROL_TRANSFORMS},
        "news_lag_trading_rows": 5,
        "cell_wall_seconds": float(time.perf_counter() - started),
        "packages": package_versions(["numpy", "pandas", "scikit-learn", "scipy", "tensorflow"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    (directory / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (directory / "integrity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not cell_complete(output_dir, model, fold, base_seed):
        raise RuntimeError("Falsification cell failed its post-write audit")
    return {**metadata, "status": "completed"}


def _expected_keys() -> set[tuple[str, str, int]]:
    return {(model, fold, int(seed)) for model in TRACK_A_MODELS for fold in FOLDS for seed in FINAL_SEEDS}


def _collect_control_predictions(output_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model in TRACK_A_MODELS:
        for fold_index, fold in enumerate(FOLDS, start=1):
            for arm in (NEWS_ONLY_ARM, *CONTROL_ARMS):
                frames = {
                    int(seed): pd.read_csv(cell_directory(output_dir, model, fold, seed) / f"predictions_{arm}.csv")
                    for seed in FINAL_SEEDS
                }
                averaged = average_seed_predictions(frames)
                averaged.insert(0, "arm", arm)
                averaged.insert(0, "test_year", 2021 + fold_index)
                averaged.insert(0, "fold", fold)
                averaged.insert(0, "model", model)
                rows.append(averaged)
    return pd.concat(rows, ignore_index=True)


def _load_reference_predictions() -> pd.DataFrame:
    path = REFERENCE_DIR / "predictions_seed_averaged.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["arm"].isin(REFERENCE_ARM_MAP)].copy()
    frame["arm"] = frame["arm"].map(REFERENCE_ARM_MAP)
    if frame.empty or frame["arm"].isna().any():
        raise ValueError("Integrated reference predictions are incomplete")
    return frame


def _daily_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_index, model in enumerate(TRACK_A_MODELS):
        model_frame = predictions.loc[predictions["model"].eq(model)]
        for contrast_index, (contrast, arms) in enumerate(CONTROL_CONTRASTS.items()):
            effects = {"squared_error_loss_delta": [], "balanced_accuracy_delta_pp": []}
            for fold in FOLDS:
                fold_frame = model_frame.loc[model_frame["fold"].eq(fold)]
                treatment = fold_frame.loc[fold_frame["arm"].eq(arms[0])]
                control = fold_frame.loc[fold_frame["arm"].eq(arms[1])]
                squared, bacc = paired_daily_effects(treatment, control)
                effects["squared_error_loss_delta"].append(squared)
                effects["balanced_accuracy_delta_pp"].append(bacc)
            for metric_index, (metric, arrays) in enumerate(effects.items()):
                result = moving_block_bootstrap(
                    arrays,
                    block_length=BOOTSTRAP_BLOCK_LENGTH,
                    replicates=BOOTSTRAP_REPLICATES,
                    seed=BOOTSTRAP_SEED + model_index * 100 + contrast_index * 10 + metric_index,
                )
                rows.append({"model": model, "contrast": contrast, "metric": metric, **result})
    return pd.DataFrame(rows)


def _quarterly_panel(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame["quarter"] = frame["Date"].dt.to_period("Q").astype(str)
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["model", "fold", "test_year", "quarter", "arm"], sort=False):
        metrics = _prediction_metrics_from_frame(group)
        rows.append(dict(zip(("model", "fold", "test_year", "quarter", "arm"), keys, strict=True)) | metrics)
    quarterly = pd.DataFrame(rows)
    contrast_rows: list[pd.DataFrame] = []
    keys = ["model", "fold", "test_year", "quarter"]
    for contrast, (treatment_arm, control_arm) in CONTROL_CONTRASTS.items():
        treatment = quarterly.loc[quarterly["arm"].eq(treatment_arm), [*keys, "balanced_accuracy"]]
        control = quarterly.loc[quarterly["arm"].eq(control_arm), [*keys, "balanced_accuracy"]]
        paired = treatment.merge(control, on=keys, validate="one_to_one", suffixes=("_treatment", "_control"))
        paired["contrast"] = contrast
        paired["balanced_accuracy_delta_pp"] = (paired["balanced_accuracy_treatment"] - paired["balanced_accuracy_control"]) * 100.0
        contrast_rows.append(paired)
    contrasts = pd.concat(contrast_rows, ignore_index=True)
    summary = (
        contrasts.groupby(["model", "contrast"], sort=False)["balanced_accuracy_delta_pp"]
        .agg(origins="size", mean_delta_pp="mean", median_delta_pp="median", positive_origins=lambda values: int(np.sum(values > 0)), negative_origins=lambda values: int(np.sum(values < 0)))
        .reset_index()
    )
    return contrasts, summary


def _prediction_metrics_from_frame(frame: pd.DataFrame) -> dict[str, float | int]:
    from models.baseline_common import binary_direction_metrics

    return binary_direction_metrics(
        frame["y_true"].to_numpy(dtype=float),
        frame["y_pred"].to_numpy(dtype=float),
        frame["Close_D"].to_numpy(dtype=float),
    )


def aggregate_experiment(*, output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    freeze_audit = verify_freeze_manifest()
    incomplete = [key for key in sorted(_expected_keys()) if not cell_complete(output_dir, *key)]
    if incomplete:
        raise ValueError(f"Falsification run has incomplete cells: {incomplete[:5]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat(
        [pd.read_csv(cell_directory(output_dir, *key) / "metrics.csv") for key in sorted(_expected_keys())],
        ignore_index=True,
    )
    registry = pd.concat(
        [pd.read_csv(cell_directory(output_dir, *key) / "fit_registry.csv") for key in sorted(_expected_keys())],
        ignore_index=True,
    )
    metrics.to_csv(output_dir / "metrics_by_seed_fold.csv", index=False)
    registry.to_csv(output_dir / "fit_registry.csv", index=False)
    control_predictions = _collect_control_predictions(output_dir)
    predictions = pd.concat([_load_reference_predictions(), control_predictions], ignore_index=True)
    if predictions.groupby(["model", "fold"])["arm"].nunique().ne(len(ANALYSIS_ARMS)).any():
        raise ValueError("Analysis predictions do not contain all six arms")
    predictions.to_csv(output_dir / "predictions_seed_averaged.csv", index=False)
    fold_metrics = fold_metrics_from_seed_averaged_predictions(predictions)
    fold_metrics.to_csv(output_dir / "fold_metrics_seed_averaged.csv", index=False)
    arm_summary = (
        fold_metrics.groupby(["model", "arm"], sort=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            direction_accuracy_mean=("direction_accuracy", "mean"),
            mcc_mean=("mcc", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            temporal_folds=("fold", "nunique"),
        )
        .reset_index()
    )
    arm_summary.to_csv(output_dir / "arm_summary.csv", index=False)
    paired = build_control_fold_contrasts(fold_metrics)
    paired.to_csv(output_dir / "paired_fold_contrasts.csv", index=False)
    inference = apply_holm_by_family(fold_level_inference(paired), pvalue_column="exact_sign_flip_pvalue")
    inference["minimum_attainable_nonzero_pvalue"] = 0.125
    inference.to_csv(output_dir / "fold_inference_holm.csv", index=False)
    bootstrap = apply_holm_by_family(_daily_bootstrap(predictions), pvalue_column="two_sided_pvalue")
    bootstrap.to_csv(output_dir / "daily_block_bootstrap_holm.csv", index=False)
    quarterly, quarterly_summary = _quarterly_panel(predictions)
    quarterly.to_csv(output_dir / "quarterly_origin_contrasts.csv", index=False)
    quarterly_summary.to_csv(output_dir / "quarterly_origin_summary.csv", index=False)
    runtime = (
        registry.groupby(["model", "arm"], sort=False)
        .agg(executed_fits=("fit_id", "size"), fit_seconds_total=("fit_seconds", "sum"), fit_seconds_mean=("fit_seconds", "mean"), inference_seconds_total=("inference_seconds", "sum"), training_sequences_min=("training_sequences", "min"), training_sequences_max=("training_sequences", "max"))
        .reset_index()
    )
    runtime.to_csv(output_dir / "runtime_summary.csv", index=False)
    primary = bootstrap.loc[
        bootstrap["contrast"].isin(("observed_news_effect", "observed_vs_shuffled"))
        & bootstrap["metric"].eq("balanced_accuracy_delta_pp")
    ].copy()
    primary.to_csv(output_dir / "paper_falsification_table.csv", index=False)
    integrity = {
        "passed": True,
        "protocol_id": PROTOCOL_ID,
        "input_manifest": freeze_audit,
        "expected_cells": len(_expected_keys()),
        "completed_cells": len(_expected_keys()),
        "new_fit_rows": len(registry),
        "analysis_arms": list(ANALYSIS_ARMS),
        "fold_metric_rows": len(fold_metrics),
        "paired_fold_rows": len(paired),
        "fold_inference_rows": len(inference),
        "bootstrap_rows": len(bootstrap),
        "quarterly_origin_rows": len(quarterly),
        "all_predictions_finite": bool(np.isfinite(predictions[["Close_D", "y_true", "y_pred"]]).all().all()),
    }
    (output_dir / "integrity_audit.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "completed_at_utc": _utc_now(),
        "evidence_status": "pre_frozen_retrospective_falsification",
        "incremental_api_cost_usd": 0,
        "models": list(TRACK_A_MODELS),
        "folds": list(FOLDS),
        "seeds": list(FINAL_SEEDS),
        "new_arms": [NEWS_ONLY_ARM, *CONTROL_ARMS],
        "reference_arms": REFERENCE_ARM_MAP,
        "primary_endpoint": "balanced_accuracy",
        "primary_falsification_contrast": "observed_vs_shuffled",
        "packages": package_versions(["numpy", "pandas", "scikit-learn", "scipy", "tensorflow"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_hashes": {"multimodal_falsification.py": _sha256(PROJECT_ROOT / "models" / "multimodal_falsification.py"), "multimodal_falsification_runner.py": _sha256(Path(__file__))},
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_cells_isolated(
    *,
    python_executable: Path = Path(sys.executable),
    output_dir: Path = OUTPUT_DIR,
    models: Iterable[str] = TRACK_A_MODELS,
    folds: Iterable[str] = FOLDS,
    seeds: Iterable[int] = FINAL_SEEDS,
    force: bool = False,
) -> dict[str, int]:
    commands = build_cell_commands(python_executable=python_executable, output_dir=output_dir, models=models, folds=folds, seeds=seeds, force=force)
    completed = skipped = 0
    for index, command in enumerate(commands, start=1):
        model = command[command.index("--model") + 1]
        fold = command[command.index("--fold") + 1]
        seed = int(command[command.index("--seed") + 1])
        if cell_complete(output_dir, model, fold, seed) and not force:
            skipped += 1
            print(f"[{index}/{len(commands)}] skip {model}/{fold}/{seed}", flush=True)
            continue
        print(f"[{index}/{len(commands)}] run {model}/{fold}/{seed}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        completed += 1
    return {"completed": completed, "skipped": skipped, "total": len(commands)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multimodal falsification controls")
    subparsers = parser.add_subparsers(dest="command", required=True)
    cell = subparsers.add_parser("cell")
    cell.add_argument("--model", required=True)
    cell.add_argument("--fold", required=True)
    cell.add_argument("--seed", required=True, type=int)
    cell.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    cell.add_argument("--force", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    run.add_argument("--force", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if args.command == "cell":
        result = run_cell(model=args.model, fold=args.fold, seed=args.seed, output_dir=args.output_dir, force=args.force)
    elif args.command == "run":
        result = run_cells_isolated(output_dir=args.output_dir, force=args.force)
    else:
        result = aggregate_experiment(output_dir=args.output_dir)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
