from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

_PROJECT_ROOT_EARLY = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "KERAS_HOME",
    str(_PROJECT_ROOT_EARLY / "runtime_cache" / "keras"),
)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(_PROJECT_ROOT_EARLY / "runtime_cache" / "mpl"),
)
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(_PROJECT_ROOT_EARLY / "runtime_cache" / "numba"),
)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from models.baseline_common import (
    PROJECT_ROOT,
    TARGET_COLUMN,
    discover_folds,
    load_fold,
    package_versions,
    predictions_frame,
    sequence_history_features,
    split_xy,
)
from models.convolutional_neural_network import (
    make_sequences,
    make_test_sequences,
)
from models.full_ta_feature_pool import FULL_TA_FEATURES
from models.neural_network_folds import inverse_scaled_target
from models.set100_robustness import (
    FEATURE_COUNT,
    FOLDS,
    MODELS,
    PROTOCOL_ID,
    SEEDS,
    TEST_YEARS,
    WINDOWS,
    evaluate_robustness_predictions,
    sha256_file,
    verify_freeze_manifest,
)
from models.shap_protocol_v2 import MODEL_BUILDERS
from models.track_a_final import TRACK_A_MODELS
from models.vmd_feature_pool import (
    VMD_CONFIG_NAME,
    VMD_FEATURES,
    VMDConfig,
    create_full_ta_vmd_folds,
    create_scaled_full_ta_vmd_nn_folds,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "set100_same_exchange_robustness_v1"
PRIVATE_DIR_NAME = "private"
CELL_DIR_NAME = "cells"
FREEZE_MANIFEST = (
    PROJECT_ROOT / "test" / "set100_same_exchange_robustness_freeze_v1.json"
)
SOURCE_FOLDS_DIR = PROJECT_ROOT / "set100_data" / "folds_point_in_time_v2"
FEATURE_FOLDS_DIR = (
    PROJECT_ROOT / "set100_data" / "full_ta_vmd_point_in_time_v2"
)
SCALED_FOLDS_DIR = (
    PROJECT_ROOT / "set100_data" / "full_ta_vmd_point_in_time_v2_nn"
)
FEATURE_AUDIT_FILE = OUTPUT_DIR / "feature_integrity_audit.json"
GENERATED_MANIFEST_FILE = OUTPUT_DIR / "generated_input_manifest.json"
RUNTIME_ENVIRONMENT_FILE = OUTPUT_DIR / "runtime_environment.json"


@dataclass(frozen=True)
class FitExecution:
    prediction: np.ndarray
    training_sequences: int
    fit_seconds: float
    inference_seconds: float
    trainable_parameters: int


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _validate_models(models: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(model) for model in models)
    if not values or len(set(values)) != len(values):
        raise ValueError("models must be non-empty and unique")
    unknown = sorted(set(values).difference(MODELS))
    if unknown:
        raise ValueError(f"Unknown SET100 models: {unknown}")
    return values


def _validate_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(seed) for seed in seeds)
    if not values or len(set(values)) != len(values):
        raise ValueError("seeds must be non-empty and unique")
    unknown = sorted(set(values).difference(SEEDS))
    if unknown:
        raise ValueError(f"Unregistered SET100 seeds: {unknown}")
    return values


def cell_directory(
    output_dir: Path,
    model: str,
    fold: str,
    seed: int,
) -> Path:
    return (
        output_dir
        / PRIVATE_DIR_NAME
        / CELL_DIR_NAME
        / model
        / fold
        / f"seed_{int(seed)}"
    )


def _cell_paths(directory: Path) -> dict[str, Path]:
    return {
        "predictions": directory / "predictions.csv",
        "metrics": directory / "metrics.json",
        "metadata": directory / "run_metadata.json",
        "audit": directory / "integrity_audit.json",
    }


def cell_complete(
    output_dir: Path,
    model: str,
    fold: str,
    seed: int,
) -> bool:
    paths = _cell_paths(cell_directory(output_dir, model, fold, seed))
    try:
        if not all(path.is_file() for path in paths.values()):
            return False
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
        metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        predictions = pd.read_csv(paths["predictions"])
        if metadata.get("protocol_id") != PROTOCOL_ID:
            return False
        if audit.get("passed") is not True:
            return False
        if "balanced_accuracy" not in metrics:
            return False
        required = {"Date", "Close_D", "y_true", "y_pred"}
        if required.difference(predictions.columns) or predictions.empty:
            return False
        numeric = predictions[["Close_D", "y_true", "y_pred"]].to_numpy(
            dtype=float
        )
        if not np.isfinite(numeric).all():
            return False
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True


def build_job_commands(
    *,
    python_executable: Path,
    output_dir: Path,
    models: Iterable[str] = MODELS,
    seeds: Iterable[int] = SEEDS,
    force: bool = False,
) -> list[list[str]]:
    model_values = _validate_models(models)
    seed_values = _validate_seeds(seeds)
    commands: list[list[str]] = []
    for model in model_values:
        for seed in seed_values:
            command = [
                str(python_executable),
                "-m",
                "models.set100_robustness_runner",
                "job",
                "--model",
                model,
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
            ]
            if force:
                command.append("--force")
            commands.append(command)
    return commands


def _manifest_rows(paths: Iterable[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]


def _fold_file_paths(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}
    ]


def _validate_feature_artifacts() -> dict[str, object]:
    source_specs = {spec.fold: spec for spec in discover_folds(SOURCE_FOLDS_DIR)}
    original_specs = {
        spec.fold: spec for spec in discover_folds(FEATURE_FOLDS_DIR)
    }
    scaled_specs = {
        spec.fold: spec for spec in discover_folds(SCALED_FOLDS_DIR)
    }
    if set(source_specs) != set(FOLDS) or set(original_specs) != set(FOLDS):
        raise ValueError("SET100 feature folds do not cover the registered folds")
    if set(scaled_specs) != set(FOLDS):
        raise ValueError("SET100 scaled folds do not cover the registered folds")
    expected_features = [*FULL_TA_FEATURES, *VMD_FEATURES]
    fold_rows: list[dict[str, object]] = []
    for fold, test_year in zip(FOLDS, TEST_YEARS, strict=True):
        source = load_fold(source_specs[fold])
        original = load_fold(original_specs[fold])
        scaled = load_fold(scaled_specs[fold])
        if original.feature_columns != expected_features:
            raise ValueError(f"{fold} feature order/count changed")
        if scaled.feature_columns != expected_features:
            raise ValueError(f"{fold} scaled feature order/count changed")
        if len(original.feature_columns) != FEATURE_COUNT:
            raise ValueError(f"{fold} does not contain {FEATURE_COUNT} features")
        source_dates = pd.to_datetime(source.test["Date"]).reset_index(drop=True)
        original_dates = pd.to_datetime(original.test["Date"]).reset_index(
            drop=True
        )
        scaled_dates = pd.to_datetime(scaled.test["Date"]).reset_index(drop=True)
        if not source_dates.equals(original_dates) or not original_dates.equals(
            scaled_dates
        ):
            raise ValueError(f"{fold} test dates changed during feature creation")
        first_test = original_dates.min()
        max_label = pd.to_datetime(original.train["Label_Date"]).max()
        if not max_label < first_test:
            raise ValueError(f"{fold} violates the label-date purge")
        scaler_path = scaled_specs[fold].train_path.parent / "minmax_scaler.json"
        scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
        if scaler.get("fit_scope") != "train_only":
            raise ValueError(f"{fold} scaler fit scope is not train-only")
        expected_scaled_columns = [*expected_features, TARGET_COLUMN]
        if scaler.get("columns") != expected_scaled_columns:
            raise ValueError(f"{fold} scaler columns changed")
        vmd_path = original_specs[fold].train_path.parent / VMD_CONFIG_NAME
        vmd = json.loads(vmd_path.read_text(encoding="utf-8"))
        if any(vmd.get(key) != value for key, value in asdict(VMDConfig()).items()):
            raise ValueError(f"{fold} VMD configuration changed")
        fold_rows.append(
            {
                "fold": fold,
                "test_year": test_year,
                "train_rows": len(original.train),
                "test_rows": len(original.test),
                "feature_count": len(original.feature_columns),
                "max_train_label_date": max_label.strftime("%Y-%m-%d"),
                "first_test_date": first_test.strftime("%Y-%m-%d"),
            }
        )
    return {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": _utc_now(),
        "passed": True,
        "feature_count": FEATURE_COUNT,
        "folds": fold_rows,
    }


def prepare_features(*, force: bool = False) -> dict[str, object]:
    freeze = verify_freeze_manifest(PROJECT_ROOT, FREEZE_MANIFEST)
    if force or not FEATURE_FOLDS_DIR.exists():
        create_full_ta_vmd_folds(
            source_dir=SOURCE_FOLDS_DIR,
            output_dir=FEATURE_FOLDS_DIR,
            config=VMDConfig(),
        )
    if force or not SCALED_FOLDS_DIR.exists():
        create_scaled_full_ta_vmd_nn_folds(
            source_dir=FEATURE_FOLDS_DIR,
            output_dir=SCALED_FOLDS_DIR,
        )
    audit = _validate_feature_artifacts()
    generated_paths = [
        *_fold_file_paths(FEATURE_FOLDS_DIR),
        *_fold_file_paths(SCALED_FOLDS_DIR),
    ]
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": _utc_now(),
        "freeze_verification": freeze,
        "files": _manifest_rows(generated_paths),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_AUDIT_FILE.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    GENERATED_MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return audit


def _load_scaler_metadata(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fit_predict(
    model: str,
    scaled_fold,
    *,
    window: int,
    seed: int,
    scaler: dict[str, object],
) -> FitExecution:
    import tensorflow as tf

    x_train, y_train, x_test, _ = split_xy(scaled_fold)
    train_features = x_train.to_numpy(dtype=float)
    train_target = y_train.to_numpy(dtype=float)
    x_sequence, y_sequence = make_sequences(
        train_features,
        train_target,
        window,
    )
    test_sequences = make_test_sequences(
        sequence_history_features(scaled_fold),
        x_test.to_numpy(dtype=float),
        window,
    )
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model_instance = MODEL_BUILDERS[model](
        (window, len(scaled_fold.feature_columns))
    )
    parameters = TRACK_A_MODELS[model].parameters
    fit_started = time.perf_counter()
    model_instance.fit(
        x_sequence,
        y_sequence,
        epochs=int(parameters["epochs"]),
        batch_size=int(parameters["batch_size"]),
        shuffle=False,
        verbose=0,
    )
    fit_seconds = time.perf_counter() - fit_started
    inference_started = time.perf_counter()
    scaled_prediction = model_instance.predict(
        test_sequences,
        verbose=0,
    ).reshape(-1)
    inference_seconds = time.perf_counter() - inference_started
    prediction = inverse_scaled_target(scaled_prediction, scaler)
    return FitExecution(
        prediction=np.asarray(prediction, dtype=float),
        training_sequences=len(x_sequence),
        fit_seconds=fit_seconds,
        inference_seconds=inference_seconds,
        trainable_parameters=int(model_instance.count_params()),
    )


def _runtime_environment() -> dict[str, object]:
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
    devices = [
        {"name": item.name, "device_type": item.device_type}
        for item in tf.config.list_physical_devices()
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "tensorflow": tf.__version__,
        "physical_devices": devices,
        "gpu_available": any(item["device_type"] == "GPU" for item in devices),
        "deterministic_operations_enabled": True,
        "tf_enable_onednn_opts": os.environ["TF_ENABLE_ONEDNN_OPTS"],
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "tensorflow"]
        ),
    }


def _save_cell(
    *,
    output_dir: Path,
    model: str,
    fold: str,
    seed: int,
    original_fold,
    execution: FitExecution,
    scaler_path: Path,
    runtime_environment: dict[str, object],
) -> dict[str, object]:
    if execution.prediction.shape != (len(original_fold.test),):
        raise ValueError(f"{model}/{fold}/{seed} prediction shape is invalid")
    if not np.isfinite(execution.prediction).all():
        raise ValueError(f"{model}/{fold}/{seed} prediction is non-finite")
    metrics = {
        "protocol_id": PROTOCOL_ID,
        "market": "SET100",
        "model": model,
        "fold": fold,
        "test_year": original_fold.spec.test_year,
        "seed": seed,
        "sequence_window": WINDOWS[model],
        "n_train": len(original_fold.train),
        "n_test": len(original_fold.test),
        "training_sequences": execution.training_sequences,
        "feature_count": len(original_fold.feature_columns),
        **evaluate_robustness_predictions(
            original_fold.test[TARGET_COLUMN].to_numpy(dtype=float),
            execution.prediction,
            original_fold.test["Close_D"].to_numpy(dtype=float),
        ),
        "fit_seconds": execution.fit_seconds,
        "inference_seconds": execution.inference_seconds,
        "total_model_seconds": (
            execution.fit_seconds + execution.inference_seconds
        ),
        "trainable_parameters": execution.trainable_parameters,
    }
    prediction = predictions_frame(original_fold, execution.prediction)
    directory = cell_directory(output_dir, model, fold, seed)
    directory.mkdir(parents=True, exist_ok=True)
    paths = _cell_paths(directory)
    prediction.to_csv(paths["predictions"], index=False)
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": _utc_now(),
        "market": "SET100",
        "model": model,
        "fold": fold,
        "seed": seed,
        "window": WINDOWS[model],
        "features": original_fold.feature_columns,
        "model_parameters": TRACK_A_MODELS[model].parameters,
        "runtime_environment": runtime_environment,
        "scaler_sha256": sha256_file(scaler_path),
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    audit = {
        "protocol_id": PROTOCOL_ID,
        "passed": True,
        "prediction_rows": len(prediction),
        "prediction_finite": True,
        "feature_count": len(original_fold.feature_columns),
        "window": WINDOWS[model],
        "seed": seed,
    }
    paths["audit"].write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    return metrics


def run_job(
    *,
    model: str,
    seed: int,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> list[dict[str, object]]:
    _validate_models((model,))
    _validate_seeds((seed,))
    verify_freeze_manifest(PROJECT_ROOT, FREEZE_MANIFEST)
    if not FEATURE_AUDIT_FILE.is_file() or not GENERATED_MANIFEST_FILE.is_file():
        raise FileNotFoundError("Run SET100 feature preparation before model jobs")
    original_specs = {
        spec.fold: spec for spec in discover_folds(FEATURE_FOLDS_DIR)
    }
    scaled_specs = {spec.fold: spec for spec in discover_folds(SCALED_FOLDS_DIR)}
    environment = _runtime_environment()
    rows: list[dict[str, object]] = []
    for fold in FOLDS:
        if cell_complete(output_dir, model, fold, seed) and not force:
            paths = _cell_paths(cell_directory(output_dir, model, fold, seed))
            rows.append(json.loads(paths["metrics"].read_text(encoding="utf-8")))
            continue
        original = load_fold(original_specs[fold])
        scaled = load_fold(scaled_specs[fold])
        if original.feature_columns != scaled.feature_columns:
            raise ValueError(f"{fold} original/scaled features differ")
        scaler_path = scaled_specs[fold].train_path.parent / "minmax_scaler.json"
        scaler = _load_scaler_metadata(scaler_path)
        execution = _fit_predict(
            model,
            scaled,
            window=WINDOWS[model],
            seed=seed,
            scaler=scaler,
        )
        rows.append(
            _save_cell(
                output_dir=output_dir,
                model=model,
                fold=fold,
                seed=seed,
                original_fold=original,
                execution=execution,
                scaler_path=scaler_path,
                runtime_environment=environment,
            )
        )
    return rows


def run_all(
    *,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict[str, object]:
    prepare_features(force=False)
    commands = build_job_commands(
        python_executable=Path(sys.executable),
        output_dir=output_dir,
        force=force,
    )
    started = time.perf_counter()
    completed_jobs = 0
    log_dir = output_dir / PRIVATE_DIR_NAME / "job_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for command in commands:
        model = command[command.index("--model") + 1]
        seed = int(command[command.index("--seed") + 1])
        if not force and all(
            cell_complete(output_dir, model, fold, seed) for fold in FOLDS
        ):
            completed_jobs += 1
            print(
                f"[{completed_jobs}/{len(commands)}] reuse {model} seed={seed}",
                flush=True,
            )
            continue
        log_path = log_dir / f"{model}_seed_{seed}.log"
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                check=True,
                cwd=PROJECT_ROOT,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        completed_jobs += 1
        elapsed = time.perf_counter() - started
        print(
            f"[{completed_jobs}/{len(commands)}] completed {model} seed={seed} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
    from models.set100_robustness_aggregate import aggregate_results

    result = aggregate_results(output_dir=output_dir)
    result["run_wall_seconds"] = time.perf_counter() - started
    (output_dir / "benchmark_completion.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> object:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--force", action="store_true")
    job = subparsers.add_parser("job")
    job.add_argument("--model", choices=MODELS, required=True)
    job.add_argument("--seed", choices=SEEDS, required=True, type=int)
    job.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    job.add_argument("--force", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    run.add_argument("--force", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_features(force=args.force)
    elif args.command == "job":
        result = run_job(
            model=args.model,
            seed=args.seed,
            output_dir=args.output_dir,
            force=args.force,
        )
    elif args.command == "run":
        result = run_all(output_dir=args.output_dir, force=args.force)
    elif args.command == "aggregate":
        from models.set100_robustness_aggregate import aggregate_results

        result = aggregate_results(output_dir=args.output_dir)
    else:
        raise ValueError(f"Unknown command: {args.command}")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
