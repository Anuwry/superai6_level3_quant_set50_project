from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_common import DATE_COLUMN, PROJECT_ROOT
from models.track_a_final import FINAL_SEEDS, TRACK_A_MODELS
from models.track_c_outer import OUTER_ARMS, REGIMES
from models.track_c_topk_validation_runner import (
    SELECTED_FEATURES_FILE,
    SELECTED_TOP_K_FILE,
    SELECTION_FREEZE_FILE,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_c" / "outer_v2"
CELL_DIR = OUTPUT_DIR / "cells"
AUDIT_OUTPUT = OUTPUT_DIR / "outer_integrity_audit.json"
EXPECTED_FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4")
EXPECTED_UNIQUE_FITS_PER_CELL = 11
EXPECTED_CONCEPTUAL_FITS_PER_CELL = 15


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_finite(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")


def audit_outer_artifacts(
    *,
    write_output: bool = True,
) -> dict[str, object]:
    expected_cells = (
        len(TRACK_A_MODELS) * len(EXPECTED_FOLDS) * len(FINAL_SEEDS)
    )
    expected_metric_rows = expected_cells * len(OUTER_ARMS)
    complete_cells = 0
    prediction_files = 0
    fit_registry_rows = 0

    for model in TRACK_A_MODELS:
        for fold in EXPECTED_FOLDS:
            for seed in FINAL_SEEDS:
                directory = CELL_DIR / model / fold / f"seed_{seed}"
                metrics_path = directory / "metrics.csv"
                registry_path = directory / "fit_registry.csv"
                metadata_path = directory / "run_metadata.json"
                required = [metrics_path, registry_path, metadata_path]
                required.extend(
                    directory / f"predictions_{arm}.csv"
                    for arm in OUTER_ARMS
                )
                missing = [str(path) for path in required if not path.is_file()]
                if missing:
                    raise FileNotFoundError(
                        f"Incomplete outer cell {model}/{fold}/{seed}: {missing}"
                    )

                metrics = pd.read_csv(metrics_path)
                if len(metrics) != len(OUTER_ARMS) or set(
                    metrics["arm"]
                ) != set(OUTER_ARMS):
                    raise ValueError(
                        f"Invalid arm metrics in {model}/{fold}/{seed}"
                    )
                if not (
                    metrics["model"].eq(model).all()
                    and metrics["fold"].eq(fold).all()
                    and metrics["base_seed"].astype(int).eq(seed).all()
                ):
                    raise ValueError("Outer cell metric identifiers differ")
                _require_finite(
                    metrics,
                    [
                        "rmse",
                        "mae",
                        "direction_accuracy",
                        "balanced_accuracy",
                        "mcc",
                        "direction_coverage",
                    ],
                    f"{model}/{fold}/{seed} metrics",
                )

                reference_dates: pd.Series | None = None
                reference_target: np.ndarray | None = None
                reference_close: np.ndarray | None = None
                for arm in OUTER_ARMS:
                    frame = pd.read_csv(directory / f"predictions_{arm}.csv")
                    required_columns = {
                        DATE_COLUMN,
                        "routing_regime",
                        "Close_D",
                        "y_true",
                        "y_pred",
                        "true_direction",
                        "pred_direction",
                    }
                    if not required_columns.issubset(frame.columns):
                        raise ValueError("Outer prediction columns are incomplete")
                    dates = pd.to_datetime(frame[DATE_COLUMN], errors="raise")
                    if dates.duplicated().any() or not dates.is_monotonic_increasing:
                        raise ValueError("Outer prediction dates are invalid")
                    _require_finite(
                        frame,
                        [
                            "Close_D",
                            "y_true",
                            "y_pred",
                            "true_direction",
                            "pred_direction",
                        ],
                        f"{model}/{fold}/{seed}/{arm} predictions",
                    )
                    if set(frame["routing_regime"]).difference(REGIMES):
                        raise ValueError("Outer prediction has unknown regime")
                    if reference_dates is None:
                        reference_dates = dates
                        reference_target = frame["y_true"].to_numpy(dtype=float)
                        reference_close = frame["Close_D"].to_numpy(dtype=float)
                    else:
                        if not dates.equals(reference_dates):
                            raise ValueError("Outer arm dates do not align")
                        if not np.array_equal(
                            frame["y_true"].to_numpy(dtype=float),
                            reference_target,
                        ) or not np.array_equal(
                            frame["Close_D"].to_numpy(dtype=float),
                            reference_close,
                        ):
                            raise ValueError("Outer arm targets do not align")
                    prediction_files += 1

                registry = pd.read_csv(registry_path)
                if (
                    len(registry) != EXPECTED_UNIQUE_FITS_PER_CELL
                    or registry["fit_id"].nunique()
                    != EXPECTED_UNIQUE_FITS_PER_CELL
                    or int(registry["arm_reference_count"].sum())
                    != EXPECTED_CONCEPTUAL_FITS_PER_CELL
                ):
                    raise ValueError("Outer fit registry count is invalid")
                _require_finite(
                    registry,
                    [
                        "training_sequences",
                        "fit_seconds",
                        "inference_seconds",
                        "trainable_parameters",
                    ],
                    f"{model}/{fold}/{seed} fit registry",
                )
                metadata = _load_json(metadata_path)
                if not (
                    metadata["model"] == model
                    and metadata["fold"] == fold
                    and int(metadata["base_seed"]) == seed
                    and int(metadata["unique_fits"])
                    == EXPECTED_UNIQUE_FITS_PER_CELL
                    and int(metadata["conceptual_arm_fits"])
                    == EXPECTED_CONCEPTUAL_FITS_PER_CELL
                ):
                    raise ValueError("Outer cell metadata is invalid")
                complete_cells += 1
                fit_registry_rows += len(registry)

    root_metrics = pd.read_csv(OUTPUT_DIR / "metrics_by_seed_fold.csv")
    metric_key = ["model", "fold", "base_seed", "arm"]
    if (
        len(root_metrics) != expected_metric_rows
        or root_metrics.duplicated(metric_key).any()
    ):
        raise ValueError("Root outer metrics are incomplete or duplicated")
    averaged = pd.read_csv(OUTPUT_DIR / "predictions_seed_averaged.csv")
    averaged_key = ["model", "fold", "arm", DATE_COLUMN]
    if averaged.duplicated(averaged_key).any():
        raise ValueError("Seed-averaged predictions contain duplicate keys")
    if not averaged["seeds_averaged"].astype(int).eq(len(FINAL_SEEDS)).all():
        raise ValueError("Seed-averaged prediction count is invalid")
    _require_finite(
        averaged,
        ["Close_D", "y_true", "y_pred"],
        "seed-averaged predictions",
    )

    root_registry = pd.read_csv(OUTPUT_DIR / "fit_registry.csv")
    if len(root_registry) != fit_registry_rows:
        raise ValueError("Root fit registry row count is invalid")
    identifier_columns = ["model", "fold", "base_seed"]
    if not set(identifier_columns).issubset(root_registry.columns):
        raise ValueError("Root fit registry lacks cell identifiers")
    if root_registry[identifier_columns].isna().any().any():
        raise ValueError("Root fit registry has missing cell identifiers")

    fold_metrics = pd.read_csv(OUTPUT_DIR / "fold_metrics_seed_averaged.csv")
    if len(fold_metrics) != len(TRACK_A_MODELS) * len(EXPECTED_FOLDS) * len(
        OUTER_ARMS
    ):
        raise ValueError("Seed-averaged fold metric count is invalid")
    arm_summary = pd.read_csv(OUTPUT_DIR / "arm_summary.csv")
    if len(arm_summary) != len(TRACK_A_MODELS) * len(OUTER_ARMS):
        raise ValueError("Outer arm summary count is invalid")

    paired = pd.read_csv(OUTPUT_DIR / "paired_fold_contrasts.csv")
    fold_inference = pd.read_csv(OUTPUT_DIR / "fold_inference.csv")
    bootstrap = pd.read_csv(OUTPUT_DIR / "daily_block_bootstrap.csv")
    holm = pd.read_csv(OUTPUT_DIR / "inference_holm_adjusted.csv")
    if len(paired) != 100 or len(fold_inference) != 125:
        raise ValueError("Fold inference artifact count is invalid")
    if len(bootstrap) != 50 or len(holm) != 175:
        raise ValueError("Bootstrap/Holm artifact count is invalid")
    for frame, columns in (
        (
            fold_inference,
            ["exact_sign_flip_pvalue", "exact_sign_flip_pvalue_holm"],
        ),
        (bootstrap, ["two_sided_pvalue", "two_sided_pvalue_holm"]),
        (holm, ["raw_pvalue", "holm_adjusted_pvalue"]),
    ):
        values = frame[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(
            (values < 0.0) | (values > 1.0)
        ):
            raise ValueError("Inference p-values are invalid")

    freeze = _load_json(SELECTION_FREEZE_FILE)
    freeze_matches = (
        _sha256(SELECTED_TOP_K_FILE) == freeze["selected_top_k_sha256"]
        and _sha256(SELECTED_FEATURES_FILE)
        == freeze["selected_features_sha256"]
        and _sha256(
            SELECTED_FEATURES_FILE.parent / "top_k_gate_audit.csv"
        )
        == freeze["top_k_gate_audit_sha256"]
    )
    if not freeze_matches:
        raise ValueError("Selection freeze hashes no longer match")
    run_metadata = _load_json(OUTPUT_DIR / "run_metadata.json")
    if not run_metadata.get("outer_results_generated"):
        raise ValueError("Outer run metadata is not marked complete")

    artifact_names = (
        "metrics_by_seed_fold.csv",
        "predictions_seed_averaged.csv",
        "fold_metrics_seed_averaged.csv",
        "arm_summary.csv",
        "fit_registry.csv",
        "paired_fold_contrasts.csv",
        "fold_inference.csv",
        "daily_block_bootstrap.csv",
        "inference_holm_adjusted.csv",
        "inference_protocol.json",
        "run_metadata.json",
    )
    payload: dict[str, object] = {
        "status": "passed",
        "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "complete_cells": complete_cells,
        "metric_rows": len(root_metrics),
        "prediction_files": prediction_files,
        "seed_averaged_prediction_rows": len(averaged),
        "fit_registry_rows": fit_registry_rows,
        "paired_fold_contrast_rows": len(paired),
        "fold_inference_rows": len(fold_inference),
        "block_bootstrap_rows": len(bootstrap),
        "holm_rows": len(holm),
        "freeze_hashes_match": freeze_matches,
        "all_predictions_finite": True,
        "all_arm_dates_aligned": True,
        "all_cells_have_expected_fit_registry": True,
        "artifact_sha256": {
            name: _sha256(OUTPUT_DIR / name) for name in artifact_names
        },
        "operational_note": (
            "A redundant deterministic worker was stopped during the final "
            "model; complete-cell structure, identifiers, dates, targets, "
            "finiteness, fit registries, and aggregate uniqueness all passed."
        ),
    }
    if write_output:
        AUDIT_OUTPUT.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    return payload


if __name__ == "__main__":
    audit_outer_artifacts()
