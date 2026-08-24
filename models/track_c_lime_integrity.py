from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_common import PROJECT_ROOT
from models.track_a_final import TRACK_A_MODELS
from models.track_c_lime_audit import LimeAuditConfig

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_c" / "dual_xai_lime_v1"
CELL_DIR = OUTPUT_DIR / "cells"
AUDIT_OUTPUT = OUTPUT_DIR / "lime_integrity_audit.json"
RUNNER_PATH = PROJECT_ROOT / "models" / "track_c_lime_outer_runner.py"
FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4")
REGIMES = ("Bull", "Sideway", "Bear")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_finite(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    values = frame[columns].to_numpy(dtype=float)
    _require(np.isfinite(values).all(), f"{label} contains non-finite values")


def audit_lime_artifacts(*, write_output: bool = True) -> dict[str, object]:
    """Fail closed on missing, stale, duplicated, or invalid LIME artifacts."""

    config = LimeAuditConfig()
    implementation_hash = _sha256(RUNNER_PATH)
    complete_cells = 0
    reproduction_errors: list[float] = []
    graph_path_differences: list[float] = []
    for model in TRACK_A_MODELS:
        for fold in FOLDS:
            directory = CELL_DIR / model / fold
            required = (
                "selected_instances.csv",
                "local_explanations.csv",
                "agreement_by_instance.csv",
                "lime_stability_by_instance.csv",
                "run_metadata.json",
            )
            _require(
                all((directory / name).is_file() for name in required),
                f"Incomplete LIME cell: {model}/{fold}",
            )
            metadata = _load_json(directory / "run_metadata.json")
            _require(metadata["model"] == model, "Cell model identifier differs")
            _require(metadata["fold"] == fold, "Cell fold identifier differs")
            _require(
                metadata.get("implementation_sha256") == implementation_hash,
                f"Stale LIME implementation hash: {model}/{fold}",
            )
            _require(
                int(metadata["selected_instances"])
                == len(REGIMES) * config.samples_per_regime_fold,
                f"Invalid selected-instance count: {model}/{fold}",
            )
            error = float(metadata["outer_prediction_max_abs_reproduction_error"])
            tolerance = float(metadata["outer_prediction_reproduction_tolerance"])
            _require(
                np.isfinite(error) and error <= tolerance,
                f"Outer reproduction gate failed: {model}/{fold}",
            )
            difference = float(
                metadata[
                    "explanation_graph_max_abs_difference_from_outer_inference_path"
                ]
            )
            _require(np.isfinite(difference), "Graph-path difference is non-finite")
            reproduction_errors.append(error)
            graph_path_differences.append(difference)
            complete_cells += 1

    selected = pd.read_csv(OUTPUT_DIR / "selected_instances.csv")
    explanations = pd.read_csv(OUTPUT_DIR / "local_explanations.csv")
    agreements = pd.read_csv(OUTPUT_DIR / "agreement_by_instance.csv")
    stability = pd.read_csv(OUTPUT_DIR / "lime_stability_by_instance.csv")
    expected_instances = len(TRACK_A_MODELS) * len(FOLDS) * len(REGIMES) * 6
    _require(len(selected) == expected_instances, "Selected-instance count differs")
    _require(selected["instance_id"].is_unique, "Instance IDs are duplicated")
    _require(
        not selected.duplicated(["model", "fold", "test_row_index"]).any(),
        "Selected model/fold/test rows are duplicated",
    )
    selection_counts = selected.groupby(["model", "fold", "regime"]).size()
    _require(
        len(selection_counts) == len(TRACK_A_MODELS) * len(FOLDS) * len(REGIMES)
        and selection_counts.eq(6).all(),
        "Regime-stratified instance counts differ",
    )
    _require(
        not selected["selection_uses_outcome"].astype(bool).any(),
        "Outcome-dependent instance selection was detected",
    )

    expected_explanations = expected_instances * 122 * 6
    _require(
        len(explanations) == expected_explanations,
        "Local-explanation row count differs",
    )
    _require(
        set(explanations["instance_id"]) == set(selected["instance_id"]),
        "Explanation instances do not match selection",
    )
    explanation_counts = explanations.groupby(
        ["instance_id", "method"], sort=False
    ).size()
    _require(
        explanation_counts.xs("SHAP", level="method").eq(122).all(),
        "SHAP feature counts differ",
    )
    _require(
        explanation_counts.xs("LIME", level="method").eq(122 * 5).all(),
        "LIME feature/repeat counts differ",
    )
    _require_finite(
        explanations,
        ["attribution", "absolute_attribution", "absolute_rank"],
        "Local explanations",
    )

    expected_agreements = expected_instances * len(config.repeat_seeds)
    _require(len(agreements) == expected_agreements, "Agreement row count differs")
    _require(
        not agreements.duplicated(["instance_id", "repeat_seed"]).any(),
        "Agreement instance/repeat keys are duplicated",
    )
    repeats = agreements.groupby("instance_id")["repeat_seed"].apply(
        lambda values: set(values.astype(int))
    )
    _require(
        repeats.map(lambda values: values == set(config.repeat_seeds)).all(),
        "LIME repeat seeds differ",
    )
    _require_finite(
        agreements,
        [
            "fidelity_r2",
            "spearman_abs",
            "top_k_jaccard",
            "sign_agreement_nonzero",
            "lime_runtime_seconds",
            "model_inference_seconds",
        ],
        "Agreement rows",
    )
    low_mask = agreements["low_fidelity"].astype(bool)
    _require(
        low_mask.equals(agreements["fidelity_r2"].lt(config.minimum_fidelity_r2)),
        "Low-fidelity flags do not match the registered threshold",
    )

    _require(len(stability) == expected_instances, "Stability row count differs")
    _require(stability["instance_id"].is_unique, "Stability IDs are duplicated")
    summary_fold = pd.read_csv(OUTPUT_DIR / "summary_by_model_fold_regime.csv")
    summary_pooled = pd.read_csv(OUTPUT_DIR / "summary_pooled_by_model_regime.csv")
    summary_close = pd.read_csv(OUTPUT_DIR / "summary_excluding_structural_close.csv")
    sensitivity = pd.read_csv(OUTPUT_DIR / "agreement_excluding_structural_close.csv")
    low_audit = pd.read_csv(OUTPUT_DIR / "low_fidelity_audit.csv")
    runtime = pd.read_csv(OUTPUT_DIR / "runtime_summary.csv")
    _require(len(summary_fold) == 60, "Fold/regime summary count differs")
    _require(len(summary_pooled) == 15, "Pooled summary count differs")
    _require(len(summary_close) == 60, "Close sensitivity summary count differs")
    _require(len(sensitivity) == expected_agreements, "Sensitivity rows differ")
    _require(len(low_audit) == int(low_mask.sum()), "Low-fidelity audit differs")
    _require(len(runtime) == complete_cells, "Runtime cell count differs")

    root_metadata = _load_json(OUTPUT_DIR / "run_metadata.json")
    _require(
        bool(root_metadata.get("lime_outer_explanations_generated")),
        "Root LIME run is not marked complete",
    )
    _require(
        int(root_metadata["audit_instances"]) == expected_instances
        and int(root_metadata["agreement_rows"]) == expected_agreements,
        "Root LIME metadata counts differ",
    )

    artifact_names = (
        "protocol.json",
        "selected_instances.csv",
        "local_explanations.csv",
        "agreement_by_instance.csv",
        "lime_stability_by_instance.csv",
        "summary_by_model_fold_regime.csv",
        "summary_pooled_by_model_regime.csv",
        "agreement_excluding_structural_close.csv",
        "summary_excluding_structural_close.csv",
        "low_fidelity_audit.csv",
        "runtime_summary.csv",
        "error_analysis_summary.csv",
        "deviation_log.csv",
        "run_metadata.json",
    )
    payload: dict[str, object] = {
        "status": "passed",
        "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "complete_cells": complete_cells,
        "audit_instances": len(selected),
        "local_explanation_rows": len(explanations),
        "agreement_rows": len(agreements),
        "low_fidelity_rows": int(low_mask.sum()),
        "low_fidelity_fraction": float(low_mask.mean()),
        "outer_reproduction_gate_passed": True,
        "outer_reproduction_max_abs_error": max(reproduction_errors),
        "explanation_graph_max_abs_difference": max(graph_path_differences),
        "implementation_hashes_match": True,
        "artifact_sha256": {
            name: _sha256(OUTPUT_DIR / name) for name in artifact_names
        },
    }
    if write_output:
        AUDIT_OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    audit_lime_artifacts()
