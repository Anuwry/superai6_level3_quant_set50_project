from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

CLAIM_HIERARCHY = {
    "balanced_accuracy": "primary",
    "direction_accuracy": "secondary",
    "mcc": "secondary",
    "rmse_mae": "secondary",
    "shap": "main_secondary",
    "lime": "supplement_diagnostic",
    "economic_proxy": "supplement_exploratory",
    "llm_intrinsic": "main_separate_intrinsic",
}

SOURCE_FILES = {
    "windows": Path("outputs/track_a_final_point_in_time_v2/locked_windows.csv"),
    "numerical": Path("outputs/track_a_final_point_in_time_v2/paper_track_a_compact.csv"),
    "falsification": Path("outputs/multimodal_falsification_v1/paper_falsification_table.csv"),
    "llm_intrinsic": Path("outputs/track_b/llm/compute_matched_v1/paper_compute_matched_table.csv"),
    "regime_shap": Path("outputs/track_c/outer_v2/inference_holm_adjusted.csv"),
    "forward": Path("outputs/track_d_q2/paper_predictive_summary.csv"),
    "set100": Path("outputs/set100_same_exchange_robustness_v1/paper_table.csv"),
    "lime": Path("outputs/track_c/dual_xai_lime_v1/summary_pooled_by_model_regime.csv"),
    "economics": Path("outputs/track_d_q2/paper_economic_primary_10bps.csv"),
    "primary_protocol": Path("test/primary_estimand_and_confirmatory_protocol_v1.md"),
    "falsification_protocol": Path("test/reliability_extension_protocol_v1.md"),
    "data_statements": Path("outputs/market_data_governance_v1/paper_data_statements.md"),
}

RESTRICTED_PARTS = {"data-raw", "data-prepared", "data-folds", "private", "set100_data"}
RESTRICTED_NAMES = {".env", "key.md"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manuscript_relative_path(relative_path: Path) -> None:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe manuscript path: {relative_path}")
    lowered = {part.lower() for part in relative_path.parts}
    if lowered.intersection(RESTRICTED_PARTS):
        raise ValueError(f"Restricted manuscript path: {relative_path}")
    if relative_path.name.lower() in RESTRICTED_NAMES:
        raise ValueError(f"Private manuscript path: {relative_path}")
    if relative_path.name.lower().startswith("predictions"):
        raise ValueError(f"Row-level predictions are not a manuscript table: {relative_path}")


def build_file_manifest(
    project_root: Path,
    relative_paths: list[Path] | tuple[Path, ...],
) -> list[dict[str, object]]:
    root = project_root.resolve()
    rows: list[dict[str, object]] = []
    for relative in sorted(set(relative_paths), key=lambda path: path.as_posix()):
        validate_manuscript_relative_path(relative)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required manuscript source is missing: {relative}")
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _required_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> pd.DataFrame:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    return frame.loc[:, columns].copy()


def _write_table(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    frame.to_csv(path, index=False)
    return {"path": path.name, "rows": len(frame), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def build_manuscript_artifacts(
    project_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    root = project_root.resolve()
    output = output_dir.resolve()
    source_manifest = build_file_manifest(root, tuple(SOURCE_FILES.values()))
    frames = {
        name: pd.read_csv(root / path)
        for name, path in SOURCE_FILES.items()
        if path.suffix == ".csv"
    }
    output.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, object]] = []

    windows = _required_columns(
        frames["windows"],
        ("model", "selected_sequence_window"),
        "locked windows",
    )
    protocol = windows.assign(
        outer_test_years="2022-2025",
        outer_folds=4,
        seeds_per_fold=5,
        primary_metric="balanced_accuracy",
        information_timezone="Asia/Bangkok",
        information_cutoff="17:00",
        numerical_features=122,
        news_features=8,
    )
    generated.append(_write_table(protocol, output / "table_1_protocol_cohort.csv"))

    numerical_columns = (
        "model",
        "selected_sequence_window",
        "full_ta_balanced_accuracy_mean",
        "vmd_balanced_accuracy_mean",
        "balanced_accuracy_delta_pp",
        "balanced_accuracy_delta_pp_ci95_lower",
        "balanced_accuracy_delta_pp_ci95_upper",
        "balanced_accuracy_exact_sign_flip_pvalue",
        "full_ta_runtime_seconds_mean",
        "vmd_runtime_seconds_mean",
    )
    numerical = _required_columns(frames["numerical"], numerical_columns, "Track A")
    generated.append(_write_table(numerical, output / "table_2_numerical_ablation.csv"))

    falsification_columns = (
        "model",
        "contrast",
        "metric",
        "point_estimate",
        "ci95_lower",
        "ci95_upper",
        "two_sided_pvalue",
        "two_sided_pvalue_holm",
        "daily_rows",
    )
    falsification = _required_columns(
        frames["falsification"], falsification_columns, "falsification table"
    )
    generated.append(_write_table(falsification, output / "table_3a_multimodal_falsification.csv"))

    llm_columns = (
        "comparison_id",
        "pairs",
        "unique_articles",
        "control_accuracy_x",
        "leader_accuracy",
        "accuracy_delta_pp",
        "accuracy_delta_pp_ci95_lower",
        "accuracy_delta_pp_ci95_upper",
        "holm_adjusted_pvalue",
    )
    llm = _required_columns(frames["llm_intrinsic"], llm_columns, "LLM intrinsic table")
    llm.insert(0, "evidence_boundary", "intrinsic_only_not_downstream_feature_source")
    generated.append(_write_table(llm, output / "table_3b_llm_intrinsic_separate.csv"))

    regime = frames["regime_shap"].loc[
        frames["regime_shap"]["metric"].eq("balanced_accuracy_delta_pp")
    ].copy()
    regime = _required_columns(
        regime,
        (
            "inference_type",
            "model",
            "contrast",
            "point_estimate",
            "ci95_lower",
            "ci95_upper",
            "raw_pvalue",
            "holm_adjusted_pvalue",
        ),
        "regime SHAP table",
    )
    generated.append(_write_table(regime, output / "table_4_regime_shap.csv"))

    forward = _required_columns(
        frames["forward"],
        ("model", "objective", "rows", "direction_accuracy", "balanced_accuracy", "mcc", "auc", "brier"),
        "Track D forward table",
    )
    forward.insert(0, "evidence_boundary", "source_contingent_partial_2026")
    generated.append(_write_table(forward, output / "table_5a_forward_robustness.csv"))

    set100 = _required_columns(
        frames["set100"],
        (
            "model",
            "sequence_window",
            "balanced_accuracy_set50_mean",
            "balanced_accuracy_set100_mean",
            "balanced_accuracy_delta_pp",
            "balanced_accuracy_delta_ci95_lower",
            "balanced_accuracy_delta_ci95_upper",
            "balanced_accuracy_holm_pvalue",
        ),
        "SET100 table",
    )
    set100.insert(0, "evidence_boundary", "same_exchange_not_external_market")
    generated.append(_write_table(set100, output / "table_5b_set100_transfer.csv"))

    lime = frames["lime"].copy()
    lime.insert(0, "evidence_role", CLAIM_HIERARCHY["lime"])
    generated.append(_write_table(lime, output / "supplement_lime_diagnostic.csv"))
    economics = frames["economics"].copy()
    economics.insert(0, "evidence_role", CLAIM_HIERARCHY["economic_proxy"])
    generated.append(_write_table(economics, output / "supplement_economic_exploratory.csv"))

    table_index = pd.DataFrame(
        [
            {"artifact": row["path"], "rows": row["rows"], "placement": "Supplement" if str(row["path"]).startswith("supplement") else "Main", "sha256": row["sha256"]}
            for row in generated
        ]
    )
    generated.append(_write_table(table_index, output / "table_index.csv"))
    (output / "claim_hierarchy.json").write_text(
        json.dumps(CLAIM_HIERARCHY, indent=2) + "\n",
        encoding="utf-8",
    )
    generated.append(
        {
            "path": "claim_hierarchy.json",
            "rows": len(CLAIM_HIERARCHY),
            "bytes": (output / "claim_hierarchy.json").stat().st_size,
            "sha256": _sha256(output / "claim_hierarchy.json"),
        }
    )
    manifest = {
        "protocol_id": "manuscript-artifacts-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": source_manifest,
        "generated": generated,
        "raw_market_data_included": False,
        "row_level_predictions_included": False,
        "claim_hierarchy": CLAIM_HIERARCHY,
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
