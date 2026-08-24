from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_ORDER = (
    "lstm",
    "cnn",
    "lstm_cnn",
    "lstm_attention",
    "lstm_cnn_attention",
)

SOURCE_PATHS = {
    "track_a": Path(
        "outputs/track_a_final_point_in_time_v2/paper_track_a_compact.csv"
    ),
    "track_a_windows": Path(
        "outputs/track_a_final_point_in_time_v2/locked_windows.csv"
    ),
    "track_b": Path(
        "outputs/track_b/four_fold_ablation_point_in_time_v2/"
        "paper_track_b_four_fold_table.csv"
    ),
    "track_c_arms": Path("outputs/track_c/outer_v2/arm_summary.csv"),
    "track_c_inference": Path(
        "outputs/track_c/outer_v2/inference_holm_adjusted.csv"
    ),
    "track_c_integrity": Path(
        "outputs/track_c/outer_v2/outer_integrity_audit.json"
    ),
    "track_d_predictive": Path(
        "outputs/track_d_q2/paper_predictive_summary.csv"
    ),
    "track_d_economic": Path(
        "outputs/track_d_q2/paper_economic_primary_10bps.csv"
    ),
    "track_d_integrity": Path("outputs/track_d_q2/integrity_audit.json"),
}

METRICS = {
    "balanced_accuracy": {
        "better_when": "higher",
        "unit": "percentage_points",
    },
    "direction_accuracy": {
        "better_when": "higher",
        "unit": "percentage_points",
    },
    "rmse": {"better_when": "lower", "unit": "index_points"},
    "auc": {"better_when": "higher", "unit": "raw"},
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required evidence artifact is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index(rows: Iterable[dict[str, str]], *keys: str) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def _delta(control: float, treatment: float, metric: str) -> float:
    difference = treatment - control
    if METRICS[metric]["unit"] == "percentage_points":
        return difference * 100.0
    return difference


def _is_improved(delta: float, metric: str) -> bool:
    if abs(delta) <= 1e-12:
        return False
    if METRICS[metric]["better_when"] == "higher":
        return delta > 0.0
    return delta < 0.0


def _row(
    *,
    track: str,
    comparison_id: str,
    comparison_label: str,
    model: str,
    metric: str,
    control_label: str,
    treatment_label: str,
    control_value: float,
    treatment_value: float,
    evidence_class: str,
    temporal_units: int,
    seeds: int,
    source_artifact: str,
    raw_pvalue: float | None = None,
    holm_adjusted_pvalue: float | None = None,
    note: str = "",
) -> dict[str, Any]:
    delta = _delta(control_value, treatment_value, metric)
    if holm_adjusted_pvalue is not None:
        statistical_status = (
            "holm_significant"
            if holm_adjusted_pvalue < 0.05
            else "not_holm_significant"
        )
    elif raw_pvalue is not None:
        statistical_status = (
            "raw_significant" if raw_pvalue < 0.05 else "not_raw_significant"
        )
    else:
        statistical_status = "descriptive_no_inference"
    return {
        "track": track,
        "comparison_id": comparison_id,
        "comparison_label": comparison_label,
        "model": model,
        "metric": metric,
        "control_label": control_label,
        "treatment_label": treatment_label,
        "control_value": control_value,
        "treatment_value": treatment_value,
        "delta": delta,
        "delta_unit": METRICS[metric]["unit"],
        "better_when": METRICS[metric]["better_when"],
        "improved": _is_improved(delta, metric),
        "improved_count_within_comparison_metric": 0,
        "raw_pvalue": raw_pvalue,
        "holm_adjusted_pvalue": holm_adjusted_pvalue,
        "statistical_status": statistical_status,
        "evidence_class": evidence_class,
        "temporal_units": temporal_units,
        "seeds": seeds,
        "source_artifact": source_artifact,
        "note": note,
    }


def _track_a_rows(repo_root: Path) -> list[dict[str, Any]]:
    source = SOURCE_PATHS["track_a"]
    indexed = _index(_read_csv(repo_root / source), "model")
    rows: list[dict[str, Any]] = []
    columns = {
        "balanced_accuracy": (
            "full_ta_balanced_accuracy_mean",
            "vmd_balanced_accuracy_mean",
            "balanced_accuracy_exact_sign_flip_pvalue",
        ),
        "direction_accuracy": (
            "full_ta_direction_accuracy_mean",
            "vmd_direction_accuracy_mean",
            "direction_exact_sign_flip_pvalue",
        ),
        "rmse": (
            "full_ta_rmse_mean",
            "vmd_rmse_mean",
            "rmse_exact_sign_flip_pvalue",
        ),
    }
    for model in MODEL_ORDER:
        item = indexed[(model,)]
        for metric, (control_column, treatment_column, p_column) in columns.items():
            rows.append(
                _row(
                    track="A",
                    comparison_id="track_a_vmd_vs_full_ta",
                    comparison_label="Causal VMD versus Full TA",
                    model=model,
                    metric=metric,
                    control_label="Full TA",
                    treatment_label="Full TA + causal VMD",
                    control_value=float(item[control_column]),
                    treatment_value=float(item[treatment_column]),
                    raw_pvalue=float(item[p_column]),
                    holm_adjusted_pvalue=None,
                    evidence_class="corrected_confirmatory_ablation_low_power",
                    temporal_units=int(item["paired_outer_folds"]),
                    seeds=int(item["seeds_per_fold"]),
                    source_artifact=source.as_posix(),
                    note="Window selected before 2022; boundary label purged in point-in-time v2.",
                )
            )
    return rows


def _track_b_rows(repo_root: Path) -> list[dict[str, Any]]:
    source = SOURCE_PATHS["track_b"]
    indexed = _index(_read_csv(repo_root / source), "model")
    rows: list[dict[str, Any]] = []
    columns = {
        "balanced_accuracy": (
            "technical_balanced_accuracy_mean",
            "news_balanced_accuracy_mean",
            "balanced_accuracy_exact_sign_flip_pvalue",
        ),
        "direction_accuracy": (
            "technical_direction_accuracy_mean",
            "news_direction_accuracy_mean",
            "direction_exact_sign_flip_pvalue",
        ),
        "rmse": (
            "technical_rmse_mean",
            "news_rmse_mean",
            "rmse_exact_sign_flip_pvalue",
        ),
    }
    for model in MODEL_ORDER:
        item = indexed[(model,)]
        for metric, (control_column, treatment_column, p_column) in columns.items():
            rows.append(
                _row(
                    track="B",
                    comparison_id="track_b_news_vs_technical",
                    comparison_label="Predicted news feature versus technical-only",
                    model=model,
                    metric=metric,
                    control_label="Technical + VMD",
                    treatment_label="Technical + VMD + predicted news",
                    control_value=float(item[control_column]),
                    treatment_value=float(item[treatment_column]),
                    raw_pvalue=float(item[p_column]),
                    holm_adjusted_pvalue=None,
                    evidence_class="corrected_paired_fusion_ablation_low_power",
                    temporal_units=int(item["paired_outer_folds"]),
                    seeds=int(item["seeds_per_fold"]),
                    source_artifact=source.as_posix(),
                    note="News fusion only; intrinsic LLM benchmark is a separate Track B result.",
                )
            )
    return rows


def _track_c_rows(repo_root: Path) -> list[dict[str, Any]]:
    arm_source = SOURCE_PATHS["track_c_arms"]
    inference_source = SOURCE_PATHS["track_c_inference"]
    arms = _index(_read_csv(repo_root / arm_source), "model", "arm")
    inference = _index(
        _read_csv(repo_root / inference_source),
        "inference_type",
        "model",
        "contrast",
        "metric",
    )
    rows: list[dict[str, Any]] = []
    metric_columns = {
        "balanced_accuracy": "balanced_accuracy_mean",
        "direction_accuracy": "direction_accuracy_mean",
        "rmse": "rmse_mean",
    }
    inference_metrics = {
        "balanced_accuracy": "balanced_accuracy_delta_pp",
        "direction_accuracy": "direction_accuracy_delta_pp",
        "rmse": "rmse_delta",
    }
    comparisons = (
        {
            "comparison_id": "track_c_regime_shap_vs_global_all_descriptive",
            "label": "End-to-end Regime-SHAP versus single Global-All",
            "control": "Global-All",
            "treatment": "Regime-SHAP",
            "contrast": None,
            "evidence_class": "descriptive_capacity_confounded",
            "note": "Descriptive only: routing capacity differs; not an isolated SHAP effect.",
        },
        {
            "comparison_id": "track_c_regime_shap_vs_regime_all",
            "label": "SHAP reduction within the same regime router",
            "control": "Regime-All",
            "treatment": "Regime-SHAP",
            "contrast": "regime_shap_reduction",
            "evidence_class": "registered_post_hoc_robustness",
            "note": "Registered paired contrast; Track C remains post-hoc robustness evidence.",
        },
        {
            "comparison_id": "track_c_regime_shap_vs_global3_all",
            "label": "Regime-SHAP versus capacity-matched Global3-All",
            "control": "Global3-All",
            "treatment": "Regime-SHAP",
            "contrast": "regime_routing",
            "evidence_class": "registered_post_hoc_capacity_control",
            "note": "Capacity-matched routing contrast; no confirmatory regime claim.",
        },
    )
    for comparison in comparisons:
        for model in MODEL_ORDER:
            control = arms[(model, comparison["control"])]
            treatment = arms[(model, comparison["treatment"])]
            for metric, column in metric_columns.items():
                raw_pvalue: float | None = None
                holm_pvalue: float | None = None
                if comparison["contrast"] is not None:
                    result = inference[
                        (
                            "four_fold_exact_sign_flip",
                            model,
                            comparison["contrast"],
                            inference_metrics[metric],
                        )
                    ]
                    raw_pvalue = float(result["raw_pvalue"])
                    holm_pvalue = float(result["holm_adjusted_pvalue"])
                rows.append(
                    _row(
                        track="C",
                        comparison_id=comparison["comparison_id"],
                        comparison_label=comparison["label"],
                        model=model,
                        metric=metric,
                        control_label=comparison["control"],
                        treatment_label=comparison["treatment"],
                        control_value=float(control[column]),
                        treatment_value=float(treatment[column]),
                        raw_pvalue=raw_pvalue,
                        holm_adjusted_pvalue=holm_pvalue,
                        evidence_class=comparison["evidence_class"],
                        temporal_units=int(treatment["temporal_folds"]),
                        seeds=5,
                        source_artifact=(
                            f"{arm_source.as_posix()} | {inference_source.as_posix()}"
                        ),
                        note=comparison["note"],
                    )
                )
    return rows


def _track_d_rows(repo_root: Path) -> list[dict[str, Any]]:
    source = SOURCE_PATHS["track_d_predictive"]
    indexed = _index(_read_csv(repo_root / source), "model", "objective")
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        control = indexed[(model, "direct")]
        treatment = indexed[(model, "multitask")]
        for metric in ("balanced_accuracy", "direction_accuracy", "auc"):
            rows.append(
                _row(
                    track="D",
                    comparison_id="track_d_multitask_vs_direct",
                    comparison_label="Multitask direction/return versus direct direction",
                    model=model,
                    metric=metric,
                    control_label="Direct direction",
                    treatment_label="Multitask direction + return",
                    control_value=float(control[metric]),
                    treatment_value=float(treatment[metric]),
                    raw_pvalue=None,
                    holm_adjusted_pvalue=None,
                    evidence_class="source_contingency_partial_2026_forward",
                    temporal_units=int(treatment["rows"]),
                    seeds=int(treatment["seeds"]),
                    source_artifact=source.as_posix(),
                    note=(
                        "Frozen model protocol, but alternative source/parser deviations; "
                        "DA gains may reflect one-sided prediction collapse."
                    ),
                )
            )
    return rows


def _annotate_group_counts(rows: list[dict[str, Any]]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["comparison_id"]), str(row["metric"]))
        counts[key] = counts.get(key, 0) + int(bool(row["improved"]))
    for row in rows:
        key = (str(row["comparison_id"]), str(row["metric"]))
        row["improved_count_within_comparison_metric"] = counts[key]


def build_improvement_rows(repo_root: Path) -> list[dict[str, Any]]:
    root = Path(repo_root)
    for relative_path in SOURCE_PATHS.values():
        path = root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Required evidence artifact is missing: {path}")
    rows = (
        _track_a_rows(root)
        + _track_b_rows(root)
        + _track_c_rows(root)
        + _track_d_rows(root)
    )
    _annotate_group_counts(rows)
    return rows


def _claim_rows() -> list[dict[str, str]]:
    return [
        {
            "component": "Track A causal VMD ablation",
            "evidence_status": "corrected confirmatory ablation; low temporal power",
            "allowed_claim": "VMD effects are model-dependent",
            "blocked_claim": "VMD universally improves direction prediction",
            "next_action": "retain mixed results and exact fold uncertainty",
        },
        {
            "component": "Track B news fusion",
            "evidence_status": "corrected paired ablation; domain shift disclosed",
            "allowed_claim": "predicted news helps selected model/metric cells",
            "blocked_claim": "news universally improves all five models",
            "next_action": "keep intrinsic text and downstream fusion claims separate",
        },
        {
            "component": "Track C regime and SHAP",
            "evidence_status": "registered post-hoc robustness; Holm families complete",
            "allowed_claim": "pipeline response differs by architecture and comparator",
            "blocked_claim": "three improved models prove SHAP or regime superiority",
            "next_action": "report descriptive and isolated contrasts separately",
        },
        {
            "component": "Track C grouped LIME",
            "evidence_status": "integrity-passed audit with limited fidelity/stability",
            "allowed_claim": "LIME provides a qualified agreement audit",
            "blocked_claim": "LIME independently validates every SHAP explanation",
            "next_action": "retain low-fidelity failures and structural-close sensitivity",
        },
        {
            "component": "Track D partial-2026 forward",
            "evidence_status": "frozen model protocol with source/parser contingency",
            "allowed_claim": "forward evaluation exposed weak discrimination and collapse",
            "blocked_claim": "pristine confirmatory holdout or deployable profitability",
            "next_action": "treat economics and XAI rankings as exploratory",
        },
        {
            "component": "Shadow deployment",
            "evidence_status": "omitted by author decision",
            "allowed_claim": "none required",
            "blocked_claim": "live-market validation completed",
            "next_action": "no action; partial-2026 forward evidence is more informative than five days",
        },
        {
            "component": "Five-model improvement wording",
            "evidence_status": "metric- and comparator-dependent",
            "allowed_claim": "name the exact models, metric, comparator, and effect size",
            "blocked_claim": "the same three of five models improve throughout the pipeline",
            "next_action": "use Balanced Accuracy as primary directional decision metric",
        },
        {
            "component": "Manuscript",
            "evidence_status": "not started in this task",
            "allowed_claim": "evidence package is a pre-writing audit artifact",
            "blocked_claim": "submission-ready manuscript completed",
            "next_action": "finish evidence and claim audit before drafting",
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty evidence table: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _improved_models(
    rows: list[dict[str, Any]], comparison_id: str, metric: str
) -> list[str]:
    selected = {
        str(row["model"])
        for row in rows
        if row["comparison_id"] == comparison_id
        and row["metric"] == metric
        and row["improved"]
    }
    return [model for model in MODEL_ORDER if model in selected]


def _write_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        ("Track A +VMD, BAcc", "track_a_vmd_vs_full_ta", "balanced_accuracy"),
        ("Track B +News, BAcc", "track_b_news_vs_technical", "balanced_accuracy"),
        (
            "Track C Regime-SHAP vs Global-All, BAcc (descriptive)",
            "track_c_regime_shap_vs_global_all_descriptive",
            "balanced_accuracy",
        ),
        (
            "Track C Regime-SHAP vs Regime-All, BAcc (isolated)",
            "track_c_regime_shap_vs_regime_all",
            "balanced_accuracy",
        ),
        ("Track D Multitask vs Direct, BAcc", "track_d_multitask_vs_direct", "balanced_accuracy"),
        ("Track D Multitask vs Direct, DA", "track_d_multitask_vs_direct", "direction_accuracy"),
    ]
    lines = [
        "# Q2 Evidence Consolidation Log",
        "",
        "Status: **COMPLETE - NON-MANUSCRIPT AUDIT ARTIFACT**",
        "",
        "This task consolidates existing corrected Track A-D evidence and does not start manuscript writing.",
        "No model was retrained and no result was overwritten.",
        "",
        "## Scope decision",
        "",
        "- Shadow deployment: omitted. A five-day pilot would add little inferential evidence beyond the existing 138-day partial-2026 forward evaluation.",
        "- The five registered architectures and their frozen windows remain unchanged.",
        "- Balanced Accuracy is the primary directional improvement metric; DA and RMSE remain separate secondary endpoints.",
        "",
        "## Does the same three models improve throughout?",
        "",
        "No. There is no stable same three models winner set across the pipeline. The set changes with the intervention, comparator, and metric:",
        "",
        "| Comparison | Improved models | Count |",
        "|---|---|---:|",
    ]
    for label, comparison_id, metric in sections:
        models = _improved_models(rows, comparison_id, metric)
        rendered = ", ".join(models) if models else "None"
        lines.append(f"| {label} | {rendered} | {len(models)}/5 |")
    lines.extend(
        [
            "",
            "A positive DA change in Track D is not accepted as improvement when BAcc remains flat or worsens, because several outputs collapse toward the majority Up class.",
            "",
            "## Evidence classification",
            "",
            "- Track A: corrected confirmatory paired ablation with four temporal units and low power.",
            "- Track B fusion: corrected paired ablation; intrinsic text/LLM evidence remains separate from downstream forecasting.",
            "- Track C: post-hoc robustness evidence. Capacity-confounded end-to-end comparisons are descriptive; registered within-router and capacity-matched contrasts carry Holm-adjusted inference.",
            "- Track D: frozen model protocol on a source-contingency partial-2026 forward set; not a pristine registered-source confirmatory holdout.",
            "",
            "## Claim rule frozen for the next stage",
            "",
            "Every improvement statement must name: model, exact comparator, metric, effect size, evidence class, and multiplicity status. Counts such as '3 of 5 improved' cannot stand alone.",
            "",
            "## Generated artifacts",
            "",
            "- `outputs/q2_evidence_package/master_evidence_matrix.csv`",
            "- `outputs/q2_evidence_package/q2_claim_status.csv`",
            "- `outputs/q2_evidence_package/source_manifest.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evidence_package(
    *,
    repo_root: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    destination = Path(output_dir)
    rows = build_improvement_rows(root)
    claims = _claim_rows()

    matrix_path = destination / "master_evidence_matrix.csv"
    claim_path = destination / "q2_claim_status.csv"
    manifest_path = destination / "source_manifest.json"
    _write_csv(matrix_path, rows)
    _write_csv(claim_path, claims)
    _write_log(Path(log_path), rows)

    sources = []
    for name, relative_path in SOURCE_PATHS.items():
        path = root / relative_path
        sources.append(
            {
                "name": name,
                "path": relative_path.as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "package_type": "non_manuscript_q2_evidence_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_registry": list(MODEL_ORDER),
        "row_count": len(rows),
        "source_artifacts": sources,
        "generated_artifacts": [
            {
                "path": matrix_path.name,
                "sha256": _sha256(matrix_path),
                "rows": len(rows),
            },
            {
                "path": claim_path.name,
                "sha256": _sha256(claim_path),
                "rows": len(claims),
            },
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "row_count": len(rows),
        "claim_count": len(claims),
        "output_dir": str(destination),
        "log_path": str(log_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the non-manuscript Q2 evidence and claim-audit package."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/q2_evidence_package"),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("test/q2_evidence_consolidation_log.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_evidence_package(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        log_path=args.log_path,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
