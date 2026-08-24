from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from models.q2_evidence_package import (
    MODEL_ORDER,
    build_improvement_rows,
    write_evidence_package,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _select(
    rows: list[dict[str, object]],
    *,
    comparison_id: str,
    metric: str,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row["comparison_id"] == comparison_id and row["metric"] == metric
    ]


def test_build_improvement_rows_preserves_five_model_registry_and_orientation():
    rows = build_improvement_rows(REPO_ROOT)

    assert len(rows) == 90
    assert {row["model"] for row in rows} == set(MODEL_ORDER)

    track_a_bacc = _select(
        rows,
        comparison_id="track_a_vmd_vs_full_ta",
        metric="balanced_accuracy",
    )
    assert {row["model"] for row in track_a_bacc if row["improved"]} == {
        "lstm_cnn"
    }

    track_b_rmse = _select(
        rows,
        comparison_id="track_b_news_vs_technical",
        metric="rmse",
    )
    assert {row["model"] for row in track_b_rmse if row["improved"]} == {
        "lstm",
        "lstm_attention",
        "lstm_cnn_attention",
    }

    track_c_bacc = _select(
        rows,
        comparison_id="track_c_regime_shap_vs_global_all_descriptive",
        metric="balanced_accuracy",
    )
    assert {row["model"] for row in track_c_bacc if row["improved"]} == {
        "cnn",
        "lstm_cnn",
        "lstm_cnn_attention",
    }

    track_d_bacc = _select(
        rows,
        comparison_id="track_d_multitask_vs_direct",
        metric="balanced_accuracy",
    )
    assert {row["model"] for row in track_d_bacc if row["improved"]} == {
        "lstm_attention"
    }


def test_track_c_registered_inference_is_separate_from_descriptive_comparison():
    rows = build_improvement_rows(REPO_ROOT)

    registered = _select(
        rows,
        comparison_id="track_c_regime_shap_vs_regime_all",
        metric="balanced_accuracy",
    )
    assert len(registered) == 5
    assert all(row["evidence_class"] == "registered_post_hoc_robustness" for row in registered)
    assert all(row["holm_adjusted_pvalue"] is not None for row in registered)
    assert all(float(row["holm_adjusted_pvalue"]) >= 0.05 for row in registered)

    descriptive = _select(
        rows,
        comparison_id="track_c_regime_shap_vs_global_all_descriptive",
        metric="balanced_accuracy",
    )
    assert len(descriptive) == 5
    assert all(
        row["evidence_class"] == "descriptive_capacity_confounded"
        for row in descriptive
    )
    assert all(row["holm_adjusted_pvalue"] is None for row in descriptive)


def test_write_evidence_package_creates_traceable_non_manuscript_artifacts(tmp_path: Path):
    output_dir = tmp_path / "evidence"
    log_path = tmp_path / "q2_evidence_log.md"

    result = write_evidence_package(
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        log_path=log_path,
    )

    matrix_path = output_dir / "master_evidence_matrix.csv"
    status_path = output_dir / "q2_claim_status.csv"
    manifest_path = output_dir / "source_manifest.json"
    assert result["row_count"] == 90
    assert matrix_path.exists()
    assert status_path.exists()
    assert manifest_path.exists()
    assert log_path.exists()

    with matrix_path.open(encoding="utf-8-sig", newline="") as handle:
        matrix_rows = list(csv.DictReader(handle))
    assert len(matrix_rows) == 90

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["package_type"] == "non_manuscript_q2_evidence_audit"
    assert len(manifest["source_artifacts"]) >= 8
    assert all(len(item["sha256"]) == 64 for item in manifest["source_artifacts"])

    log_text = log_path.read_text(encoding="utf-8")
    assert "Shadow deployment: omitted" in log_text
    assert "does not start manuscript writing" in log_text
    assert "same three models" in log_text


def test_missing_repository_sources_fail_closed(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_improvement_rows(tmp_path)
