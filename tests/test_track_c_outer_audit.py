from __future__ import annotations

from models.track_c_outer_audit import audit_outer_artifacts


def test_completed_outer_artifacts_pass_integrity_audit():
    result = audit_outer_artifacts(write_output=False)

    assert result["status"] == "passed"
    assert result["complete_cells"] == 100
    assert result["metric_rows"] == 700
    assert result["prediction_files"] == 700
    assert result["fit_registry_rows"] == 1100
    assert result["paired_fold_contrast_rows"] == 100
    assert result["fold_inference_rows"] == 125
    assert result["block_bootstrap_rows"] == 50
    assert result["freeze_hashes_match"] is True
