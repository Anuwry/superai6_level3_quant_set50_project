from __future__ import annotations

from models.track_c_lime_integrity import audit_lime_artifacts


def test_completed_lime_artifacts_pass_integrity_audit():
    result = audit_lime_artifacts(write_output=False)

    assert result["status"] == "passed"
    assert result["complete_cells"] == 20
    assert result["audit_instances"] == 360
    assert result["local_explanation_rows"] == 263520
    assert result["agreement_rows"] == 1800
    assert result["low_fidelity_rows"] == 1293
    assert result["outer_reproduction_gate_passed"] is True
    assert result["implementation_hashes_match"] is True
