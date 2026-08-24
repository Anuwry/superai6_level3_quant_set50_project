from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONTROLLING_DOCUMENTS = (
    Path("ADVISOR_PIPELINE_JOURNAL_Q_ASSESSMENT.md"),
    Path("PUBLIC_REPLICATION_PACKAGE.md"),
    Path("test/manuscript_reporting_lock_v1.md"),
    Path("test/market_data_release_status_v3.md"),
    Path("test/strong_q2_claims_register_v3.md"),
    Path("test/reliability_hardening_execution_log_v2.md"),
    Path("test/set100_same_exchange_robustness_execution_log_v1.md"),
)

STALE_ACTIVE_CLAIMS = (
    "public submission held for private evidence",
    "pending private licence evidence",
    "private licence-register evidence",
    "private license-evidence register",
    "confidential institutional entitlement evidence",
    "raw licensed records",
    "licensed local data",
    "240-file package",
    "containing 240 code",
    "release/public_replication_package_v1/",
)


def test_controlling_documents_do_not_retain_stale_access_or_package_claims() -> None:
    for relative_path in CONTROLLING_DOCUMENTS:
        content = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8").lower()
        stale = [claim for claim in STALE_ACTIVE_CLAIMS if claim in content]
        assert stale == [], f"{relative_path} contains stale claims: {stale}"


def test_controlling_documents_preserve_public_access_claim_boundary() -> None:
    required = (
        Path("test/manuscript_reporting_lock_v1.md"),
        Path("test/market_data_release_status_v3.md"),
        Path("PUBLIC_REPLICATION_PACKAGE.md"),
    )
    for relative_path in required:
        content = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8").lower()
        assert "publicly accessible" in content
        assert "provider" in content
        assert "terms" in content
        assert "raw" in content and "not" in content


def test_paper_environment_declares_full_test_dependencies() -> None:
    requirements = (PROJECT_ROOT / "requirements-paper.txt").read_text(
        encoding="utf-8"
    )
    assert "matplotlib==" in requirements
    assert "pytest-cov==" in requirements


def test_public_release_directories_are_ignored_generically() -> None:
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "release/public_replication_package_*/" in ignore
    assert ".pytest_tmp/" in ignore


def test_recorded_runtime_environments_have_explicit_lock_files() -> None:
    py312 = (PROJECT_ROOT / "requirements-paper-py312.txt").read_text(
        encoding="utf-8"
    )
    py311 = (PROJECT_ROOT / "requirements-integrated-py311.txt").read_text(
        encoding="utf-8"
    )
    assert "numpy==2.4.3" in py312
    assert "scikit-learn==1.8.0" in py312
    assert "numpy==2.1.3" in py311
    assert "pandas==2.3.3" in py311
    assert "scikit-learn==1.7.2" in py311
    assert "scipy==1.16.3" in py311
    assert "tensorflow==2.21.0" in py311


def test_reproduction_guide_explains_environment_selection() -> None:
    guide = (PROJECT_ROOT / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
    assert "requirements-paper-py312.txt" in guide
    assert "requirements-integrated-py311.txt" in guide
    assert "388" not in guide  # Never hard-code a stale test-count claim.
    pytest_config = (PROJECT_ROOT / "pytest.ini").read_text(encoding="utf-8")
    assert "--basetemp=.pytest_tmp" in pytest_config


def test_current_public_package_v3_passed_independent_audit() -> None:
    gate_path = PROJECT_ROOT / "outputs/market_data_governance_v1/release_gates_v3.json"
    audit_path = PROJECT_ROOT / "outputs/public_replication_package_v3_audit.json"
    if not gate_path.is_file() or not audit_path.is_file():
        pytest.skip("Post-build release audit is external to the public package")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert gate["clean_package_protocol"] == "public-replication-package-v3"
    assert audit["passed"] is True
    assert audit["verified_files"] == gate["clean_package_files"]
    assert audit["package_digest_matches"] is True
