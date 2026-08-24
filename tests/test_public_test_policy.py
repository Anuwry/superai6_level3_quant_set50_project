from __future__ import annotations

from pathlib import Path

from conftest import PRIVATE_ARTIFACT_REQUIREMENTS, missing_private_artifacts


def test_private_artifact_contract_is_explicit_and_path_safe(tmp_path: Path) -> None:
    assert len(PRIVATE_ARTIFACT_REQUIREMENTS) == 14

    for nodeid, requirements in PRIVATE_ARTIFACT_REQUIREMENTS.items():
        assert nodeid.startswith("tests/")
        assert requirements
        assert all(not path.is_absolute() and ".." not in path.parts for path in requirements)
        assert missing_private_artifacts(nodeid, tmp_path) == requirements


def test_ordinary_unit_test_has_no_private_artifact_requirement(tmp_path: Path) -> None:
    assert missing_private_artifacts("tests/test_example.py::test_unit", tmp_path) == ()
