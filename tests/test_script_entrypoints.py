from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    (
        "scripts/build_manuscript_artifacts.py",
        "scripts/build_public_replication_package.py",
        "scripts/audit_public_replication_package.py",
    ),
)
def test_documented_script_entrypoints_resolve_project_imports(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
