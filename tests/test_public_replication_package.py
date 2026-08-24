from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from models.public_replication_package import (
    audit_public_replication_package,
    build_public_replication_package,
    collect_default_public_sources,
    scan_file_for_secrets,
    validate_public_relative_path,
)


@pytest.mark.parametrize(
    "relative_path",
    [
        "key.md",
        ".env",
        "../outside.md",
        "data-raw/SET50.csv",
        "data-prepared/SET50.csv",
        "data-folds/fold_1/test.csv",
        "set100_data/SET100.csv",
        "outputs/example/private/calls.jsonl",
        "outputs/example/predictions_seed_averaged.csv",
        "outputs/tmp/pytest/artifact.txt",
    ],
)
def test_validate_public_relative_path_rejects_restricted_sources(
    relative_path: str,
) -> None:
    with pytest.raises(ValueError):
        validate_public_relative_path(Path(relative_path))


def test_validate_public_relative_path_accepts_code_protocol_and_aggregate() -> None:
    for relative_path in (
        ".gitignore",
        "models/example.py",
        "pytest.ini",
        "test/protocol_v1.md",
        "outputs/example/paper_table.csv",
    ):
        validate_public_relative_path(Path(relative_path))


def test_scan_file_for_secrets_rejects_openai_project_key(tmp_path: Path) -> None:
    candidate = tmp_path / "unsafe.md"
    token = "".join(("sk", "-proj-", "A" * 40))
    candidate.write_text(f"credential={token}\n", encoding="utf-8")

    findings = scan_file_for_secrets(candidate)

    assert findings == ["openai_project_key"]


def test_build_public_package_writes_audited_hash_manifest(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source_code = project / "models" / "example.py"
    aggregate = project / "outputs" / "example" / "paper_table.csv"
    source_code.parent.mkdir(parents=True)
    aggregate.parent.mkdir(parents=True)
    source_code.write_text("VALUE = 1\n", encoding="utf-8")
    aggregate.write_text("metric,value\naccuracy,0.5\n", encoding="utf-8")
    destination = project / "release" / "public_v1"

    result = build_public_replication_package(
        project,
        destination,
        [Path("models/example.py"), Path("outputs/example/paper_table.csv")],
    )

    manifest = json.loads(
        (destination / "PUBLIC_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert result["passed"] is True
    assert result["files"] == 2
    assert manifest["restricted_files"] == 0
    assert manifest["secret_findings"] == 0
    assert manifest["protocol_id"] == "public-replication-package-v3"
    assert [item["path"] for item in manifest["files"]] == [
        "models/example.py",
        "outputs/example/paper_table.csv",
    ]
    assert manifest["files"][0]["sha256"] == hashlib.sha256(
        source_code.read_bytes()
    ).hexdigest()


def test_build_public_package_fails_closed_on_secret(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "models" / "unsafe.py"
    source.parent.mkdir(parents=True)
    token = "".join(("sk", "-proj-", "B" * 40))
    source.write_text(f"TOKEN = '{token}'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="secret"):
        build_public_replication_package(
            project,
            project / "release" / "public_v1",
            [Path("models/unsafe.py")],
        )


def test_audit_public_package_verifies_exact_files_hashes_and_secrets(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = project / "models" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    destination = project / "release" / "public_v3"
    build_public_replication_package(project, destination, [Path("models/example.py")])

    audit = audit_public_replication_package(destination)

    assert audit["passed"] is True
    assert audit["protocol_id"] == "public-replication-package-v3-audit"
    assert audit["verified_files"] == 1
    assert audit["missing_files"] == []
    assert audit["unexpected_files"] == []
    assert audit["hash_mismatches"] == []
    assert audit["secret_findings"] == []


def test_audit_public_package_fails_on_tampering(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "models" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    destination = project / "release" / "public_v3"
    build_public_replication_package(project, destination, [Path("models/example.py")])
    (destination / "models" / "example.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )

    audit = audit_public_replication_package(destination)

    assert audit["passed"] is False
    assert audit["hash_mismatches"] == ["models/example.py"]


def test_build_public_package_refuses_existing_destination(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "models" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    destination = project / "release" / "public_v1"
    destination.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        build_public_replication_package(
            project,
            destination,
            [Path("models/example.py")],
        )


def test_collect_default_public_sources_uses_code_docs_and_exact_outputs(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    files = {
        "README.md": "readme\n",
        "requirements-paper.txt": "numpy\n",
        "models/example.py": "VALUE = 1\n",
        "scripts/run.py": "print('run')\n",
        "tests/test_example.py": "def test_ok(): assert True\n",
        "test/protocol_v1.md": "protocol\n",
        "test/example_freeze_v1.json": "{}\n",
        "outputs/example/paper_table.csv": "metric,value\naccuracy,0.5\n",
    }
    for relative, content in files.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    sources = collect_default_public_sources(
        project,
        required_root_files=(Path("README.md"),),
        required_output_files=(Path("outputs/example/paper_table.csv"),),
    )

    assert {path.as_posix() for path in sources} == set(files)


def test_collect_default_public_sources_accepts_reproducibility_configs(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    files = {
        ".gitignore": ".pytest_tmp/\n",
        "REPRODUCIBILITY.md": "reproduction instructions\n",
        "pytest.ini": "[pytest]\n",
        "pyrightconfig.json": "{}\n",
        "outputs/example/MANIFEST.json": "{}\n",
    }
    for relative, content in files.items():
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    sources = collect_default_public_sources(
        project,
        required_root_files=(
            Path(".gitignore"),
            Path("REPRODUCIBILITY.md"),
            Path("pytest.ini"),
            Path("pyrightconfig.json"),
        ),
        required_output_files=(Path("outputs/example/MANIFEST.json"),),
    )

    assert {path.as_posix() for path in sources} == set(files)
