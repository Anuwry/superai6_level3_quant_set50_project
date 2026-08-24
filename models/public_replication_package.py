from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

ALLOWED_SUFFIXES = {
    ".csv",
    ".ini",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOWED_FILENAMES = {".gitignore"}
RESTRICTED_PARTS = {
    ".git",
    "__pycache__",
    "data-folds",
    "data-prepared",
    "data-raw",
    "private",
    "runtime_cache",
    "runtime_tmp",
    "set100_data",
    "tmp",
}
RESTRICTED_FILENAMES = {
    ".env",
    "calls.jsonl",
    "errors.jsonl",
    "key.md",
}
SECRET_PATTERNS = {
    "openai_project_key": re.compile(
        r"(?<![A-Za-z0-9])sk-proj-[A-Za-z0-9_-]{16,}"
    ),
    "openai_legacy_key": re.compile(
        r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{32,}"
    ),
}

DEFAULT_ROOT_FILES = (
    Path(".gitignore"),
    Path("README.md"),
    Path("ADVISOR_PIPELINE_JOURNAL_Q_ASSESSMENT.md"),
    Path("JOURNAL_SCOPE_AND_SUBMISSION_STRATEGY.md"),
    Path("PUBLIC_REPLICATION_PACKAGE.md"),
    Path("REPRODUCIBILITY.md"),
    Path("data_feature_pool.md"),
    Path("pyrightconfig.json"),
    Path("pytest.ini"),
)
DEFAULT_OUTPUT_FILES = tuple(
    Path(value)
    for value in (
        "outputs/market_data_governance_v1/anomaly_summary.csv",
        "outputs/market_data_governance_v1/correction_ledger_schema.csv",
        "outputs/market_data_governance_v1/data_dictionary.csv",
        "outputs/market_data_governance_v1/integrity_checks.csv",
        "outputs/market_data_governance_v1/license_register.csv",
        "outputs/market_data_governance_v1/market_data_manifest.csv",
        "outputs/market_data_governance_v1/market_data_manifest.json",
        "outputs/market_data_governance_v1/paper_data_statements.md",
        "outputs/market_data_governance_v1/release_gates.json",
        "outputs/market_data_governance_v1/release_gates_v2.json",
        "outputs/market_data_governance_v1/repository_distribution_audit.csv",
        "outputs/set100_same_exchange_robustness_v1/benchmark_completion.json",
        "outputs/set100_same_exchange_robustness_v1/feature_integrity_audit.json",
        "outputs/set100_same_exchange_robustness_v1/generated_input_manifest.json",
        "outputs/set100_same_exchange_robustness_v1/integrity_audit.json",
        "outputs/set100_same_exchange_robustness_v1/market_deltas_by_fold.csv",
        "outputs/set100_same_exchange_robustness_v1/market_inference.csv",
        "outputs/set100_same_exchange_robustness_v1/market_inference_holm.csv",
        "outputs/set100_same_exchange_robustness_v1/model_market_summary.csv",
        "outputs/set100_same_exchange_robustness_v1/output_manifest.json",
        "outputs/set100_same_exchange_robustness_v1/paper_table.csv",
        "outputs/set100_same_exchange_robustness_v1/per_seed_metrics.csv",
        "outputs/set100_same_exchange_robustness_v1/runtime_summary.csv",
        "outputs/set100_same_exchange_robustness_v1/seed_averaged_fold_metrics.csv",
        "outputs/integrated_multimodal_posthoc_v1/arm_summary.csv",
        "outputs/integrated_multimodal_posthoc_v1/daily_block_bootstrap_holm.csv",
        "outputs/integrated_multimodal_posthoc_v1/fit_registry.csv",
        "outputs/integrated_multimodal_posthoc_v1/fold_inference_holm.csv",
        "outputs/integrated_multimodal_posthoc_v1/fold_metrics_seed_averaged.csv",
        "outputs/integrated_multimodal_posthoc_v1/integrity_audit.json",
        "outputs/integrated_multimodal_posthoc_v1/metrics_by_seed_fold.csv",
        "outputs/integrated_multimodal_posthoc_v1/paired_fold_contrasts.csv",
        "outputs/integrated_multimodal_posthoc_v1/paper_integrated_table.csv",
        "outputs/integrated_multimodal_posthoc_v1/run_metadata.json",
        "outputs/integrated_multimodal_posthoc_v1/runtime_by_cell.csv",
        "outputs/integrated_multimodal_posthoc_v1/runtime_summary.csv",
        "outputs/track_b/llm/compute_matched_v1/checkpoint_audit.json",
        "outputs/track_b/llm/compute_matched_v1/cohort_integrity_audit.json",
        "outputs/track_b/llm/compute_matched_v1/metrics_by_arm.csv",
        "outputs/track_b/llm/compute_matched_v1/output_manifest.json",
        "outputs/track_b/llm/compute_matched_v1/paired_comparisons.csv",
        "outputs/track_b/llm/compute_matched_v1/paired_comparisons_holm.csv",
        "outputs/track_b/llm/compute_matched_v1/paper_compute_matched_table.csv",
        "outputs/track_b/llm/compute_matched_v1/run_metadata.json",
        "outputs/track_b/llm/compute_matched_v1/runtime_cost_summary.csv",
        "outputs/multimodal_falsification_v1/arm_summary.csv",
        "outputs/multimodal_falsification_v1/daily_block_bootstrap_holm.csv",
        "outputs/multimodal_falsification_v1/fit_registry.csv",
        "outputs/multimodal_falsification_v1/fold_inference_holm.csv",
        "outputs/multimodal_falsification_v1/fold_metrics_seed_averaged.csv",
        "outputs/multimodal_falsification_v1/integrity_audit.json",
        "outputs/multimodal_falsification_v1/metrics_by_seed_fold.csv",
        "outputs/multimodal_falsification_v1/paired_fold_contrasts.csv",
        "outputs/multimodal_falsification_v1/paper_falsification_table.csv",
        "outputs/multimodal_falsification_v1/quarterly_origin_contrasts.csv",
        "outputs/multimodal_falsification_v1/quarterly_origin_summary.csv",
        "outputs/multimodal_falsification_v1/run_metadata.json",
        "outputs/multimodal_falsification_v1/runtime_summary.csv",
        "outputs/manuscript_tables_v1/MANIFEST.json",
        "outputs/manuscript_tables_v1/claim_hierarchy.json",
        "outputs/manuscript_tables_v1/supplement_economic_exploratory.csv",
        "outputs/manuscript_tables_v1/supplement_lime_diagnostic.csv",
        "outputs/manuscript_tables_v1/table_1_protocol_cohort.csv",
        "outputs/manuscript_tables_v1/table_2_numerical_ablation.csv",
        "outputs/manuscript_tables_v1/table_3a_multimodal_falsification.csv",
        "outputs/manuscript_tables_v1/table_3b_llm_intrinsic_separate.csv",
        "outputs/manuscript_tables_v1/table_4_regime_shap.csv",
        "outputs/manuscript_tables_v1/table_5a_forward_robustness.csv",
        "outputs/manuscript_tables_v1/table_5b_set100_transfer.csv",
        "outputs/manuscript_tables_v1/table_index.csv",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_public_relative_path(relative_path: Path) -> None:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Public source must be a safe relative path: {relative_path}")
    lowered_parts = {part.lower() for part in relative_path.parts}
    if lowered_parts.intersection(RESTRICTED_PARTS):
        raise ValueError(f"Restricted path cannot enter public package: {relative_path}")
    filename = relative_path.name.lower()
    if filename in RESTRICTED_FILENAMES or filename.startswith("predictions"):
        raise ValueError(f"Restricted file cannot enter public package: {relative_path}")
    if (
        relative_path.suffix.lower() not in ALLOWED_SUFFIXES
        and filename not in ALLOWED_FILENAMES
    ):
        raise ValueError(f"Unsupported public artifact type: {relative_path}")


def scan_file_for_secrets(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8", errors="replace")
    return [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(content)]


def collect_default_public_sources(
    project_root: Path,
    *,
    required_root_files: tuple[Path, ...] = DEFAULT_ROOT_FILES,
    required_output_files: tuple[Path, ...] = DEFAULT_OUTPUT_FILES,
) -> list[Path]:
    project_root = project_root.resolve()
    sources: set[Path] = set(required_root_files).union(required_output_files)
    for directory in ("models", "scripts", "tests"):
        root = project_root / directory
        if root.is_dir():
            sources.update(path.relative_to(project_root) for path in root.rglob("*.py"))
    protocol_root = project_root / "test"
    if protocol_root.is_dir():
        sources.update(
            path.relative_to(project_root) for path in protocol_root.glob("*.md")
        )
        for pattern in ("*freeze*.json", "*manifest*.json"):
            sources.update(
                path.relative_to(project_root) for path in protocol_root.glob(pattern)
            )
    sources.update(
        path.relative_to(project_root) for path in project_root.glob("requirements*.txt")
    )
    for relative_path in sources:
        validate_public_relative_path(relative_path)
        if not (project_root / relative_path).is_file():
            raise FileNotFoundError(f"Required public source is missing: {relative_path}")
    return sorted(sources, key=lambda value: value.as_posix())


def _manifest_entry(project_root: Path, relative_path: Path) -> dict[str, object]:
    source = project_root / relative_path
    return {
        "path": relative_path.as_posix(),
        "bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }


def _package_digest(entries: list[dict[str, object]]) -> str:
    payload = "\n".join(
        f"{entry['path']}:{entry['sha256']}:{entry['bytes']}" for entry in entries
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_public_replication_package(
    project_root: Path,
    destination: Path,
    relative_sources: Iterable[Path],
) -> dict[str, object]:
    project_root = project_root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Public package destination already exists: {destination}")

    sources = sorted(set(relative_sources), key=lambda value: value.as_posix())
    if not sources:
        raise ValueError("At least one public source is required")
    entries: list[dict[str, object]] = []
    for relative_path in sources:
        validate_public_relative_path(relative_path)
        source = (project_root / relative_path).resolve()
        if not source.is_relative_to(project_root):
            raise ValueError(f"Public source escapes project root: {relative_path}")
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"Public source is missing or unsafe: {relative_path}")
        findings = scan_file_for_secrets(source)
        if findings:
            raise ValueError(
                f"Public source contains secret-like material ({', '.join(findings)}): "
                f"{relative_path}"
            )
        entries.append(_manifest_entry(project_root, relative_path))

    for relative_path in sources:
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root / relative_path, target)

    manifest = {
        "protocol_id": "public-replication-package-v3",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "files": entries,
        "file_count": len(entries),
        "package_sha256": _package_digest(entries),
        "restricted_files": 0,
        "secret_findings": 0,
        "raw_market_data_included": False,
        "row_level_predictions_included": False,
        "private_llm_checkpoints_included": False,
        "license_policy": (
            "Code, schemas, checksums, protocols, and non-reconstructive aggregate "
            "results only; provider-hosted row-level market data are excluded and "
            "remain subject to the provider's terms of use."
        ),
    }
    (destination / "PUBLIC_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "passed": True,
        "files": len(entries),
        "package_sha256": manifest["package_sha256"],
        "destination": str(destination),
    }


def audit_public_replication_package(package_dir: Path) -> dict[str, object]:
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "PUBLIC_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Public manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise TypeError("Public manifest files field must be a list")

    expected: set[str] = set()
    missing: list[str] = []
    unexpected: list[str] = []
    hash_mismatches: list[str] = []
    size_mismatches: list[str] = []
    secret_findings: list[dict[str, object]] = []
    validated_entries: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("Public manifest contains a non-object file entry")
        relative = Path(str(entry.get("path", "")))
        validate_public_relative_path(relative)
        relative_posix = relative.as_posix()
        expected.add(relative_posix)
        path = (package_dir / relative).resolve()
        if not path.is_relative_to(package_dir) or not path.is_file() or path.is_symlink():
            missing.append(relative_posix)
            continue
        actual_size = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_size != entry.get("bytes"):
            size_mismatches.append(relative_posix)
        if actual_hash != entry.get("sha256"):
            hash_mismatches.append(relative_posix)
        findings = scan_file_for_secrets(path)
        if findings:
            secret_findings.append({"path": relative_posix, "patterns": findings})
        validated_entries.append(
            {"path": relative_posix, "bytes": actual_size, "sha256": actual_hash}
        )

    actual = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    unexpected = sorted(actual.difference(expected))
    package_digest_matches = (
        not missing
        and len(validated_entries) == len(entries)
        and _package_digest(validated_entries) == manifest.get("package_sha256")
    )
    passed = not any(
        (
            missing,
            unexpected,
            hash_mismatches,
            size_mismatches,
            secret_findings,
        )
    ) and bool(package_digest_matches)
    return {
        "protocol_id": "public-replication-package-v3-audit",
        "passed": passed,
        "source_protocol_id": manifest.get("protocol_id"),
        "verified_files": len(validated_entries),
        "missing_files": sorted(missing),
        "unexpected_files": unexpected,
        "hash_mismatches": sorted(hash_mismatches),
        "size_mismatches": sorted(size_mismatches),
        "secret_findings": secret_findings,
        "package_digest_matches": package_digest_matches,
    }
