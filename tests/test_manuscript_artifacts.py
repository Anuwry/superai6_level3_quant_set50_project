from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from models.manuscript_artifacts import (
    CLAIM_HIERARCHY,
    SOURCE_FILES,
    build_file_manifest,
    build_manuscript_artifacts,
    validate_manuscript_relative_path,
)


def test_claim_hierarchy_keeps_diagnostics_out_of_headline_results() -> None:
    assert CLAIM_HIERARCHY["balanced_accuracy"] == "primary"
    assert CLAIM_HIERARCHY["shap"] == "main_secondary"
    assert CLAIM_HIERARCHY["lime"] == "supplement_diagnostic"
    assert CLAIM_HIERARCHY["economic_proxy"] == "supplement_exploratory"
    assert CLAIM_HIERARCHY["llm_intrinsic"] == "main_separate_intrinsic"


@pytest.mark.parametrize(
    "relative_path",
    [
        "data-raw/SET50.csv",
        "outputs/example/predictions_seed_averaged.csv",
        "private/license.md",
        "key.md",
        ".env",
    ],
)
def test_manuscript_bundle_rejects_row_level_or_private_paths(
    relative_path: str,
) -> None:
    with pytest.raises(ValueError):
        validate_manuscript_relative_path(Path(relative_path))


def test_build_file_manifest_records_exact_hashes(tmp_path: Path) -> None:
    first = tmp_path / "outputs" / "table_a.csv"
    second = tmp_path / "test" / "protocol.md"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("model,value\nlstm,0.5\n", encoding="utf-8")
    second.write_text("# Frozen\n", encoding="utf-8")

    manifest = build_file_manifest(
        tmp_path,
        [Path("outputs/table_a.csv"), Path("test/protocol.md")],
    )

    assert [row["path"] for row in manifest] == [
        "outputs/table_a.csv",
        "test/protocol.md",
    ]
    assert manifest[0]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert manifest[0]["bytes"] == first.stat().st_size


def test_build_manuscript_artifacts_creates_locked_main_and_supplement_panels(
    tmp_path: Path,
) -> None:
    frames = {
        "windows": pd.DataFrame(
            [{"model": "lstm", "selected_sequence_window": 5}]
        ),
        "numerical": pd.DataFrame(
            [
                {
                    "model": "lstm",
                    "selected_sequence_window": 5,
                    "full_ta_balanced_accuracy_mean": 0.51,
                    "vmd_balanced_accuracy_mean": 0.52,
                    "balanced_accuracy_delta_pp": 1.0,
                    "balanced_accuracy_delta_pp_ci95_lower": -1.0,
                    "balanced_accuracy_delta_pp_ci95_upper": 3.0,
                    "balanced_accuracy_exact_sign_flip_pvalue": 0.5,
                    "full_ta_runtime_seconds_mean": 1.0,
                    "vmd_runtime_seconds_mean": 2.0,
                }
            ]
        ),
        "falsification": pd.DataFrame(
            [
                {
                    "model": "lstm",
                    "contrast": "observed_vs_shuffled",
                    "metric": "balanced_accuracy_delta_pp",
                    "point_estimate": 0.0,
                    "ci95_lower": -1.0,
                    "ci95_upper": 1.0,
                    "two_sided_pvalue": 1.0,
                    "two_sided_pvalue_holm": 1.0,
                    "daily_rows": 10,
                }
            ]
        ),
        "llm_intrinsic": pd.DataFrame(
            [
                {
                    "comparison_id": "leader_minus_control",
                    "pairs": 10,
                    "unique_articles": 8,
                    "control_accuracy_x": 0.5,
                    "leader_accuracy": 0.6,
                    "accuracy_delta_pp": 10.0,
                    "accuracy_delta_pp_ci95_lower": 1.0,
                    "accuracy_delta_pp_ci95_upper": 19.0,
                    "holm_adjusted_pvalue": 0.04,
                }
            ]
        ),
        "regime_shap": pd.DataFrame(
            [
                {
                    "inference_type": "four_fold_exact_sign_flip",
                    "model": "lstm",
                    "contrast": "global_shap_reduction",
                    "metric": "balanced_accuracy_delta_pp",
                    "point_estimate": 0.0,
                    "ci95_lower": -1.0,
                    "ci95_upper": 1.0,
                    "raw_pvalue": 1.0,
                    "holm_adjusted_pvalue": 1.0,
                }
            ]
        ),
        "forward": pd.DataFrame(
            [
                {
                    "model": "lstm",
                    "objective": "direct",
                    "rows": 10,
                    "direction_accuracy": 0.5,
                    "balanced_accuracy": 0.5,
                    "mcc": 0.0,
                    "auc": 0.5,
                    "brier": 0.25,
                }
            ]
        ),
        "set100": pd.DataFrame(
            [
                {
                    "model": "lstm",
                    "sequence_window": 5,
                    "balanced_accuracy_set50_mean": 0.52,
                    "balanced_accuracy_set100_mean": 0.51,
                    "balanced_accuracy_delta_pp": -1.0,
                    "balanced_accuracy_delta_ci95_lower": -2.0,
                    "balanced_accuracy_delta_ci95_upper": 0.0,
                    "balanced_accuracy_holm_pvalue": 1.0,
                }
            ]
        ),
        "lime": pd.DataFrame([{"model": "lstm", "fidelity_r2": 0.2}]),
        "economics": pd.DataFrame(
            [{"model": "lstm", "deflated_sharpe_probability": 0.4}]
        ),
    }
    for name, relative in SOURCE_FILES.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in frames:
            frames[name].to_csv(path, index=False)
        else:
            path.write_text("# Frozen protocol\n", encoding="utf-8")

    output = tmp_path / "outputs" / "manuscript_tables_v1"
    manifest = build_manuscript_artifacts(tmp_path, output)

    persisted = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["protocol_id"] == "manuscript-artifacts-v1"
    assert persisted["raw_market_data_included"] is False
    assert persisted["row_level_predictions_included"] is False
    assert len(persisted["generated"]) == 11
    assert (output / "table_3a_multimodal_falsification.csv").is_file()
    assert (output / "supplement_lime_diagnostic.csv").is_file()
    assert (output / "supplement_economic_exploratory.csv").is_file()
