from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models.integrated_multimodal import ARMS
from models.track_a_final import FINAL_SEEDS, TRACK_A_MODELS


def _frozen_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, model in enumerate(TRACK_A_MODELS):
        for arm in ARMS:
            rows.append(
                {
                    "model": model,
                    "arm": arm,
                    "balanced_accuracy_mean": 0.50 + index * 0.001,
                    "direction_accuracy_mean": 0.51,
                    "mcc_mean": 0.02,
                    "rmse_mean": 20.0,
                    "mae_mean": 15.0,
                    "temporal_folds": 4,
                }
            )
    return pd.DataFrame(rows)


def _ours_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "pit_cmm_lstm",
                "arm": arm,
                "balanced_accuracy_mean": 0.54,
                "direction_accuracy_mean": 0.53,
                "mcc_mean": 0.05,
                "rmse_mean": 18.0,
                "mae_mean": 14.0,
                "temporal_folds": 4,
            }
            for arm in ARMS
        ]
    )


def test_extension_keeps_registered_five_model_family_unchanged() -> None:
    from models.pit_cmm_extension import SIX_MODEL_ORDER

    assert tuple(TRACK_A_MODELS) == (
        "lstm",
        "cnn",
        "lstm_cnn",
        "lstm_attention",
        "lstm_cnn_attention",
    )
    assert SIX_MODEL_ORDER == (*tuple(TRACK_A_MODELS), "pit_cmm_lstm")
    assert FINAL_SEEDS == (42, 123, 456, 789, 2025)


def test_build_six_model_tables_appends_one_complete_exploratory_row() -> None:
    from models.pit_cmm_extension import (
        build_compact_six_model_comparison,
        build_six_model_tables,
    )

    all_arms, final_arm = build_six_model_tables(
        _frozen_summary(),
        _ours_summary(),
    )

    assert len(all_arms) == len(ARMS) * 6
    assert final_arm["model"].tolist() == [
        "lstm",
        "cnn",
        "lstm_cnn",
        "lstm_attention",
        "lstm_cnn_attention",
        "pit_cmm_lstm",
    ]
    assert final_arm["evidence_status"].iloc[-1] == (
        "post_freeze_exploratory_architecture_extension"
    )
    assert final_arm["evidence_status"].iloc[0] == "frozen_existing_result"

    compact = build_compact_six_model_comparison(all_arms)
    assert len(compact) == 6
    assert compact["model"].tolist()[-1] == "pit_cmm_lstm"
    assert compact["global_numeric_bacc_rank"].iloc[-1] == 1
    assert compact["final_integrated_bacc_rank"].iloc[-1] == 1


def test_build_six_model_tables_fails_closed_on_missing_arm() -> None:
    from models.pit_cmm_extension import build_six_model_tables

    incomplete = _ours_summary().iloc[:-1].copy()

    with pytest.raises(ValueError, match="exactly four arms"):
        build_six_model_tables(_frozen_summary(), incomplete)


def test_promotion_decision_uses_predeclared_lstm_contrast() -> None:
    from models.pit_cmm_extension import evaluate_promotion_gates

    fold_deltas = pd.DataFrame(
        {
            "fold": ["fold_1", "fold_2", "fold_3", "fold_4"],
            "balanced_accuracy_delta_pp": [1.4, 0.8, 1.2, -0.1],
        }
    )

    decision = evaluate_promotion_gates(
        fold_deltas,
        parameter_deltas=[0.10, 0.08, 0.12],
        complete_finite_predictions=True,
    )

    assert decision["mean_bacc_delta_pp"] == pytest.approx(0.825)
    assert decision["positive_temporal_folds"] == 3
    assert decision["passed"] is False


def test_extension_runner_contract_has_twenty_isolated_cells(tmp_path: Path) -> None:
    from models.pit_cmm_extension_runner import (
        FOLDS,
        OUTPUT_DIR,
        build_cell_commands,
    )

    commands = build_cell_commands(
        python_executable=Path("python.exe"),
        output_dir=tmp_path,
    )

    assert len(commands) == len(FOLDS) * len(FINAL_SEEDS) == 20
    assert all("models.pit_cmm_extension_runner" in command for command in commands)
    assert all("--output-dir" in command for command in commands)
    assert OUTPUT_DIR.name == "pit_cmm_lstm_extension_v1"


def test_extension_freeze_hashes_match_current_inputs() -> None:
    from models.pit_cmm_extension_runner import verify_extension_freeze

    audit = verify_extension_freeze()

    assert audit["passed"] is True
    assert audit["verified_inputs"] == 4


def test_lstm_fold_contrast_pairs_exactly_four_final_arm_rows() -> None:
    from models.pit_cmm_extension_runner import build_lstm_fold_contrast

    frozen = pd.DataFrame(
        {
            "model": ["lstm"] * 4,
            "fold": ["fold_1", "fold_2", "fold_3", "fold_4"],
            "arm": ["Regime-SHAP-Numeric-News"] * 4,
            "balanced_accuracy": [0.50, 0.51, 0.52, 0.53],
        }
    )
    ours = pd.DataFrame(
        {
            "model": ["pit_cmm_lstm"] * 4,
            "fold": ["fold_1", "fold_2", "fold_3", "fold_4"],
            "arm": ["Regime-SHAP-Numeric-News"] * 4,
            "balanced_accuracy": [0.52, 0.50, 0.55, 0.54],
        }
    )

    contrast = build_lstm_fold_contrast(frozen, ours)

    assert contrast["balanced_accuracy_delta_pp"].tolist() == pytest.approx(
        [2.0, -1.0, 3.0, 1.0]
    )
    assert contrast["fold"].tolist() == [
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
    ]
