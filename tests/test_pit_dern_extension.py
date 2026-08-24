from __future__ import annotations

from pathlib import Path

import pandas as pd

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
                "model": "pit_dern",
                "arm": arm,
                "balanced_accuracy_mean": 0.55,
                "direction_accuracy_mean": 0.54,
                "mcc_mean": 0.08,
                "rmse_mean": 18.0,
                "mae_mean": 14.0,
                "temporal_folds": 4,
            }
            for arm in ARMS
        ]
    )


def test_six_model_table_appends_pit_dern_without_changing_five() -> None:
    from models.pit_dern_extension import build_six_model_tables

    all_arms, final_arm = build_six_model_tables(
        _frozen_summary(),
        _ours_summary(),
    )

    assert len(all_arms) == 24
    assert final_arm["model"].tolist() == [
        "lstm",
        "cnn",
        "lstm_cnn",
        "lstm_attention",
        "lstm_cnn_attention",
        "pit_dern",
    ]
    assert final_arm["evidence_status"].iloc[-1] == (
        "post_freeze_exploratory_architecture_extension"
    )
    assert final_arm["evidence_status"].iloc[0] == "frozen_existing_result"


def test_promotion_requires_best_model_and_mechanism_gates() -> None:
    from models.pit_dern_extension import evaluate_promotion_gates

    decision = evaluate_promotion_gates(
        pd.DataFrame(
            {
                "fold": ["fold_1", "fold_2", "fold_3", "fold_4"],
                "balanced_accuracy_delta_pp": [1.5, 1.2, 1.6, -0.1],
            }
        ),
        ours_bacc=0.55,
        encoder_bacc=0.54,
        shuffled_control_bacc=0.53,
        parameter_delta_fraction=0.10,
        complete_finite_predictions=True,
    )

    assert decision["positive_temporal_folds"] == 3
    assert decision["passed"] is True


def test_promotion_fails_if_retrieval_does_not_beat_encoder() -> None:
    from models.pit_dern_extension import evaluate_promotion_gates

    decision = evaluate_promotion_gates(
        pd.DataFrame(
            {
                "fold": ["fold_1", "fold_2", "fold_3", "fold_4"],
                "balanced_accuracy_delta_pp": [2.0, 2.0, 2.0, 2.0],
            }
        ),
        ours_bacc=0.54,
        encoder_bacc=0.54,
        shuffled_control_bacc=0.50,
        parameter_delta_fraction=0.10,
        complete_finite_predictions=True,
    )

    assert decision["retrieval_mechanism_gate"] is False
    assert decision["passed"] is False


def test_runner_contract_has_twenty_isolated_cells(tmp_path: Path) -> None:
    from models.pit_dern_runner import FOLDS, build_cell_commands

    commands = build_cell_commands(
        python_executable=Path("python.exe"),
        output_dir=tmp_path,
    )

    assert len(commands) == len(FOLDS) * len(FINAL_SEEDS) == 20
    assert all("models.pit_dern_runner" in command for command in commands)


def test_freeze_hashes_match_current_inputs() -> None:
    from models.pit_dern_runner import verify_extension_freeze

    audit = verify_extension_freeze()

    assert audit["passed"] is True
    assert audit["verified_inputs"] == 12
