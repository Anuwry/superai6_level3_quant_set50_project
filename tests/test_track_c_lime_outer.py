from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.track_c_lime_outer import (
    build_audit_instance_id,
    summarize_agreement,
    validate_outer_lime_gate,
)
from models.track_c_lime_outer_runner import (
    _original_change_from_scaled_prediction,
)


def test_original_change_uses_the_same_inverse_scaling_as_outer_inference():
    scaler = {
        "columns": ["Close_D", "Target_Next_Close"],
        "scale": [0.1, 0.25],
        "min": [0.0, -2.0],
    }

    result = _original_change_from_scaled_prediction(
        np.array([3.0, 3.5]),
        scaler=scaler,
        current_close=np.array([19.0, 20.0]),
    )

    assert result.tolist() == pytest.approx([1.0, 2.0])


def test_outer_lime_gate_accepts_only_completed_outer_run():
    validate_outer_lime_gate(
        {
            "outer_results_generated": True,
            "models_completed": [
                "lstm",
                "cnn",
                "lstm_cnn",
                "lstm_attention",
                "lstm_cnn_attention",
            ],
            "folds_completed": ["fold_1", "fold_2", "fold_3", "fold_4"],
            "seeds_completed": [42, 123, 456, 789, 2025],
        }
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("outer_results_generated", False),
        ("models_completed", ["lstm"]),
        ("folds_completed", ["fold_1"]),
        ("seeds_completed", [42]),
    ],
)
def test_outer_lime_gate_rejects_partial_outer_run(field, value):
    metadata = {
        "outer_results_generated": True,
        "models_completed": [
            "lstm",
            "cnn",
            "lstm_cnn",
            "lstm_attention",
            "lstm_cnn_attention",
        ],
        "folds_completed": ["fold_1", "fold_2", "fold_3", "fold_4"],
        "seeds_completed": [42, 123, 456, 789, 2025],
    }
    metadata[field] = value

    with pytest.raises(ValueError, match="complete"):
        validate_outer_lime_gate(metadata)


def test_audit_instance_id_is_stable_and_unambiguous():
    first = build_audit_instance_id(
        "lstm",
        "fold_1",
        17,
        "2022-03-14",
    )
    second = build_audit_instance_id(
        "lstm",
        "fold_1",
        17,
        "2022-03-14",
    )

    assert first == second
    assert first.startswith("lstm__fold_1__")
    assert first != build_audit_instance_id(
        "lstm",
        "fold_1",
        18,
        "2022-03-14",
    )


def test_agreement_summary_excludes_low_fidelity_rows_from_agreement():
    rows = pd.DataFrame(
        {
            "model": ["lstm"] * 4,
            "regime": ["Bull"] * 4,
            "fidelity_r2": [0.90, 0.80, 0.20, -0.10],
            "low_fidelity": [False, False, True, True],
            "spearman_abs": [0.80, 0.60, -1.00, -1.00],
            "top_k_jaccard": [0.50, 0.30, 0.00, 0.00],
            "sign_agreement_nonzero": [0.70, 0.50, 0.00, 0.00],
            "lime_runtime_seconds": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = summarize_agreement(rows)

    assert len(result) == 1
    summary = result.iloc[0]
    assert summary["audit_repeats"] == 4
    assert summary["reliable_repeats"] == 2
    assert summary["low_fidelity_repeats"] == 2
    assert summary["spearman_abs_median_reliable"] == pytest.approx(0.70)
    assert summary["top_k_jaccard_median_reliable"] == pytest.approx(0.40)
    assert summary["sign_agreement_median_reliable"] == pytest.approx(0.60)
    assert summary["fidelity_r2_median_all"] == pytest.approx(0.50)
    assert summary["lime_runtime_seconds_total"] == pytest.approx(10.0)


def test_agreement_summary_reports_nan_when_no_repeat_is_reliable():
    rows = pd.DataFrame(
        {
            "model": ["cnn"],
            "regime": ["Bear"],
            "fidelity_r2": [0.1],
            "low_fidelity": [True],
            "spearman_abs": [0.9],
            "top_k_jaccard": [0.8],
            "sign_agreement_nonzero": [0.7],
            "lime_runtime_seconds": [1.5],
        }
    )

    summary = summarize_agreement(rows).iloc[0]

    assert summary["reliable_repeats"] == 0
    assert pd.isna(summary["spearman_abs_median_reliable"])
    assert pd.isna(summary["top_k_jaccard_median_reliable"])
