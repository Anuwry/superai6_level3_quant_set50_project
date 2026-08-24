from __future__ import annotations

import numpy as np
import pandas as pd

from models.pit_fcg_runner import (
    VARIANTS,
    classification_metrics,
    evaluate_inner_promotion,
    load_development_frame,
)


def test_classification_metrics_use_fixed_half_threshold_with_ties_down() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.5, 0.6, 0.9])

    result = classification_metrics(labels, probabilities)

    assert result["observations"] == 4
    assert result["balanced_accuracy"] == 1.0
    assert result["direction_accuracy"] == 1.0
    assert result["mcc"] == 1.0
    assert result["binary_crossentropy"] > 0.0
    assert result["brier_score"] > 0.0


def test_promotion_requires_all_registered_conditions() -> None:
    summary = pd.DataFrame(
        {
            "variant": VARIANTS,
            "balanced_accuracy_mean": [0.51, 0.52, 0.53, 0.54, 0.55],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "variant": ["matched_control_fcg", "matched_control_fcg"],
            "inner_fold": ["inner_2020", "inner_2021"],
            "aligned_balanced_accuracy": [0.55, 0.56],
            "placebo_balanced_accuracy": [0.50, 0.51],
            "aligned_gate_median": [0.4, 0.6],
        }
    )

    passed = evaluate_inner_promotion(
        summary,
        diagnostics,
        parameter_increase_fraction=0.05,
        integrity_passed=True,
    )
    tied = summary.copy()
    tied.loc[
        tied["variant"].eq("direct_numeric_lstm"),
        "balanced_accuracy_mean",
    ] = 0.55
    failed = evaluate_inner_promotion(
        tied,
        diagnostics,
        parameter_increase_fraction=0.05,
        integrity_passed=True,
    )

    assert passed["passed"] is True
    assert all(passed["conditions"].values())
    assert failed["passed"] is False
    assert failed["conditions"]["beats_required_inner_baselines"] is False


def test_load_development_frame_is_locked_to_2019_2021_and_122_numeric_features() -> None:
    frame, numeric_features, audit = load_development_frame()

    assert len(frame) == 727
    assert frame["Date"].dt.year.unique().tolist() == [2019, 2020, 2021]
    assert len(numeric_features) == 122
    assert len(set(numeric_features)) == 122
    assert audit["passed"] is True
    assert np.isfinite(
        frame.loc[:, [*numeric_features, "prob_bull", "prob_sideway", "prob_bear"]]
        .to_numpy(dtype=float)
    ).all()

