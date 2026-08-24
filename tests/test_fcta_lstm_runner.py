from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.fcta_lstm_runner import (
    FCTAPreparedFold,
    attention_fidelity_metrics,
    probability_from_scaled_prediction,
    promotion_decision,
    run_cell,
)


def _prepared_fold() -> FCTAPreparedFold:
    rng = np.random.default_rng(7)
    train_count = 20
    test_count = 4
    train_current = rng.uniform(0.3, 0.7, size=train_count)
    train_target = train_current + rng.normal(scale=0.02, size=train_count)
    test_current = rng.uniform(0.3, 0.7, size=test_count)
    test_target = test_current + np.array([0.02, -0.01, 0.03, -0.02])
    return FCTAPreparedFold(
        fold="toy_fold",
        test_year=2024,
        feature_columns=tuple(f"x_{index}" for index in range(7)),
        train_sequence=rng.normal(size=(train_count, 5, 7)).astype(np.float32),
        train_target_scaled=train_target.astype(np.float32),
        train_current_scaled=train_current.astype(np.float32),
        train_dates=pd.date_range("2023-01-01", periods=train_count).to_numpy(),
        test_sequence=rng.normal(size=(test_count, 5, 7)).astype(np.float32),
        test_target_scaled=test_target.astype(np.float32),
        test_current_scaled=test_current.astype(np.float32),
        test_regimes=np.asarray(["bull", "bear", "sideway", "bull"], dtype=object),
        test_dates=pd.date_range("2024-01-02", periods=test_count).to_numpy(),
        test_close=np.full(test_count, 100.0),
        test_next_close=np.array([101.0, 99.0, 102.0, 98.0]),
        scaler_metadata={
            "columns": ["Target_Next_Close"],
            "scale": [0.01],
            "min": [-9.0],
        },
        mask_counts={"bull": 38, "sideway": 130, "bear": 88},
    )


def test_probability_from_scaled_prediction_uses_current_close_threshold() -> None:
    prediction = np.array([0.6, 0.4, 0.5])
    current = np.array([0.5, 0.5, 0.5])
    probability = probability_from_scaled_prediction(
        prediction,
        current,
        temperature=0.02,
    )

    assert probability[0] > 0.5
    assert probability[1] < 0.5
    assert probability[2] == 0.5


def test_attention_fidelity_rewards_matching_counterfactual_importance() -> None:
    attention = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    importance = attention.copy()
    matched = attention_fidelity_metrics(attention, importance)
    reversed_metrics = attention_fidelity_metrics(attention, importance[:, ::-1])

    assert matched["attention_counterfactual_jsd"] < reversed_metrics[
        "attention_counterfactual_jsd"
    ]
    assert matched["top1_deletion_agreement"] == 1.0


def test_promotion_requires_prediction_and_fidelity_conditions() -> None:
    ablation = pd.DataFrame(
        {
            "variant": [
                "attention_control",
                "direction_consistency",
                "mask_augmentation",
                "fcta_lstm",
            ],
            "balanced_accuracy_mean": [0.52, 0.53, 0.525, 0.56],
            "attention_counterfactual_jsd_mean": [0.20, 0.19, 0.18, 0.10],
        }
    )
    annual = pd.DataFrame(
        {
            "variant": ["attention_control", "fcta_lstm"] * 2,
            "test_year": [2024, 2024, 2025, 2025],
            "balanced_accuracy": [0.52, 0.55, 0.51, 0.57],
        }
    )
    frozen = pd.DataFrame(
        {"model": ["LSTM-Attention"], "balanced_accuracy_mean": [0.54]}
    )

    decision = promotion_decision(ablation, annual, frozen, test_years=(2024, 2025))

    assert decision["promoted"] is True
    assert all(decision["conditions"].values())


def test_run_cell_writes_predictions_attention_and_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import models.fcta_lstm_runner as runner

    monkeypatch.setattr(runner, "EPOCHS", 1)
    monkeypatch.setattr(runner, "BATCH_SIZE", 8)
    monkeypatch.setattr(runner, "SEEDS", (42,))

    result = run_cell(
        _prepared_fold(),
        seed=42,
        variant="fcta_lstm",
        output_dir=tmp_path,
    )
    directory = tmp_path / "cells" / "2024" / "seed_42" / "fcta_lstm"

    assert result["status"] == "completed"
    assert (directory / "predictions.csv").is_file()
    assert (directory / "temporal_explanations.csv").is_file()
    assert (directory / "metrics.json").is_file()
    assert (directory / "run_metadata.json").is_file()
    assert (directory / "inference.weights.h5").is_file()
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["inference_parameters"] > 0
    assert np.isfinite(metrics["attention_counterfactual_jsd"])
