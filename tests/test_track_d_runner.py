from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.track_d_runner import (
    binary_probability_metrics,
    next_session_implementation_returns,
    run_forward,
    seed_average_predictions,
)


def test_binary_probability_metrics_report_direction_and_calibration():
    metrics = binary_probability_metrics(
        np.array([1.0, 1.0, 0.0, 0.0]),
        np.array([0.9, 0.6, 0.4, 0.2]),
    )

    assert metrics["direction_accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.0925)


def test_next_session_returns_use_next_open_not_signal_close():
    frame = pd.DataFrame(
        {
            "Date": ["2026-01-02", "2026-01-05", "2026-01-06"],
            "Open_D": [100.0, 102.0, 103.0],
            "Close_D": [101.0, 103.0, 104.0],
            "Target_Next_Close": [103.0, 104.0, 105.0],
        }
    )

    result = next_session_implementation_returns(frame)

    assert result["eligible"].tolist() == [True, True, False]
    assert result["next_open"].iloc[0] == pytest.approx(102.0)
    assert result["implementation_return"].iloc[0] == pytest.approx(
        103.0 / 102.0 - 1.0
    )


def test_seed_average_requires_identical_keys():
    first = pd.DataFrame(
        {
            "Date": ["2026-01-02", "2026-01-05"],
            "y_true": [1.0, 0.0],
            "probability": [0.7, 0.4],
        }
    )
    second = first.assign(probability=[0.9, 0.2])

    averaged = seed_average_predictions({42: first, 123: second})

    assert averaged["probability"].tolist() == pytest.approx([0.8, 0.3])
    assert averaged["seeds_averaged"].tolist() == [2, 2]


def test_seed_average_rejects_date_mismatch():
    first = pd.DataFrame(
        {"Date": ["2026-01-02"], "y_true": [1.0], "probability": [0.7]}
    )
    second = pd.DataFrame(
        {"Date": ["2026-01-05"], "y_true": [1.0], "probability": [0.8]}
    )

    with pytest.raises(ValueError, match="align"):
        seed_average_predictions({42: first, 123: second})


def test_forward_runner_builds_complete_selective_and_economic_grid(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    import models.track_d_runner as runner

    output_dir = tmp_path / "track_d"
    cell_dir = output_dir / "cells"
    output_dir.mkdir()
    (output_dir / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(
        {
            "model": ["lstm"],
            "objective": ["direct"],
            "selected_threshold": [0.5],
        }
    ).to_csv(output_dir / "selected_thresholds.csv", index=False)
    dates = pd.date_range("2026-01-02", periods=5, freq="B")
    test = pd.DataFrame(
        {
            "Date": dates,
            "Label_Date": dates + pd.offsets.BDay(1),
            "Close_D": [100.0, 101.0, 100.0, 102.0, 101.0],
            "Open_D": [99.0, 100.0, 101.0, 100.0, 102.0],
            "Target_Next_Close": [101.0, 100.0, 102.0, 101.0, 103.0],
        }
    )
    fold = SimpleNamespace(test=test)

    def fake_fit(*, model_key, objective, seed, **_kwargs):
        probabilities = np.array([0.70, 0.30, 0.60, 0.40, 0.80])
        probabilities = np.clip(probabilities + (seed - 42) * 0.0001, 0, 1)
        prediction = pd.DataFrame(
            {
                "Date": test["Date"].astype(str),
                "Label_Date": test["Label_Date"].astype(str),
                "Close_D": test["Close_D"],
                "Open_D": test["Open_D"],
                "Target_Next_Close": test["Target_Next_Close"],
                "y_true": [1.0, 0.0, 1.0, 0.0, 1.0],
                "direction_eligible": True,
                "probability": probabilities,
                "predicted_return": np.nan,
                "next_open": [100.0, 101.0, 100.0, 102.0, np.nan],
                "economic_eligible": [True, True, True, True, False],
                "implementation_return": [0.01, -0.01, 0.02, -0.01, np.nan],
                "idealized_close_return": [0.01, -0.01, 0.02, -0.01, 0.02],
            }
        )
        metric = {
            "model": model_key,
            "objective": objective,
            "seed": seed,
            "fit_seconds": 0.01,
            "inference_seconds": 0.001,
        }
        return prediction, metric

    monkeypatch.setattr(runner, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(runner, "CELL_DIR", cell_dir)
    monkeypatch.setattr(runner, "verify_freeze_manifest", lambda _path: {})
    monkeypatch.setattr(runner, "_forward_specs", lambda: ("scaled", "original"))
    monkeypatch.setattr(runner, "load_fold", lambda _spec: fold)
    monkeypatch.setattr(runner, "_fit_predict_fold", fake_fit)
    monkeypatch.setattr(runner, "package_versions", lambda _names: {})

    metadata = run_forward(
        model_keys=("lstm",),
        objectives=("direct",),
        seeds=(42, 123),
    )

    selective = pd.read_csv(output_dir / "selective_prediction_metrics.csv")
    economics = pd.read_csv(output_dir / "economic_summary.csv")
    assert metadata["forward_rows"] == 5
    assert len(selective) == 4
    assert len(economics) == 4 * 2 * 3
    assert economics["threshold_role"].eq(
        "validation_selected_primary"
    ).sum() == 2 * 3
