from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import models.tcrc_lstm_runner as runner
from models.tcrc_lstm import VARIANTS
from models.tcrc_lstm_runner import (
    INNER_YEARS,
    SEEDS,
    _fit_variant,
    _fold_metrics_from_averaged,
    _metrics,
    aggregate_experiment,
    build_inner_fold_arrays,
    evaluate_promotion,
    run_cell,
    verify_freeze_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=35)
    close = np.asarray([100.0 + index + (-1) ** index for index in range(35)])
    return pd.DataFrame(
        {
            "Date": dates,
            "Label_Date": dates + pd.offsets.BDay(1),
            "Close_D": close,
            "Target_Next_Close": close + np.where(np.arange(35) % 3, 1.0, -1.0),
            "x": np.arange(35, dtype=float),
            "news": np.linspace(0.0, 1.0, 35),
        }
    )


def test_inner_arrays_fit_scalers_on_train_and_cover_every_validation_day() -> None:
    frame = _small_frame()
    validation_start = frame.loc[24, "Date"]
    frame.loc[24:, "Date"] = pd.bdate_range("2020-01-02", periods=11)
    frame.loc[24:, "Label_Date"] = frame.loc[24:, "Date"] + pd.offsets.BDay(1)

    arrays = build_inner_fold_arrays(
        frame,
        feature_columns=("x", "news"),
        validation_year=2020,
        window=5,
    )

    assert arrays.train_x.shape == (20, 5, 2)
    assert arrays.validation_x.shape == (11, 5, 2)
    assert len(arrays.validation_dates) == 11
    assert arrays.validation_dates.min() > validation_start
    assert np.isfinite(arrays.train_x).all()
    assert arrays.train_dates.max() < arrays.validation_dates.min()
    assert arrays.return_std > 0.0


def test_inner_arrays_purge_training_labels_observed_on_validation_start() -> None:
    frame = _small_frame()
    frame.loc[24:, "Date"] = pd.bdate_range("2020-01-02", periods=11)
    frame.loc[24:, "Label_Date"] = frame.loc[24:, "Date"] + pd.offsets.BDay(1)
    frame.loc[23, "Label_Date"] = frame.loc[24, "Date"]

    arrays = build_inner_fold_arrays(
        frame,
        feature_columns=("x", "news"),
        validation_year=2020,
        window=5,
    )

    assert arrays.train_x.shape == (19, 5, 2)
    assert arrays.train_dates.max() == frame.loc[22, "Date"]


def test_promotion_requires_full_model_to_win_both_years_and_mean_margin() -> None:
    rows: list[dict[str, object]] = []
    scores = {
        "lstm_anchor": (0.50, 0.51),
        "cnn_residual": (0.51, 0.51),
        "latent_turn_gate": (0.515, 0.512),
        "supervised_turn_gate": (0.52, 0.515),
        "tcrc_full": (0.53, 0.525),
    }
    for variant in VARIANTS:
        for year, score in zip(INNER_YEARS, scores[variant], strict=True):
            rows.append(
                {
                    "variant": variant,
                    "validation_year": year,
                    "balanced_accuracy": score,
                }
            )
    decision = evaluate_promotion(pd.DataFrame(rows), integrity_passed=True)

    assert decision["passed"] is True
    assert all(decision["conditions"].values())


def test_seed_averaged_primary_direction_is_derived_from_reconstructed_return() -> None:
    predictions = pd.DataFrame(
        {
            "variant": ["tcrc_full"] * 4,
            "validation_year": [2020] * 4,
            "Close_D": [100.0] * 4,
            "y_true": [99.0, 99.0, 101.0, 101.0],
            "y_pred": [99.0, 99.0, 101.0, 101.0],
            "true_direction": [0, 0, 1, 1],
            "probability": [0.9, 0.9, 0.1, 0.1],
            "true_turn": [0, 1, 0, 1],
            "turn_valid": [True] * 4,
            "turn_probability": [0.1, 0.9, 0.1, 0.9],
        }
    )

    metrics = _fold_metrics_from_averaged(predictions)

    assert metrics.loc[0, "balanced_accuracy"] == 1.0
    assert metrics.loc[0, "direction_accuracy"] == 1.0


def test_registered_freeze_precedes_tcrc_results() -> None:
    manifest_path = PROJECT_ROOT / "test" / "tcrc_lstm_freeze_v1.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = verify_freeze_manifest(PROJECT_ROOT, manifest_path)

    assert audit["passed"] is True
    assert payload["result_access_at_freeze"] is False
    assert payload["post_hoc_architecture_motivation"] is True
    assert payload["inner_development_years"] == list(INNER_YEARS)
    assert payload["seeds"] == list(SEEDS)
    assert payload["variants"] == list(VARIANTS)


def test_actual_torch_training_loop_and_metrics_smoke() -> None:
    frame = _small_frame()
    frame.loc[24:, "Date"] = pd.bdate_range("2020-01-02", periods=11)
    frame.loc[24:, "Label_Date"] = frame.loc[24:, "Date"] + pd.offsets.BDay(1)
    arrays = build_inner_fold_arrays(
        frame,
        feature_columns=("x", "news"),
        validation_year=2020,
        window=5,
    )
    from models.tcrc_lstm import TCRCConfig

    output, diagnostics = _fit_variant(
        arrays,
        variant="tcrc_full",
        seed=42,
        config=TCRCConfig(window=5, lstm_window=3),
    )
    metrics = _metrics(arrays, output)

    assert diagnostics["training_sequences"] == len(arrays.train_x)
    assert diagnostics["validation_sequences"] == len(arrays.validation_x)
    assert np.isfinite(output["probability"] if "probability" in output else output["direction_probability"]).all()
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert metrics["observations"] == len(arrays.validation_x)


def test_cell_and_aggregate_integration_with_registered_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _small_frame()
    frame.loc[24:, "Date"] = pd.bdate_range("2020-01-02", periods=11)
    frame.loc[24:, "Label_Date"] = frame.loc[24:, "Date"] + pd.offsets.BDay(1)

    monkeypatch.setattr(runner, "INNER_YEARS", (2020,))
    monkeypatch.setattr(runner, "SEEDS", (42,))
    monkeypatch.setattr(runner, "WINDOW", 5)
    monkeypatch.setattr(runner, "LSTM_WINDOW", 3)
    monkeypatch.setattr(runner, "NEWS_FEATURES", ())
    monkeypatch.setattr(runner, "CONTEXT_FEATURES", ())
    monkeypatch.setattr(
        runner,
        "verify_freeze_manifest",
        lambda *_args, **_kwargs: {"passed": True, "files_checked": 1},
    )
    monkeypatch.setattr(
        runner,
        "load_development_frame",
        lambda: (frame, ("x", "news"), {"passed": True}),
    )

    def fake_fit(arrays, *, variant, seed, config):
        del seed, config
        actual_return = (
            (arrays.validation_next_close - arrays.validation_current_close)
            / arrays.validation_current_close
            * 100.0
        )
        raw_return = actual_return if variant == "tcrc_full" else -actual_return
        probability = np.where(raw_return > 0.0, 0.8, 0.2)
        size = len(raw_return)
        return (
            {
                "raw_return_percent": raw_return,
                "direction_probability": probability,
                "turn_probability": np.where(arrays.validation_turn > 0, 0.8, 0.2),
                "gate": np.full(size, variant == "tcrc_full", dtype=float),
                "correction": np.zeros(size),
                "attention_entropy": np.zeros(size),
            },
            {
                "device": "cpu",
                "trainable_parameters": 10,
                "fit_seconds": 0.01,
                "inference_seconds": 0.001,
                "final_training_loss": 0.5,
                "training_sequences": len(arrays.train_x),
                "validation_sequences": size,
            },
        )

    monkeypatch.setattr(runner, "_fit_variant", fake_fit)

    completed = run_cell(
        validation_year=2020,
        seed=42,
        output_dir=tmp_path,
    )
    skipped = run_cell(
        validation_year=2020,
        seed=42,
        output_dir=tmp_path,
    )
    result = aggregate_experiment(output_dir=tmp_path)

    assert completed["status"] == "completed"
    assert skipped["status"] == "skipped_complete"
    assert result["completed_cells"] == 1
    assert result["integrity_passed"] is True
    assert (tmp_path / "inner_summary.csv").is_file()
