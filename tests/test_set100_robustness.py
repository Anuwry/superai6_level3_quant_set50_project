from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.set100_robustness import (
    COMPARABLE_DELTA_METRICS,
    apply_market_holm,
    average_seed_predictions,
    build_market_fold_deltas,
    evaluate_robustness_predictions,
    fold_level_market_inference,
    validate_registered_design,
    verify_freeze_manifest,
)
from models.set100_robustness_aggregate import (
    build_paper_table,
    summarize_market_metrics,
)
from models.set100_robustness_runner import (
    FOLDS,
    MODELS,
    SEEDS,
    build_job_commands,
    cell_complete,
    cell_directory,
)


def _prediction_frame(
    predictions: list[float],
    *,
    dates: tuple[str, ...] = (
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Close_D": [100.0, 100.0, 100.0, 100.0],
            "y_true": [102.0, 98.0, 103.0, 97.0],
            "y_pred": predictions,
        }
    )


def test_verify_freeze_manifest_rejects_changed_input(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "freeze.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol_id": "set100-same-exchange-robustness-v1",
                "status": "frozen_before_any_set100_model_fit",
                "set100_model_results_seen_before_freeze": False,
                "models": {
                    "lstm": 5,
                    "cnn": 20,
                    "lstm_cnn": 20,
                    "lstm_attention": 10,
                    "lstm_cnn_attention": 20,
                },
                "seeds": [42, 123, 456, 789, 2025],
                "outer_test_years": [2022, 2023, 2024, 2025],
                "feature_count": 122,
                "frozen_inputs": [
                    {
                        "path": "source.csv",
                        "sha256": digest,
                        "bytes": source.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    verified = verify_freeze_manifest(tmp_path, manifest)
    assert verified["all_inputs_match"] is True

    source.write_text("value\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen input mismatch"):
        verify_freeze_manifest(tmp_path, manifest)


def test_evaluate_robustness_predictions_includes_auc_and_normalized_errors() -> None:
    frame = _prediction_frame([101.0, 99.0, 102.0, 98.0])

    metrics = evaluate_robustness_predictions(
        frame["y_true"].to_numpy(),
        frame["y_pred"].to_numpy(),
        frame["Close_D"].to_numpy(),
    )

    assert metrics["direction_accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["direction_coverage"] == 1.0
    assert metrics["nrmse_percent"] > 0.0
    assert metrics["nmae_percent"] > 0.0


def test_average_seed_predictions_validates_alignment_and_averages() -> None:
    seed_frames = {
        42: _prediction_frame([101.0, 99.0, 102.0, 98.0]),
        123: _prediction_frame([103.0, 97.0, 104.0, 96.0]),
    }

    averaged = average_seed_predictions(seed_frames)

    assert averaged["y_pred"].tolist() == [102.0, 98.0, 103.0, 97.0]
    assert averaged["seeds_averaged"].unique().tolist() == [2]

    misaligned = dict(seed_frames)
    misaligned[123] = _prediction_frame(
        [103.0, 97.0, 104.0, 96.0],
        dates=("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-08"),
    )
    with pytest.raises(ValueError, match="dates differ"):
        average_seed_predictions(misaligned)


def test_market_delta_inference_and_holm_use_four_folds() -> None:
    rows: list[dict[str, object]] = []
    for model_index, model in enumerate(MODELS):
        for fold_index, fold in enumerate(FOLDS):
            base = 0.50 + model_index * 0.001 + fold_index * 0.002
            common = {
                "model": model,
                "fold": fold,
                "test_year": 2022 + fold_index,
                "direction_accuracy": base,
                "mcc": base - 0.5,
                "roc_auc": base + 0.02,
                "direction_coverage": 1.0,
                "mape": 2.0,
                "nrmse_percent": 3.0,
                "nmae_percent": 2.0,
            }
            rows.append(
                {
                    **common,
                    "market": "SET50",
                    "balanced_accuracy": base,
                }
            )
            rows.append(
                {
                    **common,
                    "market": "SET100",
                    "balanced_accuracy": base + 0.01,
                    "direction_accuracy": base + 0.01,
                    "mcc": base - 0.49,
                    "roc_auc": base + 0.03,
                    "mape": 1.9,
                    "nrmse_percent": 2.9,
                    "nmae_percent": 1.9,
                }
            )
    fold_metrics = pd.DataFrame(rows)

    deltas = build_market_fold_deltas(fold_metrics)
    inference = fold_level_market_inference(deltas)
    adjusted = apply_market_holm(inference)

    assert len(deltas) == len(MODELS) * len(FOLDS)
    assert len(inference) == len(MODELS) * len(COMPARABLE_DELTA_METRICS)
    assert adjusted["holm_adjusted_pvalue"].between(0.0, 1.0).all()
    assert set(adjusted["models_in_family"]) == {len(MODELS)}
    bacc = adjusted.loc[
        adjusted["metric"].eq("balanced_accuracy_delta_pp")
    ]
    assert np.allclose(bacc["mean_delta"], 1.0)


def test_validate_registered_design_requires_all_100_cells() -> None:
    rows = [
        {"model": model, "fold": fold, "seed": seed}
        for model in MODELS
        for fold in FOLDS
        for seed in SEEDS
    ]
    design = pd.DataFrame(rows)

    summary = validate_registered_design(design)
    assert summary["expected_cells"] == 100
    assert summary["observed_cells"] == 100

    with pytest.raises(ValueError, match="Registered design is incomplete"):
        validate_registered_design(design.iloc[:-1])


def test_build_job_commands_and_cell_completion_contract(tmp_path: Path) -> None:
    commands = build_job_commands(
        python_executable=Path("python.exe"),
        output_dir=tmp_path,
    )

    assert len(commands) == len(MODELS) * len(SEEDS)
    assert commands[0][-6:] == [
        "--model",
        "lstm",
        "--seed",
        "42",
        "--output-dir",
        str(tmp_path),
    ]

    directory = cell_directory(tmp_path, "lstm", "fold_1", 42)
    directory.mkdir(parents=True)
    assert cell_complete(tmp_path, "lstm", "fold_1", 42) is False
    pd.DataFrame(
        {
            "Date": ["2022-01-04"],
            "Close_D": [100.0],
            "y_true": [101.0],
            "y_pred": [102.0],
        }
    ).to_csv(directory / "predictions.csv", index=False)
    (directory / "metrics.json").write_text(
        json.dumps({"balanced_accuracy": 0.5}),
        encoding="utf-8",
    )
    (directory / "run_metadata.json").write_text(
        json.dumps({"protocol_id": "set100-same-exchange-robustness-v1"}),
        encoding="utf-8",
    )
    (directory / "integrity_audit.json").write_text(
        json.dumps({"passed": True}),
        encoding="utf-8",
    )
    assert cell_complete(tmp_path, "lstm", "fold_1", 42) is True


def test_market_summary_and_paper_table_have_registered_rows() -> None:
    rows: list[dict[str, object]] = []
    for model_index, model in enumerate(MODELS):
        for market_index, market in enumerate(("SET50", "SET100")):
            for fold_index, fold in enumerate(FOLDS):
                value = 0.50 + model_index * 0.01 + market_index * 0.02
                rows.append(
                    {
                        "market": market,
                        "model": model,
                        "fold": fold,
                        "test_year": 2022 + fold_index,
                        "balanced_accuracy": value,
                        "direction_accuracy": value + 0.01,
                        "mcc": value - 0.5,
                        "roc_auc": value + 0.02,
                        "direction_coverage": 1.0,
                        "mape": 2.0 - market_index * 0.1,
                        "nrmse_percent": 3.0 - market_index * 0.1,
                        "nmae_percent": 2.0 - market_index * 0.1,
                    }
                )
    fold_metrics = pd.DataFrame(rows)
    summary = summarize_market_metrics(fold_metrics)
    deltas = build_market_fold_deltas(fold_metrics)
    adjusted = apply_market_holm(fold_level_market_inference(deltas))
    paper = build_paper_table(summary, adjusted)

    assert len(summary) == len(MODELS) * 2
    assert set(summary["outer_folds"]) == {4}
    assert len(paper) == len(MODELS)
    assert np.allclose(paper["balanced_accuracy_delta_pp"], 2.0)
    assert paper["balanced_accuracy_holm_pvalue"].between(0.0, 1.0).all()
