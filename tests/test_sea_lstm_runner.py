from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models import sea_lstm_runner
from models.pit_cdr_lstm_runner import PreparedFold
from models.sea_lstm_runner import (
    aggregate_experiment,
    build_ablation_table,
    promotion_decision,
    run_cell,
    run_experiment,
    validate_candidate_cohort,
)


def _ablation(scores: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant": list(scores),
            "balanced_accuracy_mean": list(scores.values()),
        }
    )


def _annual(full_2024: float, full_2025: float, control: float) -> pd.DataFrame:
    rows = []
    for year, full in ((2024, full_2024), (2025, full_2025)):
        rows.extend(
            [
                {
                    "variant": "sea_lstm",
                    "test_year": year,
                    "balanced_accuracy": full,
                },
                {
                    "variant": "standard_lstm",
                    "test_year": year,
                    "balanced_accuracy": control,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_promotion_requires_every_registered_condition() -> None:
    ablation = _ablation(
        {
            "sea_lstm": 0.56,
            "standard_lstm": 0.53,
            "positive_memory_only": 0.52,
            "negative_memory_only": 0.51,
        }
    )
    frozen = pd.DataFrame(
        {"model": ["LSTM-Attention"], "balanced_accuracy_mean": [0.55]}
    )

    passed = promotion_decision(ablation, _annual(0.56, 0.56, 0.53), frozen)
    failed = promotion_decision(ablation, _annual(0.52, 0.56, 0.53), frozen)

    assert passed["promoted"] is True
    assert all(passed["conditions"].values())
    assert failed["promoted"] is False
    assert failed["conditions"]["beats_standard_lstm_in_2024"] is False


def _toy_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2025-01-02", "2025-01-03"])
    years = dates.year
    close = np.full(4, 100.0)
    actual = np.array([101.0, 99.0, 99.0, 101.0])
    candidate_rows = []
    for variant in (
        "standard_lstm",
        "positive_memory_only",
        "negative_memory_only",
        "sea_lstm",
    ):
        for seed in (1, 2):
            for index, date in enumerate(dates):
                candidate_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "test_year": years[index],
                        "Date": date,
                        "Close_D": close[index],
                        "y_true": actual[index],
                        "probability": 0.8 if actual[index] > close[index] else 0.2,
                    }
                )
    frozen_rows = []
    for model in ("lstm", "cnn", "lstm_cnn", "lstm_attention", "lstm_cnn_attention"):
        for index, date in enumerate(dates):
            frozen_rows.append(
                {
                    "model": model,
                    "test_year": years[index],
                    "Date": date,
                    "Close_D": close[index],
                    "y_true": actual[index],
                    "y_pred": actual[index],
                }
            )
    return pd.DataFrame(candidate_rows), pd.DataFrame(frozen_rows)


def test_candidate_cohort_and_seed_averaged_ablation_are_complete() -> None:
    candidate, frozen = _toy_predictions()

    audit = validate_candidate_cohort(candidate, frozen, seeds=(1, 2))
    summary = build_ablation_table(candidate, seeds=(1, 2))

    assert audit["passed"] is True
    assert audit["rows_per_variant_seed"] == 4
    assert set(summary["variant"]) == {
        "standard_lstm",
        "positive_memory_only",
        "negative_memory_only",
        "sea_lstm",
    }
    assert summary["balanced_accuracy_mean"].eq(1.0).all()


def test_candidate_cohort_rejects_changed_test_date() -> None:
    candidate, frozen = _toy_predictions()
    candidate.loc[0, "Date"] = pd.Timestamp("2024-01-04")

    with pytest.raises(ValueError, match="dates do not match"):
        validate_candidate_cohort(candidate, frozen, seeds=(1, 2))


def _prepared_fold() -> PreparedFold:
    rng = np.random.default_rng(7)
    train_count = 20
    test_count = 4
    return PreparedFold(
        fold="toy_fold",
        test_year=2024,
        feature_columns=tuple(f"x_{index}" for index in range(7)),
        train_sequence=rng.normal(size=(train_count, 5, 7)).astype(np.float32),
        train_labels=(np.arange(train_count) % 2).astype(np.int8),
        train_regimes=np.asarray(["bull"] * train_count, dtype=object),
        train_state=np.zeros((train_count, 2), dtype=float),
        train_endpoint_positions=np.arange(train_count),
        train_dates=pd.date_range("2023-01-01", periods=train_count).to_numpy(),
        test_sequence=rng.normal(size=(test_count, 5, 7)).astype(np.float32),
        test_regimes=np.asarray(["bull", "bear", "sideway", "bull"], dtype=object),
        test_dates=pd.date_range("2024-01-02", periods=test_count).to_numpy(),
        test_close=np.full(test_count, 100.0),
        test_next_close=np.array([101.0, 99.0, 102.0, 98.0]),
        mask_counts={"bull": 3, "sideway": 7, "bear": 4},
    )


def test_run_cell_writes_complete_reproducibility_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sea_lstm_runner, "EPOCHS", 1)
    monkeypatch.setattr(sea_lstm_runner, "BATCH_SIZE", 8)

    result = run_cell(
        _prepared_fold(),
        seed=42,
        variant="sea_lstm",
        output_dir=tmp_path,
    )
    directory = tmp_path / "cells" / "2024" / "seed_42" / "sea_lstm"

    assert result["status"] == "completed"
    assert (directory / "predictions.csv").is_file()
    assert (directory / "metrics.json").is_file()
    assert (directory / "run_metadata.json").is_file()
    assert (directory / "inference.weights.h5").is_file()
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["inference_parameters"] > 0
    assert np.isfinite(metrics["mean_up_evidence"])


def _write_toy_aggregate_cells(root: Path, candidate: pd.DataFrame) -> None:
    for (year, seed, variant), group in candidate.groupby(
        ["test_year", "seed", "variant"], sort=False
    ):
        directory = root / "cells" / str(year) / f"seed_{seed}" / str(variant)
        directory.mkdir(parents=True)
        group.to_csv(directory / "predictions.csv", index=False)
        (directory / "run_metadata.json").write_text(
            json.dumps(
                {
                    "variant": variant,
                    "wall_seconds": 1.0,
                    "inference_parameters": 100,
                }
            ),
            encoding="utf-8",
        )


def test_aggregate_experiment_builds_comparison_and_promotion_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, frozen = _toy_predictions()
    labels = candidate["y_true"] > candidate["Close_D"]
    candidate.loc[candidate["variant"].eq("sea_lstm"), "probability"] = np.where(
        labels[candidate["variant"].eq("sea_lstm")], 0.9, 0.1
    )
    candidate.loc[candidate["variant"].eq("standard_lstm"), "probability"] = 0.9
    candidate.loc[candidate["variant"].eq("positive_memory_only"), "probability"] = 0.1
    negative = candidate["variant"].eq("negative_memory_only")
    candidate.loc[negative, "probability"] = np.where(labels[negative], 0.1, 0.9)
    frozen["y_pred"] = 101.0
    frozen_path = tmp_path / "frozen.csv"
    frozen.to_csv(frozen_path, index=False)
    _write_toy_aggregate_cells(tmp_path, candidate)
    monkeypatch.setattr(sea_lstm_runner, "SEEDS", (1, 2))
    monkeypatch.setattr(sea_lstm_runner, "EXPECTED_TEST_ROWS", {2024: 2, 2025: 2})
    monkeypatch.setattr(sea_lstm_runner, "FROZEN_PREDICTIONS", frozen_path)

    result = aggregate_experiment(tmp_path)

    assert result["integrity"]["passed"] is True
    assert result["decision"]["promoted"] is True
    assert len(result["six_model"]) == 6
    assert (tmp_path / "six_model_comparison_2024_2025.csv").is_file()
    assert (tmp_path / "promotion_decision.json").is_file()


def test_run_experiment_executes_registered_grid_and_writes_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int, str]] = []
    prepared = _prepared_fold()
    monkeypatch.setattr(sea_lstm_runner, "TEST_YEARS", (2024,))
    monkeypatch.setattr(sea_lstm_runner, "SEEDS", (1,))
    monkeypatch.setattr(sea_lstm_runner, "prepare_candidate_fold", lambda year: prepared)
    monkeypatch.setattr(
        sea_lstm_runner,
        "verify_freeze_manifest",
        lambda root, path: {"passed": True, "checked_files": 1},
    )
    monkeypatch.setattr(
        sea_lstm_runner,
        "run_cell",
        lambda fold, seed, variant, output_dir, force: calls.append(
            (fold.test_year, seed, variant)
        ),
    )
    fake_result = {
        "six_model": pd.DataFrame(),
        "decision": {"promoted": False},
    }
    monkeypatch.setattr(sea_lstm_runner, "aggregate_experiment", lambda path: fake_result)

    result = run_experiment(
        output_dir=tmp_path,
        variants=sea_lstm_runner.INTERNAL_VARIANTS,
    )

    assert result is fake_result
    assert len(calls) == len(sea_lstm_runner.INTERNAL_VARIANTS)
    assert (tmp_path / "run_metadata.json").is_file()
