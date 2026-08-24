from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models import integrated_multimodal_runner as runner
from models.integrated_multimodal import ARMS, FitRequest
from models.integrated_multimodal_runner import (
    aggregate_experiment,
    build_cell_commands,
    cell_complete,
    cell_directory,
    daily_contrast_effects,
    fit_execution_id,
    fold_metrics_from_seed_averaged_predictions,
    run_cells_isolated,
)
from models.track_a_final import FINAL_SEEDS, TRACK_A_MODELS


def test_full_run_builds_100_isolated_resumable_cell_commands(tmp_path: Path) -> None:
    commands = build_cell_commands(
        python_executable=Path("D:/conda_envs/my_env/python.exe"),
        output_dir=tmp_path,
    )

    assert len(commands) == len(TRACK_A_MODELS) * 4 * len(FINAL_SEEDS) == 100
    assert len({tuple(command) for command in commands}) == 100
    assert all(command[1:4] == ["-m", "models.integrated_multimodal_runner", "cell"] for command in commands)
    assert all("--output-dir" in command for command in commands)


def _write_valid_cell(directory: Path) -> None:
    directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "arm": list(ARMS),
            "n_test": [2] * 4,
            "balanced_accuracy": [0.5] * 4,
        }
    ).to_csv(directory / "metrics.csv", index=False)
    pd.DataFrame(
        {
            "fit_id": [f"fit_{index}" for index in range(8)],
            "training_sequences": [200] * 8,
            "features": [122, 130, 30, 122, 80, 38, 130, 88],
        }
    ).to_csv(directory / "fit_registry.csv", index=False)
    for arm in ARMS:
        pd.DataFrame(
            {
                "Date": ["2022-01-03", "2022-01-04"],
                "routing_regime": ["bull", "bear"],
                "Close_D": [100.0, 101.0],
                "y_true": [101.0, 100.0],
                "y_pred": [100.5, 100.5],
            }
        ).to_csv(directory / f"predictions_{arm}.csv", index=False)
    (directory / "run_metadata.json").write_text(
        json.dumps({"protocol_id": "integrated-multimodal-posthoc-v1"}),
        encoding="utf-8",
    )
    (directory / "integrity_audit.json").write_text(
        json.dumps({"passed": True}),
        encoding="utf-8",
    )


def test_cell_complete_checks_integrity_not_only_file_presence(tmp_path: Path) -> None:
    directory = cell_directory(tmp_path, "lstm", "fold_1", 42)
    _write_valid_cell(directory)

    assert cell_complete(tmp_path, "lstm", "fold_1", 42)
    (directory / "integrity_audit.json").write_text(
        json.dumps({"passed": False}),
        encoding="utf-8",
    )
    assert not cell_complete(tmp_path, "lstm", "fold_1", 42)


def test_fit_execution_id_is_stable_and_scoped_to_model_and_fold() -> None:
    request = FitRequest("global", "global", ("a", "b"), 42)

    first = fit_execution_id("lstm", "fold_1", request)
    assert first == fit_execution_id("lstm", "fold_1", request)
    assert first != fit_execution_id("cnn", "fold_1", request)
    assert first != fit_execution_id("lstm", "fold_2", request)


def test_fold_metrics_are_recomputed_from_seed_averaged_predictions() -> None:
    rows = []
    for arm in ARMS:
        for date, close, true, prediction, regime in [
            ("2022-01-03", 100.0, 101.0, 101.0, "bull"),
            ("2022-01-04", 101.0, 100.0, 100.0, "bear"),
        ]:
            rows.append(
                {
                    "model": "lstm",
                    "fold": "fold_1",
                    "test_year": 2022,
                    "arm": arm,
                    "Date": date,
                    "routing_regime": regime,
                    "Close_D": close,
                    "y_true": true,
                    "y_pred": prediction,
                    "seeds_averaged": 5,
                }
            )

    result = fold_metrics_from_seed_averaged_predictions(pd.DataFrame(rows))

    assert len(result) == 4
    assert result["balanced_accuracy"].eq(1.0).all()
    assert result["direction_accuracy"].eq(1.0).all()
    assert result["seeds_averaged"].eq(5).all()


def test_daily_interaction_is_difference_of_regime_and_global_news_effects() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2022-01-03", periods=4, freq="D"),
            "Close_D": [100.0] * 4,
            "y_true": [101.0, 99.0, 102.0, 98.0],
            "Global-Numeric": [101.0, 101.0, 101.0, 99.0],
            "Global-Numeric-News": [101.0, 99.0, 101.0, 99.0],
            "Regime-SHAP-Numeric": [99.0, 99.0, 101.0, 101.0],
            "Regime-SHAP-Numeric-News": [101.0, 99.0, 101.0, 99.0],
        }
    )

    interaction = daily_contrast_effects(frame, "routing_news_interaction")
    regime = daily_contrast_effects(frame, "regime_pipeline_news_effect")
    global_effect = daily_contrast_effects(frame, "global_news_effect")

    for metric in ("squared_error_loss_delta", "balanced_accuracy_delta_pp"):
        expected = regime[metric] - global_effect[metric]
        assert (interaction[metric] == expected).all()


def _synthetic_full_fold_metrics() -> pd.DataFrame:
    rows = []
    offsets = {
        "Global-Numeric": 0.00,
        "Global-Numeric-News": 0.01,
        "Regime-SHAP-Numeric": 0.02,
        "Regime-SHAP-Numeric-News": 0.03,
    }
    for model in TRACK_A_MODELS:
        for fold_index in range(1, 5):
            for arm, offset in offsets.items():
                rows.append(
                    {
                        "model": model,
                        "fold": f"fold_{fold_index}",
                        "test_year": 2021 + fold_index,
                        "arm": arm,
                        "seeds_averaged": 5,
                        "n_test": 2,
                        "rmse": 10.0 - offset,
                        "mae": 8.0 - offset,
                        "mape": 1.0,
                        "r2": 0.0,
                        "direction_accuracy": 0.50 + offset,
                        "balanced_accuracy": 0.50 + offset,
                        "mcc": offset,
                        "direction_coverage": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_aggregate_writes_registered_outputs_and_integrity_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = pd.DataFrame(
        {
            "model": ["lstm"] * 4,
            "fold": ["fold_1"] * 4,
            "test_year": [2022] * 4,
            "base_seed": [42] * 4,
            "arm": list(ARMS),
        }
    )
    registry = pd.DataFrame(
        {
            "fit_id": [f"fit_{index}" for index in range(8)],
            "model": ["lstm"] * 8,
            "arm": [
                "Global-Numeric",
                "Global-Numeric-News",
                "Regime-SHAP-Numeric",
                "Regime-SHAP-Numeric",
                "Regime-SHAP-Numeric",
                "Regime-SHAP-Numeric-News",
                "Regime-SHAP-Numeric-News",
                "Regime-SHAP-Numeric-News",
            ],
            "regime": [
                "global",
                "global",
                "bull",
                "sideway",
                "bear",
                "bull",
                "sideway",
                "bear",
            ],
            "features": [122, 130, 30, 122, 80, 38, 130, 88],
            "training_sequences": [220] * 8,
            "fit_seconds": [1.0] * 8,
            "inference_seconds": [0.1] * 8,
            "trainable_parameters": [100] * 8,
        }
    )
    predictions = pd.DataFrame(
        {
            "model": ["lstm"],
            "fold": ["fold_1"],
            "test_year": [2022],
            "arm": ["Global-Numeric"],
            "Date": ["2022-01-03"],
            "routing_regime": ["bull"],
            "Close_D": [100.0],
            "y_true": [101.0],
            "y_pred": [100.5],
            "seeds_averaged": [5],
        }
    )
    bootstrap_rows = []
    for model in TRACK_A_MODELS:
        for contrast in runner.CONTRASTS:
            for metric in (
                "squared_error_loss_delta",
                "balanced_accuracy_delta_pp",
            ):
                bootstrap_rows.append(
                    {
                        "model": model,
                        "contrast": contrast,
                        "metric": metric,
                        "point_estimate": 0.0,
                        "ci95_lower": -1.0,
                        "ci95_upper": 1.0,
                        "two_sided_pvalue": 0.5,
                        "replicates": 100,
                        "block_length": 10,
                        "folds": 4,
                        "daily_rows": 8,
                    }
                )

    monkeypatch.setattr(runner, "_expected_cell_keys", lambda: {("lstm", "fold_1", 42)})
    monkeypatch.setattr(runner, "cell_complete", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runner, "_collect_metrics", lambda _output: metrics)
    monkeypatch.setattr(runner, "_collect_registry", lambda _output: registry)
    monkeypatch.setattr(
        runner,
        "_collect_seed_averaged_predictions",
        lambda _output, _metrics: predictions,
    )
    monkeypatch.setattr(
        runner,
        "fold_metrics_from_seed_averaged_predictions",
        lambda _predictions: _synthetic_full_fold_metrics(),
    )
    monkeypatch.setattr(
        runner,
        "_daily_block_bootstrap",
        lambda _predictions: pd.DataFrame(bootstrap_rows),
    )
    monkeypatch.setattr(
        runner,
        "_runtime_summary",
        lambda _registry, _output: pd.DataFrame(
            {"model": ["lstm"], "executed_fits": [8]}
        ),
    )
    monkeypatch.setattr(
        runner,
        "verify_freeze_manifest",
        lambda *_args: {"passed": True, "files_checked": 24},
    )

    result = aggregate_experiment(output_dir=tmp_path)

    assert result["protocol_id"] == "integrated-multimodal-posthoc-v1"
    audit = json.loads(
        (tmp_path / "integrity_audit.json").read_text(encoding="utf-8")
    )
    assert audit["passed"] is True
    assert audit["metric_rows"] == 4
    assert audit["fit_rows"] == 8
    assert np.isfinite(
        pd.read_csv(tmp_path / "paper_integrated_table.csv")[
            "balanced_accuracy_mean_pct"
        ]
    ).all()


def test_isolated_runner_skips_complete_cells_and_executes_incomplete_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = [
        [
            "python",
            "-m",
            "models.integrated_multimodal_runner",
            "cell",
            "--model",
            "lstm",
            "--fold",
            "fold_1",
            "--seed",
            "42",
        ],
        [
            "python",
            "-m",
            "models.integrated_multimodal_runner",
            "cell",
            "--model",
            "cnn",
            "--fold",
            "fold_1",
            "--seed",
            "42",
        ],
    ]
    executed = []
    monkeypatch.setattr(runner, "build_cell_commands", lambda **_kwargs: commands)
    monkeypatch.setattr(
        runner,
        "cell_complete",
        lambda _output, model, _fold, _seed: model == "lstm",
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: executed.append((command, kwargs)),
    )

    result = run_cells_isolated(output_dir=tmp_path)

    assert result == {"total": 2, "completed": 1, "skipped": 1}
    assert len(executed) == 1
    assert executed[0][0] == commands[1]
    assert executed[0][1]["check"] is True
