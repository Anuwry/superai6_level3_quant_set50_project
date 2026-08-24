import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import models.vmd_experiments as vmd_experiments
import models.vmd_window_sweep as window_sweep
from models.convolutional_neural_network import make_sequences, make_test_sequences


def test_window_sweep_contract_covers_requested_windows_and_models():
    assert window_sweep.SEQUENCE_WINDOWS == (1, 3, 5, 10, 20)
    assert window_sweep.NEW_SEQUENCE_WINDOWS == (1, 3, 5, 10)
    assert list(window_sweep.WINDOW_SWEEP_MODELS) == [
        "lstm",
        "cnn",
        "lstm_cnn",
        "lstm_attention",
        "lstm_cnn_attention",
    ]
    assert "attention_lstm_cnn" not in window_sweep.WINDOW_SWEEP_MODELS


@pytest.mark.parametrize("window", [0, 2, 4, 21, True, "3"])
def test_validate_sequence_window_rejects_unrequested_values(window):
    with pytest.raises(ValueError):
        window_sweep.validate_sequence_window(window)


def test_all_requested_windows_create_one_step_sliding_sequences():
    features = np.arange(50, dtype=float).reshape(25, 2)
    target = np.arange(25, dtype=float)
    test_features = np.arange(12, dtype=float).reshape(6, 2)

    for window in window_sweep.SEQUENCE_WINDOWS:
        train_sequences, train_target = make_sequences(features, target, window)
        test_sequences = make_test_sequences(features, test_features, window)

        assert train_sequences.shape == (26 - window, window, 2)
        assert train_target.shape == (26 - window,)
        assert test_sequences.shape == (6, window, 2)
        np.testing.assert_array_equal(test_sequences[0, -1], test_features[0])
        if window > 1:
            np.testing.assert_array_equal(
                test_sequences[0, :-1],
                features[-(window - 1) :],
            )


def test_run_model_window_uses_isolated_output_and_requested_length(monkeypatch):
    recorded = {}
    expected = pd.DataFrame({"rmse": [1.0]})

    def fake_runner(**kwargs):
        recorded.update(kwargs)
        return expected

    monkeypatch.setattr(window_sweep, "run_vmd_sequence_model", fake_runner)

    result = window_sweep.run_model_window("cnn", 3, force=True)

    assert result is expected
    assert recorded["model_key"] == "cnn"
    assert recorded["model_name"] == "window_3"
    assert recorded["sequence_length"] == 3
    assert recorded["output_dir"] == (
        window_sweep.WINDOW_SWEEP_OUTPUT_DIR / "cnn"
    )
    assert recorded["experiment"] == "full_ta_vmd_sequence_window_sweep"


def test_reproducible_predictor_clears_session_and_resets_seed(monkeypatch):
    calls = []
    fake_tensorflow = SimpleNamespace(
        keras=SimpleNamespace(
            backend=SimpleNamespace(
                clear_session=lambda: calls.append(("clear_session", None))
            ),
            utils=SimpleNamespace(
                set_random_seed=lambda seed: calls.append(("seed", seed))
            ),
        )
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow)

    prediction = window_sweep.predict_with_reproducible_seed(
        lambda fold, window: np.array([float(window)]),
        object(),
        3,
    )

    assert calls == [("clear_session", None), ("seed", 42)]
    assert prediction.tolist() == [3.0]


def test_vmd_runner_records_model_runtime_per_fold(monkeypatch, tmp_path):
    spec = type("Spec", (), {"fold": "fold_1"})()
    fake_fold = object()
    saved = {}
    clock = iter([100.0, 101.0, 103.5, 104.0])

    monkeypatch.setattr(
        vmd_experiments,
        "discover_folds",
        lambda data_dir: [spec],
    )
    monkeypatch.setattr(vmd_experiments, "load_fold", lambda fold_spec: fake_fold)
    monkeypatch.setattr(
        vmd_experiments,
        "load_vmd_scaler_metadata",
        lambda fold_name: {},
    )
    monkeypatch.setattr(
        vmd_experiments,
        "inverse_scaled_target",
        lambda values, metadata: values,
    )
    monkeypatch.setattr(
        vmd_experiments,
        "evaluate_predictions",
        lambda fold, prediction: {
            "fold": "fold_1",
            "rmse": 1.0,
            "mae": 1.0,
            "mape": 1.0,
            "r2": 0.0,
            "direction_accuracy": 0.5,
        },
    )
    monkeypatch.setattr(
        vmd_experiments,
        "predictions_frame",
        lambda fold, prediction: pd.DataFrame({"y_pred": prediction}),
    )
    monkeypatch.setattr(vmd_experiments, "print_metrics", lambda metrics: None)
    monkeypatch.setattr(
        vmd_experiments,
        "save_run_outputs",
        lambda model_name, metrics, predictions, config, packages, output_dir: saved.update(
            {
                "metrics": metrics,
                "config": config,
            }
        ),
    )
    monkeypatch.setattr(
        vmd_experiments.time,
        "perf_counter",
        lambda: next(clock),
    )

    vmd_experiments.run_vmd_sequence_model(
        model_key="test_model",
        model_name="window_3",
        model_label="Test model",
        predictor=lambda fold, window: np.array([2.0]),
        sequence_length=3,
        model_parameters={},
        output_dir=tmp_path,
        experiment="runtime_test",
    )

    assert saved["metrics"][0]["runtime_seconds"] == pytest.approx(2.5)
    assert saved["config"]["runtime"]["clock"] == "time.perf_counter"
    assert saved["config"]["runtime"]["scope"] == (
        "model build + training + test inference"
    )
    assert saved["config"]["runtime"]["total_fit_predict_seconds"] == pytest.approx(
        2.5
    )
    assert saved["config"]["runtime"]["total_experiment_seconds"] == pytest.approx(
        4.0
    )


def test_summarize_window_metrics_compares_every_window_with_window_20():
    rows = []
    for model, offset in [("lstm", 0.0), ("cnn", 10.0)]:
        for window, rmse, direction_accuracy in [
            (1, 10.0 + offset, 0.55),
            (3, 9.0 + offset, 0.52),
            (20, 12.0 + offset, 0.50),
        ]:
            for fold_index in range(2):
                rows.append(
                    {
                        "model": model,
                        "sequence_window": window,
                        "fold": f"fold_{fold_index + 1}",
                        "rmse": rmse + fold_index,
                        "mae": rmse - 1.0 + fold_index,
                        "mape": 1.0,
                        "r2": 0.5,
                        "direction_accuracy": direction_accuracy,
                        "runtime_seconds": 2.0 + fold_index,
                    }
                )

    summary = window_sweep.summarize_window_metrics(pd.DataFrame(rows))
    lstm_window_1 = summary[
        (summary["model"] == "lstm")
        & (summary["sequence_window"] == 1)
    ].iloc[0]

    assert lstm_window_1["rmse_mean"] == pytest.approx(10.5)
    assert lstm_window_1["rmse_delta_vs_window_20"] == pytest.approx(-2.0)
    assert lstm_window_1["direction_accuracy_delta_vs_window_20_pp"] == pytest.approx(
        5.0
    )
    assert lstm_window_1["direction_accuracy_rank"] == 1
    assert lstm_window_1["runtime_seconds_total"] == pytest.approx(5.0)


def test_select_best_windows_reports_direction_and_rmse_choices():
    summary = pd.DataFrame(
        {
            "model": ["lstm", "lstm", "cnn", "cnn"],
            "sequence_window": [1, 3, 1, 3],
            "rmse_mean": [11.0, 9.0, 20.0, 21.0],
            "direction_accuracy_mean": [0.55, 0.52, 0.49, 0.53],
        }
    )

    best = window_sweep.select_best_windows(summary)

    lstm = best[best["model"] == "lstm"].iloc[0]
    cnn = best[best["model"] == "cnn"].iloc[0]
    assert lstm["best_direction_window"] == 1
    assert lstm["best_rmse_window"] == 3
    assert cnn["best_direction_window"] == 3
    assert cnn["best_rmse_window"] == 1


def test_completed_window_sweep_outputs_cover_all_combinations():
    fold_metrics = pd.read_csv(window_sweep.WINDOW_FOLD_METRICS_FILE)
    summary = pd.read_csv(window_sweep.WINDOW_SUMMARY_FILE)
    best = pd.read_csv(window_sweep.BEST_WINDOWS_FILE)
    runtime = pd.read_csv(window_sweep.WINDOW_RUNTIME_FILE)
    paper_table = pd.read_csv(window_sweep.PAPER_BEST_WINDOWS_FILE)
    runtime_environment = json.loads(
        window_sweep.RUNTIME_ENVIRONMENT_FILE.read_text(encoding="utf-8")
    )

    assert len(fold_metrics) == 5 * 5 * 4
    assert len(summary) == 5 * 5
    assert len(best) == 5
    assert len(runtime) == 5 * 5
    assert len(paper_table) == 5
    assert set(fold_metrics["model"]) == set(window_sweep.WINDOW_SWEEP_MODELS)
    assert set(fold_metrics["sequence_window"]) == set(
        window_sweep.SEQUENCE_WINDOWS
    )
    assert (
        fold_metrics.groupby(["model", "sequence_window"])["fold"]
        .nunique()
        .eq(4)
        .all()
    )
    assert np.isfinite(
        fold_metrics.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    ).all()
    assert np.isfinite(
        summary.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    ).all()
    assert (fold_metrics["runtime_seconds"] > 0.0).all()
    assert (runtime["runtime_seconds_total"] > 0.0).all()
    assert set(paper_table["model"]) == set(window_sweep.WINDOW_SWEEP_MODELS)
    assert paper_table["seed"].eq(42).all()
    assert paper_table["vmd_window"].eq(60).all()
    assert runtime_environment["runtime_clock"] == "time.perf_counter"
    assert runtime_environment["framework_startup_excluded"] is True

    for model in window_sweep.WINDOW_SWEEP_MODELS:
        for window in window_sweep.NEW_SEQUENCE_WINDOWS:
            path = (
                window_sweep.WINDOW_SWEEP_OUTPUT_DIR
                / model
                / f"window_{window}"
                / "metrics_by_fold.csv"
            )
            assert Path(path).exists()
