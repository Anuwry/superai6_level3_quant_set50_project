import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import models.full_non_ta_experiments as full_non_ta_experiments
import models.full_ta_experiments as full_ta_experiments
import models.run_all_baselines as run_all_baselines
import models.lstm_cnn_attention as lstm_cnn_attention
from models.baseline_common import FoldData, FoldSpec, TARGET_COLUMN


def test_configuration_matches_existing_sequence_model_hyperparameters():
    assert lstm_cnn_attention.MODEL_NAME == "lstm_cnn_attention"
    assert lstm_cnn_attention.BENCHMARK_SEEDS == (42, 123, 456, 789, 2025)
    assert lstm_cnn_attention.SEQUENCE_LENGTH == 20
    assert lstm_cnn_attention.EPOCHS == 20
    assert lstm_cnn_attention.BATCH_SIZE == 32
    assert lstm_cnn_attention.LSTM_UNITS == 16
    assert lstm_cnn_attention.CONV_FILTERS == 32
    assert lstm_cnn_attention.KERNEL_SIZE == 3
    assert lstm_cnn_attention.ATTENTION_HEADS == 2
    assert lstm_cnn_attention.ATTENTION_KEY_DIM == 8
    assert lstm_cnn_attention.DENSE_UNITS == 8
    assert lstm_cnn_attention.CONFIG["model_parameters"]["layer_order"] == [
        "LSTM",
        "Conv1D",
        "MultiHeadAttention",
        "GlobalAveragePooling1D",
        "Dense",
    ]


def test_builder_applies_cnn_before_causal_self_attention(monkeypatch):
    recorded_layers = []

    class FakeTensor:
        def __init__(self, name):
            self.name = name

    def layer_factory(name, constructor_args, constructor_kwargs):
        def apply(*inputs, **call_kwargs):
            recorded_layers.append(
                {
                    "name": name,
                    "constructor_args": constructor_args,
                    "constructor_kwargs": constructor_kwargs,
                    "input_names": [value.name for value in inputs],
                    "call_kwargs": call_kwargs,
                }
            )
            return FakeTensor(name)

        return apply

    class FakeLayers:
        @staticmethod
        def Input(**kwargs):
            recorded_layers.append(
                {
                    "name": "Input",
                    "constructor_args": (),
                    "constructor_kwargs": kwargs,
                    "input_names": [],
                    "call_kwargs": {},
                }
            )
            return FakeTensor("Input")

        @staticmethod
        def LSTM(*args, **kwargs):
            return layer_factory("LSTM", args, kwargs)

        @staticmethod
        def Conv1D(*args, **kwargs):
            return layer_factory("Conv1D", args, kwargs)

        @staticmethod
        def MultiHeadAttention(*args, **kwargs):
            return layer_factory("MultiHeadAttention", args, kwargs)

        @staticmethod
        def GlobalAveragePooling1D(*args, **kwargs):
            return layer_factory("GlobalAveragePooling1D", args, kwargs)

        @staticmethod
        def Dense(*args, **kwargs):
            return layer_factory("Dense", args, kwargs)

    class FakeModel:
        def __init__(self, inputs, outputs):
            self.inputs = inputs
            self.outputs = outputs
            self.compile_kwargs = None

        def compile(self, **kwargs):
            self.compile_kwargs = kwargs

    fake_tensorflow = SimpleNamespace(
        keras=SimpleNamespace(Model=FakeModel, layers=FakeLayers)
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow)

    model = lstm_cnn_attention.build_lstm_cnn_attention_model((20, 5))

    assert [layer["name"] for layer in recorded_layers] == [
        "Input",
        "LSTM",
        "Conv1D",
        "MultiHeadAttention",
        "GlobalAveragePooling1D",
        "Dense",
        "Dense",
    ]
    lstm_layer = next(layer for layer in recorded_layers if layer["name"] == "LSTM")
    conv_layer = next(layer for layer in recorded_layers if layer["name"] == "Conv1D")
    attention_layer = next(
        layer for layer in recorded_layers if layer["name"] == "MultiHeadAttention"
    )
    assert lstm_layer["constructor_kwargs"]["return_sequences"] is True
    assert conv_layer["constructor_kwargs"]["padding"] == "causal"
    assert attention_layer["input_names"] == ["Conv1D", "Conv1D"]
    assert attention_layer["call_kwargs"]["use_causal_mask"] is True
    assert model.compile_kwargs == {"optimizer": "adam", "loss": "mse"}


def test_predict_fold_uses_requested_seed_and_returns_finite_predictions(monkeypatch):
    spec = FoldSpec("fold_1", Path("train.csv"), Path("test.csv"), 2020, 2020, 2021)
    train = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=4),
            "Close_D": [1.0, 2.0, 3.0, 4.0],
            "Volume_D": [10.0, 11.0, 12.0, 13.0],
            TARGET_COLUMN: [2.0, 3.0, 4.0, 5.0],
        }
    )
    test = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=2),
            "Close_D": [5.0, 6.0],
            "Volume_D": [14.0, 15.0],
            TARGET_COLUMN: [6.0, 7.0],
        }
    )
    fold = FoldData(spec, train, test, ["Close_D", "Volume_D"])
    recorded = {}

    class FakeModel:
        def fit(self, x_values, y_values, **kwargs):
            assert x_values.shape == (2, 3, 2)
            assert y_values.tolist() == [4.0, 5.0]
            assert kwargs["shuffle"] is False

        def predict(self, x_values, **kwargs):
            assert x_values.shape == (2, 3, 2)
            return np.array([[6.1], [6.9]])

    monkeypatch.setattr(
        lstm_cnn_attention,
        "set_reproducible_seed",
        lambda seed: recorded.setdefault("seed", seed),
    )
    monkeypatch.setattr(
        lstm_cnn_attention,
        "build_lstm_cnn_attention_model",
        lambda input_shape: FakeModel(),
    )

    predictions = lstm_cnn_attention.predict_fold(
        fold,
        sequence_length=3,
        random_seed=123,
    )

    assert recorded["seed"] == 123
    assert predictions.tolist() == [6.1, 6.9]
    assert np.isfinite(predictions).all()


def test_summarize_seed_metrics_reports_seed_mean_and_cross_seed_std():
    metrics = pd.DataFrame(
        {
            "seed": [42, 42, 123, 123],
            "fold": ["fold_1", "fold_2", "fold_1", "fold_2"],
            "rmse": [10.0, 20.0, 14.0, 18.0],
            "mae": [8.0, 16.0, 10.0, 14.0],
            "mape": [1.0, 2.0, 1.4, 1.8],
            "r2": [0.1, 0.2, 0.14, 0.18],
            "direction_accuracy": [0.50, 0.60, 0.54, 0.58],
        }
    )

    by_seed, mean_std = lstm_cnn_attention.summarize_seed_metrics(metrics)

    assert by_seed["seed"].tolist() == [42, 123]
    assert by_seed["rmse"].tolist() == [15.0, 16.0]
    assert mean_std.loc[0, "rmse_mean"] == pytest.approx(15.5)
    assert mean_std.loc[0, "rmse_std"] == pytest.approx(np.std([15.0, 16.0], ddof=1))
    assert mean_std.loc[0, "direction_accuracy_mean"] == pytest.approx(0.555)


@pytest.mark.parametrize("seeds", [(), (42, 42), (42, "123")])
def test_validate_benchmark_seeds_rejects_invalid_seed_sets(seeds):
    with pytest.raises(ValueError):
        lstm_cnn_attention.validate_benchmark_seeds(seeds)


def test_new_model_is_exposed_by_all_three_benchmark_runners():
    assert lstm_cnn_attention.main in run_all_baselines.RUNNERS
    assert callable(full_non_ta_experiments.run_lstm_cnn_attention_full_non_ta)
    assert callable(full_ta_experiments.run_lstm_cnn_attention_full_ta)


def test_multi_seed_runner_saves_independent_predictions_and_summaries(
    tmp_path,
    monkeypatch,
):
    scaled_spec = FoldSpec(
        "fold_1",
        Path("scaled_train.csv"),
        Path("scaled_test.csv"),
        2020,
        2020,
        2021,
    )
    original_spec = FoldSpec(
        "fold_1",
        Path("original_train.csv"),
        Path("original_test.csv"),
        2020,
        2020,
        2021,
    )
    train = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=4),
            "Close_D": [1.0, 2.0, 3.0, 4.0],
            TARGET_COLUMN: [2.0, 3.0, 4.0, 5.0],
        }
    )
    test = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=2),
            "Close_D": [5.0, 6.0],
            TARGET_COLUMN: [6.0, 7.0],
        }
    )
    scaled_fold = FoldData(scaled_spec, train, test, ["Close_D"])
    original_fold = FoldData(original_spec, train, test, ["Close_D"])
    scaled_dir = tmp_path / "scaled"
    original_dir = tmp_path / "original"

    monkeypatch.setattr(
        lstm_cnn_attention,
        "discover_folds",
        lambda data_dir: [scaled_spec] if data_dir == scaled_dir else [original_spec],
    )
    monkeypatch.setattr(
        lstm_cnn_attention,
        "load_fold",
        lambda spec: scaled_fold if spec is scaled_spec else original_fold,
    )

    def predictor(fold, sequence_length, seed):
        assert fold is scaled_fold
        assert sequence_length == 20
        offset = 0.1 if seed == 42 else 0.2
        return np.array([6.0 + offset, 7.0 - offset])

    metrics = lstm_cnn_attention.run_multi_seed_benchmark(
        "test_model",
        scaled_dir,
        original_dir,
        lambda fold_name: {
            "columns": ["Close_D", TARGET_COLUMN],
            "scale": [1.0, 1.0],
            "min": [0.0, 0.0],
        },
        tmp_path / "outputs",
        {"model": "test"},
        seeds=(42, 123),
        predictor=predictor,
    )

    model_dir = tmp_path / "outputs" / "test_model"
    assert len(metrics) == 2
    assert metrics["seed"].tolist() == [42, 123]
    assert (model_dir / "seed_42" / "predictions_fold_1.csv").exists()
    assert (model_dir / "seed_123" / "predictions_fold_1.csv").exists()
    assert (model_dir / "metrics_by_seed_and_fold.csv").exists()
    assert (model_dir / "metrics_by_fold_mean_std.csv").exists()
    assert (model_dir / "metrics_mean_std_across_seeds.csv").exists()
    assert (model_dir / "multi_seed_run_metadata.json").exists()
