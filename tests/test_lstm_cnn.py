import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import models.lstm_cnn as lstm_cnn
from models.baseline_common import FoldData, FoldSpec, TARGET_COLUMN
from models.lstm_cnn import (
    CONFIG,
    KERNEL_SIZE,
    MODEL_NAME,
    SEQUENCE_LENGTH,
    build_lstm_cnn_model,
    predict_fold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_lstm_cnn_configuration_is_directly_comparable_to_lstm_and_cnn():
    assert MODEL_NAME == "lstm_cnn"
    assert SEQUENCE_LENGTH == 20
    assert KERNEL_SIZE == 3
    assert CONFIG["model"] == "Keras LSTM-CNN"
    assert CONFIG["model_parameters"]["layer_order"] == [
        "LSTM",
        "Conv1D",
        "GlobalAveragePooling1D",
        "Dense",
    ]
    assert CONFIG["model_parameters"]["padding"] == "causal"


def test_build_lstm_cnn_model_preserves_sequence_before_causal_convolution(monkeypatch):
    created_layers = []

    class FakeLayers:
        @staticmethod
        def Input(**kwargs):
            created_layers.append(("Input", kwargs))
            return created_layers[-1]

        @staticmethod
        def LSTM(*args, **kwargs):
            created_layers.append(("LSTM", {"args": args, **kwargs}))
            return created_layers[-1]

        @staticmethod
        def Conv1D(**kwargs):
            created_layers.append(("Conv1D", kwargs))
            return created_layers[-1]

        @staticmethod
        def GlobalAveragePooling1D():
            created_layers.append(("GlobalAveragePooling1D", {}))
            return created_layers[-1]

        @staticmethod
        def Dense(*args, **kwargs):
            created_layers.append(("Dense", {"args": args, **kwargs}))
            return created_layers[-1]

    class FakeSequential:
        def __init__(self, layers):
            self.layers = layers
            self.compile_kwargs = None

        def compile(self, **kwargs):
            self.compile_kwargs = kwargs

    fake_tensorflow = SimpleNamespace(
        keras=SimpleNamespace(Sequential=FakeSequential, layers=FakeLayers)
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow)

    model = build_lstm_cnn_model((SEQUENCE_LENGTH, 5))

    layer_names = [name for name, _ in created_layers]
    lstm_layer = next(layer for layer in created_layers if layer[0] == "LSTM")
    conv_layer = next(layer for layer in created_layers if layer[0] == "Conv1D")
    assert layer_names == [
        "Input",
        "LSTM",
        "Conv1D",
        "GlobalAveragePooling1D",
        "Dense",
        "Dense",
    ]
    assert lstm_layer[1]["return_sequences"] is True
    assert conv_layer[1]["padding"] == "causal"
    assert model.compile_kwargs == {"optimizer": "adam", "loss": "mse"}


def test_predict_fold_trains_and_returns_one_prediction_per_test_row(monkeypatch):
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

    class FakeModel:
        def fit(self, x_values, y_values, **kwargs):
            assert x_values.shape == (2, 3, 2)
            assert y_values.tolist() == [4.0, 5.0]
            assert kwargs["shuffle"] is False

        def predict(self, x_values, **kwargs):
            assert x_values.shape == (2, 3, 2)
            return np.array([[6.2], [6.8]])

    monkeypatch.setattr(lstm_cnn, "set_reproducible_seed", lambda: None)
    monkeypatch.setattr(
        lstm_cnn,
        "build_lstm_cnn_model",
        lambda input_shape: FakeModel(),
    )

    predictions = predict_fold(fold, sequence_length=3)

    assert predictions.tolist() == [6.2, 6.8]


def test_main_creates_missing_folds_and_saves_original_unit_predictions(monkeypatch):
    spec = SimpleNamespace(fold="fold_1")
    fold = SimpleNamespace()
    saved = {}
    missing_folds = PROJECT_ROOT / "__missing_lstm_cnn_test_folds__"
    assert not missing_folds.exists()
    monkeypatch.setattr(lstm_cnn, "NN_DATA_FOLDS_DIR", missing_folds)
    monkeypatch.setattr(
        lstm_cnn,
        "create_neural_network_folds",
        lambda: saved.setdefault("created", True),
    )
    monkeypatch.setattr(lstm_cnn, "discover_folds", lambda data_dir: [spec])
    monkeypatch.setattr(lstm_cnn, "load_fold", lambda current_spec: fold)
    monkeypatch.setattr(
        lstm_cnn,
        "load_scaler_metadata",
        lambda fold_name: {"columns": [TARGET_COLUMN], "scale": [1.0], "min": [0.0]},
    )
    monkeypatch.setattr(lstm_cnn, "predict_fold", lambda current_fold: np.array([10.0]))
    monkeypatch.setattr(
        lstm_cnn,
        "evaluate_predictions",
        lambda current_fold, prediction: {"fold": "fold_1", "rmse": 0.0},
    )
    monkeypatch.setattr(
        lstm_cnn,
        "predictions_frame",
        lambda current_fold, prediction: pd.DataFrame({"y_pred": prediction}),
    )
    monkeypatch.setattr(
        lstm_cnn,
        "save_run_outputs",
        lambda *args, **kwargs: saved.setdefault("output_args", (args, kwargs)),
    )
    monkeypatch.setattr(
        lstm_cnn,
        "print_metrics",
        lambda metrics: saved.setdefault("printed", metrics),
    )

    metrics = lstm_cnn.main()

    assert saved["created"] is True
    assert saved["output_args"][0][0] == MODEL_NAME
    assert metrics.loc[0, "fold"] == "fold_1"


def test_lstm_cnn_has_runner_and_visualization_notebooks():
    model_notebook = (
        PROJECT_ROOT / "models" / "lstm_cnn.ipynb"
    ).read_text(encoding="utf-8")
    baseline_notebook = (
        PROJECT_ROOT / "baseline_model_test_runner.ipynb"
    ).read_text(encoding="utf-8")
    visualization_notebook = (
        PROJECT_ROOT / "prediction_visualizations" / "lstm_cnn_predictions.ipynb"
    ).read_text(encoding="utf-8")

    json.loads(model_notebook)
    json.loads(visualization_notebook)
    assert "from models.lstm_cnn import main" in model_notebook
    assert baseline_notebook.count("lstm_cnn_metrics = run_lstm_cnn()") == 1
    assert "MODEL_ALIASES = ('lstm_cnn',)" in visualization_notebook
