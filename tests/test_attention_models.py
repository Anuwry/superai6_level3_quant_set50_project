import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import models.attention_lstm as attention_lstm
import models.attention_lstm_cnn as attention_lstm_cnn
from models.baseline_common import FoldData, FoldSpec, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTENTION_MODULES = [attention_lstm, attention_lstm_cnn]


def test_attention_model_configurations_are_causal_and_comparable():
    assert attention_lstm.MODEL_NAME == "attention_lstm"
    assert attention_lstm_cnn.MODEL_NAME == "attention_lstm_cnn"
    assert attention_lstm.SEQUENCE_LENGTH == 20
    assert attention_lstm_cnn.SEQUENCE_LENGTH == 20
    assert attention_lstm.CONFIG["model"] == "Keras Attention-LSTM"
    assert attention_lstm_cnn.CONFIG["model"] == "Keras Attention-LSTM-CNN"
    assert attention_lstm.CONFIG["model_parameters"]["causal_attention"] is True
    assert attention_lstm_cnn.CONFIG["model_parameters"]["causal_attention"] is True
    assert attention_lstm_cnn.CONFIG["model_parameters"]["padding"] == "causal"


def fake_tensorflow(recorded_layers):
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
                    "call_kwargs": {},
                }
            )
            return FakeTensor("Input")

        @staticmethod
        def LSTM(*args, **kwargs):
            return layer_factory("LSTM", args, kwargs)

        @staticmethod
        def MultiHeadAttention(*args, **kwargs):
            return layer_factory("MultiHeadAttention", args, kwargs)

        @staticmethod
        def Conv1D(*args, **kwargs):
            return layer_factory("Conv1D", args, kwargs)

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

    return SimpleNamespace(
        keras=SimpleNamespace(Model=FakeModel, layers=FakeLayers)
    )


@pytest.mark.parametrize(
    ("module", "builder_name", "expected_layers"),
    [
        (
            attention_lstm,
            "build_attention_lstm_model",
            [
                "Input",
                "LSTM",
                "MultiHeadAttention",
                "GlobalAveragePooling1D",
                "Dense",
                "Dense",
            ],
        ),
        (
            attention_lstm_cnn,
            "build_attention_lstm_cnn_model",
            [
                "Input",
                "LSTM",
                "MultiHeadAttention",
                "Conv1D",
                "GlobalAveragePooling1D",
                "Dense",
                "Dense",
            ],
        ),
    ],
)
def test_attention_builders_use_causal_self_attention(
    monkeypatch,
    module,
    builder_name,
    expected_layers,
):
    recorded_layers = []
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow(recorded_layers))

    model = getattr(module, builder_name)((20, 5))

    attention_layer = next(
        layer for layer in recorded_layers if layer["name"] == "MultiHeadAttention"
    )
    lstm_layer = next(layer for layer in recorded_layers if layer["name"] == "LSTM")
    assert [layer["name"] for layer in recorded_layers] == expected_layers
    assert lstm_layer["constructor_kwargs"]["return_sequences"] is True
    assert attention_layer["call_kwargs"]["use_causal_mask"] is True
    assert model.compile_kwargs == {"optimizer": "adam", "loss": "mse"}
    if module is attention_lstm_cnn:
        conv_layer = next(
            layer for layer in recorded_layers if layer["name"] == "Conv1D"
        )
        assert conv_layer["constructor_kwargs"]["padding"] == "causal"


@pytest.mark.parametrize(
    ("module", "builder_name"),
    [
        (attention_lstm, "build_attention_lstm_model"),
        (attention_lstm_cnn, "build_attention_lstm_cnn_model"),
    ],
)
def test_attention_predict_fold_returns_one_prediction_per_test_row(
    monkeypatch,
    module,
    builder_name,
):
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
            return np.array([[6.3], [6.7]])

    monkeypatch.setattr(module, "set_reproducible_seed", lambda: None)
    monkeypatch.setattr(module, builder_name, lambda input_shape: FakeModel())

    predictions = module.predict_fold(fold, sequence_length=3)

    assert predictions.tolist() == [6.3, 6.7]


@pytest.mark.parametrize("module", ATTENTION_MODULES)
def test_attention_main_saves_original_unit_predictions(monkeypatch, module):
    spec = SimpleNamespace(fold="fold_1")
    fold = SimpleNamespace()
    saved = {}
    missing_folds = PROJECT_ROOT / f"__missing_{module.MODEL_NAME}_test_folds__"
    assert not missing_folds.exists()
    monkeypatch.setattr(module, "NN_DATA_FOLDS_DIR", missing_folds)
    monkeypatch.setattr(
        module,
        "create_neural_network_folds",
        lambda: saved.setdefault("created", True),
    )
    monkeypatch.setattr(module, "discover_folds", lambda data_dir: [spec])
    monkeypatch.setattr(module, "load_fold", lambda current_spec: fold)
    monkeypatch.setattr(
        module,
        "load_scaler_metadata",
        lambda fold_name: {"columns": [TARGET_COLUMN], "scale": [1.0], "min": [0.0]},
    )
    monkeypatch.setattr(module, "predict_fold", lambda current_fold: np.array([10.0]))
    monkeypatch.setattr(
        module,
        "evaluate_predictions",
        lambda current_fold, prediction: {"fold": "fold_1", "rmse": 0.0},
    )
    monkeypatch.setattr(
        module,
        "predictions_frame",
        lambda current_fold, prediction: pd.DataFrame({"y_pred": prediction}),
    )
    monkeypatch.setattr(
        module,
        "save_run_outputs",
        lambda *args, **kwargs: saved.setdefault("output_args", (args, kwargs)),
    )
    monkeypatch.setattr(
        module,
        "print_metrics",
        lambda metrics: saved.setdefault("printed", metrics),
    )

    metrics = module.main()

    assert saved["created"] is True
    assert saved["output_args"][0][0] == module.MODEL_NAME
    assert metrics.loc[0, "fold"] == "fold_1"


@pytest.mark.parametrize(
    ("module_name", "model_name"),
    [
        ("attention_lstm", "attention_lstm"),
        ("attention_lstm_cnn", "attention_lstm_cnn"),
    ],
)
def test_attention_models_have_runner_and_visualization_notebooks(
    module_name,
    model_name,
):
    model_notebook = (
        PROJECT_ROOT / "models" / f"{module_name}.ipynb"
    ).read_text(encoding="utf-8")
    baseline_notebook = (
        PROJECT_ROOT / "baseline_model_test_runner.ipynb"
    ).read_text(encoding="utf-8")
    visualization_notebook = (
        PROJECT_ROOT
        / "prediction_visualizations"
        / f"{model_name}_predictions.ipynb"
    ).read_text(encoding="utf-8")

    json.loads(model_notebook)
    json.loads(visualization_notebook)
    assert f"from models.{module_name} import main" in model_notebook
    assert baseline_notebook.count(
        f"{model_name}_metrics = run_{model_name}()"
    ) == 1
    assert f"MODEL_ALIASES = ('{model_name}',)" in visualization_notebook
