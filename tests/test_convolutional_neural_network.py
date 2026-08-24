import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import models.convolutional_neural_network as cnn
from models.baseline_common import FoldData, FoldSpec, TARGET_COLUMN
from models.convolutional_neural_network import (
    CONFIG,
    KERNEL_SIZE,
    MODEL_NAME,
    SEQUENCE_LENGTH,
    build_cnn_model,
    make_sequences,
    make_test_sequences,
    predict_fold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cnn_configuration_matches_the_lstm_comparison_window():
    assert MODEL_NAME == "cnn"
    assert SEQUENCE_LENGTH == 20
    assert KERNEL_SIZE == 3
    assert CONFIG["model"] == "Keras 1D CNN"
    assert CONFIG["model_parameters"]["padding"] == "causal"


def test_make_sequences_preserves_temporal_order_and_target_alignment():
    features = np.arange(12, dtype=float).reshape(6, 2)
    target = np.arange(100, 106, dtype=float)

    x_values, y_values = make_sequences(features, target, sequence_length=3)

    assert x_values.shape == (4, 3, 2)
    assert y_values.tolist() == [102.0, 103.0, 104.0, 105.0]
    np.testing.assert_array_equal(x_values[0], features[:3])
    np.testing.assert_array_equal(x_values[-1], features[-3:])


def test_make_sequences_rejects_invalid_inputs():
    features = np.ones((3, 2), dtype=float)

    with pytest.raises(ValueError, match="same number of rows"):
        make_sequences(features, np.ones(2), sequence_length=2)

    with pytest.raises(ValueError, match="sequence_length"):
        make_sequences(features, np.ones(3), sequence_length=4)


def test_make_test_sequences_uses_only_history_available_at_each_step():
    train = np.arange(8, dtype=float).reshape(4, 2)
    test = np.arange(8, 14, dtype=float).reshape(3, 2)

    sequences = make_test_sequences(train, test, sequence_length=3)

    assert sequences.shape == (3, 3, 2)
    np.testing.assert_array_equal(sequences[0], np.vstack([train[-2:], test[:1]]))
    np.testing.assert_array_equal(sequences[-1], test)


def test_build_cnn_model_compiles_causal_conv1d(monkeypatch):
    created_layers = []

    class FakeLayers:
        @staticmethod
        def Input(**kwargs):
            created_layers.append(("Input", kwargs))
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

    model = build_cnn_model((SEQUENCE_LENGTH, 5))

    conv_layer = next(layer for layer in created_layers if layer[0] == "Conv1D")
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
            return np.array([[6.1], [6.9]])

    monkeypatch.setattr(cnn, "set_reproducible_seed", lambda: None)
    monkeypatch.setattr(cnn, "build_cnn_model", lambda input_shape: FakeModel())

    predictions = predict_fold(fold, sequence_length=3)

    assert predictions.tolist() == [6.1, 6.9]


def test_predict_fold_inserts_context_only_row_into_first_test_sequence(
    monkeypatch,
):
    spec = FoldSpec(
        "fold_1",
        Path("train.csv"),
        Path("test.csv"),
        2020,
        2020,
        2021,
    )
    train = pd.DataFrame(
        {
            "Date": pd.date_range("2020-12-28", periods=3),
            "Close_D": [1.0, 2.0, 3.0],
            "Volume_D": [10.0, 11.0, 12.0],
            TARGET_COLUMN: [2.0, 3.0, 4.0],
        }
    )
    context = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-12-31")],
            "Close_D": [4.0],
            "Volume_D": [13.0],
            TARGET_COLUMN: [5.0],
        }
    )
    test = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2021-01-04")],
            "Close_D": [5.0],
            "Volume_D": [14.0],
            TARGET_COLUMN: [6.0],
        }
    )
    fold = FoldData(
        spec,
        train,
        test,
        ["Close_D", "Volume_D"],
        context=context,
    )

    class FakeModel:
        def fit(self, *_args, **_kwargs):
            return None

        def predict(self, x_values, **_kwargs):
            np.testing.assert_array_equal(
                x_values[0],
                np.array(
                    [[3.0, 12.0], [4.0, 13.0], [5.0, 14.0]],
                    dtype=np.float32,
                ),
            )
            return np.array([[6.0]])

    monkeypatch.setattr(cnn, "set_reproducible_seed", lambda: None)
    monkeypatch.setattr(cnn, "build_cnn_model", lambda _shape: FakeModel())

    prediction = predict_fold(fold, sequence_length=3)

    assert prediction.tolist() == [6.0]


def test_main_creates_missing_folds_and_saves_original_unit_predictions(monkeypatch):
    spec = SimpleNamespace(fold="fold_1")
    fold = SimpleNamespace()
    saved = {}
    missing_folds = PROJECT_ROOT / "__missing_cnn_test_folds__"
    assert not missing_folds.exists()
    monkeypatch.setattr(cnn, "NN_DATA_FOLDS_DIR", missing_folds)
    monkeypatch.setattr(cnn, "create_neural_network_folds", lambda: saved.setdefault("created", True))
    monkeypatch.setattr(cnn, "discover_folds", lambda data_dir: [spec])
    monkeypatch.setattr(cnn, "load_fold", lambda current_spec: fold)
    monkeypatch.setattr(
        cnn,
        "load_scaler_metadata",
        lambda fold_name: {"columns": [TARGET_COLUMN], "scale": [1.0], "min": [0.0]},
    )
    monkeypatch.setattr(cnn, "predict_fold", lambda current_fold: np.array([10.0]))
    monkeypatch.setattr(
        cnn,
        "evaluate_predictions",
        lambda current_fold, prediction: {"fold": "fold_1", "rmse": 0.0},
    )
    monkeypatch.setattr(
        cnn,
        "predictions_frame",
        lambda current_fold, prediction: pd.DataFrame({"y_pred": prediction}),
    )
    monkeypatch.setattr(
        cnn,
        "save_run_outputs",
        lambda *args, **kwargs: saved.setdefault("output_args", (args, kwargs)),
    )
    monkeypatch.setattr(cnn, "print_metrics", lambda metrics: saved.setdefault("printed", metrics))

    metrics = cnn.main()

    assert saved["created"] is True
    assert saved["output_args"][0][0] == MODEL_NAME
    assert metrics.loc[0, "fold"] == "fold_1"


def test_cnn_has_a_dedicated_notebook_and_baseline_runner_cell():
    notebook = json.loads(
        (PROJECT_ROOT / "models" / "convolutional_neural_network.ipynb").read_text(encoding="utf-8")
    )
    notebook_text = json.dumps(notebook)
    baseline_text = (PROJECT_ROOT / "baseline_model_test_runner.ipynb").read_text(encoding="utf-8")

    assert "from models.convolutional_neural_network import main" in notebook_text
    assert baseline_text.count("cnn_metrics = run_cnn()") == 1


def test_cnn_has_a_saved_prediction_visualization_notebook():
    visualization_text = (
        PROJECT_ROOT / "prediction_visualizations" / "cnn_predictions.ipynb"
    ).read_text(encoding="utf-8")

    assert "MODEL_ALIASES = ('cnn',)" in visualization_text
    assert "load_predictions" in visualization_text
    assert "plot_predictions" in visualization_text
