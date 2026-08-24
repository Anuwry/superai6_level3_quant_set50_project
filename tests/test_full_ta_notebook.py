import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "full_ta_model_test_runner.ipynb"


def test_full_ta_notebook_has_separate_model_cells():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    expected_calls = [
        "run_ridge_full_ta()",
        "run_xgboost_full_ta()",
        "run_lightgbm_full_ta()",
        "run_autogluon_full_ta()",
        "run_lstm_full_ta()",
        "run_cnn_full_ta()",
        "run_lstm_cnn_full_ta()",
        "run_attention_lstm_full_ta()",
        "run_attention_lstm_cnn_full_ta()",
        "run_chronos_full_ta_reference()",
    ]

    for call in expected_calls:
        assert sum(call in source for source in code_sources) == 1


def test_full_ta_notebook_excludes_tuning_shap_and_chronos_2():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    notebook_text = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ).lower()

    assert "hyperparameter_tuning\": true" not in notebook_text
    assert "chronos-2" not in notebook_text
    assert "shap" not in notebook_text
