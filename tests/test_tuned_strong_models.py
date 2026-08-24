import json
from pathlib import Path

import optuna

import models.tuned_strong_models as tuned_models
from models.tuned_strong_models import (
    AUTOGLUON_TABULAR_TIME_LIMIT,
    AUTOGLUON_TIMESERIES_TIME_LIMIT,
    CHRONOS2_BASE_MODEL_ID,
    CHRONOS2_SMALL_MODEL_ID,
    OPTUNA_TRIALS,
)


def test_requested_trial_budgets_are_fixed():
    assert OPTUNA_TRIALS == {
        "ridge": 25,
        "xgboost": 75,
        "lightgbm": 75,
        "lstm": 40,
    }


def test_strong_model_configuration_is_fixed():
    assert AUTOGLUON_TABULAR_TIME_LIMIT == 300
    assert AUTOGLUON_TIMESERIES_TIME_LIMIT == 600
    assert CHRONOS2_SMALL_MODEL_ID == "autogluon/chronos-2-small"
    assert CHRONOS2_BASE_MODEL_ID == "amazon/chronos-2"


def test_tuning_notebook_has_separate_model_cells():
    path = Path(__file__).resolve().parents[1] / "tuned_strong_model_runner.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    expected_calls = [
        "run_ridge_optuna()",
        "run_xgboost_optuna()",
        "run_lightgbm_optuna()",
        "run_lstm_optuna()",
        "run_autogluon_tabular_strong()",
        "run_autogluon_timeseries_strong()",
        "run_chronos2_small_zero_shot()",
        "run_chronos2_base_zero_shot()",
        "run_chronos2_ensemble_strong()",
    ]

    for call in expected_calls:
        assert sum(call in source for source in code.split("\n\n")) == 1


def test_optuna_study_is_persistent_and_resumable(tmp_path, monkeypatch):
    monkeypatch.setattr(tuned_models, "OUTPUT_DIR", tmp_path)
    study = tuned_models.create_study("ridge_test", "fold_1")
    study.optimize(lambda trial: trial.suggest_float("alpha", 0.0, 1.0), n_trials=1)

    resumed = tuned_models.create_study("ridge_test", "fold_1")

    assert tuned_models.completed_trial_count(resumed) == 1
    assert tuned_models.remaining_trial_count(resumed, target_trials=3) == 2
    assert (tmp_path / "ridge_test" / "optuna" / "fold_1" / "study.db").exists()
