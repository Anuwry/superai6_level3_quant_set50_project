from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]

from models.baseline_common import OUTPUT_DIR, TARGET_COLUMN, FoldData, print_metrics, run_model_on_folds

MODEL_NAME = "autogluon"
TIME_LIMIT_SECONDS = 120
CONFIG = {
    "experiment": "naive_baseline",
    "model": "AutoGluon TabularPredictor",
    "hyperparameter_tuning": False,
    "presets": "medium_quality",
    "time_limit": TIME_LIMIT_SECONDS,
    "hyperparameters": "default",
}


def predict_fold(fold: FoldData) -> np.ndarray:
    from autogluon.tabular import TabularPredictor

    model_path = OUTPUT_DIR / MODEL_NAME / "models" / fold.spec.fold
    if model_path.exists():
        shutil.rmtree(model_path)
    train_data = fold.train.drop(columns=["Date"]).copy()
    test_data = fold.test.drop(columns=["Date", TARGET_COLUMN]).copy()
    predictor = TabularPredictor(label=TARGET_COLUMN, problem_type="regression", path=str(model_path), verbosity=1)
    predictor.fit(
        train_data=train_data,
        presets="medium_quality",
        time_limit=TIME_LIMIT_SECONDS,
        hyperparameters="default",
        hyperparameter_tune_kwargs=None,
    )
    return predictor.predict(test_data).to_numpy(dtype=float)


def main():
    metrics = run_model_on_folds(MODEL_NAME, predict_fold, CONFIG, ["numpy", "pandas", "scikit-learn", "autogluon"])
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    main()
