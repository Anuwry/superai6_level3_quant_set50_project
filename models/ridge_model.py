from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]

from models.baseline_common import FoldData, print_metrics, run_model_on_folds, split_xy

MODEL_NAME = "ridge_regression"
CONFIG = {
    "experiment": "naive_baseline",
    "model": "Ridge",
    "hyperparameter_tuning": False,
    "model_parameters": {"alpha": 1.0, "fit_intercept": True, "solver": "auto"},
}


def predict_fold(fold: FoldData) -> np.ndarray:
    x_train, y_train, x_test, _ = split_xy(fold)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, fit_intercept=True, solver="auto")),
        ]
    )
    model.fit(x_train, y_train)
    return model.predict(x_test)


def main():
    metrics = run_model_on_folds(MODEL_NAME, predict_fold, CONFIG, ["numpy", "pandas", "scikit-learn"])
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    main()
