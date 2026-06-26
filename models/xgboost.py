from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]

from models.baseline_common import RANDOM_SEED, FoldData, print_metrics, run_model_on_folds, split_xy

MODEL_NAME = "xgboost"
CONFIG = {
    "experiment": "naive_baseline",
    "model": "XGBRegressor",
    "hyperparameter_tuning": False,
    "model_parameters": {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
}


def predict_fold(fold: FoldData) -> np.ndarray:
    from xgboost import XGBRegressor

    x_train, y_train, x_test, _ = split_xy(fold)
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, verbose=False)
    return model.predict(x_test)


def main():
    metrics = run_model_on_folds(MODEL_NAME, predict_fold, CONFIG, ["numpy", "pandas", "scikit-learn", "xgboost"])
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    main()
