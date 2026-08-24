from __future__ import annotations

import numpy as np
import pandas as pd

from models.baseline_common import (
    CLOSE_COLUMN,
    DATA_FOLDS_DIR,
    PROJECT_ROOT,
    FoldData,
    print_metrics,
    run_model_on_folds,
)

MODEL_NAME = "persistence_current_close"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "persistence_baseline"
CONFIG = {
    "experiment": "persistence_baseline",
    "model": "Persistence",
    "prediction_rule": "Predicted Close(t+1) = Close(t)",
    "data": str(DATA_FOLDS_DIR),
    "hyperparameter_tuning": False,
    "training_required": False,
}


def predict_persistence_fold(fold: FoldData) -> np.ndarray:
    return fold.test[CLOSE_COLUMN].to_numpy(dtype=float)


def run_persistence_baseline() -> pd.DataFrame:
    metrics = run_model_on_folds(
        MODEL_NAME,
        predict_persistence_fold,
        CONFIG,
        ["numpy", "pandas", "scikit-learn"],
        data_dir=DATA_FOLDS_DIR,
        output_dir=OUTPUT_DIR,
    )
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    run_persistence_baseline()
