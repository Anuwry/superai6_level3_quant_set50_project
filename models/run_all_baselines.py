from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path = [path for path in sys.path if Path(path or ".").resolve() != SCRIPT_DIR]

from models import auto_gluon, chronos, lightgbm, long_short_term_memory, ridge_model, xgboost
from models.baseline_common import OUTPUT_DIR

RUNNERS = [
    ridge_model.main,
    lightgbm.main,
    xgboost.main,
    long_short_term_memory.main,
    auto_gluon.main,
    chronos.main,
]


def main():
    results = {}
    for runner in RUNNERS:
        metrics = runner()
        results[runner.__module__] = metrics
    print(f"Saved outputs to {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    main()
