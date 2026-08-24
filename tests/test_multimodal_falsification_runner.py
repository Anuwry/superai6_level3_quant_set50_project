from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.multimodal_falsification import CONTROL_ARMS, NEWS_ONLY_ARM
from models.multimodal_falsification_runner import (
    FOLDS,
    build_cell_commands,
    control_seed,
    validate_control_cell,
)
from models.track_a_final import FINAL_SEEDS, TRACK_A_MODELS


def _prediction(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2022-01-03", periods=4, freq="D"),
            "routing_regime": ["bull", "sideway", "bear", "bull"],
            "Close_D": [100.0, 101.0, 100.0, 102.0],
            "y_true": [101.0, 100.0, 102.0, 101.0],
            "y_pred": np.array([101.0, 100.0, 102.0, 101.0]) + offset,
        }
    )


def test_control_seed_is_deterministic_and_control_specific() -> None:
    first = control_seed("fold_1", "shuffled_news")
    assert first == control_seed("fold_1", "shuffled_news")
    assert first != control_seed("fold_1", "random_features")
    assert first != control_seed("fold_2", "shuffled_news")


def test_build_cell_commands_covers_five_models_four_folds_five_seeds() -> None:
    commands = build_cell_commands(
        python_executable=Path("python"),
        output_dir=Path("outputs/control"),
    )

    assert len(commands) == len(TRACK_A_MODELS) * len(FOLDS) * len(FINAL_SEEDS)
    assert len({tuple(command) for command in commands}) == len(commands)


def test_validate_control_cell_accepts_four_aligned_arms() -> None:
    arms = (NEWS_ONLY_ARM, *CONTROL_ARMS)
    metrics = pd.DataFrame(
        [{"arm": arm, "n_test": 4} for arm in arms]
    )
    registry = pd.DataFrame(
        [
            {
                "fit_id": f"fit-{index}",
                "arm": arm,
                "training_sequences": 300,
            }
            for index, arm in enumerate(arms)
        ]
    )
    predictions = {
        arm: _prediction(index / 100.0) for index, arm in enumerate(arms)
    }

    audit = validate_control_cell(metrics, registry, predictions)

    assert audit["passed"] is True
    assert audit["arms"] == 4
    assert audit["fits"] == 4
    assert audit["test_rows"] == 4
