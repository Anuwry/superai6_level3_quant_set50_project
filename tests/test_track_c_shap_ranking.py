from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.baseline_common import FoldSpec
from models.track_c_shap_ranking import (
    compute_absolute_spearman,
    selection_fold_triplets,
    validate_ranking_sequence_counts,
)


def _spec(fold: str, train_end: int, test_year: int) -> FoldSpec:
    return FoldSpec(
        fold=fold,
        train_path=Path(f"train_2012_{train_end}.csv"),
        test_path=Path(f"test_{test_year}.csv"),
        train_start_year=2012,
        train_end_year=train_end,
        test_year=test_year,
    )


def test_selection_triplets_end_before_outer_test():
    specs = [
        _spec("fold_1", 2017, 2018),
        _spec("fold_2", 2018, 2019),
        _spec("fold_3", 2019, 2020),
        _spec("fold_4", 2020, 2021),
    ]

    triplets = selection_fold_triplets(specs, first_outer_year=2022)

    assert [
        (
            item.training_rank_spec.test_year,
            item.validation_spec.test_year,
        )
        for item in triplets
    ] == [(2018, 2019), (2019, 2020), (2020, 2021)]
    assert all(
        item.validation_spec.test_year < 2022 for item in triplets
    )


def test_selection_triplets_reject_nonconsecutive_years():
    specs = [
        _spec("fold_1", 2017, 2018),
        _spec("fold_2", 2019, 2020),
    ]

    with pytest.raises(ValueError, match="consecutive"):
        selection_fold_triplets(specs, first_outer_year=2022)


def test_absolute_spearman_handles_positive_negative_and_constant_features():
    features = pd.DataFrame(
        {
            "positive": [1, 2, 3, 4, 5],
            "negative": [5, 4, 3, 2, 1],
            "constant": [1, 1, 1, 1, 1],
        }
    )
    target = np.array([10, 20, 30, 40, 50], dtype=float)

    result = compute_absolute_spearman(features, target)

    assert result["positive"] == pytest.approx(1.0)
    assert result["negative"] == pytest.approx(1.0)
    assert result["constant"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("available", "required"),
    [(0, 40), (39, 40)],
)
def test_sequence_count_gate_fails_closed(available: int, required: int):
    with pytest.raises(ValueError, match="available"):
        validate_ranking_sequence_counts(
            available,
            required=required,
            cell="fold_1/lstm/bull",
        )


def test_sequence_count_gate_accepts_exact_minimum():
    assert (
        validate_ranking_sequence_counts(
            40,
            required=40,
            cell="fold_1/lstm/bull",
        )
        == 40
    )
