from __future__ import annotations

import numpy as np
import pytest

from models.track_d_xai import (
    aggregate_feature_attributions,
    delete_feature_trajectories,
    faithfulness_percentile,
    randomization_rank_correlation,
)


def test_lag_attributions_are_summed_with_sign_per_feature():
    values = np.array(
        [
            [[1.0, -1.0], [2.0, 0.5], [-0.5, 1.5]],
            [[-1.0, 3.0], [0.5, -1.0], [1.5, -2.0]],
        ]
    )

    result = aggregate_feature_attributions(values)

    np.testing.assert_allclose(result, [[2.5, 1.0], [1.0, 0.0]])


def test_feature_deletion_replaces_complete_lag_trajectory():
    instance = np.arange(12, dtype=float).reshape(3, 4)
    reference = np.full((3, 4), -1.0)

    deleted = delete_feature_trajectories(
        instance,
        reference,
        feature_indices=(1, 3),
    )

    assert deleted[:, 1].tolist() == [-1.0, -1.0, -1.0]
    assert deleted[:, 3].tolist() == [-1.0, -1.0, -1.0]
    assert deleted[:, 0].tolist() == instance[:, 0].tolist()


def test_randomization_correlation_uses_absolute_rank():
    result = randomization_rank_correlation(
        np.array([3.0, -2.0, 1.0]),
        np.array([1.0, -2.0, 3.0]),
    )

    assert result == pytest.approx(-1.0)


def test_faithfulness_percentile_compares_against_random_deletions():
    result = faithfulness_percentile(
        top_feature_effect=0.30,
        random_feature_effects=np.array([0.10, 0.20, 0.25, 0.40]),
    )

    assert result == pytest.approx(0.75)
