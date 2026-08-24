from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.track_c_shap_selection import (
    aggregate_shap_importance,
    build_consensus_ranking,
    derive_protocol_seed,
    normalized_descending_ranks,
    purge_ranking_endpoints,
    select_registered_top_k,
)


def test_aggregate_shap_importance_uses_mean_absolute_sample_and_lag():
    values = np.array(
        [
            [[1.0, -2.0], [3.0, 0.0]],
            [[-1.0, 2.0], [1.0, -4.0]],
        ]
    )

    importance = aggregate_shap_importance(values)

    np.testing.assert_allclose(importance, [1.5, 2.0])


def test_normalized_descending_ranks_handle_ties_without_magnitude_pooling():
    importance = np.array([10.0, 5.0, 5.0, 1.0])

    result = normalized_descending_ranks(importance)

    np.testing.assert_allclose(result, [0.0, 0.5, 0.5, 1.0])


def test_consensus_ranking_averages_cell_ranks_not_raw_magnitudes():
    records = pd.DataFrame(
        [
            {
                "model": "lstm",
                "selection_fold": "fold_1",
                "regime": "global",
                "feature": "a",
                "importance": 1000.0,
            },
            {
                "model": "lstm",
                "selection_fold": "fold_1",
                "regime": "global",
                "feature": "b",
                "importance": 1.0,
            },
            {
                "model": "cnn",
                "selection_fold": "fold_1",
                "regime": "global",
                "feature": "a",
                "importance": 1.0,
            },
            {
                "model": "cnn",
                "selection_fold": "fold_1",
                "regime": "global",
                "feature": "b",
                "importance": 2.0,
            },
        ]
    )

    result = build_consensus_ranking(records)

    assert result["feature"].tolist() == ["a", "b"]
    assert result["consensus_normalized_rank"].tolist() == pytest.approx(
        [0.5, 0.5]
    )
    assert result["contributing_cells"].tolist() == [2, 2]


def test_protocol_seed_is_stable_and_cell_specific():
    first = derive_protocol_seed(42, "lstm", "fold_1", "bull")
    second = derive_protocol_seed(42, "lstm", "fold_1", "bull")
    other = derive_protocol_seed(42, "cnn", "fold_1", "bull")

    assert first == second
    assert first != other
    assert 0 <= first < 2**31


def test_purge_ranking_endpoints_excludes_label_at_validation_boundary():
    ranking = pd.DataFrame(
        {
            "Date": ["2018-12-27", "2018-12-28"],
            "Label_Date": ["2018-12-28", "2019-01-02"],
        }
    )

    keep = purge_ranking_endpoints(
        ranking,
        first_validation_date=pd.Timestamp("2019-01-02"),
    )

    assert keep.tolist() == [True, False]


def test_purge_ranking_endpoints_fails_on_invalid_label_dates():
    ranking = pd.DataFrame(
        {
            "Date": ["2018-12-27"],
            "Label_Date": ["invalid"],
        }
    )

    with pytest.raises(ValueError, match="Label_Date"):
        purge_ranking_endpoints(
            ranking,
            first_validation_date=pd.Timestamp("2019-01-02"),
        )


def test_registered_top_k_chooses_smallest_candidate_passing_all_gates():
    rows = []
    for model in ("a", "b", "c", "d", "e"):
        for fold in ("fold_1", "fold_2", "fold_3"):
            for top_k, ba, rmse in (
                (10, 0.49, 11.0),
                (20, 0.51, 10.1),
                (122, 0.50, 10.0),
            ):
                rows.append(
                    {
                        "model": model,
                        "selection_fold": fold,
                        "top_k": top_k,
                        "balanced_accuracy": ba,
                        "rmse": rmse,
                    }
                )
    metrics = pd.DataFrame(rows)
    stability = pd.Series({10: 0.60, 20: 0.70, 122: 1.0})

    selected, audit = select_registered_top_k(metrics, stability)

    assert selected == 20
    assert audit.loc[audit["top_k"].eq(20), "all_gates_pass"].item()
    assert not audit.loc[audit["top_k"].eq(10), "all_gates_pass"].item()


def test_registered_top_k_falls_back_to_all_features():
    rows = []
    for model in ("a", "b", "c", "d", "e"):
        for fold in ("fold_1", "fold_2", "fold_3"):
            rows.extend(
                [
                    {
                        "model": model,
                        "selection_fold": fold,
                        "top_k": 10,
                        "balanced_accuracy": 0.40,
                        "rmse": 20.0,
                    },
                    {
                        "model": model,
                        "selection_fold": fold,
                        "top_k": 122,
                        "balanced_accuracy": 0.50,
                        "rmse": 10.0,
                    },
                ]
            )

    selected, audit = select_registered_top_k(
        pd.DataFrame(rows),
        pd.Series({10: 0.2, 122: 1.0}),
    )

    assert selected == 122
    assert audit.loc[audit["top_k"].eq(122), "all_gates_pass"].item()
