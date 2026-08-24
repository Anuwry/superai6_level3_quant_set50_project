from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.track_c_topk_validation import (
    endpoint_regime_mask,
    scale_frame_with_metadata,
    top_features_for_regime,
)


def test_scale_frame_uses_training_metadata_without_refit():
    frame = pd.DataFrame(
        {
            "Date": ["2020-01-01"],
            "a": [10.0],
            "Target_Next_Close": [20.0],
        }
    )
    metadata = {
        "columns": ["a", "Target_Next_Close"],
        "scale": [0.1, 0.05],
        "min": [-0.5, -0.5],
    }

    result = scale_frame_with_metadata(frame, metadata)

    assert result["Date"].tolist() == ["2020-01-01"]
    assert result["a"].tolist() == pytest.approx([0.5])
    assert result["Target_Next_Close"].tolist() == pytest.approx([0.5])
    assert frame["a"].tolist() == [10.0]


def test_endpoint_regime_mask_aligns_with_sequence_window():
    labels = np.array(["bull", "bear", "sideway", "bull", "bear"])

    result = endpoint_regime_mask(labels, regime="bear", window=3)

    assert result.tolist() == [False, False, True]


def test_endpoint_regime_mask_rejects_insufficient_rows():
    with pytest.raises(ValueError, match="window"):
        endpoint_regime_mask(
            np.array(["bull", "bear"]),
            regime="bull",
            window=3,
        )


def test_top_features_follow_frozen_consensus_rank():
    ranking = pd.DataFrame(
        {
            "regime": ["global", "global", "bull"],
            "feature": ["b", "a", "c"],
            "consensus_rank": [2, 1, 1],
        }
    )

    result = top_features_for_regime(
        ranking,
        regime="global",
        top_k=2,
    )

    assert result == ("a", "b")


def test_top_features_fail_when_requested_k_is_unavailable():
    ranking = pd.DataFrame(
        {
            "regime": ["global"],
            "feature": ["a"],
            "consensus_rank": [1],
        }
    )

    with pytest.raises(ValueError, match="requested"):
        top_features_for_regime(
            ranking,
            regime="global",
            top_k=2,
        )
