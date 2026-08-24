from __future__ import annotations

import numpy as np

from models.pit_set50_crin import clipped_logit, direction_metrics


def test_clipped_logit_is_finite_at_probability_boundaries():
    values = clipped_logit(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(values).all()
    assert values[0] < 0.0 < values[-1]


def test_direction_metrics_uses_fixed_half_threshold():
    result = direction_metrics(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.7, 0.4])
    )
    assert result.observations == 4
    assert result.direction_accuracy == 0.5
    assert (result.tn, result.fp, result.fn, result.tp) == (1, 1, 1, 1)


def test_model_shapes_and_masked_attention_contract():
    from models.pit_set50_crin import (
        build_constituent_worker,
        build_reconciliation_leader,
        build_top_only_stack,
    )

    worker, diagnostics = build_constituent_worker(asset_count=3, window=5, feature_count=2)
    sequence = np.zeros((2, 3, 5, 2), dtype=np.float32)
    mask = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.float32)
    output = diagnostics(
        {"constituent_sequence": sequence, "active_member_mask": mask}, training=False
    )
    assert tuple(output["direction"].shape) == (2, 1)
    assert np.allclose(np.asarray(output["attention"]).sum(axis=1), 1.0)
    assert np.allclose(np.asarray(output["attention"])[:, 2], 0.0)
    assert worker.count_params() > 0

    leader, leader_diagnostics = build_reconciliation_leader(top_experts=5)
    leader_output = leader_diagnostics(
        {
            "top_down_scores": np.zeros((2, 5), dtype=np.float32),
            "bottom_up_logit": np.zeros((2, 1), dtype=np.float32),
            "reconciliation_context": np.ones((2, 2), dtype=np.float32),
        },
        training=False,
    )
    assert tuple(leader_output["probability"].shape) == (2, 1)
    assert np.all((np.asarray(leader_output["top_down_gate"]) >= 0.0))
    assert build_top_only_stack(top_experts=5).count_params() > 0
