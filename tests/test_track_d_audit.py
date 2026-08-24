from __future__ import annotations

from models.track_d_audit import expected_cardinalities
from models.track_d_protocol import TrackDConfig


def test_expected_cardinalities_cover_all_registered_cells():
    counts = expected_cardinalities(
        forward_rows=120,
        economic_rows=119,
        features=122,
        config=TrackDConfig(),
    )

    assert counts["validation_metric_rows"] == 30
    assert counts["forward_prediction_seed_rows"] == 120 * 5 * 2 * 5
    assert counts["forward_prediction_average_rows"] == 120 * 5 * 2
    assert counts["selective_metric_rows"] == 5 * 2 * 4
    assert counts["economic_summary_rows"] == 5 * 2 * 4 * 2 * 3
    assert counts["economic_daily_rows"] == 119 * 5 * 2 * 4 * 2 * 3
    assert counts["xai_attribution_rows"] == 5 * 30 * 3 * 122
    assert counts["xai_randomization_rows"] == 5 * 30 * 2
    assert counts["xai_deletion_rows"] == 5 * 30
    assert counts["xai_random_deletion_rows"] == 5 * 30 * 100
