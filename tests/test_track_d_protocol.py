from __future__ import annotations

import json

import numpy as np
import pytest

from models.track_d_protocol import (
    TRACK_D_PROTOCOL_VERSION,
    TrackDConfig,
    direction_labels,
    freeze_track_d_protocol,
    select_confidence_threshold,
    verify_freeze_manifest,
)


def test_direction_labels_use_next_close_minus_current_close():
    labels, eligible = direction_labels(
        np.array([101.0, 99.0, 100.0]),
        np.array([100.0, 100.0, 100.0]),
    )

    assert labels.tolist() == [1.0, 0.0, 0.0]
    assert eligible.tolist() == [True, True, False]


def test_track_d_config_locks_models_windows_and_forward_year():
    config = TrackDConfig()

    assert config.protocol_version == TRACK_D_PROTOCOL_VERSION
    assert config.forward_year == 2026
    assert config.selection_years == (2019, 2020, 2021)
    assert config.windows == {
        "lstm": 5,
        "cnn": 20,
        "lstm_cnn": 20,
        "lstm_attention": 10,
        "lstm_cnn_attention": 20,
    }
    assert config.objectives == ("direct", "multitask")


def test_track_d_config_rejects_invalid_confidence_threshold():
    with pytest.raises(ValueError, match="confidence"):
        TrackDConfig(confidence_thresholds=(0.49, 0.60))


def test_threshold_selection_applies_coverage_gate_and_stable_tie_break():
    rows = [
        {
            "threshold": 0.50,
            "mean_net_return": 0.001,
            "coverage": 1.0,
            "positive_years": 3,
        },
        {
            "threshold": 0.60,
            "mean_net_return": 0.003,
            "coverage": 0.15,
            "positive_years": 3,
        },
        {
            "threshold": 0.55,
            "mean_net_return": 0.002,
            "coverage": 0.40,
            "positive_years": 2,
        },
    ]

    selected = select_confidence_threshold(
        rows,
        minimum_coverage=0.20,
        minimum_positive_years=2,
    )

    assert selected == pytest.approx(0.55)


def test_freeze_manifest_records_no_forward_data_access(tmp_path):
    protocol_path = tmp_path / "protocol.md"
    protocol_path.write_text("frozen protocol", encoding="utf-8")
    source_path = tmp_path / "implementation.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    selection_path = tmp_path / "selected_thresholds.csv"
    selection_path.write_text(
        "model,objective,selected_threshold\nlstm,direct,0.55\n",
        encoding="utf-8",
    )
    input_path = tmp_path / "local_history.csv"
    input_path.write_text("Date,Close\n2025-12-22,839.24\n", encoding="utf-8")
    output_path = tmp_path / "freeze.json"

    payload = freeze_track_d_protocol(
        output_path=output_path,
        protocol_path=protocol_path,
        implementation_paths=(source_path,),
        selection_artifact_paths=(selection_path,),
        input_artifact_paths=(input_path,),
        market_data_accessed=False,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == saved
    assert saved["market_data_2026_accessed_before_freeze"] is False
    assert saved["protocol_sha256"]
    assert saved["implementation_sha256"][str(source_path)]
    assert saved["selection_artifact_sha256"][str(selection_path)]
    assert saved["input_artifact_sha256"][str(input_path)]


def test_freeze_manifest_requires_registered_selection_artifacts(tmp_path):
    protocol_path = tmp_path / "protocol.md"
    protocol_path.write_text("frozen protocol", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Selection artifact"):
        freeze_track_d_protocol(
            output_path=tmp_path / "freeze.json",
            protocol_path=protocol_path,
            implementation_paths=(),
            selection_artifact_paths=(tmp_path / "missing.csv",),
            market_data_accessed=False,
        )


def test_freeze_manifest_fails_if_forward_data_was_already_accessed(tmp_path):
    protocol_path = tmp_path / "protocol.md"
    protocol_path.write_text("frozen protocol", encoding="utf-8")

    with pytest.raises(ValueError, match="before freeze"):
        freeze_track_d_protocol(
            output_path=tmp_path / "freeze.json",
            protocol_path=protocol_path,
            implementation_paths=(),
            market_data_accessed=True,
        )


def test_freeze_verification_detects_post_freeze_code_change(tmp_path):
    protocol_path = tmp_path / "protocol.md"
    protocol_path.write_text("frozen protocol", encoding="utf-8")
    source_path = tmp_path / "implementation.py"
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    manifest_path = tmp_path / "freeze.json"
    freeze_track_d_protocol(
        output_path=manifest_path,
        protocol_path=protocol_path,
        implementation_paths=(source_path,),
        market_data_accessed=False,
    )
    source_path.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_freeze_manifest(manifest_path)
