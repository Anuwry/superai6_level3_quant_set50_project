from __future__ import annotations

from models.run_five_model_pipeline import (
    REGISTERED_MODELS,
    outer_commands,
    selection_commands,
    track_b_commands,
)
from models.track_a_final import TRACK_A_MODELS


def _joined(commands: list[list[str]]) -> str:
    return "\n".join(" ".join(command) for command in commands)


def test_reproduction_runner_contains_exactly_the_five_registered_models():
    assert REGISTERED_MODELS == tuple(TRACK_A_MODELS)
    assert REGISTERED_MODELS == (
        "lstm",
        "cnn",
        "lstm_cnn",
        "lstm_attention",
        "lstm_cnn_attention",
    )


def test_selection_runner_isolates_all_fifty_configurations():
    commands = selection_commands(force=True)

    assert len(commands) == 51
    assert _joined(commands).count("run_configuration") == 50


def test_outer_and_track_b_runners_isolate_each_registered_model():
    outer = _joined(outer_commands(force=True))
    track_b = _joined(track_b_commands(force=True))

    for model in REGISTERED_MODELS:
        assert f"eq('{model}')" in outer
        assert f"('{model}',)" in track_b
