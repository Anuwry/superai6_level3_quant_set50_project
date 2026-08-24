from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable, Sequence

from models.baseline_common import PROJECT_ROOT
from models.track_a_final import (
    SELECTION_SEED,
    SEQUENCE_WINDOWS,
    TRACK_A_FEATURE_SETS,
    TRACK_A_MODELS,
)

REGISTERED_MODELS = tuple(TRACK_A_MODELS)


def _python_code(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def selection_commands(*, force: bool) -> list[list[str]]:
    commands: list[list[str]] = []
    for model in REGISTERED_MODELS:
        for feature_set in TRACK_A_FEATURE_SETS:
            for window in SEQUENCE_WINDOWS:
                commands.append(
                    _python_code(
                        "import models.track_a_final as t; "
                        "t.run_configuration("
                        f"'window_selection','{model}','{feature_set}',"
                        f"{window},{SELECTION_SEED},force={force})"
                    )
                )
    commands.append(
        _python_code(
            "import models.track_a_final as t; "
            "t.run_window_selection(force=False); "
            "t.save_input_manifest()"
        )
    )
    return commands


def outer_commands(*, force: bool) -> list[list[str]]:
    commands = [
        _python_code(
            "import models.track_a_final as t; "
            "locked=t.load_locked_windows(); "
            f"selected=locked.loc[locked['model'].eq('{model}')].copy(); "
            f"t.collect_final_metrics(selected,force={force})"
        )
        for model in REGISTERED_MODELS
    ]
    commands.append(
        _python_code(
            "import models.track_a_final as t; "
            "t.run_final_outer_test(force=False); "
            "t.save_input_manifest()"
        )
    )
    return commands


def track_b_commands(*, force: bool) -> list[list[str]]:
    commands = [
        _python_code(
            "from models.track_b_four_fold_ablation import "
            "run_four_fold_ablation; "
            f"run_four_fold_ablation(models=('{model}',),force={force})"
        )
        for model in REGISTERED_MODELS
    ]
    commands.append(
        _python_code(
            "from models.track_b_four_fold_ablation import "
            "configure_fusion_module,save_yearly_and_period_summaries; "
            "configure_fusion_module().build_and_save_reports(); "
            "save_yearly_and_period_summaries()"
        )
    )
    return commands


def data_commands() -> list[list[str]]:
    return [
        [sys.executable, "-m", "models.point_in_time_data"],
        [sys.executable, "-m", "models.full_ta_feature_pool"],
        [sys.executable, "-m", "models.vmd_feature_pool"],
        [sys.executable, "-m", "models.track_a_final", "data"],
    ]


def run_commands(commands: Iterable[Sequence[str]]) -> None:
    for command in commands:
        subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            check=True,
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the registered five-model point-in-time pipeline "
            "with process isolation."
        )
    )
    parser.add_argument(
        "stage",
        choices=("data", "selection", "outer", "track-b", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.stage in {"data", "all"}:
        run_commands(data_commands())
    if args.stage in {"selection", "all"}:
        run_commands(selection_commands(force=args.force))
    if args.stage in {"outer", "all"}:
        run_commands(outer_commands(force=args.force))
    if args.stage in {"track-b", "all"}:
        run_commands(track_b_commands(force=args.force))


if __name__ == "__main__":
    main()
