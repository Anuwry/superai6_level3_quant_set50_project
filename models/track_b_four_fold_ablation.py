from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from models import track_b_fusion as fusion
from models.baseline_common import PROJECT_ROOT
from models.track_a_final import FINAL_SEEDS, TRACK_A_MODELS
from models.track_b_forward_news import DAILY_NEWS_FILE as FORWARD_DAILY_NEWS_FILE

FOUR_FOLD_TEST_YEARS = (2022, 2023, 2024, 2025)
FOUR_FOLD_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "track_b"
    / "four_fold_ablation_point_in_time_v2"
)
YEARLY_SUMMARY_NAME = "paired_summary_by_year.csv"
PERIOD_SUMMARY_NAME = "paired_summary_by_source_period.csv"

_CONFIGURATION_GLOBALS = (
    "FUSION_TEST_YEARS",
    "FUSION_OUTPUT_DIR",
    "DAILY_NEWS_FILE",
    "METRICS_FILE",
    "PERFORMANCE_FILE",
    "PAIRED_FILE",
    "PAIRED_SUMMARY_FILE",
    "PAPER_TABLE_FILE",
    "RUN_METADATA_FILE",
    "EXPERIMENT_NAME",
    "SENTIMENT_PREDICTIONS_FILE",
    "SENTIMENT_SOURCE_DESCRIPTION",
    "SENTIMENT_PREDICTION_CONTRACT",
)


class ConfiguredFusion:
    def __init__(self, output_dir: Path) -> None:
        self.FUSION_TEST_YEARS = FOUR_FOLD_TEST_YEARS
        self.FUSION_OUTPUT_DIR = output_dir
        self.DAILY_NEWS_FILE = FORWARD_DAILY_NEWS_FILE
        self.METRICS_FILE = output_dir / "metrics_by_seed_fold.csv"
        self.PERFORMANCE_FILE = output_dir / "performance_summary.csv"
        self.PAIRED_FILE = output_dir / "paired_deltas_by_seed_fold.csv"
        self.PAIRED_SUMMARY_FILE = output_dir / "paired_summary.csv"
        self.PAPER_TABLE_FILE = output_dir / "paper_track_b_four_fold_table.csv"
        self.RUN_METADATA_FILE = output_dir / "run_metadata.json"
        self.EXPERIMENT_NAME = (
            "Track B four-outer-fold technical-news paired ablation"
        )
        self.SENTIMENT_PREDICTIONS_FILE = FORWARD_DAILY_NEWS_FILE
        self.SENTIMENT_SOURCE_DESCRIPTION = (
            "headline-proxy expanding predictions through 2023 and one frozen "
            "2018-2023 fit for point-in-time SET50 headlines in 2024-2025"
        )
        self.SENTIMENT_PREDICTION_CONTRACT = (
            "expanding labelled inference for 2019-2023; frozen through 2023 "
            "for both 2024 and 2025; no pseudo-label retraining"
        )

    @contextmanager
    def activated(self):
        previous = {
            name: getattr(fusion, name) for name in _CONFIGURATION_GLOBALS
        }
        try:
            for name in _CONFIGURATION_GLOBALS:
                setattr(fusion, name, getattr(self, name))
            yield
        finally:
            for name, value in previous.items():
                setattr(fusion, name, value)

    def run_fusion_experiment(self, **kwargs) -> pd.DataFrame:
        with self.activated():
            return fusion.run_fusion_experiment(**kwargs)

    def build_and_save_reports(self):
        with self.activated():
            metrics = fusion.collect_completed_metrics()
            reports = fusion.build_and_save_reports(metrics)
            fusion._write_experiment_metadata(
                metrics,
                runtime_seconds=None,
            )
            return reports


def require_forward_daily_news(
    path: Path | None = None,
) -> Path:
    source = FORWARD_DAILY_NEWS_FILE if path is None else path
    if not source.is_file():
        raise FileNotFoundError(
            "The locked 2019-2025 daily news artifact is required before the "
            f"four-fold ablation: {source}"
        )
    return source


def configure_fusion_module(
    *,
    output_dir: Path = FOUR_FOLD_OUTPUT_DIR,
) -> ConfiguredFusion:
    return ConfiguredFusion(output_dir)


def build_yearly_and_period_summaries(
    paired: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "model",
        "test_year",
        "seed",
        "rmse_delta_news_minus_technical",
        "direction_accuracy_delta_pp",
        "balanced_accuracy_delta_pp",
        "mcc_delta_news_minus_technical",
        "runtime_seconds_delta_news_minus_technical",
    }
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ValueError(f"Paired four-fold results are missing columns: {missing}")
    yearly = (
        paired.groupby(["model", "test_year"], sort=True)
        .agg(
            seeds_per_fold=("seed", "nunique"),
            rmse_delta_mean=("rmse_delta_news_minus_technical", "mean"),
            rmse_delta_std=("rmse_delta_news_minus_technical", "std"),
            direction_accuracy_delta_pp_mean=(
                "direction_accuracy_delta_pp",
                "mean",
            ),
            direction_accuracy_delta_pp_std=(
                "direction_accuracy_delta_pp",
                "std",
            ),
            balanced_accuracy_delta_pp_mean=(
                "balanced_accuracy_delta_pp",
                "mean",
            ),
            balanced_accuracy_delta_pp_std=(
                "balanced_accuracy_delta_pp",
                "std",
            ),
            mcc_delta_mean=(
                "mcc_delta_news_minus_technical",
                "mean",
            ),
            mcc_delta_std=(
                "mcc_delta_news_minus_technical",
                "std",
            ),
            runtime_seconds_delta_mean=(
                "runtime_seconds_delta_news_minus_technical",
                "mean",
            ),
        )
        .reset_index()
    )
    yearly["period"] = yearly["test_year"].map(
        lambda year: "labelled_validation" if int(year) <= 2023 else "frozen_forward"
    )
    seed_counts = yearly.groupby(["model", "period"])["seeds_per_fold"].nunique()
    if not seed_counts.eq(1).all():
        raise ValueError("Seeds per fold are inconsistent within a source period")
    period = (
        yearly.groupby(["model", "period"], sort=True)
        .agg(
            outer_folds=("test_year", "nunique"),
            seeds_per_fold=("seeds_per_fold", "first"),
            test_year_start=("test_year", "min"),
            test_year_end=("test_year", "max"),
            rmse_delta_mean=("rmse_delta_mean", "mean"),
            rmse_delta_std_across_folds=("rmse_delta_mean", "std"),
            direction_accuracy_delta_pp_mean=(
                "direction_accuracy_delta_pp_mean",
                "mean",
            ),
            direction_accuracy_delta_pp_std_across_folds=(
                "direction_accuracy_delta_pp_mean",
                "std",
            ),
            balanced_accuracy_delta_pp_mean=(
                "balanced_accuracy_delta_pp_mean",
                "mean",
            ),
            balanced_accuracy_delta_pp_std_across_folds=(
                "balanced_accuracy_delta_pp_mean",
                "std",
            ),
            mcc_delta_mean=("mcc_delta_mean", "mean"),
            mcc_delta_std_across_folds=("mcc_delta_mean", "std"),
            runtime_seconds_delta_mean=("runtime_seconds_delta_mean", "mean"),
        )
        .reset_index()
    )
    return yearly, period


def save_yearly_and_period_summaries(
    *,
    output_dir: Path = FOUR_FOLD_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired_path = output_dir / "paired_deltas_by_seed_fold.csv"
    if not paired_path.is_file():
        raise FileNotFoundError(f"Paired four-fold results not found: {paired_path}")
    yearly, period = build_yearly_and_period_summaries(pd.read_csv(paired_path))
    yearly.to_csv(output_dir / YEARLY_SUMMARY_NAME, index=False)
    period.to_csv(output_dir / PERIOD_SUMMARY_NAME, index=False)
    return yearly, period


def run_four_fold_ablation(
    *,
    models: Iterable[str] = TRACK_A_MODELS,
    seeds: Iterable[int] = FINAL_SEEDS,
    force: bool = False,
    output_dir: Path = FOUR_FOLD_OUTPUT_DIR,
) -> pd.DataFrame:
    require_forward_daily_news()
    configured = configure_fusion_module(output_dir=output_dir)
    result = configured.run_fusion_experiment(
        models=tuple(models),
        seeds=tuple(int(seed) for seed in seeds),
        test_years=FOUR_FOLD_TEST_YEARS,
        force=force,
    )
    if isinstance(result, pd.DataFrame):
        save_yearly_and_period_summaries(output_dir=output_dir)
    return result


def _csv_values(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the locked four-outer-fold Track B paired ablation."
    )
    parser.add_argument(
        "stage",
        choices=("run", "report"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--models", default=",".join(TRACK_A_MODELS))
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in FINAL_SEEDS),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None):
    args = _parser().parse_args(argv)
    configured = configure_fusion_module()
    if args.stage == "report":
        result = configured.build_and_save_reports()
        save_yearly_and_period_summaries()
        return result
    return run_four_fold_ablation(
        models=_csv_values(args.models),
        seeds=_csv_values(args.seeds, int),
        force=args.force,
    )


if __name__ == "__main__":
    main()
