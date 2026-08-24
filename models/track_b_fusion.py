from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_common import (
    CLOSE_COLUMN,
    DATE_COLUMN,
    PROJECT_ROOT,
    TARGET_COLUMN,
    FoldData,
    FoldSpec,
    discover_folds,
    evaluate_predictions,
    get_feature_columns,
    load_fold,
    package_versions,
    predictions_frame,
    read_frame,
    validate_test_context,
)
from models.neural_network_folds import (
    inverse_scaled_target,
    scale_train_test_frames,
)
from models.track_a_analysis import exact_sign_flip_pvalue
from models.track_a_final import (
    FINAL_SEEDS,
    TRACK_A_MODELS,
    load_locked_windows,
    predict_model,
)
from models.track_b_data import (
    DAILY_FEATURE_COLUMNS,
    aggregate_daily_sentiment,
)
from models.point_in_time_data import LABEL_DATE_COLUMN
from models.vmd_feature_pool import (
    FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
)

NEWS_START_YEAR = 2019
FUSION_TEST_YEARS = (2022, 2023)
FEATURE_SETS = ("technical_vmd", "technical_vmd_news")
METRIC_COLUMNS = (
    "rmse",
    "mae",
    "mape",
    "r2",
    "direction_accuracy",
    "balanced_accuracy",
    "mcc",
    "direction_coverage",
)

LOCAL_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "track_b"
    / "local_baseline"
    / "sentiment_predictions_expanding.csv"
)
FUSION_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "track_b"
    / "fusion_point_in_time_v2"
)
DAILY_NEWS_FILE = FUSION_OUTPUT_DIR / "daily_news_features.csv"
METRICS_FILE = FUSION_OUTPUT_DIR / "metrics_by_seed_fold.csv"
PERFORMANCE_FILE = FUSION_OUTPUT_DIR / "performance_summary.csv"
PAIRED_FILE = FUSION_OUTPUT_DIR / "paired_deltas_by_seed_fold.csv"
PAIRED_SUMMARY_FILE = FUSION_OUTPUT_DIR / "paired_summary.csv"
PAPER_TABLE_FILE = FUSION_OUTPUT_DIR / "paper_track_b_fusion_table.csv"
RUN_METADATA_FILE = FUSION_OUTPUT_DIR / "run_metadata.json"
EXPERIMENT_NAME = "Track B technical-news fusion paired ablation"
SENTIMENT_PREDICTIONS_FILE = LOCAL_PREDICTIONS_FILE
SENTIMENT_SOURCE_DESCRIPTION = (
    "expanding-year out-of-sample local classifier predictions"
)
SENTIMENT_PREDICTION_CONTRACT = (
    "expanding-year and out-of-sample for every 2019-2023 pair"
)


def _validate_market_frame(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    required = {DATE_COLUMN, CLOSE_COLUMN, TARGET_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing columns: {missing}")
    result = frame.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN]).dt.normalize()
    if LABEL_DATE_COLUMN in result.columns:
        result[LABEL_DATE_COLUMN] = pd.to_datetime(
            result[LABEL_DATE_COLUMN],
            errors="coerce",
        ).dt.normalize()
        if result[LABEL_DATE_COLUMN].isna().any():
            raise ValueError(f"{context} contains invalid label dates")
    if result.empty:
        raise ValueError(f"{context} is empty")
    if result[DATE_COLUMN].duplicated().any():
        raise ValueError(f"{context} contains duplicate dates")
    if result.isna().any().any():
        raise ValueError(f"{context} contains missing values")
    return result.sort_values(DATE_COLUMN).reset_index(drop=True)


def restrict_to_news_era(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    news_start_year: int = NEWS_START_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = _validate_market_frame(train, "train frame")
    test_frame = _validate_market_frame(test, "test frame")
    restricted_train = train_frame.loc[
        train_frame[DATE_COLUMN].dt.year >= news_start_year
    ].reset_index(drop=True)
    if restricted_train.empty:
        raise ValueError("No training rows remain in the news era")
    if restricted_train[DATE_COLUMN].max() >= test_frame[DATE_COLUMN].min():
        raise ValueError("Fusion train and test dates overlap")
    if LABEL_DATE_COLUMN in restricted_train.columns:
        if (
            restricted_train[LABEL_DATE_COLUMN].max()
            >= test_frame[DATE_COLUMN].min()
        ):
            raise ValueError(
                "Fusion train contains labels observed on or after test start"
            )
    return restricted_train, test_frame


def join_news_features(
    market: pd.DataFrame,
    daily_news: pd.DataFrame,
) -> pd.DataFrame:
    market_frame = _validate_market_frame(market, "market frame")
    required = {"date", *DAILY_FEATURE_COLUMNS}
    missing = sorted(required.difference(daily_news.columns))
    if missing:
        raise ValueError(f"Daily news is missing columns: {missing}")
    news = daily_news.loc[:, ["date", *DAILY_FEATURE_COLUMNS]].copy()
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    if news["date"].duplicated().any():
        raise ValueError("Daily news contains duplicate dates")
    numeric = news.loc[:, DAILY_FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Daily news contains non-finite values")

    result = market_frame.merge(
        news,
        left_on=DATE_COLUMN,
        right_on="date",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise ValueError("Daily news must cover every market date")
    result = result.drop(columns=["date", "_merge"])
    ordered = [
        column
        for column in result.columns
        if column not in {*DAILY_FEATURE_COLUMNS, TARGET_COLUMN}
    ]
    return result.loc[:, [*ordered, *DAILY_FEATURE_COLUMNS, TARGET_COLUMN]]


def _market_dates_for_fusion(
    source_dir: Path = FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
) -> pd.DatetimeIndex:
    dates: list[pd.Series] = []
    for spec in discover_folds(source_dir):
        if spec.test_year > max(FUSION_TEST_YEARS):
            continue
        dates.extend(
            [
                pd.read_csv(spec.train_path, usecols=[DATE_COLUMN])[DATE_COLUMN],
                pd.read_csv(spec.test_path, usecols=[DATE_COLUMN])[DATE_COLUMN],
            ]
        )
    if not dates:
        raise ValueError("No market dates were found for Track B fusion")
    return pd.DatetimeIndex(
        pd.to_datetime(pd.concat(dates, ignore_index=True)).unique()
    ).sort_values()


def create_daily_news_features(
    predictions_file: Path = LOCAL_PREDICTIONS_FILE,
    output_file: Path = DAILY_NEWS_FILE,
    *,
    source_dir: Path = FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
) -> pd.DataFrame:
    if not predictions_file.is_file():
        raise FileNotFoundError(
            f"Expanding-year sentiment predictions not found: {predictions_file}"
        )
    predictions = pd.read_csv(predictions_file, parse_dates=["date"])
    daily = aggregate_daily_sentiment(
        predictions,
        _market_dates_for_fusion(source_dir),
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_file, index=False)
    return daily


def _adjusted_spec(spec: FoldSpec, train: pd.DataFrame) -> FoldSpec:
    return replace(
        spec,
        train_start_year=int(train[DATE_COLUMN].dt.year.min()),
        train_end_year=int(train[DATE_COLUMN].dt.year.max()),
    )


def _fold_data(
    spec: FoldSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    context: pd.DataFrame | None = None,
) -> FoldData:
    train_frame = _validate_market_frame(train, "fold train")
    test_frame = _validate_market_frame(test, "fold test")
    columns = get_feature_columns(train_frame)
    if columns != get_feature_columns(test_frame):
        raise ValueError("Fusion train and test feature columns differ")
    adjusted_spec = _adjusted_spec(spec, train_frame)
    context_frame = (
        _validate_market_frame(context, "fold context")
        if context is not None
        else None
    )
    if context_frame is not None:
        validate_test_context(
            train_frame,
            context_frame,
            test_frame,
            columns,
            adjusted_spec,
        )
    return FoldData(
        spec=adjusted_spec,
        train=train_frame,
        test=test_frame,
        feature_columns=columns,
        context=context_frame,
    )


def prepare_fold_pair(
    spec: FoldSpec,
    daily_news: pd.DataFrame,
) -> dict[str, FoldData]:
    source_fold = load_fold(spec)
    train, test = restrict_to_news_era(
        source_fold.train,
        source_fold.test,
    )
    context = source_fold.context
    return {
        "technical_vmd": _fold_data(spec, train, test, context),
        "technical_vmd_news": _fold_data(
            spec,
            join_news_features(train, daily_news),
            join_news_features(test, daily_news),
            (
                join_news_features(context, daily_news)
                if context is not None
                else None
            ),
        ),
    }


def _scaled_fold(fold: FoldData) -> tuple[FoldData, dict[str, object]]:
    scaled_train, scaled_test, metadata = scale_train_test_frames(
        fold.train,
        fold.test,
    )
    scaled_context = None
    if fold.context is not None:
        columns = list(metadata["columns"])
        scale = np.asarray(metadata["scale"], dtype=float)
        offset = np.asarray(metadata["min"], dtype=float)
        scaled_context = fold.context.astype(
            {column: float for column in columns},
        ).copy()
        scaled_context.loc[:, columns] = (
            fold.context.loc[:, columns].to_numpy(dtype=float) * scale
            + offset
        )
    return (
        _fold_data(
            fold.spec,
            scaled_train,
            scaled_test,
            scaled_context,
        ),
        metadata,
    )


def _result_dir(
    model: str,
    feature_set: str,
    window: int,
    seed: int,
    fold: str,
) -> Path:
    return (
        FUSION_OUTPUT_DIR
        / "runs"
        / model
        / feature_set
        / f"window_{window}"
        / f"seed_{seed}"
        / fold
    )


def _set_tensorflow_runtime(seed: int) -> None:
    import tensorflow as tf

    tf.keras.backend.clear_session()
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    tf.keras.utils.set_random_seed(seed)


def run_configuration(
    model: str,
    feature_set: str,
    window: int,
    seed: int,
    fold: FoldData,
    *,
    force: bool = False,
) -> dict[str, object]:
    if model not in TRACK_A_MODELS:
        raise ValueError(f"Unknown Track B fusion model: {model}")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown Track B fusion feature set: {feature_set}")
    output_dir = _result_dir(model, feature_set, window, seed, fold.spec.fold)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    scaled_fold, scaler = _scaled_fold(fold)
    _set_tensorflow_runtime(seed)
    started = time.perf_counter()
    scaled_prediction = predict_model(
        model,
        scaled_fold,
        sequence_window=window,
        seed=seed,
    )
    runtime_seconds = time.perf_counter() - started
    prediction = inverse_scaled_target(scaled_prediction, scaler)
    metrics: dict[str, object] = {
        "stage": "track_b_fusion",
        "model": model,
        "feature_set": feature_set,
        "sequence_window": int(window),
        "seed": int(seed),
        **evaluate_predictions(fold, prediction),
        "runtime_seconds": float(runtime_seconds),
        "news_start_year": NEWS_START_YEAR,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    predictions_frame(fold, prediction).to_csv(
        output_dir / "predictions.csv",
        index=False,
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "protocol_version": "track_b_point_in_time_v2",
                "model": model,
                "feature_set": feature_set,
                "sequence_window": int(window),
                "seed": int(seed),
                "fold": fold.spec.fold,
                "test_year": fold.spec.test_year,
                "train_period": [
                    fold.train[DATE_COLUMN].min().date().isoformat(),
                    fold.train[DATE_COLUMN].max().date().isoformat(),
                ],
                "test_period": [
                    fold.test[DATE_COLUMN].min().date().isoformat(),
                    fold.test[DATE_COLUMN].max().date().isoformat(),
                ],
                "context_feature_dates": (
                    fold.context[DATE_COLUMN]
                    .dt.strftime("%Y-%m-%d")
                    .tolist()
                    if fold.context is not None
                    else []
                ),
                "context_contract": (
                    "features used only in evaluation sequences; target "
                    "excluded from fitting"
                ),
                "scaler_fit_scope": "current fold train only",
                "daily_news_assignment": "strictly next trading day",
                "sentiment_source": (
                    SENTIMENT_SOURCE_DESCRIPTION
                    if feature_set == "technical_vmd_news"
                    else "none"
                ),
                "runtime_scope": "model build + training + test inference",
                "packages": package_versions(
                    ["numpy", "pandas", "scikit-learn", "tensorflow"]
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def _selected_windows() -> dict[str, int]:
    locked = load_locked_windows()
    return {
        str(row.model): int(row.selected_sequence_window)
        for row in locked.itertuples(index=False)
    }


def _selected_specs(
    source_dir: Path = FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
    test_years: Iterable[int] = FUSION_TEST_YEARS,
) -> list[FoldSpec]:
    years = frozenset(int(year) for year in test_years)
    unknown = sorted(years.difference(FUSION_TEST_YEARS))
    if unknown:
        raise ValueError(f"Fusion test years must be in {FUSION_TEST_YEARS}: {unknown}")
    specs = [spec for spec in discover_folds(source_dir) if spec.test_year in years]
    if {spec.test_year for spec in specs} != years:
        raise ValueError("Not every requested fusion test year has a fold")
    return specs


def run_fusion_experiment(
    *,
    models: Iterable[str] = TRACK_A_MODELS,
    seeds: Iterable[int] = FINAL_SEEDS,
    test_years: Iterable[int] = FUSION_TEST_YEARS,
    force: bool = False,
) -> pd.DataFrame:
    model_keys = tuple(models)
    unknown_models = sorted(set(model_keys).difference(TRACK_A_MODELS))
    if unknown_models or not model_keys:
        raise ValueError(f"Unknown or empty model selection: {unknown_models}")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("At least one seed is required")

    daily_news = (
        pd.read_csv(DAILY_NEWS_FILE, parse_dates=["date"])
        if DAILY_NEWS_FILE.is_file()
        else create_daily_news_features()
    )
    windows = _selected_windows()
    rows: list[dict[str, object]] = []
    experiment_started = time.perf_counter()
    for spec in _selected_specs(test_years=test_years):
        folds = prepare_fold_pair(spec, daily_news)
        for model in model_keys:
            for feature_set in FEATURE_SETS:
                for seed in seed_values:
                    print(
                        "Track B fusion: "
                        f"{model}, {feature_set}, {spec.fold}, seed={seed}",
                        flush=True,
                    )
                    rows.append(
                        run_configuration(
                            model,
                            feature_set,
                            windows[model],
                            seed,
                            folds[feature_set],
                            force=force,
                        )
                    )
    metrics = pd.DataFrame(rows)
    FUSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_FILE, index=False)
    build_and_save_reports(metrics)
    _write_experiment_metadata(
        metrics,
        runtime_seconds=time.perf_counter() - experiment_started,
    )
    return metrics


def collect_completed_metrics() -> pd.DataFrame:
    paths = sorted((FUSION_OUTPUT_DIR / "runs").glob("**/metrics.json"))
    if not paths:
        raise FileNotFoundError("No completed Track B fusion runs were found")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    metrics = pd.DataFrame(rows).sort_values(
        ["model", "feature_set", "seed", "test_year"]
    )
    return metrics.reset_index(drop=True)


def _confidence_interval(values: np.ndarray) -> tuple[float, float]:
    from scipy.stats import t

    data = np.asarray(values, dtype=float)
    if len(data) < 2:
        return np.nan, np.nan
    mean = float(data.mean())
    margin = float(t.ppf(0.975, len(data) - 1) * data.std(ddof=1) / np.sqrt(len(data)))
    return mean - margin, mean + margin


def build_paired_fusion_summary(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["model", "seed", "fold", "test_year", "sequence_window"]
    required = {*keys, "feature_set", *METRIC_COLUMNS, "runtime_seconds"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Fusion metrics are missing columns: {missing}")

    fold_means = (
        metrics.groupby(
            ["model", "feature_set", "sequence_window", "fold"],
            sort=False,
        )[[*METRIC_COLUMNS]]
        .mean()
        .reset_index()
    )
    performance = fold_means.groupby(
        ["model", "feature_set", "sequence_window"],
        sort=False,
    )[[*METRIC_COLUMNS]].agg(["mean", "std"])
    performance.columns = [
        f"{metric}_{statistic}" for metric, statistic in performance.columns
    ]
    performance = performance.reset_index()
    runtime = (
        metrics.groupby(
            ["model", "feature_set", "sequence_window"],
            sort=False,
        )["runtime_seconds"]
        .agg(["mean", "std", "sum"])
        .rename(
            columns={
                "mean": "runtime_seconds_mean",
                "std": "runtime_seconds_std",
                "sum": "runtime_seconds_total",
            }
        )
        .reset_index()
    )
    performance = performance.merge(
        runtime,
        on=["model", "feature_set", "sequence_window"],
        validate="one_to_one",
    )

    technical = metrics.loc[
        metrics["feature_set"].eq("technical_vmd"),
        [*keys, *METRIC_COLUMNS, "runtime_seconds"],
    ]
    news = metrics.loc[
        metrics["feature_set"].eq("technical_vmd_news"),
        [*keys, *METRIC_COLUMNS, "runtime_seconds"],
    ]
    paired = technical.merge(
        news,
        on=keys,
        suffixes=("_technical", "_news"),
        validate="one_to_one",
    )
    if len(paired) * 2 != len(metrics):
        raise ValueError("Fusion metrics are not fully paired")
    for metric in (
        "rmse",
        "mae",
        "mape",
        "r2",
        "mcc",
        "direction_coverage",
        "runtime_seconds",
    ):
        paired[f"{metric}_delta_news_minus_technical"] = (
            paired[f"{metric}_news"] - paired[f"{metric}_technical"]
        )
    paired["direction_accuracy_delta_pp"] = (
        paired["direction_accuracy_news"] - paired["direction_accuracy_technical"]
    ) * 100.0
    paired["balanced_accuracy_delta_pp"] = (
        paired["balanced_accuracy_news"]
        - paired["balanced_accuracy_technical"]
    ) * 100.0

    delta_columns = [
        "rmse_delta_news_minus_technical",
        "direction_accuracy_delta_pp",
        "balanced_accuracy_delta_pp",
        "mcc_delta_news_minus_technical",
        "direction_coverage_delta_news_minus_technical",
        "runtime_seconds_delta_news_minus_technical",
    ]
    fold_deltas = (
        paired.groupby(["model", "fold"], sort=False)[delta_columns]
        .mean()
        .reset_index()
    )
    summaries: list[dict[str, object]] = []
    for model, model_folds in fold_deltas.groupby("model", sort=False):
        rmse = model_folds["rmse_delta_news_minus_technical"].to_numpy(float)
        direction = model_folds["direction_accuracy_delta_pp"].to_numpy(float)
        balanced = model_folds["balanced_accuracy_delta_pp"].to_numpy(float)
        mcc = model_folds[
            "mcc_delta_news_minus_technical"
        ].to_numpy(float)
        coverage = model_folds[
            "direction_coverage_delta_news_minus_technical"
        ].to_numpy(float)
        runtime_values = model_folds[
            "runtime_seconds_delta_news_minus_technical"
        ].to_numpy(float)
        rmse_ci = _confidence_interval(rmse)
        direction_ci = _confidence_interval(direction)
        balanced_ci = _confidence_interval(balanced)
        summaries.append(
            {
                "model": model,
                "paired_runs": int((paired["model"] == model).sum()),
                "paired_outer_folds": len(model_folds),
                "seeds_per_fold": int(
                    paired.loc[paired["model"].eq(model), "seed"].nunique()
                ),
                "rmse_delta_mean": float(rmse.mean()),
                "rmse_delta_std": float(rmse.std(ddof=1)),
                "rmse_delta_ci95_lower": rmse_ci[0],
                "rmse_delta_ci95_upper": rmse_ci[1],
                "rmse_exact_sign_flip_pvalue": exact_sign_flip_pvalue(rmse),
                "direction_accuracy_delta_pp_mean": float(direction.mean()),
                "direction_accuracy_delta_pp_std": float(direction.std(ddof=1)),
                "direction_accuracy_delta_pp_ci95_lower": direction_ci[0],
                "direction_accuracy_delta_pp_ci95_upper": direction_ci[1],
                "direction_exact_sign_flip_pvalue": exact_sign_flip_pvalue(direction),
                "balanced_accuracy_delta_pp_mean": float(balanced.mean()),
                "balanced_accuracy_delta_pp_std": float(
                    balanced.std(ddof=1)
                ),
                "balanced_accuracy_delta_pp_ci95_lower": balanced_ci[0],
                "balanced_accuracy_delta_pp_ci95_upper": balanced_ci[1],
                "balanced_accuracy_exact_sign_flip_pvalue": (
                    exact_sign_flip_pvalue(balanced)
                ),
                "mcc_delta_news_minus_technical_mean": float(mcc.mean()),
                "direction_coverage_delta_mean": float(coverage.mean()),
                "runtime_seconds_delta_mean": float(runtime_values.mean()),
            }
        )
    return performance, paired, pd.DataFrame(summaries)


def build_fusion_paper_table(
    performance: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["model", "sequence_window"]
    technical = performance.loc[performance["feature_set"].eq("technical_vmd")].drop(
        columns="feature_set"
    )
    news = performance.loc[performance["feature_set"].eq("technical_vmd_news")].drop(
        columns="feature_set"
    )
    wide = technical.merge(
        news,
        on=keys,
        suffixes=("_technical", "_news"),
        validate="one_to_one",
    ).merge(paired_summary, on="model", validate="one_to_one")
    return wide.loc[
        :,
        [
            "model",
            "sequence_window",
            "rmse_mean_technical",
            "rmse_mean_news",
            "rmse_delta_mean",
            "rmse_delta_ci95_lower",
            "rmse_delta_ci95_upper",
            "rmse_exact_sign_flip_pvalue",
            "direction_accuracy_mean_technical",
            "direction_accuracy_mean_news",
            "direction_accuracy_delta_pp_mean",
            "direction_accuracy_delta_pp_ci95_lower",
            "direction_accuracy_delta_pp_ci95_upper",
            "direction_exact_sign_flip_pvalue",
            "balanced_accuracy_mean_technical",
            "balanced_accuracy_mean_news",
            "balanced_accuracy_delta_pp_mean",
            "balanced_accuracy_delta_pp_ci95_lower",
            "balanced_accuracy_delta_pp_ci95_upper",
            "balanced_accuracy_exact_sign_flip_pvalue",
            "mcc_mean_technical",
            "mcc_mean_news",
            "mcc_delta_news_minus_technical_mean",
            "direction_coverage_mean_technical",
            "direction_coverage_mean_news",
            "runtime_seconds_mean_technical",
            "runtime_seconds_mean_news",
            "paired_outer_folds",
            "seeds_per_fold",
        ],
    ].rename(
        columns={
            "sequence_window": "selected_sequence_window",
            "rmse_mean_technical": "technical_rmse_mean",
            "rmse_mean_news": "news_rmse_mean",
            "rmse_delta_mean": "rmse_delta_news_minus_technical",
            "direction_accuracy_mean_technical": ("technical_direction_accuracy_mean"),
            "direction_accuracy_mean_news": "news_direction_accuracy_mean",
            "direction_accuracy_delta_pp_mean": ("direction_accuracy_delta_pp"),
            "balanced_accuracy_mean_technical": (
                "technical_balanced_accuracy_mean"
            ),
            "balanced_accuracy_mean_news": "news_balanced_accuracy_mean",
            "balanced_accuracy_delta_pp_mean": (
                "balanced_accuracy_delta_pp"
            ),
            "mcc_mean_technical": "technical_mcc_mean",
            "mcc_mean_news": "news_mcc_mean",
            "mcc_delta_news_minus_technical_mean": (
                "mcc_delta_news_minus_technical"
            ),
            "direction_coverage_mean_technical": (
                "technical_direction_coverage_mean"
            ),
            "direction_coverage_mean_news": (
                "news_direction_coverage_mean"
            ),
            "runtime_seconds_mean_technical": ("technical_runtime_seconds_mean"),
            "runtime_seconds_mean_news": "news_runtime_seconds_mean",
        }
    )


def build_and_save_reports(
    metrics: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = collect_completed_metrics() if metrics is None else metrics
    performance, paired, summary = build_paired_fusion_summary(values)
    paper = build_fusion_paper_table(performance, summary)
    FUSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    values.to_csv(METRICS_FILE, index=False)
    performance.to_csv(PERFORMANCE_FILE, index=False)
    paired.to_csv(PAIRED_FILE, index=False)
    summary.to_csv(PAIRED_SUMMARY_FILE, index=False)
    paper.to_csv(PAPER_TABLE_FILE, index=False)
    return performance, paired, summary, paper


def _write_experiment_metadata(
    metrics: pd.DataFrame,
    *,
    runtime_seconds: float | None,
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": EXPERIMENT_NAME,
        "protocol_version": "track_b_point_in_time_v2",
        "models": list(TRACK_A_MODELS),
        "feature_sets": list(FEATURE_SETS),
        "locked_windows": _selected_windows(),
        "seeds": sorted(int(value) for value in metrics["seed"].unique()),
        "test_years": sorted(int(value) for value in metrics["test_year"].unique()),
        "news_start_year": NEWS_START_YEAR,
        "training_scope": (
            "2019 through test_year-1; identical rows for technical and "
            "technical+news arms"
        ),
        "daily_news_assignment": "strictly next trading day",
        "sentiment_predictions": str(SENTIMENT_PREDICTIONS_FILE),
        "sentiment_prediction_contract": SENTIMENT_PREDICTION_CONTRACT,
        "scaling": "MinMaxScaler fit separately on each fold train only",
        "market_source_dir": str(
            FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR
        ),
        "boundary_context": (
            "feature-only rows between supervised train and test are "
            "included in test sequences but excluded from model fitting"
        ),
        "direction_metric_contract": {
            "task": "binary Up versus Down",
            "actual_ties": "excluded",
            "predicted_no_change": "abstention",
        },
        "completed_fits": len(metrics),
        "fit_runtime_seconds_sum": float(metrics["runtime_seconds"].sum()),
        "command_runtime_seconds": (
            None if runtime_seconds is None else float(runtime_seconds)
        ),
        "runtime_scope": (
            "fit_runtime_seconds_sum is the sum of model build, fit, and "
            "evaluation inference time; process startup and report recovery "
            "are excluded"
        ),
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "scipy", "tensorflow"]
        ),
    }
    RUN_METADATA_FILE.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _parse_csv_values(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main() -> object:
    parser = argparse.ArgumentParser(
        description="Run leakage-controlled Track B news fusion."
    )
    parser.add_argument(
        "stage",
        choices=["prepare", "run", "report", "all"],
        nargs="?",
        default="all",
    )
    parser.add_argument("--models", default=",".join(TRACK_A_MODELS))
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in FINAL_SEEDS),
    )
    parser.add_argument(
        "--years",
        default=",".join(str(year) for year in FUSION_TEST_YEARS),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        return create_daily_news_features()
    if args.stage == "report":
        return build_and_save_reports()
    if args.stage in {"run", "all"}:
        create_daily_news_features()
        return run_fusion_experiment(
            models=_parse_csv_values(args.models),
            seeds=_parse_csv_values(args.seeds, int),
            test_years=_parse_csv_values(args.years, int),
            force=args.force,
        )
    raise AssertionError("Unreachable stage")


if __name__ == "__main__":
    main()
