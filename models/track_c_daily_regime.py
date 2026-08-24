from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from models.baseline_common import (
    CLOSE_COLUMN,
    DATE_COLUMN,
    PROJECT_ROOT,
    discover_folds,
)
from models.vmd_feature_pool import (
    FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
)

LAGGED_CLOSE_COLUMN = "Close_D_lag1"
SEMANTIC_REGIMES = ("bull", "sideway", "bear")
LEGACY_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_c" / "daily_regime_v2"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "track_c" / "daily_regime_point_in_time_v2"
)
DEFAULT_HMM_BASELINE_DIR = PROJECT_ROOT / "outputs" / "track_c" / "regime_labeling"


@dataclass(frozen=True)
class DailyRegimeConfig:
    horizons: tuple[int, ...] = (1, 3, 5, 10, 20, 60)
    weights: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.25)
    volatility_horizons: tuple[int, ...] = (20, 20, 20, 20, 20, 60)
    adx_column: str = "ADX_14"
    smoothing_span: int = 3
    sideway_quantile: float = 0.35
    membership_temperature: float = 0.35
    minimum_training_regime_share: float = 0.15

    def validate(self) -> DailyRegimeConfig:
        lengths = {
            len(self.horizons),
            len(self.weights),
            len(self.volatility_horizons),
        }
        if len(lengths) != 1:
            raise ValueError(
                "horizons, weights, and volatility_horizons must have the same length"
            )
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and increasing")
        if self.horizons != (1, 3, 5, 10, 20, 60):
            raise ValueError("horizons are locked to (1, 3, 5, 10, 20, 60) for Track C")
        if any(weight <= 0 for weight in self.weights) or not math.isclose(
            sum(self.weights),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("weights must be positive and sum to one")
        if any(horizon <= 1 for horizon in self.volatility_horizons):
            raise ValueError("volatility_horizons must be greater than one")
        if self.volatility_horizons != (20, 20, 20, 20, 20, 60):
            raise ValueError(
                "volatility_horizons are locked to "
                "(20, 20, 20, 20, 20, 60) for Track C"
            )
        if self.adx_column != "ADX_14":
            raise ValueError("adx_column is locked to ADX_14 for Track C")
        if self.smoothing_span < 1:
            raise ValueError("smoothing_span must be positive")
        if not 0.10 <= self.sideway_quantile <= 0.60:
            raise ValueError("sideway_quantile must be between 0.10 and 0.60")
        if self.membership_temperature <= 0:
            raise ValueError("membership_temperature must be positive")
        if not 0.0 < self.minimum_training_regime_share < 1.0 / 3.0:
            raise ValueError(
                "minimum_training_regime_share must be between zero and one-third"
            )
        return self


@dataclass(frozen=True)
class DailyFoldRegimeResult:
    train_labels: pd.DataFrame
    test_labels: pd.DataFrame
    semantic_profiles: pd.DataFrame
    model_parameters: dict[str, object]
    metadata: dict[str, object]


def _return_column(horizon: int) -> str:
    return f"Return_{horizon}D"


def _volatility_column(horizon: int) -> str:
    return f"Volatility_{horizon}"


def _required_columns(config: DailyRegimeConfig) -> tuple[str, ...]:
    columns = {
        DATE_COLUMN,
        CLOSE_COLUMN,
        LAGGED_CLOSE_COLUMN,
        config.adx_column,
    }
    columns.update(_return_column(horizon) for horizon in config.horizons)
    columns.update(
        _volatility_column(horizon) for horizon in config.volatility_horizons
    )
    return tuple(sorted(columns))


def _validate_daily_frame(
    frame: pd.DataFrame,
    *,
    config: DailyRegimeConfig,
    context: str,
) -> pd.DataFrame:
    required = set(_required_columns(config))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{context} missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{context} is empty")
    result = frame.loc[:, list(frame.columns)].copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], errors="raise")
    if result[DATE_COLUMN].isna().any():
        raise ValueError(f"{context} contains invalid dates")
    if result[DATE_COLUMN].duplicated().any():
        raise ValueError(f"{context} contains duplicate dates")
    result = result.sort_values(DATE_COLUMN).reset_index(drop=True)

    numeric_columns = sorted(required.difference({DATE_COLUMN}))
    numeric = result.loc[:, numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{context} contains non-finite regime inputs")
    if (result[[CLOSE_COLUMN, LAGGED_CLOSE_COLUMN]] <= 0).any().any():
        raise ValueError(f"{context} close prices must be strictly positive")
    volatility_columns = sorted(
        {_volatility_column(horizon) for horizon in config.volatility_horizons}
    )
    if (result[volatility_columns] <= 0).any().any():
        raise ValueError(f"{context} volatility inputs must be strictly positive")
    if ((result[config.adx_column] < 0) | (result[config.adx_column] > 100)).any():
        raise ValueError(f"{context} ADX values must be between zero and 100")
    return result


def build_daily_regime_features(
    frame: pd.DataFrame,
    *,
    config: DailyRegimeConfig,
) -> pd.DataFrame:
    locked = config.validate()
    market = _validate_daily_frame(
        frame,
        config=locked,
        context="Daily regime frame",
    )
    result = market[
        [
            DATE_COLUMN,
            CLOSE_COLUMN,
            LAGGED_CLOSE_COLUMN,
            *[_return_column(horizon) for horizon in locked.horizons],
            *sorted(
                {_volatility_column(horizon) for horizon in locked.volatility_horizons}
            ),
            locked.adx_column,
        ]
    ].copy()
    result["log_return_1d"] = np.log(
        result[CLOSE_COLUMN].to_numpy(dtype=float)
        / result[LAGGED_CLOSE_COLUMN].to_numpy(dtype=float)
    )

    weighted_terms: list[pd.Series] = []
    for horizon, weight, volatility_horizon in zip(
        locked.horizons,
        locked.weights,
        locked.volatility_horizons,
        strict=True,
    ):
        trend = result[_return_column(horizon)] / (
            result[_volatility_column(volatility_horizon)] * math.sqrt(horizon)
        )
        result[f"trend_z_{horizon}d"] = trend
        weighted_terms.append(weight * trend)
    result["composite_trend_score"] = sum(weighted_terms)
    result["directional_strength"] = (result[locked.adx_column] / 100.0).clip(
        lower=0.0, upper=1.0
    )
    result["semantic_score_raw"] = (
        result["composite_trend_score"] * result["directional_strength"]
    )
    result["semantic_score"] = (
        result["semantic_score_raw"]
        .ewm(
            span=locked.smoothing_span,
            adjust=False,
            min_periods=1,
        )
        .mean()
    )
    feature_columns = [
        "log_return_1d",
        *[f"trend_z_{horizon}d" for horizon in locked.horizons],
        "composite_trend_score",
        "directional_strength",
        "semantic_score_raw",
        "semantic_score",
    ]
    if not np.isfinite(result[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Daily semantic regime features contain non-finite values")
    return result


def score_memberships(
    semantic_scores: np.ndarray,
    *,
    threshold: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(semantic_scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("semantic_scores must be a finite non-empty 1D array")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    normalized = scores / threshold
    logits = np.column_stack(
        [
            normalized - 1.0,
            np.zeros_like(normalized),
            -normalized - 1.0,
        ]
    )
    logits = logits / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(logits)
    probabilities = exponential / exponential.sum(axis=1, keepdims=True)
    labels = np.where(
        scores > threshold,
        "bull",
        np.where(scores < -threshold, "bear", "sideway"),
    )
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1),
        1.0,
    ):
        raise ValueError("Regime memberships are invalid")
    return probabilities, labels


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.clip(probabilities, 1e-15, 1.0)
    return -np.sum(safe * np.log(safe), axis=1) / math.log(probabilities.shape[1])


def _labels_frame(
    features: pd.DataFrame,
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    fold_name: str,
    split: str,
    config: DailyRegimeConfig,
) -> pd.DataFrame:
    keep = [
        DATE_COLUMN,
        CLOSE_COLUMN,
        "log_return_1d",
        *[_return_column(horizon) for horizon in config.horizons],
        config.adx_column,
        *[f"trend_z_{horizon}d" for horizon in config.horizons],
        "composite_trend_score",
        "directional_strength",
        "semantic_score_raw",
        "semantic_score",
    ]
    result = features.loc[:, keep].reset_index(drop=True).copy()
    result["fold"] = fold_name
    result["split"] = split
    for index, regime in enumerate(SEMANTIC_REGIMES):
        result[f"prob_{regime}"] = probabilities[:, index]
    result["routing_regime"] = labels
    result["routing_confidence"] = probabilities.max(axis=1)
    result["routing_entropy"] = _entropy(probabilities)
    result["routing_target"] = "current_regime_at_t_for_direction_t_plus_1"
    return result


def _semantic_profiles(
    train_labels: pd.DataFrame,
    *,
    fold_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for regime in SEMANTIC_REGIMES:
        selected = train_labels.loc[train_labels["routing_regime"].eq(regime)]
        rows.append(
            {
                "fold": fold_name,
                "regime": regime,
                "rows": len(selected),
                "share": len(selected) / len(train_labels),
                "mean_return_1d": selected["Return_1D"].mean(),
                "mean_return_20d": selected["Return_20D"].mean(),
                "median_return_20d": selected["Return_20D"].median(),
                "mean_adx_14": selected["ADX_14"].mean(),
                "mean_semantic_score": selected["semantic_score"].mean(),
                "mean_absolute_semantic_score": selected["semantic_score"].abs().mean(),
                "mean_routing_confidence": selected["routing_confidence"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _quality_metadata(
    profiles: pd.DataFrame,
    *,
    minimum_training_regime_share: float,
) -> dict[str, object]:
    indexed = profiles.set_index("regime")
    mean_returns = indexed["mean_return_20d"]
    absolute_scores = indexed["mean_absolute_semantic_score"]
    ordered_returns = bool(
        mean_returns["bear"] < mean_returns["sideway"] < mean_returns["bull"]
    )
    sideway_return_nearest_zero = bool(
        abs(mean_returns["sideway"])
        < min(abs(mean_returns["bull"]), abs(mean_returns["bear"]))
    )
    sideway_score_smallest = bool(
        absolute_scores["sideway"]
        < min(absolute_scores["bull"], absolute_scores["bear"])
    )
    minimum_share = float(indexed["share"].min())
    gate = bool(
        minimum_share >= minimum_training_regime_share
        and ordered_returns
        and sideway_return_nearest_zero
        and sideway_score_smallest
    )
    return {
        "minimum_training_regime_share": minimum_share,
        "ordered_training_mean_return_20d": ordered_returns,
        "sideway_return_nearest_zero": sideway_return_nearest_zero,
        "sideway_score_smallest": sideway_score_smallest,
        "quality_gate_passed": gate,
    }


def fit_fold_daily_regimes(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    config: DailyRegimeConfig,
    fold_name: str,
) -> DailyFoldRegimeResult:
    locked = config.validate()
    train_market = _validate_daily_frame(
        train,
        config=locked,
        context=f"{fold_name} train",
    )
    test_market = _validate_daily_frame(
        test,
        config=locked,
        context=f"{fold_name} test",
    )
    if train_market[DATE_COLUMN].max() >= test_market[DATE_COLUMN].min():
        raise ValueError(f"{fold_name} train and test dates overlap")

    combined = pd.concat([train_market, test_market], ignore_index=True)
    features = build_daily_regime_features(combined, config=locked)
    train_count = len(train_market)
    train_features = features.iloc[:train_count].reset_index(drop=True)
    test_features = features.iloc[train_count:].reset_index(drop=True)
    threshold = float(
        train_features["semantic_score"]
        .abs()
        .quantile(
            locked.sideway_quantile,
        )
    )
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"{fold_name} produced an invalid sideway threshold")

    probabilities, labels = score_memberships(
        features["semantic_score"].to_numpy(dtype=float),
        threshold=threshold,
        temperature=locked.membership_temperature,
    )
    train_labels = _labels_frame(
        train_features,
        probabilities[:train_count],
        labels[:train_count],
        fold_name=fold_name,
        split="train",
        config=locked,
    )
    test_labels = _labels_frame(
        test_features,
        probabilities[train_count:],
        labels[train_count:],
        fold_name=fold_name,
        split="test",
        config=locked,
    )
    profiles = _semantic_profiles(train_labels, fold_name=fold_name)
    quality = _quality_metadata(
        profiles,
        minimum_training_regime_share=locked.minimum_training_regime_share,
    )
    metadata: dict[str, object] = {
        "protocol_version": "track_c_daily_regime_point_in_time_v2",
        "fold": fold_name,
        "fit_scope": "fold_training_only",
        "threshold_fit_scope": "fold_training_only",
        "routing_target": "current_regime_at_t_for_direction_t_plus_1",
        "target_columns_used": False,
        "train_rows": len(train_labels),
        "test_rows": len(test_labels),
        "sideway_threshold": threshold,
        **quality,
    }
    parameters: dict[str, object] = {
        "config": asdict(locked),
        "sideway_threshold": threshold,
        "score_definition": (
            "EWMA_3(sum(weight_h * Return_h / "
            "(Volatility_ref_h * sqrt(h))) * ADX_14 / 100)"
        ),
        "hard_rule": (
            "Bull if score > threshold; Bear if score < -threshold; "
            "otherwise Sideway"
        ),
        "threshold_fit_scope": "fold_training_only",
    }
    return DailyFoldRegimeResult(
        train_labels=train_labels,
        test_labels=test_labels,
        semantic_profiles=profiles,
        model_parameters=parameters,
        metadata=metadata,
    )


def _observed_run_lengths(labels: pd.Series) -> dict[str, float]:
    values = labels.astype(str).reset_index(drop=True)
    groups = values.ne(values.shift()).cumsum()
    runs = (
        pd.DataFrame({"regime": values, "group": groups})
        .groupby(["group", "regime"], sort=False)
        .size()
    )
    return {
        regime: (
            float(runs.xs(regime, level="regime").mean())
            if regime in runs.index.get_level_values("regime")
            else math.nan
        )
        for regime in SEMANTIC_REGIMES
    }


def _distribution_rows(
    frame: pd.DataFrame,
    *,
    fold_name: str,
    split: str,
) -> list[dict[str, object]]:
    run_lengths = _observed_run_lengths(frame["routing_regime"])
    rows: list[dict[str, object]] = []
    for regime in SEMANTIC_REGIMES:
        selected = frame.loc[frame["routing_regime"].eq(regime)]
        rows.append(
            {
                "fold": fold_name,
                "split": split,
                "regime": regime,
                "rows": len(selected),
                "share": len(selected) / len(frame),
                "mean_routing_confidence": selected["routing_confidence"].mean(),
                "mean_return_1d": selected["Return_1D"].mean(),
                "mean_return_20d": selected["Return_20D"].mean(),
                "mean_adx_14": selected["ADX_14"].mean(),
                "observed_mean_run_days": run_lengths[regime],
            }
        )
    return rows


def _switch_count(labels: pd.Series) -> int:
    return int(labels.ne(labels.shift()).sum() - 1)


def _protocol_development_rows(
    first_training_frame: pd.DataFrame,
    *,
    config: DailyRegimeConfig,
) -> list[dict[str, object]]:
    features = build_daily_regime_features(
        first_training_frame,
        config=config,
    )
    development = features.loc[features[DATE_COLUMN] < pd.Timestamp("2020-01-01")]
    validation = features.loc[features[DATE_COLUMN] >= pd.Timestamp("2020-01-01")]
    if development.empty or validation.empty:
        return []
    rows: list[dict[str, object]] = []
    for quantile in (0.30, 0.35, 0.40):
        threshold = float(development["semantic_score"].abs().quantile(quantile))
        for split_name, split_frame in (
            ("development_2012_2019", development),
            ("validation_2020_2021", validation),
        ):
            _, labels = score_memberships(
                split_frame["semantic_score"].to_numpy(dtype=float),
                threshold=threshold,
                temperature=config.membership_temperature,
            )
            evaluated = split_frame.assign(regime=labels)
            for regime in SEMANTIC_REGIMES:
                selected = evaluated.loc[evaluated["regime"].eq(regime)]
                rows.append(
                    {
                        "sideway_quantile": quantile,
                        "selected_protocol": math.isclose(
                            quantile,
                            config.sideway_quantile,
                        ),
                        "split": split_name,
                        "regime": regime,
                        "threshold_fit_on_development_only": threshold,
                        "rows": len(selected),
                        "share": len(selected) / len(evaluated),
                        "mean_return_20d": selected["Return_20D"].mean(),
                        "mean_adx_14": selected[config.adx_column].mean(),
                        "mean_absolute_semantic_score": selected["semantic_score"]
                        .abs()
                        .mean(),
                    }
                )
    return rows


def _hmm_comparison_rows(
    v2_test: pd.DataFrame,
    *,
    hmm_test_path: Path,
    fold_name: str,
) -> list[dict[str, object]]:
    if not hmm_test_path.is_file():
        return []
    baseline = pd.read_csv(hmm_test_path)
    required = {DATE_COLUMN, "routing_regime"}
    if not required.issubset(baseline.columns):
        raise ValueError(f"Invalid HMM baseline artifact: {hmm_test_path}")
    comparison = v2_test[[DATE_COLUMN, "routing_regime"]].copy()
    comparison[DATE_COLUMN] = pd.to_datetime(comparison[DATE_COLUMN])
    baseline_subset = baseline[[DATE_COLUMN, "routing_regime"]].copy()
    baseline_subset[DATE_COLUMN] = pd.to_datetime(baseline_subset[DATE_COLUMN])
    merged = comparison.merge(
        baseline_subset,
        on=DATE_COLUMN,
        how="inner",
        suffixes=("_v2", "_hmm_v1"),
        validate="one_to_one",
    )
    return [
        {
            "fold": fold_name,
            "hmm_v1_regime": hmm_regime,
            "daily_v2_regime": v2_regime,
            "rows": int(
                (
                    merged["routing_regime_hmm_v1"].eq(hmm_regime)
                    & merged["routing_regime_v2"].eq(v2_regime)
                ).sum()
            ),
            "comparison_rows": len(merged),
            "overall_agreement": float(
                merged["routing_regime_hmm_v1"].eq(merged["routing_regime_v2"]).mean()
            ),
        }
        for hmm_regime in SEMANTIC_REGIMES
        for v2_regime in SEMANTIC_REGIMES
    ]


def _causality_audit_row(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    config: DailyRegimeConfig,
    threshold: float,
    fold_name: str,
) -> dict[str, object]:
    train_market = _validate_daily_frame(
        train,
        config=config,
        context=f"{fold_name} causality train",
    )
    test_market = _validate_daily_frame(
        test,
        config=config,
        context=f"{fold_name} causality test",
    )
    combined = pd.concat([train_market, test_market], ignore_index=True)
    cutoff = len(train_market) + max(1, len(test_market) // 2)
    full_features = build_daily_regime_features(combined, config=config)
    prefix_features = build_daily_regime_features(
        combined.iloc[:cutoff].copy(),
        config=config,
    )
    numeric_columns = list(prefix_features.select_dtypes(include=[np.number]).columns)
    feature_difference = np.abs(
        full_features.loc[: cutoff - 1, numeric_columns].to_numpy(dtype=float)
        - prefix_features[numeric_columns].to_numpy(dtype=float)
    )
    full_memberships, full_labels = score_memberships(
        full_features["semantic_score"].to_numpy(dtype=float),
        threshold=threshold,
        temperature=config.membership_temperature,
    )
    prefix_memberships, prefix_labels = score_memberships(
        prefix_features["semantic_score"].to_numpy(dtype=float),
        threshold=threshold,
        temperature=config.membership_temperature,
    )
    return {
        "fold": fold_name,
        "prefix_rows": cutoff,
        "feature_max_abs_difference": float(feature_difference.max()),
        "membership_max_abs_difference": float(
            np.abs(full_memberships[:cutoff] - prefix_memberships).max()
        ),
        "label_mismatches": int(
            np.count_nonzero(full_labels[:cutoff] != prefix_labels)
        ),
    }


def _sensitivity_rows(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    selected_labels: pd.Series,
    config: DailyRegimeConfig,
    fold_name: str,
) -> list[dict[str, object]]:
    variants = (
        (
            "sideway_quantile_0.30",
            replace(config, sideway_quantile=0.30),
        ),
        (
            "sideway_quantile_0.40",
            replace(config, sideway_quantile=0.40),
        ),
        ("smoothing_span_2", replace(config, smoothing_span=2)),
        ("smoothing_span_5", replace(config, smoothing_span=5)),
    )
    selected = selected_labels.astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for variant_name, variant_config in variants:
        variant = fit_fold_daily_regimes(
            train,
            test,
            config=variant_config,
            fold_name=fold_name,
        )
        labels = variant.test_labels["routing_regime"].astype(str).to_numpy()
        shares = (
            pd.Series(labels)
            .value_counts(normalize=True)
            .reindex(SEMANTIC_REGIMES, fill_value=0.0)
        )
        rows.append(
            {
                "fold": fold_name,
                "variant": variant_name,
                "sideway_quantile": variant_config.sideway_quantile,
                "smoothing_span": variant_config.smoothing_span,
                "agreement_with_selected": float(np.mean(labels == selected)),
                "adjusted_rand_with_selected": float(
                    adjusted_rand_score(selected, labels)
                ),
                "bull_share": float(shares["bull"]),
                "sideway_share": float(shares["sideway"]),
                "bear_share": float(shares["bear"]),
                "quality_gate_passed": variant.metadata["quality_gate_passed"],
            }
        )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_daily_regime_labeling(
    *,
    source_dir: Path = FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: DailyRegimeConfig | None = None,
    hmm_baseline_dir: Path | None = DEFAULT_HMM_BASELINE_DIR,
) -> dict[str, object]:
    locked = (DailyRegimeConfig() if config is None else config).validate()
    started_at = _utc_now()
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    distributions: list[dict[str, object]] = []
    profiles: list[pd.DataFrame] = []
    thresholds: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    causality_audits: list[dict[str, object]] = []
    sensitivity: list[dict[str, object]] = []
    input_files: list[dict[str, object]] = []
    specs = discover_folds(source_dir)
    first_training_frame: pd.DataFrame | None = None

    for spec in specs:
        train = pd.read_csv(spec.train_path)
        test = pd.read_csv(spec.test_path)
        if first_training_frame is None:
            first_training_frame = train.copy()
        fold_started = time.perf_counter()
        result = fit_fold_daily_regimes(
            train,
            test,
            config=locked,
            fold_name=spec.fold,
        )
        fold_runtime = time.perf_counter() - fold_started
        fold_dir = output_dir / spec.fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        result.train_labels.to_csv(
            fold_dir / "train_regimes.csv",
            index=False,
        )
        result.test_labels.to_csv(
            fold_dir / "test_regimes.csv",
            index=False,
        )
        _write_json(
            fold_dir / "model_parameters.json",
            result.model_parameters,
        )

        summaries.append(
            {
                **result.metadata,
                "train_start": result.train_labels[DATE_COLUMN]
                .min()
                .date()
                .isoformat(),
                "train_end": result.train_labels[DATE_COLUMN].max().date().isoformat(),
                "test_start": result.test_labels[DATE_COLUMN].min().date().isoformat(),
                "test_end": result.test_labels[DATE_COLUMN].max().date().isoformat(),
                "train_routing_switches": _switch_count(
                    result.train_labels["routing_regime"]
                ),
                "test_routing_switches": _switch_count(
                    result.test_labels["routing_regime"]
                ),
                "fold_runtime_seconds": fold_runtime,
            }
        )
        distributions.extend(
            _distribution_rows(
                result.train_labels,
                fold_name=spec.fold,
                split="train",
            )
        )
        distributions.extend(
            _distribution_rows(
                result.test_labels,
                fold_name=spec.fold,
                split="test",
            )
        )
        profiles.append(result.semantic_profiles)
        thresholds.append(
            {
                "fold": spec.fold,
                "training_end": result.train_labels[DATE_COLUMN]
                .max()
                .date()
                .isoformat(),
                "sideway_quantile": locked.sideway_quantile,
                "sideway_threshold": result.metadata["sideway_threshold"],
            }
        )
        causality_audits.append(
            _causality_audit_row(
                train,
                test,
                config=locked,
                threshold=float(result.metadata["sideway_threshold"]),
                fold_name=spec.fold,
            )
        )
        sensitivity.extend(
            _sensitivity_rows(
                train,
                test,
                selected_labels=result.test_labels["routing_regime"],
                config=locked,
                fold_name=spec.fold,
            )
        )
        if hmm_baseline_dir is not None:
            comparisons.extend(
                _hmm_comparison_rows(
                    result.test_labels,
                    hmm_test_path=(hmm_baseline_dir / spec.fold / "test_regimes.csv"),
                    fold_name=spec.fold,
                )
            )
        input_files.extend(
            [
                {
                    "fold": spec.fold,
                    "split": "train",
                    "path": _display_path(spec.train_path),
                    "sha256": _sha256(spec.train_path),
                    "rows": len(train),
                },
                {
                    "fold": spec.fold,
                    "split": "test",
                    "path": _display_path(spec.test_path),
                    "sha256": _sha256(spec.test_path),
                    "rows": len(test),
                },
            ]
        )

    pd.DataFrame(summaries).to_csv(
        output_dir / "fold_summary.csv",
        index=False,
    )
    pd.DataFrame(distributions).to_csv(
        output_dir / "regime_distribution.csv",
        index=False,
    )
    pd.concat(profiles, ignore_index=True).to_csv(
        output_dir / "semantic_profiles.csv",
        index=False,
    )
    pd.DataFrame(thresholds).to_csv(
        output_dir / "threshold_stability.csv",
        index=False,
    )
    pd.DataFrame(causality_audits).to_csv(
        output_dir / "causality_audit.csv",
        index=False,
    )
    pd.DataFrame(sensitivity).to_csv(
        output_dir / "sensitivity_analysis.csv",
        index=False,
    )
    development_rows = (
        []
        if first_training_frame is None
        else _protocol_development_rows(
            first_training_frame,
            config=locked,
        )
    )
    pd.DataFrame(development_rows).to_csv(
        output_dir / "protocol_development.csv",
        index=False,
    )
    if comparisons:
        pd.DataFrame(comparisons).to_csv(
            output_dir / "hmm_baseline_comparison.csv",
            index=False,
        )

    metadata: dict[str, object] = {
        "started_at": started_at,
        "completed_at": _utc_now(),
        "experiment": (
            "Track C causal daily multi-timescale semantic regime labeling v2"
        ),
        "completed_folds": len(specs),
        "total_runtime_seconds": time.perf_counter() - started,
        "source_dir": _display_path(source_dir),
        "output_dir": _display_path(output_dir),
        "protocol": {
            **asdict(locked),
            "score_definition": (
                "EWMA_3(sum(weight_h * Return_h / "
                "(Volatility_ref_h * sqrt(h))) * ADX_14 / 100)"
            ),
            "sideway_definition": (
                "absolute semantic score inside a symmetric training-only "
                "35th-percentile deadband"
            ),
            "threshold_fit_scope": "each fold training period only",
            "label_time_purge_rule": (
                "source folds retain Label_Date < first evaluation Date"
            ),
            "routing_target": ("current_regime_at_t_for_direction_t_plus_1"),
            "membership_interpretation": (
                "distance-based soft memberships; not calibrated posteriors"
            ),
            "target_columns_used": False,
            "outer_test_used_for_thresholds": False,
            "future_rows_used_for_features": False,
            "post_hoc_protocol_correction": True,
            "post_hoc_note": (
                "The v2 construct was introduced after the HMM v1 outer "
                "diagnostic exposed a non-semantic Sideway state. Exact v2 "
                "weights and deadband were screened only on 2012-2019 "
                "development and 2020-2021 validation data; 2022-2025 results "
                "must still be described as post-hoc robustness evidence."
            ),
        },
        "input_files": input_files,
        "artifacts": {
            name: _display_path(output_dir / name)
            for name in (
                "fold_summary.csv",
                "regime_distribution.csv",
                "semantic_profiles.csv",
                "threshold_stability.csv",
                "causality_audit.csv",
                "sensitivity_analysis.csv",
                "protocol_development.csv",
            )
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "paper_sources": [
            {
                "citation": (
                    "Wilder (1978), New Concepts in Technical Trading Systems"
                ),
                "use": "Average Directional Index (ADX)",
            },
            {
                "citation": (
                    "Moskowitz, Ooi, and Pedersen (2012), " "Time Series Momentum"
                ),
                "doi_or_url": ("https://doi.org/10.1016/j.jfineco.2011.11.003"),
                "use": "directional return signals over time",
            },
            {
                "citation": "Hamilton (1989)",
                "doi_or_url": (
                    "https://ideas.repec.org/a/ecm/emetrp/" "v57y1989i2p357-84.html"
                ),
                "use": "HMM v1 baseline",
            },
        ],
    }
    if comparisons:
        metadata["artifacts"]["hmm_baseline_comparison.csv"] = _display_path(
            output_dir / "hmm_baseline_comparison.csv"
        )
    _write_json(output_dir / "run_metadata.json", metadata)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Build causal daily Bull/Sideway/Bear labels for Track C.")
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--sideway-quantile", type=float, default=0.35)
    parser.add_argument("--smoothing-span", type=int, default=3)
    parser.add_argument(
        "--no-hmm-comparison",
        action="store_true",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    config = DailyRegimeConfig(
        sideway_quantile=args.sideway_quantile,
        smoothing_span=args.smoothing_span,
    )
    return run_daily_regime_labeling(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        config=config,
        hmm_baseline_dir=(None if args.no_hmm_comparison else DEFAULT_HMM_BASELINE_DIR),
    )


if __name__ == "__main__":
    main()
