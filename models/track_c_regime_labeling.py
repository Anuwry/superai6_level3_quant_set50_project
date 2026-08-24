from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from models.baseline_common import (
    CLOSE_COLUMN,
    DATE_COLUMN,
    PROJECT_ROOT,
    discover_folds,
)
from models.vmd_feature_pool import FULL_TA_VMD_DATA_FOLDS_DIR

LAGGED_CLOSE_COLUMN = "Close_D_lag1"
OBSERVATION_COLUMNS = ("ewma_return", "ewma_volatility")
SEMANTIC_REGIMES = ("bull", "sideway", "bear")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_c" / "regime_labeling"


@dataclass(frozen=True)
class RegimeConfig:
    state_count: int = 3
    span: int = 30
    covariance_type: str = "full"
    restart_seeds: tuple[int, ...] = (42, 123, 456, 789, 2025)
    max_iterations: int = 200
    tolerance: float = 1e-4
    minimum_covariance: float = 1e-5
    minimum_training_regime_share: float = 0.05
    minimum_self_transition_probability: float = 0.80

    def validate(self) -> RegimeConfig:
        if self.state_count != 3:
            raise ValueError("state_count must be 3 for Bull/Sideway/Bear")
        if self.span < 2:
            raise ValueError("span must be at least 2")
        if self.covariance_type != "full":
            raise ValueError("Only full covariance is supported")
        if not self.restart_seeds:
            raise ValueError("At least one restart seed is required")
        if len(set(self.restart_seeds)) != len(self.restart_seeds):
            raise ValueError("restart_seeds must be unique")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0 or self.minimum_covariance <= 0:
            raise ValueError("Tolerance and minimum covariance must be positive")
        if not 0.0 < self.minimum_training_regime_share < 1.0 / 3.0:
            raise ValueError(
                "minimum_training_regime_share must be between zero and one-third"
            )
        if not 0.0 < self.minimum_self_transition_probability < 1.0:
            raise ValueError(
                "minimum_self_transition_probability must be between zero and one"
            )
        return self


@dataclass(frozen=True)
class FittedRegimeModel:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    start_probability: np.ndarray
    transition_matrix: np.ndarray
    standardized_means: np.ndarray
    covariances: np.ndarray
    raw_state_means: np.ndarray
    state_map: dict[int, str]
    chosen_seed: int
    train_log_likelihood: float
    bic: float
    converged: bool
    iterations: int
    runtime_seconds: float
    restart_diagnostics: pd.DataFrame


@dataclass(frozen=True)
class FoldRegimeResult:
    train_labels: pd.DataFrame
    test_labels: pd.DataFrame
    state_profiles: pd.DataFrame
    restart_diagnostics: pd.DataFrame
    semantic_transition_matrix: np.ndarray
    model_parameters: dict[str, object]
    metadata: dict[str, object]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _validate_market_frame(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    required = {DATE_COLUMN, CLOSE_COLUMN, LAGGED_CLOSE_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{context} is empty")

    result = frame.loc[:, [DATE_COLUMN, CLOSE_COLUMN, LAGGED_CLOSE_COLUMN]].copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], errors="coerce")
    if result[DATE_COLUMN].isna().any():
        raise ValueError(f"{context} contains invalid dates")
    if result[DATE_COLUMN].duplicated().any():
        raise ValueError(f"{context} contains duplicate dates")
    result = result.sort_values(DATE_COLUMN).reset_index(drop=True)

    prices = result[[CLOSE_COLUMN, LAGGED_CLOSE_COLUMN]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"{context} contains invalid close prices")
    return result


def build_causal_observations(
    frame: pd.DataFrame,
    *,
    span: int,
) -> pd.DataFrame:
    if span < 2:
        raise ValueError("span must be at least 2")
    market = _validate_market_frame(frame, context="Market frame")
    log_return = np.log(
        market[CLOSE_COLUMN].to_numpy(dtype=float)
        / market[LAGGED_CLOSE_COLUMN].to_numpy(dtype=float)
    )
    returns = pd.Series(log_return, index=market.index, dtype=float)
    ewma_return = returns.ewm(span=span, adjust=False, min_periods=1).mean()
    ewma_second_moment = (
        returns.pow(2).ewm(span=span, adjust=False, min_periods=1).mean()
    )
    ewma_variance = (ewma_second_moment - ewma_return.pow(2)).clip(lower=0.0)
    result = market.assign(
        log_return=returns,
        ewma_return=ewma_return,
        ewma_volatility=np.sqrt(ewma_variance),
    )
    values = result[["log_return", "ewma_return", "ewma_volatility"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("Causal regime observations contain non-finite values")
    return result


def map_states_by_return(raw_state_means: np.ndarray) -> dict[int, str]:
    means = np.asarray(raw_state_means, dtype=float)
    if means.shape != (3, len(OBSERVATION_COLUMNS)):
        raise ValueError(
            "raw_state_means must have shape " f"(3, {len(OBSERVATION_COLUMNS)})"
        )
    if not np.isfinite(means).all():
        raise ValueError("State means contain non-finite values")
    order = np.argsort(means[:, 0], kind="stable")
    return {
        int(order[0]): "bear",
        int(order[1]): "sideway",
        int(order[2]): "bull",
    }


def _validate_hmm_parameters(
    *,
    observations: np.ndarray,
    start_probability: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(observations, dtype=float)
    start = np.asarray(start_probability, dtype=float)
    transition = np.asarray(transition_matrix, dtype=float)
    state_means = np.asarray(means, dtype=float)
    state_covariances = np.asarray(covariances, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("observations must be a non-empty 2D array")
    state_count = len(start)
    feature_count = values.shape[1]
    if transition.shape != (state_count, state_count):
        raise ValueError("transition_matrix has an invalid shape")
    if state_means.shape != (state_count, feature_count):
        raise ValueError("means has an invalid shape")
    if state_covariances.shape != (
        state_count,
        feature_count,
        feature_count,
    ):
        raise ValueError("covariances has an invalid shape")
    arrays = (values, start, transition, state_means, state_covariances)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("HMM inputs contain non-finite values")
    if (start < 0).any() or not np.isclose(start.sum(), 1.0):
        raise ValueError("start_probability must sum to one")
    if (transition < 0).any() or not np.allclose(
        transition.sum(axis=1),
        1.0,
    ):
        raise ValueError("transition_matrix rows must sum to one")
    return arrays


def _gaussian_log_emissions(
    observations: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    minimum_covariance: float,
) -> np.ndarray:
    sample_count, feature_count = observations.shape
    state_count = means.shape[0]
    result = np.empty((sample_count, state_count), dtype=float)
    identity = np.eye(feature_count)
    constant = feature_count * math.log(2.0 * math.pi)
    for state in range(state_count):
        covariance = (covariances[state] + covariances[state].T) / 2.0
        covariance = covariance + minimum_covariance * identity
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError(f"State {state} covariance is not positive definite")
        difference = observations - means[state]
        solved = np.linalg.solve(covariance, difference.T).T
        mahalanobis = np.einsum("ij,ij->i", difference, solved)
        result[:, state] = -0.5 * (constant + log_determinant + mahalanobis)
    return result


def filter_gaussian_hmm(
    observations: np.ndarray,
    *,
    start_probability: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    minimum_covariance: float = 1e-9,
) -> np.ndarray:
    (
        values,
        start,
        transition,
        state_means,
        state_covariances,
    ) = _validate_hmm_parameters(
        observations=observations,
        start_probability=start_probability,
        transition_matrix=transition_matrix,
        means=means,
        covariances=covariances,
    )
    emissions = _gaussian_log_emissions(
        values,
        state_means,
        state_covariances,
        minimum_covariance=minimum_covariance,
    )
    log_start = np.full_like(start, -np.inf)
    np.log(start, out=log_start, where=start > 0)
    log_transition = np.full_like(transition, -np.inf)
    np.log(
        transition,
        out=log_transition,
        where=transition > 0,
    )
    log_filtered = np.empty_like(emissions)
    first = log_start + emissions[0]
    log_filtered[0] = first - logsumexp(first)
    for index in range(1, len(values)):
        prior = logsumexp(
            log_filtered[index - 1][:, np.newaxis] + log_transition,
            axis=0,
        )
        current = prior + emissions[index]
        log_filtered[index] = current - logsumexp(current)
    probabilities = np.exp(log_filtered)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1),
        1.0,
    ):
        raise ValueError("Filtered HMM probabilities are invalid")
    return probabilities


def _parameter_count(state_count: int, feature_count: int) -> int:
    start_parameters = state_count - 1
    transition_parameters = state_count * (state_count - 1)
    mean_parameters = state_count * feature_count
    covariance_parameters = state_count * feature_count * (feature_count + 1) // 2
    return (
        start_parameters
        + transition_parameters
        + mean_parameters
        + covariance_parameters
    )


def _fit_regime_model(
    train_observations: pd.DataFrame,
    *,
    config: RegimeConfig,
) -> FittedRegimeModel:
    from hmmlearn.hmm import GaussianHMM

    locked = config.validate()
    values = train_observations.loc[:, OBSERVATION_COLUMNS].to_numpy(dtype=float)
    if len(values) < locked.state_count * 10:
        raise ValueError("Training observations are insufficient for a 3-state HMM")
    scaler = StandardScaler().fit(values)
    standardized = scaler.transform(values)

    candidates: list[tuple[float, int, GaussianHMM, float]] = []
    diagnostics: list[dict[str, object]] = []
    for seed in locked.restart_seeds:
        started = time.perf_counter()
        model = GaussianHMM(
            n_components=locked.state_count,
            covariance_type=locked.covariance_type,
            min_covar=locked.minimum_covariance,
            n_iter=locked.max_iterations,
            tol=locked.tolerance,
            random_state=int(seed),
            implementation="log",
        )
        model.fit(standardized)
        log_likelihood = float(model.score(standardized))
        elapsed = time.perf_counter() - started
        converged = bool(model.monitor_.converged)
        iterations = int(model.monitor_.iter)
        diagnostics.append(
            {
                "seed": int(seed),
                "train_log_likelihood": log_likelihood,
                "converged": converged,
                "iterations": iterations,
                "runtime_seconds": elapsed,
            }
        )
        candidates.append((log_likelihood, int(seed), model, elapsed))

    train_log_likelihood, chosen_seed, best, best_runtime = max(
        candidates,
        key=lambda item: (item[0], -item[1]),
    )
    raw_state_means = scaler.inverse_transform(best.means_)
    state_map = map_states_by_return(raw_state_means)
    best_filtered = filter_gaussian_hmm(
        standardized,
        start_probability=best.startprob_,
        transition_matrix=best.transmat_,
        means=best.means_,
        covariances=best.covars_,
        minimum_covariance=locked.minimum_covariance,
    ).argmax(axis=1)
    adjusted_rand_values: list[float] = []
    for _, _, candidate, _ in candidates:
        candidate_filtered = filter_gaussian_hmm(
            standardized,
            start_probability=candidate.startprob_,
            transition_matrix=candidate.transmat_,
            means=candidate.means_,
            covariances=candidate.covars_,
            minimum_covariance=locked.minimum_covariance,
        ).argmax(axis=1)
        adjusted_rand_values.append(
            float(adjusted_rand_score(best_filtered, candidate_filtered))
        )
    diagnostics_frame = pd.DataFrame(diagnostics)
    diagnostics_frame["delta_log_likelihood_from_best"] = (
        diagnostics_frame["train_log_likelihood"] - train_log_likelihood
    )
    diagnostics_frame["adjusted_rand_vs_selected"] = adjusted_rand_values
    parameter_count = _parameter_count(
        locked.state_count,
        len(OBSERVATION_COLUMNS),
    )
    bic = -2.0 * train_log_likelihood + parameter_count * math.log(len(values))
    return FittedRegimeModel(
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=scaler.scale_.copy(),
        start_probability=best.startprob_.copy(),
        transition_matrix=best.transmat_.copy(),
        standardized_means=best.means_.copy(),
        covariances=best.covars_.copy(),
        raw_state_means=raw_state_means,
        state_map=state_map,
        chosen_seed=chosen_seed,
        train_log_likelihood=train_log_likelihood,
        bic=float(bic),
        converged=bool(best.monitor_.converged),
        iterations=int(best.monitor_.iter),
        runtime_seconds=best_runtime,
        restart_diagnostics=diagnostics_frame,
    )


def _semantic_state_order(state_map: dict[int, str]) -> list[int]:
    reverse = {label: state for state, label in state_map.items()}
    if set(reverse) != set(SEMANTIC_REGIMES):
        raise ValueError("State map must contain Bull, Sideway, and Bear")
    return [reverse[label] for label in SEMANTIC_REGIMES]


def _semantic_probabilities(
    hidden_probabilities: np.ndarray,
    fitted: FittedRegimeModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_order = _semantic_state_order(fitted.state_map)
    filtered = hidden_probabilities[:, state_order]
    semantic_transition = fitted.transition_matrix[np.ix_(state_order, state_order)]
    next_probabilities = filtered @ semantic_transition
    return filtered, next_probabilities, semantic_transition


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.clip(probabilities, 1e-15, 1.0)
    return -np.sum(safe * np.log(safe), axis=1) / math.log(probabilities.shape[1])


def _labels_frame(
    observations: pd.DataFrame,
    filtered: np.ndarray,
    next_probabilities: np.ndarray,
    *,
    fold_name: str,
    split: str,
) -> pd.DataFrame:
    current_index = filtered.argmax(axis=1)
    next_index = next_probabilities.argmax(axis=1)
    result = observations[
        [
            DATE_COLUMN,
            CLOSE_COLUMN,
            "log_return",
            "ewma_return",
            "ewma_volatility",
        ]
    ].copy()
    result["fold"] = fold_name
    result["split"] = split
    for index, regime in enumerate(SEMANTIC_REGIMES):
        result[f"prob_{regime}_filtered"] = filtered[:, index]
    result["regime_filtered"] = np.asarray(SEMANTIC_REGIMES)[current_index]
    result["regime_confidence_filtered"] = filtered.max(axis=1)
    result["regime_entropy_filtered"] = _entropy(filtered)
    for index, regime in enumerate(SEMANTIC_REGIMES):
        result[f"prob_{regime}_next"] = next_probabilities[:, index]
    result["routing_regime"] = np.asarray(SEMANTIC_REGIMES)[next_index]
    result["routing_confidence"] = next_probabilities.max(axis=1)
    result["routing_entropy"] = _entropy(next_probabilities)
    result["routing_target"] = "next_trading_day_regime"
    return result


def _state_profiles(
    fitted: FittedRegimeModel,
    *,
    fold_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for state in range(len(fitted.start_probability)):
        self_probability = float(fitted.transition_matrix[state, state])
        expected_duration = (
            math.inf if self_probability >= 1.0 else 1.0 / (1.0 - self_probability)
        )
        rows.append(
            {
                "fold": fold_name,
                "hidden_state": state,
                "regime": fitted.state_map[state],
                "ewma_return_mean": fitted.raw_state_means[state, 0],
                "ewma_volatility_mean": fitted.raw_state_means[state, 1],
                "self_transition_probability": self_probability,
                "expected_duration_days": expected_duration,
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def _model_parameters_payload(
    fitted: FittedRegimeModel,
    *,
    config: RegimeConfig,
) -> dict[str, object]:
    return {
        "config": asdict(config),
        "scaler_mean": fitted.scaler_mean.tolist(),
        "scaler_scale": fitted.scaler_scale.tolist(),
        "start_probability": fitted.start_probability.tolist(),
        "transition_matrix_hidden_order": fitted.transition_matrix.tolist(),
        "standardized_state_means": fitted.standardized_means.tolist(),
        "state_covariances": fitted.covariances.tolist(),
        "raw_state_means": fitted.raw_state_means.tolist(),
        "hidden_state_to_regime": {
            str(key): value for key, value in fitted.state_map.items()
        },
        "chosen_seed": fitted.chosen_seed,
        "train_log_likelihood": fitted.train_log_likelihood,
        "bic": fitted.bic,
        "converged": fitted.converged,
        "iterations": fitted.iterations,
    }


def fit_fold_regimes(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    config: RegimeConfig,
    fold_name: str,
) -> FoldRegimeResult:
    locked = config.validate()
    train_market = _validate_market_frame(train, context=f"{fold_name} train")
    test_market = _validate_market_frame(test, context=f"{fold_name} test")
    if train_market[DATE_COLUMN].max() >= test_market[DATE_COLUMN].min():
        raise ValueError(f"{fold_name} train and test dates overlap")

    combined = pd.concat([train_market, test_market], ignore_index=True)
    observations = build_causal_observations(combined, span=locked.span)
    train_count = len(train_market)
    train_observations = observations.iloc[:train_count].reset_index(drop=True)
    test_observations = observations.iloc[train_count:].reset_index(drop=True)
    fitted = _fit_regime_model(train_observations, config=locked)

    standardized_all = (
        observations.loc[:, OBSERVATION_COLUMNS].to_numpy(dtype=float)
        - fitted.scaler_mean
    ) / fitted.scaler_scale
    hidden_filtered = filter_gaussian_hmm(
        standardized_all,
        start_probability=fitted.start_probability,
        transition_matrix=fitted.transition_matrix,
        means=fitted.standardized_means,
        covariances=fitted.covariances,
        minimum_covariance=locked.minimum_covariance,
    )
    filtered, next_probabilities, semantic_transition = _semantic_probabilities(
        hidden_filtered, fitted
    )
    train_labels = _labels_frame(
        train_observations,
        filtered[:train_count],
        next_probabilities[:train_count],
        fold_name=fold_name,
        split="train",
    )
    test_labels = _labels_frame(
        test_observations,
        filtered[train_count:],
        next_probabilities[train_count:],
        fold_name=fold_name,
        split="test",
    )
    restart_diagnostics = fitted.restart_diagnostics.assign(
        fold=fold_name,
        selected=lambda frame: frame["seed"].eq(fitted.chosen_seed),
    )
    train_regime_shares = (
        train_labels["routing_regime"]
        .value_counts(normalize=True)
        .reindex(SEMANTIC_REGIMES, fill_value=0.0)
    )
    state_profiles = _state_profiles(fitted, fold_name=fold_name)
    minimum_train_share = float(train_regime_shares.min())
    minimum_self_transition = float(state_profiles["self_transition_probability"].min())
    quality_gate_passed = bool(
        fitted.converged
        and minimum_train_share >= locked.minimum_training_regime_share
        and minimum_self_transition >= locked.minimum_self_transition_probability
    )
    metadata = {
        "fold": fold_name,
        "fit_scope": "fold_training_only",
        "inference": "causal_forward_filter",
        "routing_probability": "P(S_t+1 | observations_through_t)",
        "target_columns_used": False,
        "train_rows": len(train_labels),
        "test_rows": len(test_labels),
        "chosen_seed": fitted.chosen_seed,
        "train_log_likelihood": fitted.train_log_likelihood,
        "bic": fitted.bic,
        "converged": fitted.converged,
        "iterations": fitted.iterations,
        "selected_restart_runtime_seconds": fitted.runtime_seconds,
        "minimum_training_regime_share": minimum_train_share,
        "minimum_self_transition_probability": minimum_self_transition,
        "quality_gate_passed": quality_gate_passed,
    }
    return FoldRegimeResult(
        train_labels=train_labels,
        test_labels=test_labels,
        state_profiles=state_profiles,
        restart_diagnostics=restart_diagnostics,
        semantic_transition_matrix=semantic_transition,
        model_parameters=_model_parameters_payload(fitted, config=locked),
        metadata=metadata,
    )


def _observed_run_lengths(labels: pd.Series) -> dict[str, float]:
    values = labels.astype(str).reset_index(drop=True)
    groups = values.ne(values.shift()).cumsum()
    runs = (
        pd.DataFrame({"regime": values, "group": groups})
        .groupby(
            ["group", "regime"],
            sort=False,
        )
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
                "mean_log_return": selected["log_return"].mean(),
                "mean_ewma_volatility": selected["ewma_volatility"].mean(),
                "observed_mean_run_days": run_lengths[regime],
            }
        )
    return rows


def _transition_rows(
    matrix: np.ndarray,
    *,
    fold_name: str,
) -> list[dict[str, object]]:
    return [
        {
            "fold": fold_name,
            "from_regime": source,
            "to_regime": destination,
            "probability": float(matrix[source_index, destination_index]),
        }
        for source_index, source in enumerate(SEMANTIC_REGIMES)
        for destination_index, destination in enumerate(SEMANTIC_REGIMES)
    ]


def run_regime_labeling(
    *,
    source_dir: Path = FULL_TA_VMD_DATA_FOLDS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config: RegimeConfig | None = None,
) -> dict[str, object]:
    locked = (RegimeConfig() if config is None else config).validate()
    started_at = _utc_now()
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_summaries: list[dict[str, object]] = []
    distributions: list[dict[str, object]] = []
    state_profiles: list[pd.DataFrame] = []
    transitions: list[dict[str, object]] = []
    restart_diagnostics: list[pd.DataFrame] = []
    input_files: list[dict[str, object]] = []

    specs = discover_folds(source_dir)
    for spec in specs:
        train = pd.read_csv(spec.train_path)
        test = pd.read_csv(spec.test_path)
        fold_started = time.perf_counter()
        result = fit_fold_regimes(
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

        train_switches = int(
            result.train_labels["routing_regime"]
            .ne(result.train_labels["routing_regime"].shift())
            .sum()
            - 1
        )
        test_switches = int(
            result.test_labels["routing_regime"]
            .ne(result.test_labels["routing_regime"].shift())
            .sum()
            - 1
        )
        fold_summaries.append(
            {
                **result.metadata,
                "train_start": result.train_labels[DATE_COLUMN]
                .min()
                .date()
                .isoformat(),
                "train_end": result.train_labels[DATE_COLUMN].max().date().isoformat(),
                "test_start": result.test_labels[DATE_COLUMN].min().date().isoformat(),
                "test_end": result.test_labels[DATE_COLUMN].max().date().isoformat(),
                "train_routing_switches": train_switches,
                "test_routing_switches": test_switches,
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
        state_profiles.append(result.state_profiles)
        transitions.extend(
            _transition_rows(
                result.semantic_transition_matrix,
                fold_name=spec.fold,
            )
        )
        restart_diagnostics.append(result.restart_diagnostics)
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

    pd.DataFrame(fold_summaries).to_csv(
        output_dir / "fold_summary.csv",
        index=False,
    )
    pd.DataFrame(distributions).to_csv(
        output_dir / "regime_distribution.csv",
        index=False,
    )
    pd.concat(state_profiles, ignore_index=True).to_csv(
        output_dir / "state_profiles.csv",
        index=False,
    )
    pd.DataFrame(transitions).to_csv(
        output_dir / "transition_matrices.csv",
        index=False,
    )
    pd.concat(restart_diagnostics, ignore_index=True).to_csv(
        output_dir / "restart_diagnostics.csv",
        index=False,
    )

    elapsed = time.perf_counter() - started
    metadata = {
        "started_at": started_at,
        "completed_at": _utc_now(),
        "experiment": "Track C leakage-free three-state HMM regime labeling",
        "completed_folds": len(specs),
        "total_runtime_seconds": elapsed,
        "source_dir": _display_path(source_dir),
        "output_dir": _display_path(output_dir),
        "protocol": {
            **asdict(locked),
            "state_count": locked.state_count,
            "observations": list(OBSERVATION_COLUMNS),
            "return_definition": "log(Close_D / Close_D_lag1)",
            "volatility_definition": ("sqrt(EWMA(return^2) - EWMA(return)^2)"),
            "scaling": "StandardScaler fit on each fold training observations only",
            "fit_scope": "each outer fold training period only",
            "inference": "causal forward-filtered probabilities",
            "routing_target": "next_trading_day_regime",
            "routing_probability": "P(S_t+1 | observations_through_t)",
            "state_mapping": (
                "lowest training EWMA-return mean=Bear; middle=Sideway; " "highest=Bull"
            ),
            "target_columns_used": False,
            "outer_test_used_for_selection": False,
            "smoothed_or_viterbi_test_labels": False,
        },
        "input_files": input_files,
        "artifacts": {
            name: _display_path(output_dir / name)
            for name in (
                "fold_summary.csv",
                "regime_distribution.csv",
                "state_profiles.csv",
                "transition_matrices.csv",
                "restart_diagnostics.csv",
            )
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "scipy": importlib.metadata.version("scipy"),
            "hmmlearn": importlib.metadata.version("hmmlearn"),
        },
        "paper_sources": [
            {
                "citation": "Hamilton (1989)",
                "doi_or_url": (
                    "https://ideas.repec.org/a/ecm/emetrp/" "v57y1989i2p357-84.html"
                ),
            },
            {
                "citation": "Wang, Lin, and Mikhelson (2020)",
                "doi_or_url": "https://doi.org/10.3390/jrfm13120311",
            },
            {
                "citation": "Werge (2021)",
                "doi_or_url": "https://arxiv.org/abs/2107.05535",
            },
        ],
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Build leakage-free Bull/Sideway/Bear HMM labels for Track C.")
    )
    parser.add_argument("--span", type=int, default=30)
    parser.add_argument(
        "--restart-seeds",
        default="42,123,456,789,2025",
    )
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=FULL_TA_VMD_DATA_FOLDS_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    seeds = tuple(
        int(value.strip()) for value in args.restart_seeds.split(",") if value.strip()
    )
    config = RegimeConfig(
        span=args.span,
        restart_seeds=seeds,
        max_iterations=args.max_iterations,
    )
    return run_regime_labeling(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
