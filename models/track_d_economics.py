from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm


def positions_from_probabilities(
    probabilities: np.ndarray,
    *,
    threshold: float,
    strategy: str,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Probabilities must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Probabilities must be in [0, 1]")
    if not 0.5 <= threshold < 1.0:
        raise ValueError("threshold must be in [0.5, 1)")
    if strategy not in {"long_flat", "long_short"}:
        raise ValueError(f"Unknown strategy: {strategy}")
    positions = np.zeros(len(values), dtype=float)
    positions[values >= threshold] = 1.0
    if strategy == "long_short":
        positions[values < 1.0 - threshold] = -1.0
    return positions


def backtest_positions(
    positions: np.ndarray,
    implementation_returns: np.ndarray,
    *,
    cost_bps: float,
) -> pd.DataFrame:
    position = np.asarray(positions, dtype=float).reshape(-1)
    returns = np.asarray(implementation_returns, dtype=float).reshape(-1)
    if position.shape != returns.shape:
        raise ValueError("Position and return shapes differ")
    if not np.isfinite(position).all() or not np.isfinite(returns).all():
        raise ValueError("Backtest inputs must be finite")
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("cost_bps must be finite and non-negative")
    previous = np.concatenate([[0.0], position[:-1]])
    position_change = np.abs(position - previous)
    round_trip_units = np.abs(position)
    gross_return = position * returns
    trading_cost = round_trip_units * float(cost_bps) / 10_000.0
    net_return = gross_return - trading_cost
    return pd.DataFrame(
        {
            "position": position,
            "implementation_return": returns,
            "position_change": position_change,
            "round_trip_units": round_trip_units,
            "gross_return": gross_return,
            "trading_cost": trading_cost,
            "net_return": net_return,
        }
    )


def annualized_sharpe(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype=float).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all():
        return float("nan")
    deviation = float(values.std(ddof=1))
    if deviation <= np.finfo(float).eps:
        return 0.0
    return float(np.sqrt(252.0) * values.mean() / deviation)


def deflated_sharpe_ratio(
    *,
    observed_sharpe: float,
    return_count: int,
    return_skewness: float,
    return_kurtosis: float,
    trials: int,
    sharpe_std_across_trials: float,
) -> float:
    if return_count < 3 or trials < 1:
        raise ValueError("return_count and trials are too small")
    inputs = np.asarray(
        [
            observed_sharpe,
            return_skewness,
            return_kurtosis,
            sharpe_std_across_trials,
        ],
        dtype=float,
    )
    if not np.isfinite(inputs).all() or sharpe_std_across_trials < 0.0:
        raise ValueError("Deflated Sharpe inputs are invalid")
    euler_gamma = 0.5772156649015329
    expected_maximum = 0.0
    if trials > 1 and sharpe_std_across_trials > 0.0:
        expected_maximum = sharpe_std_across_trials * (
            (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / trials)
            + euler_gamma * norm.ppf(1.0 - 1.0 / (trials * math.e))
        )
    variance_term = (
        1.0
        - return_skewness * observed_sharpe
        + ((return_kurtosis - 1.0) / 4.0) * observed_sharpe**2
    ) / (return_count - 1)
    if variance_term <= 0.0:
        return 0.0
    statistic = (observed_sharpe - expected_maximum) / math.sqrt(variance_term)
    return float(np.clip(norm.cdf(statistic), 0.0, 1.0))
