from __future__ import annotations

import numpy as np
import pandas as pd


def scale_frame_with_metadata(
    frame: pd.DataFrame,
    metadata: dict[str, object],
) -> pd.DataFrame:
    required_metadata = {"columns", "scale", "min"}
    missing_metadata = sorted(required_metadata.difference(metadata))
    if missing_metadata:
        raise ValueError(f"Scaler metadata missing keys: {missing_metadata}")
    columns = list(metadata["columns"])
    missing_columns = sorted(set(columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Frame missing scaler columns: {missing_columns}")
    scale = np.asarray(metadata["scale"], dtype=np.float64)
    offset = np.asarray(metadata["min"], dtype=np.float64)
    if scale.shape != (len(columns),) or offset.shape != (len(columns),):
        raise ValueError("Scaler metadata shapes do not match columns")
    if not np.isfinite(scale).all() or not np.isfinite(offset).all():
        raise ValueError("Scaler metadata must be finite")
    values = frame.loc[:, columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Frame contains non-finite scaler inputs")
    result = frame.loc[:, list(frame.columns)].copy()
    result.loc[:, columns] = values * scale + offset
    return result


def endpoint_regime_mask(
    labels: np.ndarray,
    *,
    regime: str,
    window: int,
) -> np.ndarray:
    values = np.asarray(labels, dtype=object)
    if values.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if window < 1 or len(values) < window:
        raise ValueError("labels do not contain enough rows for the window")
    return values[window - 1 :] == regime


def top_features_for_regime(
    ranking: pd.DataFrame,
    *,
    regime: str,
    top_k: int,
) -> tuple[str, ...]:
    required = {"regime", "feature", "consensus_rank"}
    missing = sorted(required.difference(ranking.columns))
    if missing:
        raise ValueError(f"Consensus ranking missing columns: {missing}")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    selected = ranking.loc[ranking["regime"].eq(regime)].sort_values(
        ["consensus_rank", "feature"]
    )
    if len(selected) < top_k:
        raise ValueError(
            f"{regime} has {len(selected)} features; requested {top_k}"
        )
    features = tuple(selected.head(top_k)["feature"].astype(str))
    if len(set(features)) != top_k:
        raise ValueError("Consensus ranking contains duplicate features")
    return features
