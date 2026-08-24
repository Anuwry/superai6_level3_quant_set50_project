from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT_EARLY = Path(__file__).resolve().parents[1]
os.environ.setdefault("KERAS_HOME", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT_EARLY / "runtime_cache" / "mpl"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from models.baseline_common import (
    CLOSE_COLUMN,
    DATE_COLUMN,
    TARGET_COLUMN,
    discover_folds,
    package_versions,
)
from models.pit_set50_crin import (
    CONSTITUENT_FEATURES,
    MIN_ACTIVE_MEMBERS,
    MODEL_KEY,
    MODEL_LABEL,
    PROTOCOL_ID,
    WINDOW,
    build_constituent_worker,
    build_reconciliation_leader,
    build_top_only_stack,
    clipped_logit,
    direction_metrics,
)
from models.set50_constituent_data import (
    MEMBERSHIP_FILE,
    OUTPUT_DIR as PRICE_DIR,
    PRICE_COLUMNS,
    sha256_file,
    validate_membership,
)
from models.track_a_final import FINAL_SEEDS, TRACK_A_MODELS
from models.vmd_feature_pool import FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR

OUTPUT_DIR = _PROJECT_ROOT_EARLY / "outputs" / "pit_set50_crin_2024_2025_v1"
FROZEN_PREDICTIONS = (
    _PROJECT_ROOT_EARLY
    / "outputs"
    / "final_five_model_prediction_visuals_v1"
    / "final_arm_prediction_series.csv"
)
FREEZE_FILE = _PROJECT_ROOT_EARLY / "test" / "pit_set50_crin_freeze_v1.json"
TEST_YEARS = (2024, 2025)
BOTTOM_TRAIN_END_YEAR = 2020
BOTTOM_VALIDATION_YEAR = 2021
META_START_YEAR = 2022
BOTTOM_EPOCHS = 18
LEADER_EPOCHS = 35
BATCH_SIZE = 32
PATIENCE = 4
MODEL_ORDER = tuple(TRACK_A_MODELS)
VARIANTS = ("majority_vote", "top_only_stack", "bottom_only", MODEL_KEY)


@dataclass(frozen=True)
class PreparedSamples:
    dates: np.ndarray
    sequence: np.ndarray
    active_mask: np.ndarray
    direction: np.ndarray
    next_breadth: np.ndarray
    coverage: np.ndarray
    active_count: np.ndarray


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _derived_seed(seed: int, *parts: object) -> int:
    material = "|".join([PROTOCOL_ID, str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") % (2**31)


def verify_freeze() -> dict[str, Any]:
    payload = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID or not payload.get(
        "frozen_before_outer_execution"
    ):
        raise ValueError("PIT-SET50-CRIN freeze metadata is invalid")
    hashes = payload.get("input_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("PIT-SET50-CRIN freeze has no input hashes")
    checked: dict[str, str] = {}
    root = _PROJECT_ROOT_EARLY.resolve()
    for relative, expected in hashes.items():
        path = (root / str(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("A frozen input path escapes the project") from error
        if not path.is_file():
            raise FileNotFoundError(f"Frozen input is missing: {relative}")
        actual = sha256_file(path)
        if actual != str(expected):
            raise ValueError(f"Frozen input hash mismatch: {relative}")
        checked[str(relative)] = actual
    return {"passed": True, "checked_files": len(checked), "input_sha256": checked}


def _configure_tensorflow(seed: int):
    import tensorflow as tf

    random.seed(int(seed))
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    return tf


def _fold_spec(test_year: int):
    specs = {spec.test_year: spec for spec in discover_folds(FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR)}
    if test_year not in specs:
        raise ValueError(f"No registered fold for test year {test_year}")
    return specs[test_year]


def _index_frame(test_year: int) -> pd.DataFrame:
    spec = _fold_spec(test_year)
    train = pd.read_csv(spec.train_path)
    test = pd.read_csv(spec.test_path)
    frame = pd.concat([train, test], ignore_index=True)
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="raise")
    frame = frame.sort_values(DATE_COLUMN).drop_duplicates(DATE_COLUMN, keep="last")
    required = [DATE_COLUMN, CLOSE_COLUMN, TARGET_COLUMN]
    if frame[required].isna().any().any():
        raise ValueError("Index fold has missing dates, closes, or targets")
    if not frame[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError("Index fold dates are not chronological")
    return frame.loc[:, required].reset_index(drop=True)


def _load_membership() -> pd.DataFrame:
    return validate_membership(pd.read_csv(MEMBERSHIP_FILE))


def _base_symbols(membership: pd.DataFrame, test_year: int) -> set[str]:
    expected_version = {
        2024: "2024_h1",
        2025: "2025_h1_pre_symbol_change",
    }[test_year]
    symbols = set(
        membership.loc[membership["membership_version"].eq(expected_version), "symbol"]
    )
    if len(symbols) != 50:
        raise ValueError(f"Forecast-origin membership for {test_year} is not 50 symbols")
    return symbols


def _active_symbols(
    membership: pd.DataFrame,
    *,
    date: pd.Timestamp,
    test_year: int,
    base: set[str],
) -> set[str]:
    if date.year < test_year:
        return base
    rows = membership.loc[
        membership["effective_from"].le(date) & membership["effective_to"].ge(date)
    ]
    symbols = set(rows["symbol"])
    if len(symbols) != 50:
        raise ValueError(f"No unique 50-member point-in-time universe for {date.date()}")
    return symbols


def _load_price_frame(symbol: str) -> pd.DataFrame | None:
    path = PRICE_DIR / f"{symbol}.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    missing = sorted(set(PRICE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{symbol} price file is missing columns: {missing}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame = frame.sort_values("Date").drop_duplicates("Date", keep="last")
    if frame["Close"].dropna().le(0.0).any():
        raise ValueError(f"{symbol} has a non-positive close")
    return frame


def _symbol_features(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    values = frame.set_index("Date").sort_index()
    close = pd.to_numeric(values["Close"], errors="coerce")
    high = pd.to_numeric(values["High"], errors="coerce")
    low = pd.to_numeric(values["Low"], errors="coerce")
    volume = pd.to_numeric(values["Volume"], errors="coerce")
    log_close = np.log(close.where(close.gt(0.0)))
    return_1d = log_close.diff().clip(-0.25, 0.25)
    return_5d = (log_close.diff(5) / np.sqrt(5.0)).clip(-0.50, 0.50)
    intraday_range = ((high - low) / close).clip(0.0, 0.25)
    log_volume = np.log1p(volume.where(volume.ge(0.0)))
    rolling_mean = log_volume.rolling(20, min_periods=10).mean()
    rolling_std = log_volume.rolling(20, min_periods=10).std(ddof=0).replace(0.0, np.nan)
    volume_z = ((log_volume - rolling_mean) / rolling_std).clip(-5.0, 5.0)
    feature_frame = pd.DataFrame(
        {
            CONSTITUENT_FEATURES[0]: return_1d,
            CONSTITUENT_FEATURES[1]: return_5d,
            CONSTITUENT_FEATURES[2]: intraday_range,
            CONSTITUENT_FEATURES[3]: volume_z,
        }
    ).reindex(calendar)
    aligned_close = close.reindex(calendar)
    observed = aligned_close.notna()
    available = observed & observed.rolling(WINDOW, min_periods=1).sum().ge(15)
    next_return = np.log(aligned_close.shift(-1) / aligned_close)
    next_valid = next_return.notna()
    next_up = next_return.gt(0.0)
    return feature_frame.fillna(0.0), available, next_up, next_valid


def _constituent_panel(
    calendar: pd.DatetimeIndex, symbols: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    feature_values: list[np.ndarray] = []
    availability: list[np.ndarray] = []
    next_up: list[np.ndarray] = []
    next_valid: list[np.ndarray] = []
    missing_files: list[str] = []
    for symbol in symbols:
        frame = _load_price_frame(symbol)
        if frame is None:
            feature_values.append(np.zeros((len(calendar), len(CONSTITUENT_FEATURES)), dtype=np.float32))
            availability.append(np.zeros(len(calendar), dtype=bool))
            next_up.append(np.zeros(len(calendar), dtype=bool))
            next_valid.append(np.zeros(len(calendar), dtype=bool))
            missing_files.append(symbol)
            continue
        features, available, up, valid = _symbol_features(frame, calendar)
        feature_values.append(features.to_numpy(dtype=np.float32))
        availability.append(available.to_numpy(dtype=bool))
        next_up.append(up.to_numpy(dtype=bool))
        next_valid.append(valid.to_numpy(dtype=bool))
    features_array = np.stack(feature_values, axis=1)
    availability_array = np.stack(availability, axis=1)
    next_up_array = np.stack(next_up, axis=1)
    next_valid_array = np.stack(next_valid, axis=1)
    if not np.isfinite(features_array).all():
        raise ValueError("Constituent feature panel contains non-finite values")
    return (
        features_array,
        availability_array,
        next_up_array,
        next_valid_array,
        {"symbols": len(symbols), "missing_price_files": missing_files},
    )


def prepare_samples(test_year: int) -> tuple[PreparedSamples, tuple[str, ...], dict[str, Any]]:
    index = _index_frame(test_year)
    membership = _load_membership()
    symbols = tuple(sorted(membership["symbol"].unique()))
    base = _base_symbols(membership, test_year)
    calendar = pd.DatetimeIndex(index[DATE_COLUMN])
    features, available, next_up, next_valid, panel_audit = _constituent_panel(calendar, symbols)
    symbol_index = {symbol: position for position, symbol in enumerate(symbols)}
    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    directions: list[int] = []
    breadth: list[float] = []
    coverage: list[float] = []
    active_counts: list[int] = []
    dates: list[np.datetime64] = []
    for position in range(WINDOW - 1, len(index)):
        date = index.at[position, DATE_COLUMN]
        active_symbols = _active_symbols(
            membership, date=date, test_year=test_year, base=base
        )
        membership_mask = np.zeros(len(symbols), dtype=bool)
        membership_mask[[symbol_index[symbol] for symbol in active_symbols]] = True
        active = membership_mask & available[position]
        label_mask = active & next_valid[position]
        count = int(active.sum())
        label_count = int(label_mask.sum())
        if count < MIN_ACTIVE_MEMBERS:
            continue
        if date.year < test_year and label_count < MIN_ACTIVE_MEMBERS:
            continue
        current = float(index.at[position, CLOSE_COLUMN])
        target = float(index.at[position, TARGET_COLUMN])
        sequences.append(features[position - WINDOW + 1 : position + 1].transpose(1, 0, 2))
        masks.append(active.astype(np.float32))
        directions.append(int(target > current))
        breadth.append(
            float(next_up[position, label_mask].mean()) if label_count > 0 else 0.5
        )
        coverage.append(count / 50.0)
        active_counts.append(count)
        dates.append(np.datetime64(date, "ns"))
    result = PreparedSamples(
        dates=np.asarray(dates),
        sequence=np.asarray(sequences, dtype=np.float32),
        active_mask=np.asarray(masks, dtype=np.float32),
        direction=np.asarray(directions, dtype=np.float32),
        next_breadth=np.asarray(breadth, dtype=np.float32),
        coverage=np.asarray(coverage, dtype=np.float32),
        active_count=np.asarray(active_counts, dtype=np.int16),
    )
    if len(result.dates) < 500 or result.sequence.shape[1:] != (
        len(symbols),
        WINDOW,
        len(CONSTITUENT_FEATURES),
    ):
        raise ValueError("Prepared constituent samples violate the frozen shape contract")
    years = pd.DatetimeIndex(result.dates).year
    if not set((BOTTOM_VALIDATION_YEAR, META_START_YEAR, test_year)).issubset(set(years)):
        raise ValueError("Prepared samples do not cover required training and evaluation years")
    audit = {
        **panel_audit,
        "samples": len(result.dates),
        "first_date": str(pd.Timestamp(result.dates.min()).date()),
        "last_date": str(pd.Timestamp(result.dates.max()).date()),
        "coverage_min": float(result.coverage.min()),
        "coverage_mean": float(result.coverage.mean()),
        "active_count_min": int(result.active_count.min()),
        "forecast_origin_universe": sorted(base),
        "forecast_origin_universe_size": len(base),
        "membership_mode": "forecast-origin H1 backcast before test; effective-date PIT mask during test",
    }
    return result, symbols, audit


def _top_down_frame() -> pd.DataFrame:
    frame = pd.read_csv(FROZEN_PREDICTIONS)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if set(frame["model"].unique()) != set(MODEL_ORDER):
        raise ValueError("Frozen top-down model cohort is not the registered five")
    base = frame[["Date", "Close_D", "y_true"]].drop_duplicates()
    if base["Date"].duplicated().any():
        raise ValueError("Frozen predictions disagree on date-level labels")
    scores = frame.assign(score=frame["y_pred"] / frame["Close_D"] - 1.0).pivot(
        index="Date", columns="model", values="score"
    )
    scores = scores.loc[:, MODEL_ORDER]
    result = base.set_index("Date").join(scores).reset_index()
    if result[[*MODEL_ORDER, "Close_D", "y_true"]].isna().any().any():
        raise ValueError("Frozen top-down prediction panel is incomplete")
    result["direction"] = (result["y_true"] > result["Close_D"]).astype(int)
    return result.sort_values("Date").reset_index(drop=True)


def _aligned_top_inputs(samples: PreparedSamples) -> tuple[np.ndarray, np.ndarray]:
    top = _top_down_frame().set_index("Date")
    dates = pd.DatetimeIndex(samples.dates)
    aligned = top.reindex(dates)
    eligible = dates.year >= META_START_YEAR
    if aligned.loc[eligible, [*MODEL_ORDER, "direction"]].isna().any().any():
        missing = dates[eligible & aligned["direction"].isna()].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Top-down predictions do not align with constituent samples: {missing[:5]}")
    direction = np.zeros(len(samples.dates), dtype=np.float32)
    direction[eligible] = aligned.loc[eligible, "direction"].to_numpy(dtype=np.float32)
    if not np.array_equal(direction[eligible], samples.direction[eligible]):
        raise ValueError("Index direction labels disagree after alignment")
    scores = np.zeros((len(samples.dates), len(MODEL_ORDER)), dtype=np.float32)
    scores[eligible] = aligned.loc[eligible, MODEL_ORDER].to_numpy(dtype=np.float32)
    return scores, direction


def _split_indices(samples: PreparedSamples, test_year: int) -> dict[str, np.ndarray]:
    years = pd.DatetimeIndex(samples.dates).year.to_numpy()
    splits = {
        "bottom_train": np.flatnonzero(years <= BOTTOM_TRAIN_END_YEAR),
        "bottom_validation": np.flatnonzero(years == BOTTOM_VALIDATION_YEAR),
        "meta_train": np.flatnonzero((years >= META_START_YEAR) & (years <= test_year - 2)),
        "meta_validation": np.flatnonzero(years == test_year - 1),
        "test": np.flatnonzero(years == test_year),
    }
    if any(len(indices) < 100 for indices in splits.values()):
        raise ValueError(f"A temporal split is too small: { {key: len(value) for key, value in splits.items()} }")
    ordered = [splits[key] for key in ("bottom_train", "bottom_validation", "meta_train", "meta_validation", "test")]
    for left, right in zip(ordered, ordered[1:]):
        if int(left.max()) >= int(right.min()):
            raise ValueError("Temporal training/validation/test splits overlap")
    return splits


def _early_stopping():
    import tensorflow as tf

    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        min_delta=1e-4,
        restore_best_weights=True,
    )


def _worker_inputs(samples: PreparedSamples, indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "constituent_sequence": samples.sequence[indices],
        "active_member_mask": samples.active_mask[indices],
    }


def _fit_bottom_worker(
    samples: PreparedSamples,
    symbols: tuple[str, ...],
    splits: dict[str, np.ndarray],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tf = _configure_tensorflow(seed)
    tf.keras.backend.clear_session()
    _configure_tensorflow(seed)
    model, diagnostics = build_constituent_worker(asset_count=len(symbols))
    train_indices = splits["bottom_train"]
    validation_indices = splits["bottom_validation"]
    started = time.perf_counter()
    history = model.fit(
        _worker_inputs(samples, train_indices),
        {
            "direction": samples.direction[train_indices],
            "next_breadth": samples.next_breadth[train_indices],
        },
        validation_data=(
            _worker_inputs(samples, validation_indices),
            {
                "direction": samples.direction[validation_indices],
                "next_breadth": samples.next_breadth[validation_indices],
            },
        ),
        epochs=BOTTOM_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[_early_stopping()],
        verbose=0,
    )
    fit_seconds = time.perf_counter() - started
    prediction_indices = np.concatenate(
        [splits["meta_train"], splits["meta_validation"], splits["test"]]
    )
    started = time.perf_counter()
    output = diagnostics(_worker_inputs(samples, prediction_indices), training=False)
    inference_seconds = time.perf_counter() - started
    probability = np.asarray(output["direction"]).reshape(-1)
    attention = np.asarray(output["attention"])
    if not np.isfinite(probability).all() or attention.shape != (
        len(prediction_indices),
        len(symbols),
    ):
        raise ValueError("Bottom worker produced invalid predictions or attention")
    metadata = {
        "fit_seconds": float(fit_seconds),
        "inference_seconds": float(inference_seconds),
        "trainable_parameters": int(model.count_params()),
        "epochs_ran": len(history.history["loss"]),
        "final_training_loss": float(history.history["loss"][-1]),
        "final_validation_loss": float(history.history["val_loss"][-1]),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "prediction_samples": len(prediction_indices),
        "prediction_indices": prediction_indices.tolist(),
    }
    return probability, attention, metadata


def _standardize_top(
    top_scores: np.ndarray, fit_indices: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    fit = np.asarray(top_scores[fit_indices], dtype=np.float64)
    mean = fit.mean(axis=0)
    std = fit.std(axis=0, ddof=0)
    std = np.where(std > np.finfo(float).eps, std, 1.0)
    scaled = np.clip((top_scores - mean) / std, -10.0, 10.0).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("Standardized top-down scores are non-finite")
    return scaled, {
        "fit_scope": "meta_train_only",
        "mean": dict(zip(MODEL_ORDER, mean.tolist())),
        "std": dict(zip(MODEL_ORDER, std.tolist())),
    }


def _leader_inputs(
    scaled_top: np.ndarray,
    bottom_probability: np.ndarray,
    coverage: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    selected_top = scaled_top[indices]
    bottom = bottom_probability[indices]
    context = np.column_stack(
        [coverage[indices], np.std(selected_top, axis=1)]
    ).astype(np.float32)
    return {
        "top_down_scores": selected_top,
        "bottom_up_logit": clipped_logit(bottom).astype(np.float32).reshape(-1, 1),
        "reconciliation_context": context,
    }


def _fit_leaders(
    samples: PreparedSamples,
    top_scores: np.ndarray,
    bottom_probability: np.ndarray,
    splits: dict[str, np.ndarray],
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    tf = _configure_tensorflow(seed)
    scaled_top, scaler = _standardize_top(top_scores, splits["meta_train"])
    train_indices = splits["meta_train"]
    validation_indices = splits["meta_validation"]
    test_indices = splits["test"]
    tf.keras.backend.clear_session()
    _configure_tensorflow(seed)
    leader, diagnostics = build_reconciliation_leader(top_experts=len(MODEL_ORDER))
    started = time.perf_counter()
    history = leader.fit(
        _leader_inputs(scaled_top, bottom_probability, samples.coverage, train_indices),
        samples.direction[train_indices],
        validation_data=(
            _leader_inputs(scaled_top, bottom_probability, samples.coverage, validation_indices),
            samples.direction[validation_indices],
        ),
        epochs=LEADER_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[_early_stopping()],
        verbose=0,
    )
    leader_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    full_output = diagnostics(
        _leader_inputs(scaled_top, bottom_probability, samples.coverage, test_indices),
        training=False,
    )
    leader_inference_seconds = time.perf_counter() - started

    tf.keras.backend.clear_session()
    _configure_tensorflow(seed + 1)
    top_only = build_top_only_stack(top_experts=len(MODEL_ORDER))
    started = time.perf_counter()
    top_history = top_only.fit(
        scaled_top[train_indices],
        samples.direction[train_indices],
        validation_data=(scaled_top[validation_indices], samples.direction[validation_indices]),
        epochs=LEADER_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[_early_stopping()],
        verbose=0,
    )
    top_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    top_probability = np.asarray(top_only(scaled_top[test_indices], training=False)).reshape(-1)
    top_inference_seconds = time.perf_counter() - started
    full_probability = np.asarray(full_output["probability"]).reshape(-1)
    probability = {
        MODEL_KEY: full_probability,
        "top_only_stack": top_probability,
        "bottom_only": bottom_probability[test_indices],
        "majority_vote": np.mean(top_scores[test_indices] > 0.0, axis=1),
    }
    diagnostics_values = {
        "top_down_gate": np.asarray(full_output["top_down_gate"]).reshape(-1),
        "top_down_logit": np.asarray(full_output["top_down_logit"]).reshape(-1),
        "correction": np.asarray(full_output["correction"]).reshape(-1),
    }
    for name, values in {**probability, **diagnostics_values}.items():
        if len(values) != len(test_indices) or not np.isfinite(values).all():
            raise ValueError(f"Leader output {name} is invalid")
    metadata = {
        "scaler": scaler,
        "leader_fit_seconds": float(leader_fit_seconds),
        "leader_inference_seconds": float(leader_inference_seconds),
        "leader_trainable_parameters": int(leader.count_params()),
        "leader_epochs_ran": len(history.history["loss"]),
        "leader_final_training_loss": float(history.history["loss"][-1]),
        "leader_final_validation_loss": float(history.history["val_loss"][-1]),
        "top_only_fit_seconds": float(top_fit_seconds),
        "top_only_inference_seconds": float(top_inference_seconds),
        "top_only_trainable_parameters": int(top_only.count_params()),
        "top_only_epochs_ran": len(top_history.history["loss"]),
        "meta_train_samples": len(train_indices),
        "meta_validation_samples": len(validation_indices),
        "test_samples": len(test_indices),
    }
    return probability, diagnostics_values, metadata


def _cell_dir(output_dir: Path, test_year: int, seed: int) -> Path:
    return output_dir / "cells" / str(test_year) / f"seed_{seed}"


def cell_complete(output_dir: Path, test_year: int, seed: int) -> bool:
    directory = _cell_dir(output_dir, test_year, seed)
    return all((directory / name).is_file() for name in ("predictions.csv", "runtime.json", "attention_by_symbol.csv"))


def run_cell(
    *,
    test_year: int,
    seed: int,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> Path:
    if test_year not in TEST_YEARS or seed not in FINAL_SEEDS:
        raise ValueError("Cell is outside the registered 2024-2025 / five-seed grid")
    freeze_audit = verify_freeze()
    directory = _cell_dir(output_dir, test_year, seed)
    if cell_complete(output_dir, test_year, seed) and not force:
        return directory
    started_total = time.perf_counter()
    samples, symbols, data_audit = prepare_samples(test_year)
    top_scores, _ = _aligned_top_inputs(samples)
    splits = _split_indices(samples, test_year)
    bottom_seed = _derived_seed(seed, test_year, "bottom")
    bottom_compact, attention_compact, bottom_metadata = _fit_bottom_worker(
        samples, symbols, splits, seed=bottom_seed
    )
    compact_indices = np.asarray(bottom_metadata.pop("prediction_indices"), dtype=int)
    bottom_probability = np.full(len(samples.dates), np.nan, dtype=np.float32)
    bottom_probability[compact_indices] = bottom_compact
    if np.isnan(bottom_probability[np.concatenate([splits["meta_train"], splits["meta_validation"], splits["test"]])]).any():
        raise ValueError("Bottom predictions do not cover the reconciliation period")
    leader_seed = _derived_seed(seed, test_year, "leader")
    probability, leader_diagnostics, leader_metadata = _fit_leaders(
        samples,
        top_scores,
        bottom_probability,
        splits,
        seed=leader_seed,
    )
    test_indices = splits["test"]
    prediction_rows: list[pd.DataFrame] = []
    for variant in VARIANTS:
        prediction_rows.append(
            pd.DataFrame(
                {
                    "protocol_id": PROTOCOL_ID,
                    "model": MODEL_KEY,
                    "variant": variant,
                    "test_year": test_year,
                    "seed": seed,
                    "Date": pd.DatetimeIndex(samples.dates[test_indices]).strftime("%Y-%m-%d"),
                    "y_true": samples.direction[test_indices].astype(int),
                    "probability": probability[variant],
                    "predicted_direction": (probability[variant] >= 0.5).astype(int),
                    "active_members": samples.active_count[test_indices],
                    "coverage": samples.coverage[test_indices],
                }
            )
        )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    test_attention_start = int(np.flatnonzero(np.isin(compact_indices, test_indices))[0])
    test_attention = attention_compact[test_attention_start : test_attention_start + len(test_indices)]
    active = samples.active_mask[test_indices]
    active_attention = np.where(active > 0.0, test_attention, np.nan)
    attention_frame = pd.DataFrame(
        {
            "symbol": symbols,
            "mean_attention_when_active": np.nanmean(active_attention, axis=0),
            "active_test_days": np.sum(active > 0.0, axis=0).astype(int),
        }
    ).sort_values(["mean_attention_when_active", "symbol"], ascending=[False, True])
    leader_frame = pd.DataFrame(
        {
            "Date": pd.DatetimeIndex(samples.dates[test_indices]).strftime("%Y-%m-%d"),
            **leader_diagnostics,
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(directory / "predictions.csv", index=False)
    attention_frame.to_csv(directory / "attention_by_symbol.csv", index=False)
    leader_frame.to_csv(directory / "leader_diagnostics.csv", index=False)
    runtime = {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": _utc_now(),
        "test_year": test_year,
        "seed": seed,
        "bottom_seed": bottom_seed,
        "leader_seed": leader_seed,
        "total_seconds": time.perf_counter() - started_total,
        "bottom": bottom_metadata,
        "leader": leader_metadata,
        "data_audit": data_audit,
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "freeze_audit": freeze_audit,
    }
    (directory / "runtime.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    gc.collect()
    return directory


def _aggregate_predictions(output_dir: Path) -> pd.DataFrame:
    paths = sorted((output_dir / "cells").glob("*/seed_*/predictions.csv"))
    if len(paths) != len(TEST_YEARS) * len(FINAL_SEEDS):
        raise ValueError("Cannot aggregate an incomplete registered grid")
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    keys = ["model", "variant", "test_year", "Date"]
    invariants = raw.groupby(keys).agg(
        label_values=("y_true", "nunique"),
        coverage_values=("coverage", "nunique"),
        seeds=("seed", "nunique"),
    )
    if (
        invariants["label_values"].ne(1).any()
        or invariants["coverage_values"].ne(1).any()
        or invariants["seeds"].ne(len(FINAL_SEEDS)).any()
    ):
        raise ValueError("Seed predictions violate aggregation invariants")
    averaged = (
        raw.groupby(keys, as_index=False)
        .agg(
            y_true=("y_true", "first"),
            probability=("probability", "mean"),
            probability_sd=("probability", "std"),
            active_members=("active_members", "first"),
            coverage=("coverage", "first"),
            seeds_averaged=("seed", "nunique"),
        )
        .sort_values(["variant", "test_year", "Date"])
    )
    return averaged


def _variant_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, test_year), group in predictions.groupby(["variant", "test_year"], sort=False):
        metric = direction_metrics(group["y_true"], group["probability"])
        rows.append({"model": MODEL_KEY, "variant": variant, "test_year": test_year, **asdict(metric)})
    return pd.DataFrame(rows)


def _frozen_model_metrics(common_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(FROZEN_PREDICTIONS)
    frame = frame.loc[frame["test_year"].isin(TEST_YEARS)].copy()
    common = common_predictions[["test_year", "Date"]].drop_duplicates()
    common["Date"] = common["Date"].astype(str)
    frame["Date"] = frame["Date"].astype(str)
    frame = frame.merge(common, on=["test_year", "Date"], how="inner", validate="many_to_one")
    frame["probability"] = (frame["y_pred"] > frame["Close_D"]).astype(float)
    frame["direction"] = (frame["y_true"] > frame["Close_D"]).astype(int)
    rows = []
    for (model, test_year), group in frame.groupby(["model", "test_year"], sort=False):
        metric = direction_metrics(group["direction"], group["probability"])
        rows.append({"model": model, "test_year": test_year, **asdict(metric)})
    return pd.DataFrame(rows), frame


def _mean_fold_summary(metrics: pd.DataFrame, *, label_column: str) -> pd.DataFrame:
    return (
        metrics.groupby(label_column, as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            direction_accuracy_mean=("direction_accuracy", "mean"),
            direction_accuracy_sd=("direction_accuracy", "std"),
            mcc_mean=("mcc", "mean"),
            predicted_up_share_mean=("predicted_up_share", "mean"),
            temporal_folds=("test_year", "nunique"),
            observations=("observations", "sum"),
        )
    )


def _runtime_summary(output_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((output_dir / "cells").glob("*/seed_*/runtime.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "test_year": payload["test_year"],
                "seed": payload["seed"],
                "total_seconds": payload["total_seconds"],
                "bottom_fit_seconds": payload["bottom"]["fit_seconds"],
                "leader_fit_seconds": payload["leader"]["leader_fit_seconds"],
                "top_only_fit_seconds": payload["leader"]["top_only_fit_seconds"],
                "bottom_parameters": payload["bottom"]["trainable_parameters"],
                "leader_parameters": payload["leader"]["leader_trainable_parameters"],
            }
        )
    return pd.DataFrame(rows)


def aggregate(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    verify_freeze()
    averaged = _aggregate_predictions(output_dir)
    variant_metrics = _variant_metrics(averaged)
    frozen_metrics, frozen_predictions = _frozen_model_metrics(averaged)
    frozen_summary = _mean_fold_summary(frozen_metrics, label_column="model")
    full_metrics = variant_metrics.loc[variant_metrics["variant"].eq(MODEL_KEY)]
    ours_summary = _mean_fold_summary(full_metrics, label_column="model")
    six = pd.concat([frozen_summary, ours_summary], ignore_index=True)
    labels = {**{key: value.label for key, value in TRACK_A_MODELS.items()}, MODEL_KEY: MODEL_LABEL}
    six.insert(1, "label", six["model"].map(labels))
    six = six.sort_values("balanced_accuracy_mean", ascending=False).reset_index(drop=True)
    six.insert(0, "rank", np.arange(1, len(six) + 1))
    ablation = _mean_fold_summary(variant_metrics, label_column="variant").sort_values(
        "balanced_accuracy_mean", ascending=False
    )
    runtime = _runtime_summary(output_dir)
    averaged.to_csv(output_dir / "predictions_seed_averaged.csv", index=False)
    variant_metrics.to_csv(output_dir / "variant_metrics_by_year.csv", index=False)
    frozen_metrics.to_csv(output_dir / "frozen_five_metrics_2024_2025.csv", index=False)
    six.to_csv(output_dir / "six_model_comparison_2024_2025.csv", index=False)
    ablation.to_csv(output_dir / "crin_ablation_2024_2025.csv", index=False)
    runtime.to_csv(output_dir / "runtime_by_cell.csv", index=False)
    runtime_summary = pd.DataFrame(
        [
            {
                "cells": len(runtime),
                "total_runtime_seconds": runtime["total_seconds"].sum(),
                "cell_runtime_seconds_mean": runtime["total_seconds"].mean(),
                "bottom_fit_seconds_total": runtime["bottom_fit_seconds"].sum(),
                "leader_fit_seconds_total": runtime["leader_fit_seconds"].sum(),
                "top_only_fit_seconds_total": runtime["top_only_fit_seconds"].sum(),
                "bottom_parameters": int(runtime["bottom_parameters"].iloc[0]),
                "leader_parameters": int(runtime["leader_parameters"].iloc[0]),
            }
        ]
    )
    runtime_summary.to_csv(output_dir / "runtime_summary.csv", index=False)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": _utc_now(),
        "status": "provisional_internal_extension",
        "test_years": list(TEST_YEARS),
        "seeds": list(FINAL_SEEDS),
        "model_order": list(MODEL_ORDER),
        "variants": list(VARIANTS),
        "window": WINDOW,
        "constituent_features": list(CONSTITUENT_FEATURES),
        "minimum_active_members": MIN_ACTIVE_MEMBERS,
        "bottom_training_end_year": BOTTOM_TRAIN_END_YEAR,
        "bottom_validation_year": BOTTOM_VALIDATION_YEAR,
        "meta_start_year": META_START_YEAR,
        "raw_market_data_distribution": "prohibited",
        "paper_readiness_blocker": (
            "Replace provisional Yahoo constituent rows with institution-authorized data and rerun."
        ),
        "input_sha256": {
            str(MEMBERSHIP_FILE.relative_to(_PROJECT_ROOT_EARLY)): sha256_file(MEMBERSHIP_FILE),
            str(FROZEN_PREDICTIONS.relative_to(_PROJECT_ROOT_EARLY)): sha256_file(FROZEN_PREDICTIONS),
            str((PRICE_DIR / "manifest.json").relative_to(_PROJECT_ROOT_EARLY)): sha256_file(PRICE_DIR / "manifest.json"),
        },
        "packages": package_versions(
            ["tensorflow", "keras", "numpy", "pandas", "scikit-learn"]
        ),
        "platform": platform.platform(),
        "outputs": [
            "six_model_comparison_2024_2025.csv",
            "crin_ablation_2024_2025.csv",
            "variant_metrics_by_year.csv",
            "predictions_seed_averaged.csv",
            "runtime_summary.csv",
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "six_model_comparison": six,
        "ablation": ablation,
        "runtime": runtime_summary,
        "frozen_prediction_rows": len(frozen_predictions),
    }


def run_registered_grid(
    *,
    output_dir: Path = OUTPUT_DIR,
    test_years: tuple[int, ...] = TEST_YEARS,
    seeds: tuple[int, ...] = FINAL_SEEDS,
    force: bool = False,
) -> dict[str, Any] | None:
    total = len(test_years) * len(seeds)
    index = 0
    for test_year in test_years:
        for seed in seeds:
            index += 1
            if cell_complete(output_dir, test_year, seed) and not force:
                print(f"[{index}/{total}] skip complete year={test_year} seed={seed}", flush=True)
                continue
            print(f"[{index}/{total}] run year={test_year} seed={seed}", flush=True)
            run_cell(
                test_year=test_year,
                seed=seed,
                output_dir=output_dir,
                force=force,
            )
    if test_years == TEST_YEARS and seeds == FINAL_SEEDS:
        return aggregate(output_dir)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PIT-SET50-CRIN on 2024-2025")
    subparsers = parser.add_subparsers(dest="command", required=True)
    cell = subparsers.add_parser("cell")
    cell.add_argument("--test-year", type=int, choices=TEST_YEARS, required=True)
    cell.add_argument("--seed", type=int, choices=FINAL_SEEDS, required=True)
    cell.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    cell.add_argument("--force", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--test-year", action="append", type=int, choices=TEST_YEARS)
    run.add_argument("--seed", action="append", type=int, choices=FINAL_SEEDS)
    run.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    run.add_argument("--force", action="store_true")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.command == "cell":
        print(run_cell(test_year=args.test_year, seed=args.seed, output_dir=args.output_dir, force=args.force))
    elif args.command == "aggregate":
        result = aggregate(args.output_dir)
        print(result["six_model_comparison"].to_string(index=False))
    else:
        years = TEST_YEARS if args.test_year is None else tuple(args.test_year)
        seeds = FINAL_SEEDS if args.seed is None else tuple(args.seed)
        result = run_registered_grid(
            output_dir=args.output_dir,
            test_years=years,
            seeds=seeds,
            force=args.force,
        )
        if result is not None:
            print(result["six_model_comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
