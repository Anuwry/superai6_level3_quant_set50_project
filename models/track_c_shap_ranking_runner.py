from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_common import (
    CLOSE_COLUMN,
    DATE_COLUMN,
    PROJECT_ROOT,
    TARGET_COLUMN,
    discover_folds,
    load_fold,
    package_versions,
    sequence_history_features,
)
from models.convolutional_neural_network import (
    make_sequences,
    make_test_sequences,
)
from models.neural_network_folds import SCALER_METADATA_NAME
from models.shap_protocol_v2 import (
    BACKGROUND_CAP,
    EXPLANATION_CAP,
    MODEL_BUILDERS,
    NSAMPLES,
    RANDOM_SEED,
    build_original_change_model,
    evenly_spaced_indices,
    normalize_single_output_shap,
)
from models.track_a_data import (
    FULL_TA_VMD_POINT_IN_TIME_SELECTION_DIR,
    FULL_TA_VMD_POINT_IN_TIME_SELECTION_NN_DIR,
)
from models.track_a_final import (
    FIRST_OUTER_TEST_YEAR,
    TRACK_A_MODELS,
    load_locked_windows,
)
from models.track_c_daily_regime import (
    DailyRegimeConfig,
    fit_fold_daily_regimes,
)
from models.track_c_shap_ranking import (
    compute_absolute_spearman,
    selection_fold_triplets,
    validate_ranking_sequence_counts,
)
from models.track_c_shap_selection import (
    aggregate_shap_importance,
    build_consensus_ranking,
    derive_protocol_seed,
    normalized_descending_ranks,
    purge_ranking_endpoints,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_c" / "shap_selection_v2"
CELL_DIR = OUTPUT_DIR / "cells"
MIN_TRAIN_REGIME_SEQUENCES = 200
MIN_RANKING_SEQUENCES = 40
TOP_K_GRID = (10, 20, 30, 40, 60, 80, 100, 122)
REGIMES = ("global", "bull", "sideway", "bear")
PROTOCOL_VERSION = "track-c-shap-point-in-time-v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_scaler(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scale_value(
    metadata: dict[str, object],
    column: str,
) -> tuple[float, float]:
    columns = list(metadata["columns"])
    index = columns.index(column)
    return (
        float(list(metadata["scale"])[index]),
        float(list(metadata["min"])[index]),
    )


def _model_window_map() -> dict[str, int]:
    locked = load_locked_windows()
    return {
        str(row.model): int(row.selected_sequence_window)
        for row in locked.itertuples(index=False)
    }


def _validate_models(model_keys: Iterable[str]) -> tuple[str, ...]:
    keys = tuple(model_keys)
    if not keys:
        raise ValueError("At least one model is required")
    unknown = sorted(set(keys).difference(TRACK_A_MODELS))
    if unknown:
        raise ValueError(f"Unknown Track C models: {unknown}")
    return keys


def _fit_global_model(
    model_key: str,
    *,
    train_features: np.ndarray,
    train_target: np.ndarray,
    window: int,
    seed: int,
):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    x_train, y_train = make_sequences(
        train_features,
        train_target,
        window,
    )
    builder = MODEL_BUILDERS[model_key]
    model = builder((window, train_features.shape[1]))
    parameters = TRACK_A_MODELS[model_key].parameters
    model.fit(
        x_train,
        y_train,
        epochs=int(parameters["epochs"]),
        batch_size=int(parameters["batch_size"]),
        shuffle=False,
        verbose=0,
    )
    return model, x_train, y_train


def _regime_endpoint_counts(
    labels: np.ndarray,
    *,
    window: int,
) -> dict[str, int]:
    endpoints = np.asarray(labels, dtype=object)[window - 1 :]
    return {
        regime: int(np.sum(endpoints == regime))
        for regime in REGIMES[1:]
    }


def _cell_path(model_key: str, fold_name: str, regime: str) -> Path:
    return CELL_DIR / model_key / fold_name / f"{regime}_importance.csv"


def _metadata_path(model_key: str, fold_name: str) -> Path:
    return CELL_DIR / model_key / fold_name / "model_fold_metadata.json"


def _cell_is_complete(model_key: str, fold_name: str) -> bool:
    return all(
        _cell_path(model_key, fold_name, regime).is_file()
        for regime in REGIMES
    ) and _metadata_path(model_key, fold_name).is_file()


def _explain_cell(
    explainer,
    sequences: np.ndarray,
    *,
    model_key: str,
    fold_name: str,
    regime: str,
) -> tuple[np.ndarray, dict[str, object]]:
    import time as runtime_time

    available = validate_ranking_sequence_counts(
        len(sequences),
        required=MIN_RANKING_SEQUENCES,
        cell=f"{fold_name}/{model_key}/{regime}",
    )
    selected_indices = evenly_spaced_indices(
        available,
        EXPLANATION_CAP,
    )
    explained = np.asarray(
        sequences[selected_indices],
        dtype=np.float32,
    )
    seed = derive_protocol_seed(
        RANDOM_SEED,
        model_key,
        fold_name,
        regime,
    )
    started = runtime_time.perf_counter()
    raw_values = explainer.shap_values(
        explained,
        nsamples=NSAMPLES,
        rseed=seed,
    )
    runtime_seconds = runtime_time.perf_counter() - started
    values = normalize_single_output_shap(
        raw_values,
        explained.shape,
    )
    importance = aggregate_shap_importance(values)
    return importance, {
        "available_sequences": available,
        "explained_sequences": len(explained),
        "rseed": seed,
        "nsamples": NSAMPLES,
        "runtime_seconds": float(runtime_seconds),
        "finite": True,
    }


def _write_importance(
    *,
    model_key: str,
    fold_name: str,
    regime: str,
    features: Sequence[str],
    importance: np.ndarray,
    metadata: dict[str, object],
) -> None:
    output_path = _cell_path(model_key, fold_name, regime)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "model": model_key,
            "selection_fold": fold_name,
            "regime": regime,
            "feature": list(features),
            "importance": np.asarray(importance, dtype=float),
            "normalized_rank": normalized_descending_ranks(importance),
            "available_sequences": metadata["available_sequences"],
            "explained_sequences": metadata["explained_sequences"],
            "rseed": metadata["rseed"],
            "nsamples": metadata["nsamples"],
            "runtime_seconds": metadata["runtime_seconds"],
            "fallback": metadata.get("fallback", "none"),
        }
    )
    frame.to_csv(output_path, index=False)


def _run_model_fold(
    model_key: str,
    *,
    scaled_spec,
    original_spec,
    validation_spec,
    window: int,
) -> None:
    if scaled_spec.fold != original_spec.fold:
        raise ValueError("Scaled and original selection folds differ")
    if _cell_is_complete(model_key, scaled_spec.fold):
        return

    import shap

    scaled = load_fold(scaled_spec)
    original = load_fold(original_spec)
    validation = pd.read_csv(validation_spec.test_path)
    first_validation_date = pd.to_datetime(
        validation[DATE_COLUMN],
        errors="raise",
    ).min()
    if original.feature_columns != scaled.feature_columns:
        raise ValueError("Scaled and original feature pools differ")
    if len(scaled.feature_columns) != TOP_K_GRID[-1]:
        raise ValueError(
            f"Expected 122 numerical features; found {len(scaled.feature_columns)}"
        )
    scaled_dates = pd.to_datetime(scaled.test[DATE_COLUMN]).reset_index(drop=True)
    original_dates = pd.to_datetime(original.test[DATE_COLUMN]).reset_index(
        drop=True
    )
    if not scaled_dates.equals(original_dates):
        raise ValueError("Scaled and original ranking dates differ")

    eligible = purge_ranking_endpoints(
        original.test,
        first_validation_date=first_validation_date,
    )
    regimes = fit_fold_daily_regimes(
        original.train,
        original.test,
        config=DailyRegimeConfig(),
        fold_name=original.spec.fold,
    )
    routing = regimes.test_labels["routing_regime"].to_numpy(dtype=object)
    if len(routing) != len(eligible):
        raise ValueError("Regime labels do not align with ranking endpoints")

    train_features = scaled.train.loc[
        :,
        scaled.feature_columns,
    ].to_numpy(dtype=float)
    train_target = scaled.train[TARGET_COLUMN].to_numpy(dtype=float)
    model_started = time.perf_counter()
    level_model, train_sequences, _ = _fit_global_model(
        model_key,
        train_features=train_features,
        train_target=train_target,
        window=window,
        seed=RANDOM_SEED,
    )
    model_fit_seconds = time.perf_counter() - model_started
    rank_sequences = make_test_sequences(
        sequence_history_features(scaled),
        scaled.test.loc[:, scaled.feature_columns].to_numpy(dtype=float),
        window,
    )
    eligible_sequences = rank_sequences[eligible]
    eligible_routing = routing[eligible]

    scaler_path = (
        scaled_spec.train_path.parent / SCALER_METADATA_NAME
    )
    scaler = _load_scaler(scaler_path)
    close_scale, close_offset = _scale_value(scaler, CLOSE_COLUMN)
    target_scale, target_offset = _scale_value(scaler, TARGET_COLUMN)
    change_model = build_original_change_model(
        level_model,
        close_feature_index=scaled.feature_columns.index(CLOSE_COLUMN),
        close_scale=close_scale,
        close_offset=close_offset,
        target_scale=target_scale,
        target_offset=target_offset,
    )
    background_indices = evenly_spaced_indices(
        len(train_sequences),
        BACKGROUND_CAP,
    )
    background = np.asarray(
        train_sequences[background_indices],
        dtype=np.float32,
    )
    explainer = shap.GradientExplainer(change_model, background)

    importance_rows: dict[str, dict[str, object]] = {}
    global_importance, global_metadata = _explain_cell(
        explainer,
        eligible_sequences,
        model_key=model_key,
        fold_name=scaled_spec.fold,
        regime="global",
    )
    _write_importance(
        model_key=model_key,
        fold_name=scaled_spec.fold,
        regime="global",
        features=scaled.feature_columns,
        importance=global_importance,
        metadata=global_metadata,
    )
    importance_rows["global"] = global_metadata

    training_regime_counts = _regime_endpoint_counts(
        regimes.train_labels["routing_regime"].to_numpy(dtype=object),
        window=window,
    )
    for regime in REGIMES[1:]:
        selected = eligible_sequences[eligible_routing == regime]
        fallback_reason = "none"
        if training_regime_counts[regime] < MIN_TRAIN_REGIME_SEQUENCES:
            fallback_reason = (
                "global_importance_due_to_training_regime_count_"
                f"{training_regime_counts[regime]}"
            )
        elif len(selected) < MIN_RANKING_SEQUENCES:
            fallback_reason = (
                "global_importance_due_to_ranking_regime_count_"
                f"{len(selected)}"
            )
        if fallback_reason != "none":
            metadata = {
                **global_metadata,
                "available_sequences": len(selected),
                "explained_sequences": 0,
                "runtime_seconds": 0.0,
                "fallback": fallback_reason,
            }
            importance = global_importance.copy()
        else:
            importance, metadata = _explain_cell(
                explainer,
                selected,
                model_key=model_key,
                fold_name=scaled_spec.fold,
                regime=regime,
            )
            metadata["fallback"] = "none"
        _write_importance(
            model_key=model_key,
            fold_name=scaled_spec.fold,
            regime=regime,
            features=scaled.feature_columns,
            importance=importance,
            metadata=metadata,
        )
        importance_rows[regime] = metadata

    metadata_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": _utc_now(),
        "model": model_key,
        "selection_fold": scaled_spec.fold,
        "training_years": [
            scaled_spec.train_start_year,
            scaled_spec.train_end_year,
        ],
        "ranking_year": scaled_spec.test_year,
        "validation_year": validation_spec.test_year,
        "window": window,
        "seed": RANDOM_SEED,
        "features": len(scaled.feature_columns),
        "training_sequences": len(train_sequences),
        "training_regime_sequences": training_regime_counts,
        "ranking_rows_before_label_purge": len(original.test),
        "ranking_rows_after_label_purge": int(np.sum(eligible)),
        "ranking_boundary_purged_rows": int(np.sum(~eligible)),
        "background_sequences": len(background),
        "model_fit_seconds": float(model_fit_seconds),
        "cells": importance_rows,
        "input_files": {
            "scaled_train": str(scaled_spec.train_path),
            "scaled_rank": str(scaled_spec.test_path),
            "original_train": str(original_spec.train_path),
            "original_rank": str(original_spec.test_path),
            "validation": str(validation_spec.test_path),
            "scaler": str(scaler_path),
        },
    }
    metadata_path = _metadata_path(model_key, scaled_spec.fold)
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2),
        encoding="utf-8",
    )


def _importance_records() -> pd.DataFrame:
    paths = sorted(CELL_DIR.glob("*/*/*_importance.csv"))
    if not paths:
        raise FileNotFoundError("No SHAP importance cells were generated")
    return pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )


def _spearman_records() -> pd.DataFrame:
    original_specs = discover_folds(
        FULL_TA_VMD_POINT_IN_TIME_SELECTION_DIR
    )
    triplets = selection_fold_triplets(
        original_specs,
        first_outer_year=FIRST_OUTER_TEST_YEAR,
    )
    rows: list[dict[str, object]] = []
    for triplet in triplets:
        spec = triplet.training_rank_spec
        fold = load_fold(spec)
        validation = pd.read_csv(triplet.validation_spec.test_path)
        first_validation_date = pd.to_datetime(
            validation[DATE_COLUMN],
            errors="raise",
        ).min()
        eligible = purge_ranking_endpoints(
            fold.test,
            first_validation_date=first_validation_date,
        )
        regime_result = fit_fold_daily_regimes(
            fold.train,
            fold.test,
            config=DailyRegimeConfig(),
            fold_name=spec.fold,
        )
        routing = regime_result.test_labels[
            "routing_regime"
        ].to_numpy(dtype=object)
        next_return = (
            fold.test[TARGET_COLUMN].to_numpy(dtype=float)
            / fold.test[CLOSE_COLUMN].to_numpy(dtype=float)
            - 1.0
        )
        for regime in REGIMES:
            mask = eligible.copy()
            if regime != "global":
                mask &= routing == regime
            if int(mask.sum()) < MIN_RANKING_SEQUENCES:
                mask = eligible.copy()
                fallback = "global_due_to_minimum_ranking_count"
            else:
                fallback = "none"
            scores = compute_absolute_spearman(
                fold.test.loc[mask, fold.feature_columns],
                next_return[mask],
            )
            ranks = normalized_descending_ranks(
                scores.to_numpy(dtype=float)
            )
            for feature, score, rank in zip(
                scores.index,
                scores.to_numpy(dtype=float),
                ranks,
                strict=True,
            ):
                rows.append(
                    {
                        "model": "spearman_filter",
                        "selection_fold": spec.fold,
                        "regime": regime,
                        "feature": feature,
                        "importance": score,
                        "normalized_rank": rank,
                        "available_sequences": int(mask.sum()),
                        "fallback": fallback,
                    }
                )
    return pd.DataFrame(rows)


def _fold_consensus(records: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold_name, selected in records.groupby("selection_fold", sort=True):
        consensus = build_consensus_ranking(selected)
        consensus["selection_fold"] = fold_name
        frames.append(consensus)
    return pd.concat(frames, ignore_index=True)


def _rank_stability(fold_consensus: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for regime, regime_frame in fold_consensus.groupby("regime", sort=False):
        feature_sets: dict[tuple[str, int], set[str]] = {}
        for fold_name, fold_frame in regime_frame.groupby(
            "selection_fold",
            sort=True,
        ):
            ordered = fold_frame.sort_values("consensus_rank")
            for top_k in TOP_K_GRID:
                feature_sets[(str(fold_name), top_k)] = set(
                    ordered.head(top_k)["feature"].astype(str)
                )
        folds = sorted(regime_frame["selection_fold"].unique())
        for left, right in combinations(folds, 2):
            for top_k in TOP_K_GRID:
                left_set = feature_sets[(str(left), top_k)]
                right_set = feature_sets[(str(right), top_k)]
                rows.append(
                    {
                        "regime": regime,
                        "left_fold": left,
                        "right_fold": right,
                        "top_k": top_k,
                        "jaccard": (
                            len(left_set & right_set)
                            / len(left_set | right_set)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _write_protocol() -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "selection_seed": RANDOM_SEED,
        "background_cap": BACKGROUND_CAP,
        "explanation_cap": EXPLANATION_CAP,
        "minimum_training_regime_sequences": MIN_TRAIN_REGIME_SEQUENCES,
        "minimum_ranking_sequences": MIN_RANKING_SEQUENCES,
        "shap_nsamples": NSAMPLES,
        "top_k_grid": list(TOP_K_GRID),
        "attribution_target": (
            "predicted next-close minus current close in original units"
        ),
        "consensus": "mean normalized rank across models and temporal folds",
        "raw_magnitude_pooled_across_models": False,
        "outer_data_used": False,
        "lime_used_for_selection": False,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "protocol.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def run_shap_rankings(
    *,
    model_keys: Iterable[str] = TRACK_A_MODELS,
    force: bool = False,
) -> dict[str, object]:
    import tensorflow as tf

    keys = _validate_models(model_keys)
    tf.config.experimental.enable_op_determinism()
    started_at = _utc_now()
    started = time.perf_counter()
    _write_protocol()
    scaled_specs = discover_folds(
        FULL_TA_VMD_POINT_IN_TIME_SELECTION_NN_DIR
    )
    original_specs = discover_folds(
        FULL_TA_VMD_POINT_IN_TIME_SELECTION_DIR
    )
    scaled_triplets = selection_fold_triplets(
        scaled_specs,
        first_outer_year=FIRST_OUTER_TEST_YEAR,
    )
    original_triplets = selection_fold_triplets(
        original_specs,
        first_outer_year=FIRST_OUTER_TEST_YEAR,
    )
    windows = _model_window_map()

    for model_key in keys:
        for scaled_triplet, original_triplet in zip(
            scaled_triplets,
            original_triplets,
            strict=True,
        ):
            fold_name = scaled_triplet.training_rank_spec.fold
            if force:
                for regime in REGIMES:
                    _cell_path(model_key, fold_name, regime).unlink(
                        missing_ok=True
                    )
                _metadata_path(model_key, fold_name).unlink(missing_ok=True)
            _run_model_fold(
                model_key,
                scaled_spec=scaled_triplet.training_rank_spec,
                original_spec=original_triplet.training_rank_spec,
                validation_spec=original_triplet.validation_spec,
                window=windows[model_key],
            )

    records = _importance_records()
    expected_cells = len(TRACK_A_MODELS) * len(scaled_triplets) * len(REGIMES)
    observed_cells = records[
        ["model", "selection_fold", "regime"]
    ].drop_duplicates()
    if set(records["model"]) == set(TRACK_A_MODELS) and len(
        observed_cells
    ) != expected_cells:
        raise ValueError(
            f"Expected {expected_cells} SHAP cells; found {len(observed_cells)}"
        )
    records.to_csv(OUTPUT_DIR / "importance_by_cell.csv", index=False)
    consensus = build_consensus_ranking(records)
    consensus.to_csv(OUTPUT_DIR / "consensus_ranking.csv", index=False)
    fold_consensus = _fold_consensus(records)
    fold_consensus.to_csv(
        OUTPUT_DIR / "fold_consensus_ranking.csv",
        index=False,
    )
    stability = _rank_stability(fold_consensus)
    stability.to_csv(OUTPUT_DIR / "rank_stability.csv", index=False)

    spearman = _spearman_records()
    spearman.to_csv(
        OUTPUT_DIR / "spearman_importance_by_fold.csv",
        index=False,
    )
    spearman_consensus = build_consensus_ranking(spearman)
    spearman_consensus.to_csv(
        OUTPUT_DIR / "spearman_consensus_ranking.csv",
        index=False,
    )
    cell_runtime = (
        records[
            [
                "model",
                "selection_fold",
                "regime",
                "runtime_seconds",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    cell_runtime.to_csv(OUTPUT_DIR / "runtime_by_cell.csv", index=False)
    fallbacks = (
        records[
            [
                "model",
                "selection_fold",
                "regime",
                "available_sequences",
                "explained_sequences",
                "fallback",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    fallbacks.to_csv(OUTPUT_DIR / "fallback_audit.csv", index=False)
    pd.DataFrame(
        columns=["created_at", "reason", "affected_outputs", "classification"]
    ).to_csv(OUTPUT_DIR / "deviation_log.csv", index=False)

    input_paths = sorted(
        {
            spec.train_path
            for spec in (*scaled_specs, *original_specs)
        }
        | {
            spec.test_path
            for spec in (*scaled_specs, *original_specs)
        },
        key=str,
    )
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "ranking_generated": True,
        "outer_results_generated": False,
        "outer_results_viewed": False,
        "lime_outer_explanations_generated": False,
        "models_completed": sorted(records["model"].unique()),
        "selection_folds": sorted(records["selection_fold"].unique()),
        "regimes": list(REGIMES),
        "feature_count": int(records["feature"].nunique()),
        "shap_cells": len(observed_cells),
        "fallback_cells": int(
            fallbacks["fallback"].ne("none").sum()
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "packages": package_versions(
            [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "tensorflow",
                "shap",
            ]
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "input_files": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in input_paths
        ],
        "artifacts": [
            "protocol.json",
            "importance_by_cell.csv",
            "consensus_ranking.csv",
            "fold_consensus_ranking.csv",
            "rank_stability.csv",
            "spearman_importance_by_fold.csv",
            "spearman_consensus_ranking.csv",
            "runtime_by_cell.csv",
            "fallback_audit.csv",
            "deviation_log.csv",
        ],
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run registered Track C consensus SHAP rankings."
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(TRACK_A_MODELS),
        dest="models",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    return run_shap_rankings(
        model_keys=(TRACK_A_MODELS if args.models is None else args.models),
        force=args.force,
    )


if __name__ == "__main__":
    main()
