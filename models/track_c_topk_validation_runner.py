from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_common import (
    CLOSE_COLUMN,
    DATE_COLUMN,
    PROJECT_ROOT,
    TARGET_COLUMN,
    binary_direction_metrics,
    discover_folds,
    load_fold,
    package_versions,
    regression_metrics,
)
from models.convolutional_neural_network import (
    make_sequences,
    make_test_sequences,
)
from models.neural_network_folds import (
    SCALER_METADATA_NAME,
    inverse_scaled_target,
)
from models.shap_protocol_v2 import MODEL_BUILDERS, RANDOM_SEED
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
from models.track_c_shap_ranking import selection_fold_triplets
from models.track_c_shap_ranking_runner import (
    MIN_TRAIN_REGIME_SEQUENCES,
    REGIMES,
    TOP_K_GRID,
)
from models.track_c_shap_ranking_runner import (
    OUTPUT_DIR as RANKING_OUTPUT_DIR,
)
from models.track_c_shap_selection import select_registered_top_k
from models.track_c_topk_validation import (
    endpoint_regime_mask,
    scale_frame_with_metadata,
    top_features_for_regime,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_c" / "topk_validation_v2"
CELL_DIR = OUTPUT_DIR / "cells"
PROTOCOL_VERSION = "track-c-shap-point-in-time-v2"
SELECTED_TOP_K_FILE = OUTPUT_DIR / "selected_top_k.json"
SELECTED_FEATURES_FILE = OUTPUT_DIR / "selected_features.csv"
SELECTION_FREEZE_FILE = OUTPUT_DIR / "selection_frozen.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_models(model_keys: Iterable[str]) -> tuple[str, ...]:
    keys = tuple(model_keys)
    if not keys:
        raise ValueError("At least one model is required")
    unknown = sorted(set(keys).difference(TRACK_A_MODELS))
    if unknown:
        raise ValueError(f"Unknown Track C models: {unknown}")
    return keys


def _window_map() -> dict[str, int]:
    return {
        str(row.model): int(row.selected_sequence_window)
        for row in load_locked_windows().itertuples(index=False)
    }


def _source_order(
    selected: Sequence[str],
    source_columns: Sequence[str],
) -> tuple[str, ...]:
    chosen = set(selected)
    ordered = tuple(feature for feature in source_columns if feature in chosen)
    if len(ordered) != len(selected):
        raise ValueError("Selected features are missing from the source pool")
    return ordered


def _fit_model(
    model_key: str,
    *,
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
    input_shape: tuple[int, int],
    seed: int,
):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = MODEL_BUILDERS[model_key](input_shape)
    parameters = TRACK_A_MODELS[model_key].parameters
    model.fit(
        train_sequences,
        train_targets,
        epochs=int(parameters["epochs"]),
        batch_size=int(parameters["batch_size"]),
        shuffle=False,
        verbose=0,
    )
    return model


def _prediction_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_close: np.ndarray,
) -> dict[str, float | int]:
    return {
        **regression_metrics(y_true, y_pred),
        **binary_direction_metrics(y_true, y_pred, current_close),
    }


def _cell_dir(model_key: str, fold_name: str, top_k: int) -> Path:
    return CELL_DIR / model_key / fold_name / f"top_k_{top_k}"


def _cell_complete(model_key: str, fold_name: str, top_k: int) -> bool:
    directory = _cell_dir(model_key, fold_name, top_k)
    return all(
        (directory / name).is_file()
        for name in (
            "metrics.csv",
            "global_predictions.csv",
            "regime_predictions.csv",
            "run_metadata.json",
        )
    )


def _prepare_validation_fold(
    scaled_spec,
    original_spec,
    validation_spec,
) -> dict[str, object]:
    scaled = load_fold(scaled_spec)
    original = load_fold(original_spec)
    validation_original = pd.read_csv(validation_spec.test_path)
    validation_original[DATE_COLUMN] = pd.to_datetime(
        validation_original[DATE_COLUMN],
        errors="raise",
    )
    validation_original = validation_original.sort_values(
        DATE_COLUMN
    ).reset_index(drop=True)
    scaler_path = scaled_spec.train_path.parent / SCALER_METADATA_NAME
    scaler = _load_json(scaler_path)
    validation_scaled = scale_frame_with_metadata(
        validation_original,
        scaler,
    )
    rank_scaled = scaled.test.reset_index(drop=True)
    rank_original = original.test.reset_index(drop=True)
    rank_dates = pd.to_datetime(rank_scaled[DATE_COLUMN]).reset_index(drop=True)
    if not rank_dates.equals(
        pd.to_datetime(rank_original[DATE_COLUMN]).reset_index(drop=True)
    ):
        raise ValueError("Scaled and original ranking dates differ")

    regime_evaluation = pd.concat(
        [rank_original, validation_original],
        ignore_index=True,
    )
    regimes = fit_fold_daily_regimes(
        original.train,
        regime_evaluation,
        config=DailyRegimeConfig(),
        fold_name=original.spec.fold,
    )
    validation_regimes = regimes.test_labels.iloc[
        len(rank_original) :
    ]["routing_regime"].to_numpy(dtype=object)
    if len(validation_regimes) != len(validation_original):
        raise ValueError("Validation regimes do not align with validation rows")
    return {
        "scaled": scaled,
        "original": original,
        "validation_original": validation_original,
        "validation_scaled": validation_scaled,
        "validation_regimes": validation_regimes,
        "train_regimes": regimes.train_labels[
            "routing_regime"
        ].to_numpy(dtype=object),
        "scaler": scaler,
        "scaler_path": scaler_path,
    }


def _train_and_predict(
    model_key: str,
    *,
    feature_columns: Sequence[str],
    train_frame: pd.DataFrame,
    rank_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    window: int,
    endpoint_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    train_features = train_frame.loc[
        :,
        list(feature_columns),
    ].to_numpy(dtype=float)
    train_target = train_frame[TARGET_COLUMN].to_numpy(dtype=float)
    x_train, y_train = make_sequences(
        train_features,
        train_target,
        window,
    )
    if endpoint_mask is not None:
        if endpoint_mask.shape != (len(x_train),):
            raise ValueError("Training regime mask does not align with sequences")
        x_fit = x_train[endpoint_mask]
        y_fit = y_train[endpoint_mask]
    else:
        x_fit = x_train
        y_fit = y_train
    if len(x_fit) < MIN_TRAIN_REGIME_SEQUENCES:
        raise ValueError(
            f"Training subset has {len(x_fit)} sequences; "
            f"{MIN_TRAIN_REGIME_SEQUENCES} required"
        )
    validation_sequences = make_test_sequences(
        rank_frame.loc[:, list(feature_columns)].to_numpy(dtype=float),
        validation_frame.loc[:, list(feature_columns)].to_numpy(dtype=float),
        window,
    )
    started = time.perf_counter()
    model = _fit_model(
        model_key,
        train_sequences=x_fit,
        train_targets=y_fit,
        input_shape=(window, len(feature_columns)),
        seed=RANDOM_SEED,
    )
    fit_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    prediction = model.predict(
        validation_sequences,
        verbose=0,
    ).reshape(-1)
    inference_seconds = time.perf_counter() - inference_started
    if prediction.shape != (len(validation_frame),):
        raise ValueError("Validation prediction shape is invalid")
    if not np.isfinite(prediction).all():
        raise ValueError("Validation prediction contains non-finite values")
    return prediction, {
        "training_sequences": len(x_fit),
        "validation_sequences": len(validation_sequences),
        "fit_seconds": float(fit_seconds),
        "inference_seconds": float(inference_seconds),
    }


def _run_validation_cell(
    model_key: str,
    *,
    scaled_spec,
    original_spec,
    validation_spec,
    window: int,
    top_k: int,
    consensus: pd.DataFrame,
    force: bool,
) -> None:
    output_dir = _cell_dir(model_key, scaled_spec.fold, top_k)
    if _cell_complete(model_key, scaled_spec.fold, top_k) and not force:
        return
    prepared = _prepare_validation_fold(
        scaled_spec,
        original_spec,
        validation_spec,
    )
    scaled = prepared["scaled"]
    validation_original = prepared["validation_original"]
    validation_scaled = prepared["validation_scaled"]
    validation_regimes = np.asarray(
        prepared["validation_regimes"],
        dtype=object,
    )
    train_regimes = np.asarray(prepared["train_regimes"], dtype=object)
    scaler = prepared["scaler"]
    feature_pool = tuple(scaled.feature_columns)
    y_true = validation_original[TARGET_COLUMN].to_numpy(dtype=float)
    current_close = validation_original[CLOSE_COLUMN].to_numpy(dtype=float)

    global_ranked = top_features_for_regime(
        consensus,
        regime="global",
        top_k=top_k,
    )
    global_features = _source_order(global_ranked, feature_pool)
    global_scaled_prediction, global_runtime = _train_and_predict(
        model_key,
        feature_columns=global_features,
        train_frame=scaled.train,
        rank_frame=scaled.test,
        validation_frame=validation_scaled,
        window=window,
    )
    global_prediction = inverse_scaled_target(
        global_scaled_prediction,
        scaler,
    )
    metrics_rows: list[dict[str, object]] = [
        {
            "model": model_key,
            "selection_fold": scaled_spec.fold,
            "training_end_year": scaled_spec.train_end_year,
            "ranking_year": scaled_spec.test_year,
            "validation_year": validation_spec.test_year,
            "scope": "global",
            "regime": "global",
            "top_k": top_k,
            "window": window,
            "n_train_sequences": global_runtime["training_sequences"],
            "n_validation": len(y_true),
            **_prediction_metrics(
                y_true=y_true,
                y_pred=global_prediction,
                current_close=current_close,
            ),
            **global_runtime,
        }
    ]
    global_predictions = pd.DataFrame(
        {
            DATE_COLUMN: validation_original[DATE_COLUMN],
            "regime": validation_regimes,
            "Close_D": current_close,
            "y_true": y_true,
            "y_pred": global_prediction,
        }
    )

    regime_prediction = np.full(len(validation_original), np.nan, dtype=float)
    regime_runtime_rows: list[dict[str, object]] = []
    for regime in REGIMES[1:]:
        ranked = top_features_for_regime(
            consensus,
            regime=regime,
            top_k=top_k,
        )
        features = _source_order(ranked, feature_pool)
        endpoint_mask = endpoint_regime_mask(
            train_regimes,
            regime=regime,
            window=window,
        )
        scaled_prediction, runtime = _train_and_predict(
            model_key,
            feature_columns=features,
            train_frame=scaled.train,
            rank_frame=scaled.test,
            validation_frame=validation_scaled,
            window=window,
            endpoint_mask=endpoint_mask,
        )
        prediction = inverse_scaled_target(scaled_prediction, scaler)
        evaluation_mask = validation_regimes == regime
        if not np.any(evaluation_mask):
            raise ValueError(
                f"{scaled_spec.fold} validation contains no {regime} rows"
            )
        regime_prediction[evaluation_mask] = prediction[evaluation_mask]
        metrics_rows.append(
            {
                "model": model_key,
                "selection_fold": scaled_spec.fold,
                "training_end_year": scaled_spec.train_end_year,
                "ranking_year": scaled_spec.test_year,
                "validation_year": validation_spec.test_year,
                "scope": "regime",
                "regime": regime,
                "top_k": top_k,
                "window": window,
                "n_train_sequences": runtime["training_sequences"],
                "n_validation": int(evaluation_mask.sum()),
                **_prediction_metrics(
                    y_true=y_true[evaluation_mask],
                    y_pred=prediction[evaluation_mask],
                    current_close=current_close[evaluation_mask],
                ),
                **runtime,
            }
        )
        regime_runtime_rows.append({"regime": regime, **runtime})
    if not np.isfinite(regime_prediction).all():
        raise ValueError("Regime validation routing left missing predictions")
    metrics_rows.append(
        {
            "model": model_key,
            "selection_fold": scaled_spec.fold,
            "training_end_year": scaled_spec.train_end_year,
            "ranking_year": scaled_spec.test_year,
            "validation_year": validation_spec.test_year,
            "scope": "regime_combined",
            "regime": "combined",
            "top_k": top_k,
            "window": window,
            "n_train_sequences": int(
                sum(row["training_sequences"] for row in regime_runtime_rows)
            ),
            "n_validation": len(y_true),
            **_prediction_metrics(
                y_true=y_true,
                y_pred=regime_prediction,
                current_close=current_close,
            ),
            "fit_seconds": float(
                sum(row["fit_seconds"] for row in regime_runtime_rows)
            ),
            "inference_seconds": float(
                sum(row["inference_seconds"] for row in regime_runtime_rows)
            ),
            "validation_sequences": len(y_true),
        }
    )
    regime_predictions = pd.DataFrame(
        {
            DATE_COLUMN: validation_original[DATE_COLUMN],
            "regime": validation_regimes,
            "Close_D": current_close,
            "y_true": y_true,
            "y_pred": regime_prediction,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )
    global_predictions.to_csv(
        output_dir / "global_predictions.csv",
        index=False,
    )
    regime_predictions.to_csv(
        output_dir / "regime_predictions.csv",
        index=False,
    )
    metadata = {
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
        "top_k": top_k,
        "window": window,
        "seed": RANDOM_SEED,
        "global_features": list(global_features),
        "regime_features": {
            regime: list(
                _source_order(
                    top_features_for_regime(
                        consensus,
                        regime=regime,
                        top_k=top_k,
                    ),
                    feature_pool,
                )
            )
            for regime in REGIMES[1:]
        },
        "feature_order": "original source-column order",
        "scaler_fit_scope": "training_only",
        "regime_threshold_fit_scope": "training_only",
        "outer_data_used": False,
        "lime_used_for_selection": False,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _collect_metrics() -> pd.DataFrame:
    paths = sorted(CELL_DIR.glob("*/*/top_k_*/metrics.csv"))
    if not paths:
        raise FileNotFoundError("No top-k validation metrics found")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _median_stability(regime: str) -> pd.Series:
    stability = pd.read_csv(RANKING_OUTPUT_DIR / "rank_stability.csv")
    selected = stability.loc[stability["regime"].eq(regime)]
    return selected.groupby("top_k")["jaccard"].median()


def _select_and_freeze(metrics: pd.DataFrame) -> dict[str, int]:
    selections: dict[str, int] = {}
    audits: list[pd.DataFrame] = []
    for regime in REGIMES:
        if regime == "global":
            selected_metrics = metrics.loc[
                metrics["scope"].eq("global")
            ].copy()
        else:
            selected_metrics = metrics.loc[
                metrics["scope"].eq("regime")
                & metrics["regime"].eq(regime)
            ].copy()
        selected_k, audit = select_registered_top_k(
            selected_metrics,
            _median_stability(regime),
        )
        selections[regime] = selected_k
        audit.insert(0, "regime", regime)
        audits.append(audit)
    gate_audit = pd.concat(audits, ignore_index=True)
    gate_audit.to_csv(
        OUTPUT_DIR / "top_k_gate_audit.csv",
        index=False,
    )

    consensus = pd.read_csv(RANKING_OUTPUT_DIR / "consensus_ranking.csv")
    spearman = pd.read_csv(
        RANKING_OUTPUT_DIR / "spearman_consensus_ranking.csv"
    )
    feature_rows: list[dict[str, object]] = []
    for selector, ranking in (
        ("shap", consensus),
        ("spearman", spearman),
    ):
        for regime, top_k in selections.items():
            selected = ranking.loc[ranking["regime"].eq(regime)].sort_values(
                ["consensus_rank", "feature"]
            ).head(top_k)
            if len(selected) != top_k:
                raise ValueError(
                    f"{selector}/{regime} does not contain top_k features"
                )
            for row in selected.itertuples(index=False):
                feature_rows.append(
                    {
                        "selector": selector,
                        "regime": regime,
                        "selected_top_k": top_k,
                        "feature": str(row.feature),
                        "selector_rank": int(row.consensus_rank),
                        "consensus_normalized_rank": float(
                            row.consensus_normalized_rank
                        ),
                    }
                )
    features = pd.DataFrame(feature_rows)
    features.to_csv(SELECTED_FEATURES_FILE, index=False)
    SELECTED_TOP_K_FILE.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "selected_at": _utc_now(),
                "rule": (
                    "smallest k within one SE plus paired BA, model, RMSE, "
                    "and temporal Jaccard guardrails"
                ),
                "selected_top_k": selections,
                "spearman_k_matched_to_shap": True,
                "outer_data_used": False,
                "lime_used_for_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    freeze_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "frozen_at": _utc_now(),
        "selection_complete": True,
        "outer_execution_authorized": True,
        "outer_results_generated": False,
        "lime_outer_explanations_generated": False,
        "selected_top_k_sha256": _sha256(SELECTED_TOP_K_FILE),
        "selected_features_sha256": _sha256(SELECTED_FEATURES_FILE),
        "top_k_gate_audit_sha256": _sha256(
            OUTPUT_DIR / "top_k_gate_audit.csv"
        ),
    }
    SELECTION_FREEZE_FILE.write_text(
        json.dumps(freeze_payload, indent=2),
        encoding="utf-8",
    )
    return selections


def run_topk_validation(
    *,
    model_keys: Iterable[str] = TRACK_A_MODELS,
    force: bool = False,
) -> dict[str, object]:
    import tensorflow as tf

    keys = _validate_models(model_keys)
    ranking_metadata = _load_json(
        RANKING_OUTPUT_DIR / "run_metadata.json"
    )
    if not ranking_metadata.get("ranking_generated"):
        raise ValueError("SHAP ranking is not complete")
    if set(ranking_metadata["models_completed"]) != set(TRACK_A_MODELS):
        raise ValueError("SHAP ranking does not cover all five models")
    tf.config.experimental.enable_op_determinism()
    started_at = _utc_now()
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    consensus = pd.read_csv(RANKING_OUTPUT_DIR / "consensus_ranking.csv")
    scaled_triplets = selection_fold_triplets(
        discover_folds(FULL_TA_VMD_POINT_IN_TIME_SELECTION_NN_DIR),
        first_outer_year=FIRST_OUTER_TEST_YEAR,
    )
    original_triplets = selection_fold_triplets(
        discover_folds(FULL_TA_VMD_POINT_IN_TIME_SELECTION_DIR),
        first_outer_year=FIRST_OUTER_TEST_YEAR,
    )
    windows = _window_map()
    for model_key in keys:
        for scaled_triplet, original_triplet in zip(
            scaled_triplets,
            original_triplets,
            strict=True,
        ):
            for top_k in TOP_K_GRID:
                _run_validation_cell(
                    model_key,
                    scaled_spec=scaled_triplet.training_rank_spec,
                    original_spec=original_triplet.training_rank_spec,
                    validation_spec=original_triplet.validation_spec,
                    window=windows[model_key],
                    top_k=top_k,
                    consensus=consensus,
                    force=force,
                )
    metrics = _collect_metrics()
    metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    expected_cells = len(TRACK_A_MODELS) * len(scaled_triplets) * len(TOP_K_GRID)
    observed_cells = metrics[
        ["model", "selection_fold", "top_k"]
    ].drop_duplicates()
    full_run = set(metrics["model"]) == set(TRACK_A_MODELS)
    if full_run and len(observed_cells) != expected_cells:
        raise ValueError(
            f"Expected {expected_cells} validation cells; "
            f"found {len(observed_cells)}"
        )
    selections = _select_and_freeze(metrics) if full_run else {}
    runtime = (
        metrics.groupby(
            ["model", "selection_fold", "scope", "regime", "top_k"],
            sort=False,
        )[["fit_seconds", "inference_seconds"]]
        .sum()
        .reset_index()
    )
    runtime.to_csv(OUTPUT_DIR / "runtime_by_cell.csv", index=False)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "models_completed": sorted(metrics["model"].unique()),
        "selection_folds": sorted(metrics["selection_fold"].unique()),
        "top_k_grid": list(TOP_K_GRID),
        "validation_cells": len(observed_cells),
        "selected_top_k": selections,
        "selection_frozen": bool(full_run),
        "outer_data_used": False,
        "lime_used_for_selection": False,
        "ranking_inspected_before_validation_execution": True,
        "ranking_inspection_effect": (
            "none; grid, gate rule, selectors, and model windows were frozen"
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "packages": package_versions(
            [
                "numpy",
                "pandas",
                "scikit-learn",
                "tensorflow",
            ]
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "artifacts": [
            "validation_metrics.csv",
            "runtime_by_cell.csv",
            "top_k_gate_audit.csv",
            "selected_top_k.json",
            "selected_features.csv",
            "selection_frozen.json",
        ],
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen Track C top-k temporal validation."
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
    return run_topk_validation(
        model_keys=(TRACK_A_MODELS if args.models is None else args.models),
        force=args.force,
    )


if __name__ == "__main__":
    main()
