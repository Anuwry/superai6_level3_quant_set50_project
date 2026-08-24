from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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
    sequence_history_features,
)
from models.convolutional_neural_network import (
    make_sequences,
    make_test_sequences,
)
from models.neural_network_folds import (
    SCALER_METADATA_NAME,
    inverse_scaled_target,
)
from models.shap_protocol_v2 import MODEL_BUILDERS
from models.track_a_final import (
    FINAL_SEEDS,
    TRACK_A_MODELS,
    load_locked_windows,
)
from models.track_c_inference import (
    CONTRASTS,
    apply_holm_by_family,
    average_seed_predictions,
    build_paired_fold_contrasts,
    fold_level_inference,
    moving_block_bootstrap,
    paired_daily_effects,
)
from models.track_c_outer import (
    OUTER_ARMS,
    REGIMES,
    capacity_matched_subseeds,
    route_regime_predictions,
    selected_feature_lookup,
)
from models.track_c_topk_validation import endpoint_regime_mask
from models.track_c_topk_validation_runner import (
    SELECTED_FEATURES_FILE,
    SELECTION_FREEZE_FILE,
)
from models.vmd_feature_pool import (
    FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
    FULL_TA_VMD_POINT_IN_TIME_NN_DATA_FOLDS_DIR,
)

REGIME_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "track_c" / "daily_regime_point_in_time_v2"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_c" / "outer_v2"
CELL_DIR = OUTPUT_DIR / "cells"
PROTOCOL_VERSION = "track-c-shap-point-in-time-v2"
INFERENCE_PROTOCOL_VERSION = "track-c-outer-inference-v1"
INFERENCE_ADDENDUM = (
    PROJECT_ROOT / "test" / "track_c_outer_inference_addendum.md"
)
BOOTSTRAP_BLOCK_LENGTH = 10
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_731


@dataclass(frozen=True)
class FitRequest:
    scope: str
    regime: str
    features: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class FitResult:
    prediction: np.ndarray
    training_sequences: int
    fit_seconds: float
    inference_seconds: float
    trainable_parameters: int
    fit_id: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise ValueError("Selected features are missing from the outer pool")
    return ordered


def _fit_id(request: FitRequest) -> str:
    material = "|".join(
        [
            request.scope,
            request.regime,
            str(request.seed),
            *request.features,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _load_regimes(fold_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_dir = REGIME_OUTPUT_DIR / fold_name
    train = pd.read_csv(fold_dir / "train_regimes.csv")
    test = pd.read_csv(fold_dir / "test_regimes.csv")
    for frame in (train, test):
        frame[DATE_COLUMN] = pd.to_datetime(
            frame[DATE_COLUMN],
            errors="raise",
        )
    return train, test


def _validate_regime_alignment(
    market: pd.DataFrame,
    regime_frame: pd.DataFrame,
    *,
    split: str,
) -> np.ndarray:
    market_dates = pd.to_datetime(market[DATE_COLUMN]).reset_index(drop=True)
    regime_dates = pd.to_datetime(
        regime_frame[DATE_COLUMN]
    ).reset_index(drop=True)
    if not market_dates.equals(regime_dates):
        raise ValueError(f"{split} market and regime dates do not align")
    labels = regime_frame["routing_regime"].astype(str).to_numpy(dtype=object)
    if set(labels).difference(REGIMES):
        raise ValueError(f"{split} contains unknown regimes")
    return labels


def _fit_request(
    model_key: str,
    request: FitRequest,
    *,
    scaled_fold,
    train_regimes: np.ndarray,
    window: int,
    scaler: dict[str, object],
) -> FitResult:
    import tensorflow as tf

    train_features = scaled_fold.train.loc[
        :,
        list(request.features),
    ].to_numpy(dtype=float)
    train_target = scaled_fold.train[TARGET_COLUMN].to_numpy(dtype=float)
    x_train, y_train = make_sequences(
        train_features,
        train_target,
        window,
    )
    if request.scope == "regime":
        mask = endpoint_regime_mask(
            train_regimes,
            regime=request.regime,
            window=window,
        )
        x_fit = x_train[mask]
        y_fit = y_train[mask]
    elif request.scope == "global":
        x_fit = x_train
        y_fit = y_train
    else:
        raise ValueError(f"Unknown fit scope: {request.scope}")
    if len(x_fit) < 200:
        raise ValueError(
            f"{request.scope}/{request.regime} has only {len(x_fit)} "
            "training sequences"
        )

    test_sequences = make_test_sequences(
        sequence_history_features(scaled_fold)[
            :,
            [scaled_fold.feature_columns.index(name) for name in request.features],
        ],
        scaled_fold.test.loc[
            :,
            list(request.features),
        ].to_numpy(dtype=float),
        window,
    )
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(request.seed)
    model = MODEL_BUILDERS[model_key]((window, len(request.features)))
    parameters = TRACK_A_MODELS[model_key].parameters
    fit_started = time.perf_counter()
    model.fit(
        x_fit,
        y_fit,
        epochs=int(parameters["epochs"]),
        batch_size=int(parameters["batch_size"]),
        shuffle=False,
        verbose=0,
    )
    fit_seconds = time.perf_counter() - fit_started
    inference_started = time.perf_counter()
    scaled_prediction = model.predict(
        test_sequences,
        verbose=0,
    ).reshape(-1)
    inference_seconds = time.perf_counter() - inference_started
    prediction = inverse_scaled_target(scaled_prediction, scaler)
    if prediction.shape != (len(scaled_fold.test),):
        raise ValueError("Outer prediction shape is invalid")
    if not np.isfinite(prediction).all():
        raise ValueError("Outer prediction contains non-finite values")
    return FitResult(
        prediction=prediction,
        training_sequences=len(x_fit),
        fit_seconds=float(fit_seconds),
        inference_seconds=float(inference_seconds),
        trainable_parameters=int(model.count_params()),
        fit_id=_fit_id(request),
    )


def _arm_requests(
    *,
    base_seed: int,
    feature_pool: tuple[str, ...],
    selected: dict[tuple[str, str], tuple[str, ...]],
) -> dict[str, tuple[FitRequest, ...]]:
    subseeds = capacity_matched_subseeds(base_seed)
    return {
        "Global-All": (
            FitRequest("global", "global", feature_pool, base_seed),
        ),
        "Global3-All": tuple(
            FitRequest("global", regime, feature_pool, subseeds[regime])
            for regime in REGIMES
        ),
        "Global-SHAP": (
            FitRequest(
                "global",
                "global",
                selected[("shap", "global")],
                base_seed,
            ),
        ),
        "Global-Spearman": (
            FitRequest(
                "global",
                "global",
                selected[("spearman", "global")],
                base_seed,
            ),
        ),
        "Regime-All": tuple(
            FitRequest("regime", regime, feature_pool, subseeds[regime])
            for regime in REGIMES
        ),
        "Regime-SHAP": tuple(
            FitRequest(
                "regime",
                regime,
                selected[("shap", regime)],
                subseeds[regime],
            )
            for regime in REGIMES
        ),
        "Regime-Spearman": tuple(
            FitRequest(
                "regime",
                regime,
                selected[("spearman", regime)],
                subseeds[regime],
            )
            for regime in REGIMES
        ),
    }


def _prediction_metrics(
    original_fold,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    y_true = original_fold.test[TARGET_COLUMN].to_numpy(dtype=float)
    current_close = original_fold.test[CLOSE_COLUMN].to_numpy(dtype=float)
    return {
        **regression_metrics(y_true, prediction),
        **binary_direction_metrics(y_true, prediction, current_close),
    }


def _cell_dir(
    model_key: str,
    fold_name: str,
    base_seed: int,
) -> Path:
    return CELL_DIR / model_key / fold_name / f"seed_{base_seed}"


def _cell_complete(
    model_key: str,
    fold_name: str,
    base_seed: int,
) -> bool:
    directory = _cell_dir(model_key, fold_name, base_seed)
    return (
        (directory / "metrics.csv").is_file()
        and (directory / "fit_registry.csv").is_file()
        and (directory / "run_metadata.json").is_file()
        and all(
            (directory / f"predictions_{arm}.csv").is_file()
            for arm in OUTER_ARMS
        )
    )


def _run_outer_cell(
    model_key: str,
    *,
    scaled_spec,
    original_spec,
    base_seed: int,
    window: int,
    selected: dict[tuple[str, str], tuple[str, ...]],
    force: bool,
) -> None:
    if _cell_complete(model_key, scaled_spec.fold, base_seed) and not force:
        return
    scaled_fold = load_fold(scaled_spec)
    original_fold = load_fold(original_spec)
    if scaled_fold.feature_columns != original_fold.feature_columns:
        raise ValueError("Scaled and original outer feature pools differ")
    train_regime_frame, test_regime_frame = _load_regimes(
        scaled_spec.fold
    )
    train_regimes = _validate_regime_alignment(
        original_fold.train,
        train_regime_frame,
        split="train",
    )
    test_regimes = _validate_regime_alignment(
        original_fold.test,
        test_regime_frame,
        split="test",
    )
    scaler = _load_json(
        scaled_spec.train_path.parent / SCALER_METADATA_NAME
    )
    feature_pool = tuple(scaled_fold.feature_columns)
    ordered_selected = {
        key: _source_order(features, feature_pool)
        for key, features in selected.items()
    }
    requests_by_arm = _arm_requests(
        base_seed=base_seed,
        feature_pool=feature_pool,
        selected=ordered_selected,
    )
    cache: dict[FitRequest, FitResult] = {}
    request_use_count: dict[FitRequest, int] = {}
    arm_predictions: dict[str, np.ndarray] = {}
    metrics_rows: list[dict[str, object]] = []
    cell_started = time.perf_counter()

    for arm in OUTER_ARMS:
        requests = requests_by_arm[arm]
        results: list[FitResult] = []
        reused = 0
        for request in requests:
            if request in cache:
                reused += 1
            else:
                cache[request] = _fit_request(
                    model_key,
                    request,
                    scaled_fold=scaled_fold,
                    train_regimes=train_regimes,
                    window=window,
                    scaler=scaler,
                )
            request_use_count[request] = request_use_count.get(request, 0) + 1
            results.append(cache[request])
        if arm == "Global3-All":
            prediction = np.mean(
                np.stack([result.prediction for result in results]),
                axis=0,
            )
        elif arm.startswith("Regime-"):
            prediction = route_regime_predictions(
                test_regimes,
                {
                    request.regime: result.prediction
                    for request, result in zip(
                        requests,
                        results,
                        strict=True,
                    )
                },
            )
        else:
            prediction = results[0].prediction.copy()
        arm_predictions[arm] = prediction
        metrics_rows.append(
            {
                "model": model_key,
                "fold": scaled_spec.fold,
                "test_year": original_spec.test_year,
                "base_seed": base_seed,
                "arm": arm,
                "window": window,
                "models_in_arm": len(requests),
                "unique_fits_executed_when_reached": len(requests) - reused,
                "identical_fits_reused_when_reached": reused,
                "conceptual_fit_seconds": float(
                    sum(result.fit_seconds for result in results)
                ),
                "conceptual_inference_seconds": float(
                    sum(result.inference_seconds for result in results)
                ),
                "conceptual_trainable_parameters": int(
                    sum(result.trainable_parameters for result in results)
                ),
                "n_test": len(prediction),
                **_prediction_metrics(original_fold, prediction),
            }
        )

    output_dir = _cell_dir(model_key, scaled_spec.fold, base_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )
    for arm, prediction in arm_predictions.items():
        frame = pd.DataFrame(
            {
                DATE_COLUMN: original_fold.test[DATE_COLUMN],
                "routing_regime": test_regimes,
                "Close_D": original_fold.test[CLOSE_COLUMN],
                "y_true": original_fold.test[TARGET_COLUMN],
                "y_pred": prediction,
            }
        )
        frame["true_direction"] = np.sign(
            frame["y_true"] - frame["Close_D"]
        )
        frame["pred_direction"] = np.sign(
            frame["y_pred"] - frame["Close_D"]
        )
        frame.to_csv(
            output_dir / f"predictions_{arm}.csv",
            index=False,
        )
    registry_rows = []
    for request, result in cache.items():
        registry_rows.append(
            {
                "fit_id": result.fit_id,
                "scope": request.scope,
                "regime": request.regime,
                "seed": request.seed,
                "features": len(request.features),
                "feature_hash": hashlib.sha256(
                    "|".join(request.features).encode("utf-8")
                ).hexdigest(),
                "training_sequences": result.training_sequences,
                "fit_seconds": result.fit_seconds,
                "inference_seconds": result.inference_seconds,
                "trainable_parameters": result.trainable_parameters,
                "arm_reference_count": request_use_count[request],
            }
        )
    pd.DataFrame(registry_rows).to_csv(
        output_dir / "fit_registry.csv",
        index=False,
    )
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": _utc_now(),
        "model": model_key,
        "fold": scaled_spec.fold,
        "test_year": original_spec.test_year,
        "base_seed": base_seed,
        "capacity_matched_subseeds": capacity_matched_subseeds(base_seed),
        "window": window,
        "arms": list(OUTER_ARMS),
        "unique_fits": len(cache),
        "conceptual_arm_fits": sum(
            len(requests) for requests in requests_by_arm.values()
        ),
        "cell_wall_seconds": float(time.perf_counter() - cell_started),
        "feature_order": "original source-column order",
        "regime_threshold_source": str(
            REGIME_OUTPUT_DIR / scaled_spec.fold / "model_parameters.json"
        ),
        "selection_freeze": str(SELECTION_FREEZE_FILE),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _collect_metrics() -> pd.DataFrame:
    paths = sorted(CELL_DIR.glob("*/*/seed_*/metrics.csv"))
    if not paths:
        raise FileNotFoundError("No Track C outer metrics were generated")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _collect_fit_registry() -> pd.DataFrame:
    paths = sorted(CELL_DIR.glob("*/*/seed_*/fit_registry.csv"))
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        seed_label = path.parent.name
        if not seed_label.startswith("seed_"):
            raise ValueError(f"Unexpected seed directory: {path.parent}")
        frame.insert(0, "model", path.parents[2].name)
        frame.insert(1, "fold", path.parents[1].name)
        frame.insert(2, "base_seed", int(seed_label.removeprefix("seed_")))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _collect_seed_averaged_predictions(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (model, fold, arm), group in metrics.groupby(
        ["model", "fold", "arm"],
        sort=False,
    ):
        frames_by_seed: dict[int, pd.DataFrame] = {}
        for seed in sorted(group["base_seed"].astype(int).unique()):
            path = (
                _cell_dir(str(model), str(fold), seed)
                / f"predictions_{arm}.csv"
            )
            frames_by_seed[seed] = pd.read_csv(path)
        averaged = average_seed_predictions(frames_by_seed)
        averaged.insert(0, "model", model)
        averaged.insert(1, "fold", fold)
        averaged.insert(2, "test_year", int(group["test_year"].iloc[0]))
        averaged.insert(3, "arm", arm)
        rows.append(averaged)
    return pd.concat(rows, ignore_index=True)


def _fold_metrics_from_seed_averaged_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, fold, test_year, arm), group in predictions.groupby(
        ["model", "fold", "test_year", "arm"],
        sort=False,
    ):
        averaged = group["y_pred"].to_numpy(dtype=float)
        y_true = group["y_true"].to_numpy(dtype=float)
        close = group["Close_D"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "fold": fold,
                "arm": arm,
                "test_year": int(test_year),
                "seeds_averaged": int(group["seeds_averaged"].iloc[0]),
                "n_test": len(group),
                **regression_metrics(y_true, averaged),
                **binary_direction_metrics(y_true, averaged, close),
            }
        )
    return pd.DataFrame(rows)


def _seed_averaged_fold_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    predictions = _collect_seed_averaged_predictions(metrics)
    return _fold_metrics_from_seed_averaged_predictions(predictions)


def _daily_block_bootstrap(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in TRACK_A_MODELS:
        model_predictions = predictions.loc[
            predictions["model"].eq(model)
        ]
        for contrast, (treatment_arm, control_arm) in CONTRASTS.items():
            effects: dict[str, list[np.ndarray]] = {
                "squared_error_loss_delta": [],
                "balanced_accuracy_delta_pp": [],
            }
            test_years: list[int] = []
            for fold in sorted(model_predictions["fold"].unique()):
                fold_predictions = model_predictions.loc[
                    model_predictions["fold"].eq(fold)
                ]
                treatment = fold_predictions.loc[
                    fold_predictions["arm"].eq(treatment_arm)
                ]
                control = fold_predictions.loc[
                    fold_predictions["arm"].eq(control_arm)
                ]
                if treatment.empty or control.empty:
                    raise ValueError(
                        f"Missing daily predictions for {model}/{contrast}/{fold}"
                    )
                squared_error, balanced_accuracy = paired_daily_effects(
                    treatment,
                    control,
                )
                effects["squared_error_loss_delta"].append(squared_error)
                effects["balanced_accuracy_delta_pp"].append(
                    balanced_accuracy
                )
                test_years.append(int(treatment["test_year"].iloc[0]))
            for metric, fold_effects in effects.items():
                result = moving_block_bootstrap(
                    fold_effects,
                    block_length=BOOTSTRAP_BLOCK_LENGTH,
                    replicates=BOOTSTRAP_REPLICATES,
                    seed=BOOTSTRAP_SEED,
                )
                rows.append(
                    {
                        "model": model,
                        "contrast": contrast,
                        "treatment_arm": treatment_arm,
                        "control_arm": control_arm,
                        "metric": metric,
                        "test_years": ",".join(
                            str(year) for year in test_years
                        ),
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def _write_inference_outputs(
    *,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, object]:
    if not INFERENCE_ADDENDUM.is_file():
        raise FileNotFoundError("Track C inference addendum is missing")
    paired = build_paired_fold_contrasts(fold_metrics)
    paired.to_csv(
        OUTPUT_DIR / "paired_fold_contrasts.csv",
        index=False,
    )
    fold_inference = apply_holm_by_family(
        fold_level_inference(paired),
        pvalue_column="exact_sign_flip_pvalue",
    )
    fold_inference["minimum_attainable_nonzero_pvalue"] = 0.125
    fold_inference.to_csv(
        OUTPUT_DIR / "fold_inference.csv",
        index=False,
    )
    block_bootstrap = apply_holm_by_family(
        _daily_block_bootstrap(predictions),
        pvalue_column="two_sided_pvalue",
    )
    block_bootstrap.to_csv(
        OUTPUT_DIR / "daily_block_bootstrap.csv",
        index=False,
    )

    fold_holm = fold_inference[
        [
            "model",
            "contrast",
            "metric",
            "mean_delta",
            "ci95_lower",
            "ci95_upper",
            "exact_sign_flip_pvalue",
            "exact_sign_flip_pvalue_holm",
        ]
    ].rename(
        columns={
            "mean_delta": "point_estimate",
            "exact_sign_flip_pvalue": "raw_pvalue",
            "exact_sign_flip_pvalue_holm": "holm_adjusted_pvalue",
        }
    )
    fold_holm.insert(0, "inference_type", "four_fold_exact_sign_flip")
    block_holm = block_bootstrap[
        [
            "model",
            "contrast",
            "metric",
            "point_estimate",
            "ci95_lower",
            "ci95_upper",
            "two_sided_pvalue",
            "two_sided_pvalue_holm",
        ]
    ].rename(
        columns={
            "two_sided_pvalue": "raw_pvalue",
            "two_sided_pvalue_holm": "holm_adjusted_pvalue",
        }
    )
    block_holm.insert(
        0,
        "inference_type",
        "daily_circular_moving_block_bootstrap",
    )
    holm = pd.concat([fold_holm, block_holm], ignore_index=True)
    holm.to_csv(
        OUTPUT_DIR / "inference_holm_adjusted.csv",
        index=False,
    )
    protocol = {
        "protocol_version": INFERENCE_PROTOCOL_VERSION,
        "created_at": _utc_now(),
        "addendum": str(INFERENCE_ADDENDUM),
        "addendum_sha256": _sha256(INFERENCE_ADDENDUM),
        "contrasts": {
            key: {"treatment": value[0], "control": value[1]}
            for key, value in CONTRASTS.items()
        },
        "seed_aggregation": "mean predictions within model-arm-fold",
        "primary_units": "four temporal outer folds",
        "fold_interval": "Student-t 95% interval",
        "fold_test": "two-sided exact sign-flip",
        "holm_family": (
            "five architectures within each contrast-metric family"
        ),
        "bootstrap": {
            "type": "circular moving-block within each fold",
            "block_length": BOOTSTRAP_BLOCK_LENGTH,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 95%",
            "fold_aggregation": "equal-weight mean",
        },
    }
    (OUTPUT_DIR / "inference_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    return {
        "paired_fold_rows": len(paired),
        "fold_inference_rows": len(fold_inference),
        "block_bootstrap_rows": len(block_bootstrap),
        "holm_rows": len(holm),
        "inference_protocol_sha256": _sha256(
            OUTPUT_DIR / "inference_protocol.json"
        ),
    }


def run_outer_experiment(
    *,
    model_keys: Iterable[str] = TRACK_A_MODELS,
    seeds: Iterable[int] = FINAL_SEEDS,
    force: bool = False,
) -> dict[str, object]:
    import tensorflow as tf

    keys = _validate_models(model_keys)
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("Outer seeds must be non-empty and unique")
    if not SELECTION_FREEZE_FILE.is_file():
        raise FileNotFoundError("Top-k selection freeze is missing")
    freeze = _load_json(SELECTION_FREEZE_FILE)
    if not freeze.get("outer_execution_authorized"):
        raise ValueError("Top-k selection freeze does not authorize outer execution")
    selected = selected_feature_lookup(
        pd.read_csv(SELECTED_FEATURES_FILE)
    )
    tf.config.experimental.enable_op_determinism()
    started_at = _utc_now()
    started = time.perf_counter()
    scaled_specs = discover_folds(
        FULL_TA_VMD_POINT_IN_TIME_NN_DATA_FOLDS_DIR
    )
    original_specs = discover_folds(
        FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR
    )
    windows = _window_map()
    for model_key in keys:
        for scaled_spec, original_spec in zip(
            scaled_specs,
            original_specs,
            strict=True,
        ):
            if scaled_spec.fold != original_spec.fold:
                raise ValueError("Scaled and original outer folds differ")
            for base_seed in seed_values:
                _run_outer_cell(
                    model_key,
                    scaled_spec=scaled_spec,
                    original_spec=original_spec,
                    base_seed=base_seed,
                    window=windows[model_key],
                    selected=selected,
                    force=force,
                )
    metrics = _collect_metrics()
    metrics.to_csv(OUTPUT_DIR / "metrics_by_seed_fold.csv", index=False)
    registry = _collect_fit_registry()
    registry.to_csv(OUTPUT_DIR / "fit_registry.csv", index=False)
    registry_complete = (
        set(metrics["model"]) == set(TRACK_A_MODELS)
        and set(metrics["base_seed"].astype(int)) == set(FINAL_SEEDS)
    )
    expected_rows = (
        len(TRACK_A_MODELS)
        * len(scaled_specs)
        * len(FINAL_SEEDS)
        * len(OUTER_ARMS)
    )
    expected_cells = (
        len(TRACK_A_MODELS) * len(scaled_specs) * len(FINAL_SEEDS)
    )
    rows_per_cell = metrics.groupby(
        ["model", "fold", "base_seed"],
        sort=False,
    )["arm"].agg(["size", "nunique"])
    structure_complete = (
        len(rows_per_cell) == expected_cells
        and rows_per_cell["size"].eq(len(OUTER_ARMS)).all()
        and rows_per_cell["nunique"].eq(len(OUTER_ARMS)).all()
    )
    full_run = (
        registry_complete
        and len(metrics) == expected_rows
        and structure_complete
    )
    if registry_complete and len(metrics) >= expected_rows and not full_run:
        raise ValueError("Outer metrics have duplicate or incomplete arm cells")
    seed_averaged_predictions = _collect_seed_averaged_predictions(
        metrics
    )
    seed_averaged_predictions.to_csv(
        OUTPUT_DIR / "predictions_seed_averaged.csv",
        index=False,
    )
    fold_metrics = _fold_metrics_from_seed_averaged_predictions(
        seed_averaged_predictions
    )
    fold_metrics.to_csv(
        OUTPUT_DIR / "fold_metrics_seed_averaged.csv",
        index=False,
    )
    arm_summary = (
        fold_metrics.groupby(["model", "arm"], sort=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            direction_accuracy_mean=("direction_accuracy", "mean"),
            mcc_mean=("mcc", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            temporal_folds=("fold", "nunique"),
        )
        .reset_index()
    )
    arm_summary.to_csv(OUTPUT_DIR / "arm_summary.csv", index=False)
    inference_metadata: dict[str, object] = {
        "status": "not generated for a partial run"
    }
    if full_run:
        inference_metadata = _write_inference_outputs(
            fold_metrics=fold_metrics,
            predictions=seed_averaged_predictions,
        )
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "models_completed": sorted(
            str(model) for model in metrics["model"].unique()
        ),
        "folds_completed": sorted(
            str(fold) for fold in metrics["fold"].unique()
        ),
        "seeds_completed": sorted(
            int(seed)
            for seed in metrics["base_seed"].astype(int).unique()
        ),
        "arms": list(OUTER_ARMS),
        "metrics_rows": len(metrics),
        "selection_freeze": freeze,
        "outer_results_generated": bool(full_run),
        "lime_outer_explanations_generated": False,
        "runtime_seconds": float(time.perf_counter() - started),
        "executed_unique_fits": len(registry),
        "conceptual_arm_fit_references": int(
            registry["arm_reference_count"].sum()
        ),
        "inference": inference_metadata,
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "tensorflow"]
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "artifacts": [
            "metrics_by_seed_fold.csv",
            "predictions_seed_averaged.csv",
            "fold_metrics_seed_averaged.csv",
            "arm_summary.csv",
            "fit_registry.csv",
            "paired_fold_contrasts.csv",
            "fold_inference.csv",
            "daily_block_bootstrap.csv",
            "inference_holm_adjusted.csv",
            "inference_protocol.json",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen seven-arm Track C outer experiment."
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(TRACK_A_MODELS),
        dest="models",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    return run_outer_experiment(
        model_keys=(TRACK_A_MODELS if args.models is None else args.models),
        seeds=(FINAL_SEEDS if args.seeds is None else args.seeds),
        force=args.force,
    )


if __name__ == "__main__":
    main()
