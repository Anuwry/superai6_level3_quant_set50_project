from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
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
from models.neural_network_folds import (
    SCALER_METADATA_NAME,
    inverse_scaled_target,
)
from models.shap_protocol_v2 import (
    BACKGROUND_CAP,
    MODEL_BUILDERS,
    NSAMPLES,
    build_original_change_model,
    evenly_spaced_indices,
    normalize_single_output_shap,
)
from models.track_a_final import TRACK_A_MODELS, load_locked_windows
from models.track_c_lime_audit import (
    LimeAuditConfig,
    aggregate_local_shap,
    compare_local_explanations,
    explain_instance_with_lime,
    lime_repeat_stability,
    select_regime_audit_indices,
)
from models.track_c_lime_outer import (
    boolean_mask,
    build_audit_instance_id,
    summarize_agreement,
    validate_outer_lime_gate,
)
from models.track_c_shap_selection import derive_protocol_seed
from models.vmd_feature_pool import (
    FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
    FULL_TA_VMD_POINT_IN_TIME_NN_DATA_FOLDS_DIR,
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "track_c" / "dual_xai_lime_v1"
)
CELL_DIR = OUTPUT_DIR / "cells"
OUTER_DIR = PROJECT_ROOT / "outputs" / "track_c" / "outer_v2"
REGIME_DIR = (
    PROJECT_ROOT / "outputs" / "track_c" / "daily_regime_point_in_time_v2"
)
ADDENDUM_PATH = PROJECT_ROOT / "test" / "track_c_dual_xai_lime_addendum.md"
CLOSE_SENSITIVITY_PATH = (
    PROJECT_ROOT
    / "test"
    / "track_c_dual_xai_structural_close_sensitivity.md"
)
AUDITED_OUTER_ARM = "Global-All"
AUDITED_BASE_SEED = 42
REGIME_DISPLAY = {
    "bull": "Bull",
    "sideway": "Sideway",
    "bear": "Bear",
}


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


def _config_hash(config: LimeAuditConfig) -> str:
    material = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _implementation_hash() -> str:
    """Invalidate resumable cells when the audit implementation changes."""

    return _sha256(Path(__file__))


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


def _cell_dir(model_key: str, fold_name: str) -> Path:
    return CELL_DIR / model_key / fold_name


def _cell_complete(
    model_key: str,
    fold_name: str,
    *,
    config_hash: str,
    implementation_hash: str,
) -> bool:
    directory = _cell_dir(model_key, fold_name)
    required = (
        "selected_instances.csv",
        "local_explanations.csv",
        "agreement_by_instance.csv",
        "lime_stability_by_instance.csv",
        "run_metadata.json",
    )
    if not all((directory / name).is_file() for name in required):
        return False
    metadata = _load_json(directory / "run_metadata.json")
    return (
        metadata.get("config_sha256") == config_hash
        and metadata.get("implementation_sha256") == implementation_hash
    )


def _ranking(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(-np.abs(array), kind="stable")
    ranks = np.empty(len(array), dtype=int)
    ranks[order] = np.arange(1, len(array) + 1)
    return ranks


def _predict_change(change_model, sequences: np.ndarray) -> np.ndarray:
    values = change_model(
        np.asarray(sequences, dtype=np.float32),
        training=False,
    )
    result = np.asarray(values.numpy(), dtype=float).reshape(-1)
    if result.shape != (len(sequences),) or not np.isfinite(result).all():
        raise ValueError("Change model produced invalid predictions")
    return result


def _original_change_from_scaled_prediction(
    scaled_prediction: np.ndarray,
    *,
    scaler: dict[str, object],
    current_close: np.ndarray,
) -> np.ndarray:
    """Mirror the frozen outer inference path before comparing predictions."""

    original_level = inverse_scaled_target(scaled_prediction, scaler).reshape(-1)
    close = np.asarray(current_close, dtype=float).reshape(-1)
    if original_level.shape != close.shape:
        raise ValueError("Predicted level and current close shapes differ")
    change = original_level - close
    if not np.isfinite(change).all():
        raise ValueError("Outer-style change prediction is non-finite")
    return change


def _validate_outer_prediction_reproduction(
    *,
    model_key: str,
    fold_name: str,
    predicted_change: np.ndarray,
    dates: pd.Series,
    tolerance: float = 1e-4,
) -> float:
    path = (
        OUTER_DIR
        / "cells"
        / model_key
        / fold_name
        / f"seed_{AUDITED_BASE_SEED}"
        / f"predictions_{AUDITED_OUTER_ARM}.csv"
    )
    expected = pd.read_csv(path)
    expected_dates = pd.to_datetime(expected[DATE_COLUMN]).reset_index(drop=True)
    actual_dates = pd.to_datetime(dates).reset_index(drop=True)
    if not expected_dates.equals(actual_dates):
        raise ValueError("LIME audit and outer prediction dates do not align")
    expected_change = (
        expected["y_pred"].to_numpy(dtype=float)
        - expected["Close_D"].to_numpy(dtype=float)
    )
    maximum_error = float(
        np.max(np.abs(expected_change - predicted_change))
    )
    if maximum_error > tolerance:
        raise ValueError(
            "Refitted audit model does not reproduce frozen outer predictions: "
            f"max absolute change error={maximum_error:.8g}"
        )
    return maximum_error


def _load_test_regimes(
    fold_name: str,
    expected_dates: pd.Series,
) -> np.ndarray:
    frame = pd.read_csv(REGIME_DIR / fold_name / "test_regimes.csv")
    regime_dates = pd.to_datetime(frame[DATE_COLUMN]).reset_index(drop=True)
    market_dates = pd.to_datetime(expected_dates).reset_index(drop=True)
    if not regime_dates.equals(market_dates):
        raise ValueError("LIME audit and regime dates do not align")
    labels = frame["routing_regime"].astype(str)
    unknown = sorted(set(labels).difference(REGIME_DISPLAY))
    if unknown:
        raise ValueError(f"Unknown outer regimes: {unknown}")
    return labels.map(REGIME_DISPLAY).to_numpy(dtype=object)


def _append_explanation_rows(
    rows: list[dict[str, object]],
    *,
    instance_id: str,
    model_key: str,
    fold_name: str,
    test_year: int,
    date: str,
    regime: str,
    method: str,
    repeat_seed: int | None,
    features: Sequence[str],
    values: np.ndarray,
) -> None:
    ranks = _ranking(values)
    for feature, value, rank in zip(
        features,
        np.asarray(values, dtype=float),
        ranks,
        strict=True,
    ):
        rows.append(
            {
                "instance_id": instance_id,
                "model": model_key,
                "fold": fold_name,
                "test_year": test_year,
                "date": date,
                "regime": regime,
                "method": method,
                "repeat_seed": repeat_seed,
                "feature": feature,
                "attribution": float(value),
                "absolute_attribution": float(abs(value)),
                "absolute_rank": int(rank),
            }
        )


def _fit_audit_model(
    model_key: str,
    *,
    scaled_fold,
    window: int,
):
    import tensorflow as tf

    features = tuple(scaled_fold.feature_columns)
    train_features = scaled_fold.train.loc[
        :,
        list(features),
    ].to_numpy(dtype=float)
    train_target = scaled_fold.train[TARGET_COLUMN].to_numpy(dtype=float)
    x_train, y_train = make_sequences(
        train_features,
        train_target,
        window,
    )
    test_sequences = make_test_sequences(
        sequence_history_features(scaled_fold),
        scaled_fold.test.loc[:, list(features)].to_numpy(dtype=float),
        window,
    )
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(AUDITED_BASE_SEED)
    model = MODEL_BUILDERS[model_key]((window, len(features)))
    parameters = TRACK_A_MODELS[model_key].parameters
    started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        epochs=int(parameters["epochs"]),
        batch_size=int(parameters["batch_size"]),
        shuffle=False,
        verbose=0,
    )
    return (
        model,
        x_train.astype(np.float32),
        test_sequences.astype(np.float32),
        float(time.perf_counter() - started),
    )


def _run_audit_cell(
    model_key: str,
    *,
    scaled_spec,
    original_spec,
    window: int,
    config: LimeAuditConfig,
    force: bool,
) -> None:
    import shap

    config_sha256 = _config_hash(config)
    implementation_sha256 = _implementation_hash()
    fold_name = scaled_spec.fold
    if _cell_complete(
        model_key,
        fold_name,
        config_hash=config_sha256,
        implementation_hash=implementation_sha256,
    ) and not force:
        return
    if fold_name != original_spec.fold:
        raise ValueError("Scaled and original audit folds differ")

    cell_started = time.perf_counter()
    scaled_fold = load_fold(scaled_spec)
    original_fold = load_fold(original_spec)
    if scaled_fold.feature_columns != original_fold.feature_columns:
        raise ValueError("Scaled and original audit feature pools differ")
    features = tuple(scaled_fold.feature_columns)
    if len(features) != 122:
        raise ValueError(f"Expected 122 audit features; found {len(features)}")
    non_structural_mask = np.asarray(
        [feature != CLOSE_COLUMN for feature in features],
        dtype=bool,
    )
    if int(non_structural_mask.sum()) != 121:
        raise ValueError("Structural Close_D sensitivity mask is invalid")
    model, train_sequences, test_sequences, fit_seconds = _fit_audit_model(
        model_key,
        scaled_fold=scaled_fold,
        window=window,
    )
    scaler = _load_json(
        scaled_spec.train_path.parent / SCALER_METADATA_NAME
    )
    close_scale, close_offset = _scale_value(scaler, CLOSE_COLUMN)
    target_scale, target_offset = _scale_value(scaler, TARGET_COLUMN)
    change_model = build_original_change_model(
        model,
        close_feature_index=features.index(CLOSE_COLUMN),
        close_scale=close_scale,
        close_offset=close_offset,
        target_scale=target_scale,
        target_offset=target_offset,
    )
    outer_style_scaled_prediction = model.predict(
        test_sequences,
        verbose=0,
    ).reshape(-1)
    outer_style_change = _original_change_from_scaled_prediction(
        outer_style_scaled_prediction,
        scaler=scaler,
        current_close=original_fold.test[CLOSE_COLUMN].to_numpy(dtype=float),
    )
    reproduction_error = _validate_outer_prediction_reproduction(
        model_key=model_key,
        fold_name=fold_name,
        predicted_change=outer_style_change,
        dates=original_fold.test[DATE_COLUMN],
    )
    all_test_change = _predict_change(change_model, test_sequences)
    explanation_path_difference = float(
        np.max(np.abs(all_test_change - outer_style_change))
    )

    background_indices = evenly_spaced_indices(
        len(train_sequences),
        BACKGROUND_CAP,
    )
    background = train_sequences[background_indices]
    regimes = _load_test_regimes(
        fold_name,
        original_fold.test[DATE_COLUMN],
    )
    selected_by_regime = select_regime_audit_indices(
        regimes,
        samples_per_regime=config.samples_per_regime_fold,
    )
    selected_indices = np.array(
        [
            index
            for regime in ("Bull", "Sideway", "Bear")
            for index in selected_by_regime[regime]
        ],
        dtype=int,
    )
    selected_sequences = test_sequences[selected_indices]
    selected_regimes = regimes[selected_indices]

    shap_started = time.perf_counter()
    explainer = shap.GradientExplainer(change_model, background)
    shap_seed = derive_protocol_seed(
        AUDITED_BASE_SEED,
        "lime_outer_audit",
        model_key,
        fold_name,
    )
    raw_shap = explainer.shap_values(
        selected_sequences,
        nsamples=NSAMPLES,
        rseed=shap_seed,
    )
    local_shap = aggregate_local_shap(
        normalize_single_output_shap(
            raw_shap,
            selected_sequences.shape,
        )
    )
    shap_seconds = float(time.perf_counter() - shap_started)

    selected_rows: list[dict[str, object]] = []
    explanation_rows: list[dict[str, object]] = []
    agreement_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []

    def predict_change(sequences: np.ndarray) -> np.ndarray:
        return _predict_change(change_model, sequences)

    for local_index, test_index in enumerate(selected_indices):
        date = str(
            pd.Timestamp(
                original_fold.test.iloc[test_index][DATE_COLUMN]
            ).date()
        )
        regime = str(selected_regimes[local_index])
        instance_id = build_audit_instance_id(
            model_key,
            fold_name,
            int(test_index),
            date,
        )
        selected_rows.append(
            {
                "instance_id": instance_id,
                "model": model_key,
                "fold": fold_name,
                "test_year": original_spec.test_year,
                "test_row_index": int(test_index),
                "date": date,
                "regime": regime,
                "selection_rule": (
                    "six evenly spaced endpoints within each regime"
                ),
                "selection_uses_outcome": False,
                "current_close": float(
                    original_fold.test.iloc[test_index][CLOSE_COLUMN]
                ),
                "next_close": float(
                    original_fold.test.iloc[test_index][TARGET_COLUMN]
                ),
                "black_box_predicted_change": float(
                    all_test_change[test_index]
                ),
                "true_direction": int(
                    np.sign(
                        float(
                            original_fold.test.iloc[test_index][TARGET_COLUMN]
                        )
                        - float(
                            original_fold.test.iloc[test_index][CLOSE_COLUMN]
                        )
                    )
                ),
                "predicted_direction": int(
                    np.sign(all_test_change[test_index])
                ),
                "direction_correct": bool(
                    np.sign(
                        float(
                            original_fold.test.iloc[test_index][TARGET_COLUMN]
                        )
                        - float(
                            original_fold.test.iloc[test_index][CLOSE_COLUMN]
                        )
                    )
                    == np.sign(all_test_change[test_index])
                ),
            }
        )
        shap_vector = local_shap[local_index]
        _append_explanation_rows(
            explanation_rows,
            instance_id=instance_id,
            model_key=model_key,
            fold_name=fold_name,
            test_year=original_spec.test_year,
            date=date,
            regime=regime,
            method="SHAP",
            repeat_seed=None,
            features=features,
            values=shap_vector,
        )
        repeated_coefficients = []
        for repeat_seed in config.repeat_seeds:
            lime = explain_instance_with_lime(
                predict_change,
                selected_sequences[local_index],
                background,
                n_perturbations=config.n_perturbations,
                seed=repeat_seed,
                ridge_alpha=config.ridge_alpha,
                presence_probability=config.presence_probability,
                kernel_width_multiplier=config.kernel_width_multiplier,
            )
            lime_vector = lime.surrogate.coefficients
            repeated_coefficients.append(lime_vector)
            _append_explanation_rows(
                explanation_rows,
                instance_id=instance_id,
                model_key=model_key,
                fold_name=fold_name,
                test_year=original_spec.test_year,
                date=date,
                regime=regime,
                method="LIME",
                repeat_seed=repeat_seed,
                features=features,
                values=lime_vector,
            )
            agreement = compare_local_explanations(
                shap_vector,
                lime_vector,
                top_k=config.top_k,
            )
            agreement_without_close = compare_local_explanations(
                shap_vector[non_structural_mask],
                lime_vector[non_structural_mask],
                top_k=config.top_k,
            )
            agreement_rows.append(
                {
                    "instance_id": instance_id,
                    "model": model_key,
                    "fold": fold_name,
                    "test_year": original_spec.test_year,
                    "date": date,
                    "regime": regime,
                    "repeat_seed": repeat_seed,
                    "fidelity_r2": lime.surrogate.fidelity_r2,
                    "minimum_fidelity_r2": config.minimum_fidelity_r2,
                    "low_fidelity": bool(
                        lime.surrogate.fidelity_r2
                        < config.minimum_fidelity_r2
                    ),
                    "black_box_prediction": lime.black_box_prediction,
                    "surrogate_local_prediction": (
                        lime.surrogate.local_prediction
                    ),
                    "local_prediction_absolute_error": float(
                        abs(
                            lime.black_box_prediction
                            - lime.surrogate.local_prediction
                        )
                    ),
                    "shap_runtime_seconds_cell": shap_seconds,
                    "shap_runtime_seconds_per_instance_amortized": (
                        shap_seconds / len(selected_indices)
                    ),
                    "lime_runtime_seconds": lime.runtime_seconds,
                    "model_inference_seconds": lime.inference_seconds,
                    **agreement,
                    **{
                        f"{key}_excluding_structural_close": value
                        for key, value in agreement_without_close.items()
                    },
                }
            )
        stability = lime_repeat_stability(
            np.asarray(repeated_coefficients, dtype=float),
            top_k=config.top_k,
        )
        stability_without_close = lime_repeat_stability(
            np.asarray(repeated_coefficients, dtype=float)[
                :,
                non_structural_mask,
            ],
            top_k=config.top_k,
        )
        stability_rows.append(
            {
                "instance_id": instance_id,
                "model": model_key,
                "fold": fold_name,
                "test_year": original_spec.test_year,
                "date": date,
                "regime": regime,
                **stability,
                **{
                    f"{key}_excluding_structural_close": value
                    for key, value in stability_without_close.items()
                },
            }
        )

    output_dir = _cell_dir(model_key, fold_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_rows).to_csv(
        output_dir / "selected_instances.csv",
        index=False,
    )
    pd.DataFrame(explanation_rows).to_csv(
        output_dir / "local_explanations.csv",
        index=False,
    )
    pd.DataFrame(agreement_rows).to_csv(
        output_dir / "agreement_by_instance.csv",
        index=False,
    )
    pd.DataFrame(stability_rows).to_csv(
        output_dir / "lime_stability_by_instance.csv",
        index=False,
    )
    metadata = {
        "protocol_version": config.protocol_version,
        "created_at": _utc_now(),
        "model": model_key,
        "fold": fold_name,
        "test_year": original_spec.test_year,
        "window": window,
        "audited_outer_arm": AUDITED_OUTER_ARM,
        "audited_base_seed": AUDITED_BASE_SEED,
        "config": asdict(config),
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
        "feature_count": len(features),
        "background_sequences": len(background),
        "background_selection": "deterministic evenly spaced train sequences",
        "selected_instances": len(selected_indices),
        "selected_instances_per_regime": config.samples_per_regime_fold,
        "shap_nsamples": NSAMPLES,
        "shap_rseed": shap_seed,
        "fit_seconds": fit_seconds,
        "shap_seconds": shap_seconds,
        "lime_seconds": float(
            sum(row["lime_runtime_seconds"] for row in agreement_rows)
        ),
        "cell_wall_seconds": float(time.perf_counter() - cell_started),
        "outer_prediction_max_abs_reproduction_error": reproduction_error,
        "outer_prediction_reproduction_tolerance": 1e-4,
        "explanation_graph_max_abs_difference_from_outer_inference_path": (
            explanation_path_difference
        ),
        "outcome_used_for_instance_selection": False,
        "lime_used_for_feature_selection": False,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _collect_cell_csv(filename: str) -> pd.DataFrame:
    paths = sorted(CELL_DIR.glob(f"*/*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No LIME audit artifact found: {filename}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def run_lime_outer_audit(
    *,
    model_keys: Iterable[str] = TRACK_A_MODELS,
    config: LimeAuditConfig | None = None,
    force: bool = False,
) -> dict[str, object]:
    import tensorflow as tf

    keys = _validate_models(model_keys)
    audit_config = LimeAuditConfig() if config is None else config
    outer_metadata_path = OUTER_DIR / "run_metadata.json"
    outer_metadata = _load_json(outer_metadata_path)
    validate_outer_lime_gate(outer_metadata)
    if not ADDENDUM_PATH.is_file():
        raise FileNotFoundError("The frozen LIME protocol addendum is missing")
    if not CLOSE_SENSITIVITY_PATH.is_file():
        raise FileNotFoundError(
            "The frozen structural-close sensitivity is missing"
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
            _run_audit_cell(
                model_key,
                scaled_spec=scaled_spec,
                original_spec=original_spec,
                window=windows[model_key],
                config=audit_config,
                force=force,
            )

    selected = _collect_cell_csv("selected_instances.csv")
    explanations = _collect_cell_csv("local_explanations.csv")
    agreements = _collect_cell_csv("agreement_by_instance.csv")
    stability = _collect_cell_csv("lime_stability_by_instance.csv")
    selected.to_csv(OUTPUT_DIR / "selected_instances.csv", index=False)
    explanations.to_csv(OUTPUT_DIR / "local_explanations.csv", index=False)
    agreements.to_csv(
        OUTPUT_DIR / "agreement_by_instance.csv",
        index=False,
    )
    stability.to_csv(
        OUTPUT_DIR / "lime_stability_by_instance.csv",
        index=False,
    )
    summarize_agreement(agreements).to_csv(
        OUTPUT_DIR / "summary_by_model_fold_regime.csv",
        index=False,
    )
    summarize_agreement(
        agreements.drop(columns="fold")
    ).to_csv(
        OUTPUT_DIR / "summary_pooled_by_model_regime.csv",
        index=False,
    )
    sensitivity_columns = [
        "instance_id",
        "model",
        "fold",
        "test_year",
        "date",
        "regime",
        "repeat_seed",
        "fidelity_r2",
        "minimum_fidelity_r2",
        "low_fidelity",
        "lime_runtime_seconds",
        "spearman_abs_excluding_structural_close",
        "top_k_jaccard_excluding_structural_close",
        "sign_agreement_nonzero_excluding_structural_close",
        "n_sign_compared_excluding_structural_close",
    ]
    sensitivity = agreements[sensitivity_columns].rename(
        columns={
            "spearman_abs_excluding_structural_close": "spearman_abs",
            "top_k_jaccard_excluding_structural_close": "top_k_jaccard",
            (
                "sign_agreement_nonzero_excluding_structural_close"
            ): "sign_agreement_nonzero",
            (
                "n_sign_compared_excluding_structural_close"
            ): "n_sign_compared",
        }
    )
    sensitivity.to_csv(
        OUTPUT_DIR / "agreement_excluding_structural_close.csv",
        index=False,
    )
    summarize_agreement(sensitivity).to_csv(
        OUTPUT_DIR / "summary_excluding_structural_close.csv",
        index=False,
    )
    low_fidelity = boolean_mask(agreements["low_fidelity"])
    agreements.loc[low_fidelity].to_csv(
        OUTPUT_DIR / "low_fidelity_audit.csv",
        index=False,
    )

    runtime_rows = []
    for path in sorted(CELL_DIR.glob("*/*/run_metadata.json")):
        cell_metadata = _load_json(path)
        lime_seconds = float(cell_metadata["lime_seconds"])
        audit_instances = int(cell_metadata["selected_instances"])
        cell_agreements = agreements.loc[
            agreements["model"].eq(cell_metadata["model"])
            & agreements["fold"].eq(cell_metadata["fold"])
        ]
        runtime_rows.append(
            {
                "model": cell_metadata["model"],
                "fold": cell_metadata["fold"],
                "test_year": cell_metadata["test_year"],
                "fit_seconds": cell_metadata["fit_seconds"],
                "shap_seconds": cell_metadata["shap_seconds"],
                "lime_seconds": lime_seconds,
                "model_inference_seconds": float(
                    cell_agreements["model_inference_seconds"].sum()
                ),
                "cell_wall_seconds": cell_metadata["cell_wall_seconds"],
                "audit_instances": audit_instances,
                "lime_repeats": (
                    audit_instances * len(audit_config.repeat_seeds)
                ),
                "lime_instances_per_second": (
                    audit_instances / lime_seconds
                    if lime_seconds > 0.0
                    else float("nan")
                ),
            }
        )
    pd.DataFrame(runtime_rows).to_csv(
        OUTPUT_DIR / "runtime_summary.csv",
        index=False,
    )

    error_rows = agreements.merge(
        selected[["instance_id", "direction_correct"]],
        on="instance_id",
        how="left",
        validate="many_to_one",
    )
    reliable_error_rows = error_rows.loc[
        ~boolean_mask(error_rows["low_fidelity"])
    ]
    error_analysis = (
        reliable_error_rows.groupby(
            ["model", "direction_correct"],
            sort=False,
        )
        .agg(
            reliable_repeats=("repeat_seed", "size"),
            fidelity_r2_median=("fidelity_r2", "median"),
            spearman_abs_median=("spearman_abs", "median"),
            top_k_jaccard_median=("top_k_jaccard", "median"),
            sign_agreement_median=(
                "sign_agreement_nonzero",
                "median",
            ),
        )
        .reset_index()
    )
    error_analysis.to_csv(
        OUTPUT_DIR / "error_analysis_summary.csv",
        index=False,
    )

    full_run = (
        set(selected["model"]) == set(TRACK_A_MODELS)
        and set(selected["fold"])
        == {"fold_1", "fold_2", "fold_3", "fold_4"}
    )
    expected_instances = (
        len(TRACK_A_MODELS)
        * len(scaled_specs)
        * len(REGIME_DISPLAY)
        * audit_config.samples_per_regime_fold
    )
    if full_run and len(selected) != expected_instances:
        raise ValueError(
            f"Expected {expected_instances} audit instances; "
            f"found {len(selected)}"
        )
    protocol = {
        "protocol_version": audit_config.protocol_version,
        "frozen_addendum": str(ADDENDUM_PATH),
        "frozen_addendum_sha256": _sha256(ADDENDUM_PATH),
        "structural_close_sensitivity": str(CLOSE_SENSITIVITY_PATH),
        "structural_close_sensitivity_sha256": _sha256(
            CLOSE_SENSITIVITY_PATH
        ),
        "config": asdict(audit_config),
        "config_sha256": _config_hash(audit_config),
        "audited_outer_arm": AUDITED_OUTER_ARM,
        "audited_base_seed": AUDITED_BASE_SEED,
        "feature_space": "full 122-feature numerical Global-All pool",
        "explanation_target": (
            "predicted next-close minus current close in original units"
        ),
        "selection_role": "post-selection audit only",
        "outcome_used_for_instance_selection": False,
        "agreement_policy": (
            "retain all rows; exclude fidelity R2 below 0.70 from "
            "substantive SHAP-LIME agreement summaries"
        ),
    }
    (OUTPUT_DIR / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "recorded_at": _utc_now(),
                "deviation": "none",
                "reason": "not applicable",
                "ranking_or_outer_viewed": True,
                "classification": "no deviation",
                "affected_artifacts": "none",
            }
        ]
    ).to_csv(
        OUTPUT_DIR / "deviation_log.csv",
        index=False,
    )
    metadata = {
        "protocol_version": audit_config.protocol_version,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "models_completed": sorted(selected["model"].unique()),
        "folds_completed": sorted(selected["fold"].unique()),
        "outer_results_verified_complete": True,
        "outer_run_metadata_sha256": _sha256(outer_metadata_path),
        "lime_outer_explanations_generated": bool(full_run),
        "audit_instances": len(selected),
        "local_explanation_rows": len(explanations),
        "agreement_rows": len(agreements),
        "low_fidelity_rows": int(low_fidelity.sum()),
        "runtime_seconds": float(time.perf_counter() - started),
        "packages": package_versions(
            [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "shap",
                "tensorflow",
            ]
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "artifacts": [
            "protocol.json",
            "selected_instances.csv",
            "local_explanations.csv",
            "agreement_by_instance.csv",
            "lime_stability_by_instance.csv",
            "summary_by_model_fold_regime.csv",
            "summary_pooled_by_model_regime.csv",
            "agreement_excluding_structural_close.csv",
            "summary_excluding_structural_close.csv",
            "low_fidelity_audit.csv",
            "runtime_summary.csv",
            "error_analysis_summary.csv",
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
        description="Run the frozen post-selection Track C LIME audit."
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
    return run_lime_outer_audit(
        model_keys=(TRACK_A_MODELS if args.models is None else args.models),
        force=args.force,
    )


if __name__ == "__main__":
    main()
