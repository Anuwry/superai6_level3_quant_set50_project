from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from models.integrated_multimodal import ARMS
from models.pit_cmm_lstm import MODEL_KEY, MODEL_LABEL
from models.track_a_final import TRACK_A_MODELS

SIX_MODEL_ORDER = (*tuple(TRACK_A_MODELS), MODEL_KEY)
FINAL_ARM = "Regime-SHAP-Numeric-News"
EVIDENCE_STATUS = "post_freeze_exploratory_architecture_extension"

MODEL_LABELS = {
    **{key: model.label for key, model in TRACK_A_MODELS.items()},
    MODEL_KEY: MODEL_LABEL,
}

REQUIRED_SUMMARY_COLUMNS = {
    "model",
    "arm",
    "balanced_accuracy_mean",
    "direction_accuracy_mean",
    "mcc_mean",
    "rmse_mean",
    "mae_mean",
    "temporal_folds",
}


def _validate_summary(
    frame: pd.DataFrame,
    *,
    expected_models: Sequence[str],
    name: str,
) -> pd.DataFrame:
    missing = sorted(REQUIRED_SUMMARY_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} summary is missing columns: {missing}")
    keys = frame.loc[:, ["model", "arm"]]
    if keys.duplicated().any():
        raise ValueError(f"{name} summary contains duplicate model/arm rows")
    if set(frame["model"].astype(str)) != set(expected_models):
        raise ValueError(f"{name} summary contains an unexpected model family")
    expected = {
        (model, arm) for model in expected_models for arm in ARMS
    }
    actual = set(
        frame.loc[:, ["model", "arm"]].itertuples(index=False, name=None)
    )
    if actual != expected:
        raise ValueError(f"{name} must contain exactly four arms per model")
    if not frame["temporal_folds"].eq(4).all():
        raise ValueError(f"{name} must contain four temporal folds per row")
    numeric = frame.loc[
        :,
        [
            "balanced_accuracy_mean",
            "direction_accuracy_mean",
            "mcc_mean",
            "rmse_mean",
            "mae_mean",
        ],
    ]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} summary contains non-finite metrics")
    return frame.copy()


def build_six_model_tables(
    frozen_summary: pd.DataFrame,
    pit_cmm_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = _validate_summary(
        frozen_summary,
        expected_models=tuple(TRACK_A_MODELS),
        name="frozen",
    )
    ours = _validate_summary(
        pit_cmm_summary,
        expected_models=(MODEL_KEY,),
        name="PIT-CMM-LSTM",
    )
    frozen["evidence_status"] = "frozen_existing_result"
    ours["evidence_status"] = EVIDENCE_STATUS
    combined = pd.concat([frozen, ours], ignore_index=True)
    combined["model_label"] = combined["model"].map(MODEL_LABELS)
    combined["model_order"] = pd.Categorical(
        combined["model"],
        categories=SIX_MODEL_ORDER,
        ordered=True,
    )
    combined["arm_order"] = pd.Categorical(
        combined["arm"],
        categories=ARMS,
        ordered=True,
    )
    combined = (
        combined.sort_values(["model_order", "arm_order"])
        .drop(columns=["model_order", "arm_order"])
        .reset_index(drop=True)
    )
    combined["balanced_accuracy_mean_pct"] = (
        combined["balanced_accuracy_mean"] * 100.0
    )
    combined["direction_accuracy_mean_pct"] = (
        combined["direction_accuracy_mean"] * 100.0
    )
    final_arm = combined.loc[combined["arm"].eq(FINAL_ARM)].reset_index(
        drop=True
    )
    if final_arm["model"].tolist() != list(SIX_MODEL_ORDER):
        raise ValueError("Final-arm table does not preserve the six-model order")
    return combined, final_arm


def build_compact_six_model_comparison(
    all_arms: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "model",
        "model_label",
        "arm",
        "balanced_accuracy_mean",
        "direction_accuracy_mean",
        "mcc_mean",
        "rmse_mean",
        "mae_mean",
        "evidence_status",
    }
    missing = sorted(required.difference(all_arms.columns))
    if missing:
        raise ValueError(f"Six-model arm table is missing columns: {missing}")
    if len(all_arms) != len(SIX_MODEL_ORDER) * len(ARMS):
        raise ValueError("Six-model arm table has incorrect cardinality")
    keys = ["model", "model_label", "evidence_status"]
    bacc = all_arms.pivot(
        index=keys,
        columns="arm",
        values="balanced_accuracy_mean",
    ).reset_index()
    bacc = bacc.rename(
        columns={
            "Global-Numeric": "global_numeric_bacc",
            "Global-Numeric-News": "global_numeric_news_bacc",
            "Regime-SHAP-Numeric": "regime_shap_numeric_bacc",
            "Regime-SHAP-Numeric-News": "final_integrated_bacc",
        }
    )
    final_metrics = all_arms.loc[
        all_arms["arm"].eq(FINAL_ARM),
        [
            *keys,
            "direction_accuracy_mean",
            "mcc_mean",
            "rmse_mean",
            "mae_mean",
        ],
    ]
    compact = bacc.merge(
        final_metrics,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    compact["global_numeric_bacc_rank"] = (
        compact["global_numeric_bacc"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    compact["final_integrated_bacc_rank"] = (
        compact["final_integrated_bacc"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    order = {model: index for index, model in enumerate(SIX_MODEL_ORDER)}
    compact["_order"] = compact["model"].map(order)
    return compact.sort_values("_order").drop(columns="_order").reset_index(
        drop=True
    )


def evaluate_promotion_gates(
    fold_deltas: pd.DataFrame,
    *,
    parameter_deltas: Sequence[float],
    complete_finite_predictions: bool,
) -> dict[str, object]:
    required = {"fold", "balanced_accuracy_delta_pp"}
    missing = sorted(required.difference(fold_deltas.columns))
    if missing:
        raise ValueError(f"fold_deltas are missing columns: {missing}")
    if len(fold_deltas) != 4 or fold_deltas["fold"].nunique() != 4:
        raise ValueError("Promotion evaluation requires four temporal folds")
    deltas = fold_deltas["balanced_accuracy_delta_pp"].to_numpy(dtype=float)
    parameters = np.asarray(tuple(parameter_deltas), dtype=float)
    if parameters.shape != (3,) or not np.isfinite(parameters).all():
        raise ValueError("parameter_deltas must contain three finite values")
    if not np.isfinite(deltas).all():
        raise ValueError("Fold BAcc deltas must be finite")

    mean_delta = float(deltas.mean())
    positive_folds = int(np.count_nonzero(deltas > 0.0))
    parameter_budget_passed = bool(np.max(np.abs(parameters)) <= 0.15)
    gates = {
        "mean_bacc_delta_gate": mean_delta >= 1.0,
        "temporal_consistency_gate": positive_folds >= 3,
        "parameter_budget_gate": parameter_budget_passed,
        "complete_finite_prediction_gate": bool(complete_finite_predictions),
    }
    return {
        "mean_bacc_delta_pp": mean_delta,
        "positive_temporal_folds": positive_folds,
        "maximum_parameter_delta_fraction": float(np.max(np.abs(parameters))),
        **gates,
        "passed": bool(all(gates.values())),
    }
