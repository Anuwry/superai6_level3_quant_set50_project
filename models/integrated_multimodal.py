from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_common import DATE_COLUMN, FoldSpec
from models.track_a_analysis import exact_sign_flip_pvalue
from models.track_b_data import DAILY_FEATURE_COLUMNS
from models.track_b_fusion import prepare_fold_pair
from models.track_c_inference import holm_adjust
from models.track_c_outer import REGIMES, capacity_matched_subseeds

PROTOCOL_ID = "integrated-multimodal-posthoc-v1"
ARMS = (
    "Global-Numeric",
    "Global-Numeric-News",
    "Regime-SHAP-Numeric",
    "Regime-SHAP-Numeric-News",
)
NEWS_FEATURES = tuple(DAILY_FEATURE_COLUMNS)
CONTRASTS = (
    "global_news_effect",
    "regime_shap_effect_without_news",
    "regime_pipeline_news_effect",
    "final_integrated_effect",
    "routing_news_interaction",
)
METRIC_DELTAS = (
    "balanced_accuracy_delta_pp",
    "direction_accuracy_delta_pp",
    "mcc_delta",
    "rmse_delta",
    "mae_delta",
)

DIRECT_CONTRASTS: dict[str, tuple[str, str]] = {
    "global_news_effect": ("Global-Numeric-News", "Global-Numeric"),
    "regime_shap_effect_without_news": (
        "Regime-SHAP-Numeric",
        "Global-Numeric",
    ),
    "regime_pipeline_news_effect": (
        "Regime-SHAP-Numeric-News",
        "Regime-SHAP-Numeric",
    ),
    "final_integrated_effect": (
        "Regime-SHAP-Numeric-News",
        "Global-Numeric",
    ),
}


@dataclass(frozen=True)
class FitRequest:
    scope: str
    regime: str
    features: tuple[str, ...]
    seed: int


def _unique_nonempty(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicates")
    return result


def build_arm_feature_sets(
    numerical_pool: Sequence[str],
    selected_by_regime: Mapping[str, Sequence[str]],
    *,
    news_features: Sequence[str] = NEWS_FEATURES,
) -> dict[str, dict[str, tuple[str, ...]]]:
    numerical = _unique_nonempty(numerical_pool, name="numerical_pool")
    news = _unique_nonempty(news_features, name="news_features")
    overlap = sorted(set(numerical).intersection(news))
    if overlap:
        raise ValueError(f"Numerical and news features overlap: {overlap}")
    missing_regimes = sorted(set(REGIMES).difference(selected_by_regime))
    extra_regimes = sorted(set(selected_by_regime).difference(REGIMES))
    if missing_regimes or extra_regimes:
        raise ValueError(
            "selected_by_regime must contain exactly bull, sideway, and bear"
        )

    numerical_set = set(numerical)
    selected: dict[str, tuple[str, ...]] = {}
    for regime in REGIMES:
        raw = _unique_nonempty(
            selected_by_regime[regime],
            name=f"selected_by_regime[{regime}]",
        )
        missing = sorted(set(raw).difference(numerical_set))
        if missing:
            raise ValueError(
                f"Selected features are missing from the numerical pool: {missing}"
            )
        chosen = set(raw)
        selected[regime] = tuple(
            feature for feature in numerical if feature in chosen
        )

    return {
        "Global-Numeric": {"global": numerical},
        "Global-Numeric-News": {"global": (*numerical, *news)},
        "Regime-SHAP-Numeric": dict(selected),
        "Regime-SHAP-Numeric-News": {
            regime: (*selected[regime], *news) for regime in REGIMES
        },
    }


def build_fit_requests(
    *,
    base_seed: int,
    arm_features: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, tuple[FitRequest, ...]]:
    if tuple(arm_features) != ARMS:
        raise ValueError(f"arm_features must follow the registered arms: {ARMS}")
    subseeds = capacity_matched_subseeds(int(base_seed))
    requests: dict[str, tuple[FitRequest, ...]] = {}
    for arm in ARMS:
        feature_groups = arm_features[arm]
        if arm.startswith("Global-"):
            if tuple(feature_groups) != ("global",):
                raise ValueError(f"{arm} must contain one global feature group")
            requests[arm] = (
                FitRequest(
                    "global",
                    "global",
                    tuple(feature_groups["global"]),
                    int(base_seed),
                ),
            )
        else:
            if set(feature_groups) != set(REGIMES):
                raise ValueError(f"{arm} must contain all registered regimes")
            requests[arm] = tuple(
                FitRequest(
                    "regime",
                    regime,
                    tuple(feature_groups[regime]),
                    subseeds[regime],
                )
                for regime in REGIMES
            )
    flattened = [request for group in requests.values() for request in group]
    if len(flattened) != 8 or len(set(flattened)) != 8:
        raise ValueError("The integrated design must produce eight unique fits")
    return requests


def subset_aligned_regimes(
    market: pd.DataFrame,
    regime_frame: pd.DataFrame,
    *,
    split: str,
) -> np.ndarray:
    if DATE_COLUMN not in market:
        raise ValueError(f"{split} market frame is missing {DATE_COLUMN}")
    required = {DATE_COLUMN, "routing_regime"}
    missing_columns = sorted(required.difference(regime_frame.columns))
    if missing_columns:
        raise ValueError(f"{split} regimes are missing columns: {missing_columns}")
    market_dates = pd.to_datetime(
        market[DATE_COLUMN], errors="raise"
    ).dt.normalize()
    if market_dates.duplicated().any():
        raise ValueError(f"{split} market dates contain duplicates")
    regimes = regime_frame.loc[:, [DATE_COLUMN, "routing_regime"]].copy()
    regimes[DATE_COLUMN] = pd.to_datetime(
        regimes[DATE_COLUMN], errors="raise"
    ).dt.normalize()
    if regimes[DATE_COLUMN].duplicated().any():
        raise ValueError(f"{split} regime dates contain duplicates")
    indexed = regimes.set_index(DATE_COLUMN)
    missing_dates = market_dates.loc[~market_dates.isin(indexed.index)]
    if not missing_dates.empty:
        formatted = missing_dates.dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{split} has missing regime dates: {formatted[:5]}")
    labels = indexed.loc[market_dates, "routing_regime"].astype(str).to_numpy()
    unknown = sorted(set(labels).difference(REGIMES))
    if unknown:
        raise ValueError(f"{split} contains unknown regimes: {unknown}")
    return labels.astype(object)


def validate_regime_training_capacity(
    train_regimes: Sequence[str] | np.ndarray,
    *,
    window: int,
    minimum: int,
) -> dict[str, int]:
    labels = np.asarray(train_regimes, dtype=object)
    if labels.ndim != 1 or len(labels) <= window:
        raise ValueError("train_regimes do not contain enough rows for the window")
    if isinstance(window, bool) or window < 1:
        raise ValueError("window must be a positive integer")
    if isinstance(minimum, bool) or minimum < 1:
        raise ValueError("minimum must be a positive integer")
    unknown = sorted(set(labels).difference(REGIMES))
    if unknown:
        raise ValueError(f"Unknown training regimes: {unknown}")
    endpoints = labels[window:]
    counts = {
        regime: int(np.count_nonzero(endpoints == regime)) for regime in REGIMES
    }
    too_small = {regime: count for regime, count in counts.items() if count < minimum}
    if too_small:
        raise ValueError(
            f"Regime experts have fewer than {minimum} sequences: {too_small}"
        )
    return counts


def prepare_integrated_fold(
    spec: FoldSpec,
    daily_news: pd.DataFrame,
) -> dict[str, object]:
    pair = prepare_fold_pair(spec, daily_news)
    numerical = pair["technical_vmd"]
    multimodal = pair["technical_vmd_news"]
    if len(multimodal.feature_columns) - len(numerical.feature_columns) != len(
        NEWS_FEATURES
    ):
        raise ValueError("Integrated fold does not contain the eight-feature news block")
    if tuple(multimodal.feature_columns[-len(NEWS_FEATURES) :]) != NEWS_FEATURES:
        raise ValueError("News features are not the final ordered feature block")
    return {
        "Global-Numeric": numerical,
        "Global-Numeric-News": multimodal,
        "Regime-SHAP-Numeric": numerical,
        "Regime-SHAP-Numeric-News": multimodal,
    }


def _validate_fold_metrics(fold_metrics: pd.DataFrame) -> None:
    required = {
        "model",
        "fold",
        "test_year",
        "arm",
        "balanced_accuracy",
        "direction_accuracy",
        "mcc",
        "rmse",
        "mae",
    }
    missing = sorted(required.difference(fold_metrics.columns))
    if missing:
        raise ValueError(f"Fold metrics are missing columns: {missing}")
    duplicates = fold_metrics.duplicated(["model", "fold", "arm"])
    if duplicates.any():
        raise ValueError("Fold metrics contain duplicate model/fold/arm rows")
    if set(fold_metrics["arm"]) != set(ARMS):
        raise ValueError("Fold metrics do not contain exactly the registered arms")


def _direct_contrast(
    fold_metrics: pd.DataFrame,
    *,
    contrast: str,
    treatment_arm: str,
    control_arm: str,
) -> pd.DataFrame:
    keys = ["model", "fold", "test_year"]
    metric_columns = [
        "balanced_accuracy",
        "direction_accuracy",
        "mcc",
        "rmse",
        "mae",
    ]
    treatment = fold_metrics.loc[
        fold_metrics["arm"].eq(treatment_arm), [*keys, *metric_columns]
    ]
    control = fold_metrics.loc[
        fold_metrics["arm"].eq(control_arm), [*keys, *metric_columns]
    ]
    paired = treatment.merge(
        control,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_treatment", "_control"),
    )
    if len(paired) != len(treatment) or len(paired) != len(control):
        raise ValueError(f"Incomplete fold pairing for {contrast}")
    paired.insert(3, "contrast", contrast)
    paired.insert(4, "treatment_arm", treatment_arm)
    paired.insert(5, "control_arm", control_arm)
    paired["balanced_accuracy_delta_pp"] = (
        paired["balanced_accuracy_treatment"]
        - paired["balanced_accuracy_control"]
    ) * 100.0
    paired["direction_accuracy_delta_pp"] = (
        paired["direction_accuracy_treatment"]
        - paired["direction_accuracy_control"]
    ) * 100.0
    for metric in ("mcc", "rmse", "mae"):
        paired[f"{metric}_delta"] = (
            paired[f"{metric}_treatment"] - paired[f"{metric}_control"]
        )
    return paired


def build_integrated_fold_contrasts(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    _validate_fold_metrics(fold_metrics)
    direct = {
        contrast: _direct_contrast(
            fold_metrics,
            contrast=contrast,
            treatment_arm=arms[0],
            control_arm=arms[1],
        )
        for contrast, arms in DIRECT_CONTRASTS.items()
    }
    regime_news = direct["regime_pipeline_news_effect"]
    global_news = direct["global_news_effect"]
    keys = ["model", "fold", "test_year"]
    interaction = regime_news[[*keys, *METRIC_DELTAS]].merge(
        global_news[[*keys, *METRIC_DELTAS]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_regime", "_global"),
    )
    if len(interaction) != len(regime_news) or len(interaction) != len(global_news):
        raise ValueError("Incomplete fold pairing for routing_news_interaction")
    interaction.insert(3, "contrast", "routing_news_interaction")
    interaction.insert(4, "treatment_arm", "difference_of_differences")
    interaction.insert(5, "control_arm", "zero_interaction")
    for metric in METRIC_DELTAS:
        interaction[metric] = (
            interaction[f"{metric}_regime"] - interaction[f"{metric}_global"]
        )
    columns = [
        "model",
        "fold",
        "test_year",
        "contrast",
        "treatment_arm",
        "control_arm",
        *METRIC_DELTAS,
    ]
    ordered = [direct[name].loc[:, columns] for name in DIRECT_CONTRASTS]
    ordered.append(interaction.loc[:, columns])
    return pd.concat(ordered, ignore_index=True)


def integrated_fold_inference(paired: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "contrast", "fold", *METRIC_DELTAS}
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ValueError(f"Paired contrasts are missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for (model, contrast), group in paired.groupby(
        ["model", "contrast"], sort=False
    ):
        if len(group) != 4 or group["fold"].nunique() != 4:
            raise ValueError(f"{model}/{contrast} must contain four unique folds")
        for metric in METRIC_DELTAS:
            values = group[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Fold deltas contain non-finite values")
            mean = float(values.mean())
            if np.allclose(values, mean, rtol=0.0, atol=0.0):
                lower = upper = mean
            else:
                from scipy.stats import t

                standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
                margin = float(t.ppf(0.975, len(values) - 1) * standard_error)
                lower, upper = mean - margin, mean + margin
            rows.append(
                {
                    "model": model,
                    "contrast": contrast,
                    "metric": metric,
                    "outer_folds": 4,
                    "mean_delta": mean,
                    "std_delta": float(values.std(ddof=1)),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "positive_folds": int(np.count_nonzero(values > 0.0)),
                    "negative_folds": int(np.count_nonzero(values < 0.0)),
                    "zero_folds": int(np.count_nonzero(values == 0.0)),
                    "exact_sign_flip_pvalue": exact_sign_flip_pvalue(values),
                }
            )
    return pd.DataFrame(rows)


def apply_integrated_holm(inference: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "contrast", "metric", "exact_sign_flip_pvalue"}
    missing = sorted(required.difference(inference.columns))
    if missing:
        raise ValueError(f"Inference is missing columns: {missing}")
    result = inference.copy()
    result["holm_adjusted_pvalue"] = np.nan
    result["holm_family_size"] = 0
    for index in result.groupby(["contrast", "metric"]).groups.values():
        positions = list(index)
        if len(positions) != 5:
            raise ValueError("Every Holm family must contain exactly five models")
        adjusted = holm_adjust(
            result.loc[positions, "exact_sign_flip_pvalue"].to_numpy(dtype=float)
        )
        result.loc[positions, "holm_adjusted_pvalue"] = adjusted
        result.loc[positions, "holm_family_size"] = len(positions)
    return result


def validate_cell_integrity(
    metrics: pd.DataFrame,
    fit_registry: pd.DataFrame,
    predictions_by_arm: Mapping[str, pd.DataFrame],
    *,
    minimum_training_sequences: int,
) -> dict[str, object]:
    if metrics["arm"].duplicated().any() or set(metrics["arm"]) != set(ARMS):
        raise ValueError("Cell metrics must contain each registered arm exactly once")
    missing_prediction_arms = sorted(set(ARMS).difference(predictions_by_arm))
    extra_prediction_arms = sorted(set(predictions_by_arm).difference(ARMS))
    if missing_prediction_arms:
        raise ValueError(f"Cell has missing prediction arms: {missing_prediction_arms}")
    if extra_prediction_arms:
        raise ValueError(f"Cell has unexpected prediction arms: {extra_prediction_arms}")
    if len(fit_registry) != 8 or fit_registry["fit_id"].duplicated().any():
        raise ValueError("Cell fit registry must contain eight unique fits")
    counts = fit_registry["training_sequences"].to_numpy(dtype=int)
    if np.any(counts < minimum_training_sequences):
        raise ValueError(
            f"Cell contains fewer than {minimum_training_sequences} training sequences"
        )
    required_prediction_columns = {
        DATE_COLUMN,
        "Close_D",
        "y_true",
        "y_pred",
        "routing_regime",
    }
    reference: pd.DataFrame | None = None
    for arm in ARMS:
        frame = predictions_by_arm[arm].reset_index(drop=True).copy()
        missing = sorted(required_prediction_columns.difference(frame.columns))
        if missing:
            raise ValueError(f"{arm} predictions are missing columns: {missing}")
        frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="raise")
        if frame[DATE_COLUMN].duplicated().any():
            raise ValueError(f"{arm} predictions contain duplicate dates")
        numeric = frame[["Close_D", "y_true", "y_pred"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{arm} predictions contain non-finite values")
        if reference is None:
            reference = frame
            continue
        if not frame[DATE_COLUMN].equals(reference[DATE_COLUMN]):
            raise ValueError("Prediction arm dates do not align")
        for column in ("Close_D", "y_true"):
            if not np.allclose(
                frame[column].to_numpy(dtype=float),
                reference[column].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("Prediction arm targets do not align")
        if not frame["routing_regime"].astype(str).equals(
            reference["routing_regime"].astype(str)
        ):
            raise ValueError("Prediction arm regimes do not align")
    return {
        "passed": True,
        "arms": len(ARMS),
        "unique_fits": len(fit_registry),
        "test_rows": len(reference) if reference is not None else 0,
        "minimum_training_sequences": int(counts.min()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze_manifest(
    project_root: Path,
    manifest_path: Path,
) -> dict[str, object]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Freeze manifest protocol_id is incorrect")
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Freeze manifest does not contain inputs")
    for entry in inputs:
        if not isinstance(entry, dict):
            raise TypeError("Freeze manifest input entry is invalid")
        path = project_root / str(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Frozen input does not exist: {path}")
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"Frozen input size changed: {path}")
        actual = _sha256(path)
        if actual != str(entry["sha256"]):
            raise ValueError(f"Frozen input hash changed: {path}")
    return {"passed": True, "files_checked": len(inputs)}
