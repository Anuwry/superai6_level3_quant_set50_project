from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from models.baseline_common import DATE_COLUMN, PROJECT_ROOT, discover_folds
from models.integrated_multimodal import (
    ARMS,
    CONTRASTS,
    NEWS_FEATURES,
    apply_integrated_holm,
    build_arm_feature_sets,
    build_fit_requests,
    build_integrated_fold_contrasts,
    integrated_fold_inference,
    prepare_integrated_fold,
    subset_aligned_regimes,
    validate_cell_integrity,
    validate_regime_training_capacity,
    verify_freeze_manifest,
)
from models.track_b_forward_news import DAILY_NEWS_FILE
from models.vmd_feature_pool import FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR


def _small_arm_features() -> dict[str, dict[str, tuple[str, ...]]]:
    return build_arm_feature_sets(
        ("a", "b", "c", "d"),
        {
            "bull": ("a",),
            "sideway": ("a", "b", "c", "d"),
            "bear": ("c", "d"),
        },
        news_features=("news_x", "news_y"),
    )


def test_build_arm_feature_sets_keeps_frozen_numerical_order_and_news_block() -> None:
    features = _small_arm_features()

    assert tuple(features) == ARMS
    assert features["Global-Numeric"] == {"global": ("a", "b", "c", "d")}
    assert features["Global-Numeric-News"]["global"] == (
        "a",
        "b",
        "c",
        "d",
        "news_x",
        "news_y",
    )
    assert features["Regime-SHAP-Numeric"] == {
        "bull": ("a",),
        "sideway": ("a", "b", "c", "d"),
        "bear": ("c", "d"),
    }
    assert features["Regime-SHAP-Numeric-News"]["bear"] == (
        "c",
        "d",
        "news_x",
        "news_y",
    )

    with pytest.raises(ValueError, match="missing from the numerical pool"):
        build_arm_feature_sets(
            ("a", "b"),
            {"bull": ("missing",), "sideway": ("a",), "bear": ("b",)},
            news_features=("news",),
        )
    with pytest.raises(ValueError, match="overlap"):
        build_arm_feature_sets(
            ("a", "news"),
            {"bull": ("a",), "sideway": ("a",), "bear": ("a",)},
            news_features=("news",),
        )


def test_build_fit_requests_registers_eight_unique_fits_per_cell() -> None:
    requests = build_fit_requests(base_seed=42, arm_features=_small_arm_features())

    assert {arm: len(values) for arm, values in requests.items()} == {
        "Global-Numeric": 1,
        "Global-Numeric-News": 1,
        "Regime-SHAP-Numeric": 3,
        "Regime-SHAP-Numeric-News": 3,
    }
    flattened = [request for values in requests.values() for request in values]
    assert len(flattened) == len(set(flattened)) == 8
    for arm in ("Regime-SHAP-Numeric", "Regime-SHAP-Numeric-News"):
        assert {request.regime for request in requests[arm]} == {
            "bull",
            "sideway",
            "bear",
        }


def test_subset_aligned_regimes_uses_exact_market_dates() -> None:
    market = pd.DataFrame(
        {DATE_COLUMN: pd.to_datetime(["2020-01-02", "2020-01-03"])}
    )
    regimes = pd.DataFrame(
        {
            DATE_COLUMN: pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03"]
            ),
            "routing_regime": ["bear", "bull", "sideway"],
        }
    )

    assert subset_aligned_regimes(market, regimes, split="train").tolist() == [
        "bull",
        "sideway",
    ]
    with pytest.raises(ValueError, match="missing regime dates"):
        subset_aligned_regimes(
            market,
            regimes.iloc[:2],
            split="train",
        )
    with pytest.raises(ValueError, match="duplicate"):
        subset_aligned_regimes(
            market,
            pd.concat([regimes, regimes.iloc[[1]]], ignore_index=True),
            split="train",
        )


def _fold_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    offsets = {
        "Global-Numeric": 0.00,
        "Global-Numeric-News": 0.02,
        "Regime-SHAP-Numeric": 0.01,
        "Regime-SHAP-Numeric-News": 0.04,
    }
    for model_index in range(5):
        for fold in range(1, 5):
            for arm, offset in offsets.items():
                rows.append(
                    {
                        "model": f"model_{model_index}",
                        "fold": f"fold_{fold}",
                        "test_year": 2021 + fold,
                        "arm": arm,
                        "balanced_accuracy": 0.50 + offset,
                        "direction_accuracy": 0.51 + offset,
                        "mcc": offset,
                        "rmse": 10.0 - offset,
                        "mae": 8.0 - offset,
                    }
                )
    return pd.DataFrame(rows)


def test_contrasts_include_two_by_two_interaction_with_registered_orientation() -> None:
    paired = build_integrated_fold_contrasts(_fold_metrics())

    assert set(paired["contrast"]) == set(CONTRASTS)
    row = paired.loc[
        paired["contrast"].eq("routing_news_interaction")
    ].iloc[0]
    assert row["balanced_accuracy_delta_pp"] == pytest.approx(1.0)
    assert row["direction_accuracy_delta_pp"] == pytest.approx(1.0)
    assert row["mcc_delta"] == pytest.approx(0.01)
    assert row["rmse_delta"] == pytest.approx(-0.01)
    assert row["mae_delta"] == pytest.approx(-0.01)


def test_four_fold_inference_and_holm_adjust_exactly_five_models_per_family() -> None:
    inference = integrated_fold_inference(
        build_integrated_fold_contrasts(_fold_metrics())
    )
    adjusted = apply_integrated_holm(inference)

    assert len(adjusted) == 5 * 5 * 5
    assert adjusted.groupby(["contrast", "metric"]).size().eq(5).all()
    assert adjusted["holm_family_size"].eq(5).all()
    assert adjusted["holm_adjusted_pvalue"].between(0.0, 1.0).all()


def test_regime_capacity_fails_closed_below_registered_minimum() -> None:
    regimes = np.array(
        ["bull"] * 205 + ["sideway"] * 220 + ["bear"] * 210,
        dtype=object,
    )
    counts = validate_regime_training_capacity(
        regimes,
        window=5,
        minimum=200,
    )
    assert counts == {"bull": 200, "sideway": 220, "bear": 210}

    with pytest.raises(ValueError, match="fewer than 200"):
        validate_regime_training_capacity(
            regimes,
            window=20,
            minimum=200,
        )


def test_cell_integrity_requires_all_arms_and_eight_fit_rows() -> None:
    metrics = pd.DataFrame(
        {
            "arm": list(ARMS),
            "n_test": [2] * len(ARMS),
            "balanced_accuracy": [0.5] * len(ARMS),
        }
    )
    registry = pd.DataFrame(
        {
            "fit_id": [f"fit_{index}" for index in range(8)],
            "training_sequences": [200] * 8,
            "features": [4, 6, 1, 4, 2, 3, 6, 4],
        }
    )
    predictions = {
        arm: pd.DataFrame(
            {
                DATE_COLUMN: pd.to_datetime(["2022-01-03", "2022-01-04"]),
                "Close_D": [100.0, 101.0],
                "y_true": [101.0, 100.0],
                "y_pred": [100.5, 100.5],
                "routing_regime": ["bull", "bear"],
            }
        )
        for arm in ARMS
    }

    audit = validate_cell_integrity(
        metrics,
        registry,
        predictions,
        minimum_training_sequences=200,
    )
    assert audit["passed"] is True
    with pytest.raises(ValueError, match="missing prediction arms"):
        validate_cell_integrity(
            metrics,
            registry,
            {arm: frame for arm, frame in predictions.items() if arm != ARMS[-1]},
            minimum_training_sequences=200,
        )


def test_frozen_manifest_matches_current_inputs() -> None:
    manifest = PROJECT_ROOT / "test" / "integrated_multimodal_freeze_v1.json"
    audit = verify_freeze_manifest(PROJECT_ROOT, manifest)

    assert audit["passed"] is True
    assert audit["files_checked"] == len(
        json.loads(manifest.read_text(encoding="utf-8"))["inputs"]
    )


def test_actual_fold_preparation_has_common_dates_and_122_plus_8_features() -> None:
    spec = discover_folds(FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR)[0]
    daily_news = pd.read_csv(DAILY_NEWS_FILE)

    prepared = prepare_integrated_fold(spec, daily_news)

    assert len(prepared["Global-Numeric"].feature_columns) == 122
    assert len(prepared["Global-Numeric-News"].feature_columns) == 130
    assert tuple(prepared["Global-Numeric-News"].feature_columns[-8:]) == tuple(
        NEWS_FEATURES
    )
    assert prepared["Global-Numeric"].train[DATE_COLUMN].dt.year.min() == 2019
    pd.testing.assert_series_equal(
        prepared["Global-Numeric"].train[DATE_COLUMN],
        prepared["Global-Numeric-News"].train[DATE_COLUMN],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        prepared["Global-Numeric"].test[DATE_COLUMN],
        prepared["Global-Numeric-News"].test[DATE_COLUMN],
        check_names=False,
    )
