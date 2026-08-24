import json

import numpy as np
import pandas as pd
import pytest

import models.track_a_data as track_a_data
import models.track_a_final as track_a
from models.baseline_common import DATE_COLUMN, TARGET_COLUMN, discover_folds
from models.point_in_time_data import CONTEXT_FILE_NAME, LABEL_DATE_COLUMN


def _source_frame() -> pd.DataFrame:
    dates = pd.to_datetime(
        [f"{year}-{month:02d}-01" for year in range(2017, 2022) for month in (1, 7)]
    )
    values = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            DATE_COLUMN: dates.strftime("%Y-%m-%d"),
            LABEL_DATE_COLUMN: (
                dates + pd.offsets.BDay(1)
            ).strftime("%Y-%m-%d"),
            "Close_D": 100.0 + values,
            "Feature_A": values,
            TARGET_COLUMN: 101.0 + values,
        }
    )


def test_track_a_contract_uses_only_pretest_selection_years():
    assert track_a.SEQUENCE_WINDOWS == (1, 3, 5, 10, 20)
    assert track_a.SELECTION_YEARS == (2018, 2019, 2020, 2021)
    assert max(track_a.SELECTION_YEARS) < track_a.FIRST_OUTER_TEST_YEAR
    assert track_a.FINAL_SEEDS == (42, 123, 456, 789, 2025)
    assert list(track_a.TRACK_A_MODELS) == [
        "lstm",
        "cnn",
        "lstm_cnn",
        "lstm_attention",
        "lstm_cnn_attention",
    ]
    assert set(track_a.TRACK_A_FEATURE_SETS) == {"full_ta", "full_ta_vmd"}


def test_create_pretest_selection_folds_is_strictly_temporal(tmp_path):
    source_dir = tmp_path / "source"
    source_fold = source_dir / "fold_1"
    source_fold.mkdir(parents=True)
    frame = _source_frame()
    frame.to_csv(source_fold / "train_2017_2021.csv", index=False)
    frame.iloc[-1:].to_csv(source_fold / "test_2022.csv", index=False)

    output_dir = tmp_path / "selection"
    track_a_data.create_pretest_selection_folds(
        source_dir,
        output_dir,
        selection_years=(2018, 2019, 2020, 2021),
        first_test_year=2022,
    )

    specs = discover_folds(output_dir)
    assert [spec.test_year for spec in specs] == [2018, 2019, 2020, 2021]
    for spec in specs:
        train = pd.read_csv(spec.train_path, parse_dates=[DATE_COLUMN])
        validation = pd.read_csv(spec.test_path, parse_dates=[DATE_COLUMN])
        assert train[DATE_COLUMN].max() < validation[DATE_COLUMN].min()
        assert set(validation[DATE_COLUMN].dt.year) == {spec.test_year}
        assert spec.test_year < 2022


def test_create_pretest_selection_folds_purges_boundary_label(tmp_path):
    source_dir = tmp_path / "source"
    source_fold = source_dir / "fold_1"
    source_fold.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            DATE_COLUMN: [
                "2020-12-30",
                "2020-12-31",
                "2021-01-04",
                "2021-01-05",
            ],
            LABEL_DATE_COLUMN: [
                "2020-12-31",
                "2021-01-04",
                "2021-01-05",
                "2021-01-06",
            ],
            "Close_D": [98.0, 99.0, 100.0, 101.0],
            TARGET_COLUMN: [99.0, 100.0, 101.0, 102.0],
        }
    )
    frame.to_csv(source_fold / "train_2020_2021.csv", index=False)
    frame.iloc[-1:].to_csv(source_fold / "test_2022.csv", index=False)

    output_dir = tmp_path / "selection"
    track_a_data.create_pretest_selection_folds(
        source_dir,
        output_dir,
        selection_years=(2021,),
        first_test_year=2022,
    )

    spec = discover_folds(output_dir)[0]
    train = pd.read_csv(spec.train_path)
    audit = json.loads(
        (spec.train_path.parent / "point_in_time_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert train[DATE_COLUMN].tolist() == ["2020-12-30"]
    context = pd.read_csv(spec.train_path.parent / CONTEXT_FILE_NAME)
    assert context[DATE_COLUMN].tolist() == ["2020-12-31"]
    assert context[LABEL_DATE_COLUMN].tolist() == ["2021-01-04"]
    assert audit["removed_rows"] == 1


def test_scaled_selection_folds_fit_scaler_on_selection_train_only(tmp_path):
    source_dir = tmp_path / "source"
    fold_dir = source_dir / "fold_1"
    fold_dir.mkdir(parents=True)
    frame = _source_frame()
    train = frame[pd.to_datetime(frame[DATE_COLUMN]).dt.year < 2021]
    validation = frame[pd.to_datetime(frame[DATE_COLUMN]).dt.year == 2021]
    train.to_csv(fold_dir / "train_2017_2020.csv", index=False)
    validation.to_csv(fold_dir / "test_2021.csv", index=False)

    output_dir = tmp_path / "scaled"
    track_a_data.create_scaled_selection_folds(source_dir, output_dir)

    spec = discover_folds(output_dir)[0]
    scaled_train = pd.read_csv(spec.train_path).drop(
        columns=[DATE_COLUMN, LABEL_DATE_COLUMN]
    )
    metadata = json.loads(
        (spec.train_path.parent / "minmax_scaler.json").read_text(encoding="utf-8")
    )
    assert metadata["fit_scope"] == "selection_train_only"
    assert metadata["selection_year"] == 2021
    assert scaled_train.min().min() >= -1e-12
    assert scaled_train.max().max() <= 1.0 + 1e-12


def test_scaled_selection_folds_transform_context_without_fitting_on_it(
    tmp_path,
):
    source_dir = tmp_path / "source"
    fold_dir = source_dir / "fold_1"
    fold_dir.mkdir(parents=True)
    train = pd.DataFrame(
        {
            DATE_COLUMN: ["2020-12-29"],
            LABEL_DATE_COLUMN: ["2020-12-30"],
            "Close_D": [10.0],
            TARGET_COLUMN: [11.0],
        }
    )
    context = pd.DataFrame(
        {
            DATE_COLUMN: ["2020-12-30"],
            LABEL_DATE_COLUMN: ["2021-01-04"],
            "Close_D": [20.0],
            TARGET_COLUMN: [21.0],
        }
    )
    validation = pd.DataFrame(
        {
            DATE_COLUMN: ["2021-01-04"],
            LABEL_DATE_COLUMN: ["2021-01-05"],
            "Close_D": [30.0],
            TARGET_COLUMN: [31.0],
        }
    )
    train.to_csv(fold_dir / "train_2020_2020.csv", index=False)
    context.to_csv(fold_dir / CONTEXT_FILE_NAME, index=False)
    validation.to_csv(fold_dir / "test_2021.csv", index=False)

    output_dir = tmp_path / "scaled"
    track_a_data.create_scaled_selection_folds(source_dir, output_dir)

    scaled_context = pd.read_csv(
        output_dir / "fold_1" / CONTEXT_FILE_NAME
    )
    assert scaled_context["Close_D"].tolist() == [10.0]


def _selection_metrics() -> pd.DataFrame:
    rows = []
    direction_by_window = {
        1: (0.60, 0.50),
        3: (0.54, 0.54),
        5: (0.51, 0.51),
        10: (0.50, 0.50),
        20: (0.49, 0.49),
    }
    for window, feature_scores in direction_by_window.items():
        for feature_set, score in zip(
            ("full_ta", "full_ta_vmd"),
            feature_scores,
            strict=True,
        ):
            for fold_index, year in enumerate(track_a.SELECTION_YEARS, start=1):
                rows.append(
                    {
                        "stage": "window_selection",
                        "model": "lstm",
                        "feature_set": feature_set,
                        "sequence_window": window,
                        "seed": 42,
                        "fold": f"fold_{fold_index}",
                        "test_year": year,
                        "rmse": 10.0 + window,
                        "mae": 9.0 + window,
                        "mape": 1.0,
                        "r2": 0.8,
                        "direction_accuracy": score,
                        "balanced_accuracy": score,
                        "mcc": 2.0 * score - 1.0,
                        "direction_coverage": 1.0,
                        "runtime_seconds": 2.0,
                    }
                )
    return pd.DataFrame(rows)


def test_select_locked_windows_uses_symmetric_pretest_validation():
    locked, summary = track_a.select_locked_windows(_selection_metrics())

    assert len(summary) == len(track_a.SEQUENCE_WINDOWS)
    assert locked.loc[0, "model"] == "lstm"
    assert locked.loc[0, "selected_sequence_window"] == 1
    assert locked.loc[0, "selection_feature_sets"] == 2
    assert locked.loc[0, "selection_years"] == 4
    assert locked.loc[0, "selection_direction_accuracy_mean"] == pytest.approx(0.55)


def test_select_locked_windows_uses_balanced_accuracy_as_primary_metric():
    metrics = _selection_metrics()
    metrics.loc[
        metrics["sequence_window"] == 1,
        "balanced_accuracy",
    ] = 0.49
    metrics.loc[
        metrics["sequence_window"] == 3,
        "balanced_accuracy",
    ] = 0.60

    locked, _ = track_a.select_locked_windows(metrics)

    assert locked.loc[0, "selected_sequence_window"] == 3
    assert locked.loc[0, "selection_balanced_accuracy_mean"] == pytest.approx(
        0.60
    )


def test_select_locked_windows_rejects_outer_test_data():
    metrics = _selection_metrics()
    metrics.loc[0, "test_year"] = track_a.FIRST_OUTER_TEST_YEAR

    with pytest.raises(ValueError, match="outer test"):
        track_a.select_locked_windows(metrics)


def test_exact_sign_flip_pvalue_is_two_sided_and_exact():
    assert track_a.exact_sign_flip_pvalue(np.ones(4)) == pytest.approx(0.125)
    assert track_a.exact_sign_flip_pvalue(np.array([1.0, -1.0])) == 1.0


def _final_metrics() -> pd.DataFrame:
    rows = []
    for seed in (42, 123):
        for fold_index, test_year in enumerate((2022, 2023), start=1):
            for feature_set, rmse, direction in (
                ("full_ta", 12.0 + fold_index, 0.50),
                ("full_ta_vmd", 10.0 + fold_index, 0.55),
            ):
                rows.append(
                    {
                        "stage": "outer_test",
                        "model": "lstm",
                        "feature_set": feature_set,
                        "sequence_window": 3,
                        "seed": seed,
                        "fold": f"fold_{fold_index}",
                        "test_year": test_year,
                        "rmse": rmse,
                        "mae": rmse - 1.0,
                        "mape": 1.0,
                        "r2": 0.8,
                        "direction_accuracy": direction,
                        "balanced_accuracy": direction,
                        "mcc": 2.0 * direction - 1.0,
                        "direction_coverage": 1.0,
                        "runtime_seconds": 5.0,
                    }
                )
    return pd.DataFrame(rows)


def test_build_paired_deltas_matches_same_model_seed_fold_and_window():
    paired = track_a.build_paired_deltas(_final_metrics())

    assert len(paired) == 4
    assert paired["rmse_delta_vmd_minus_full_ta"].eq(-2.0).all()
    assert np.allclose(paired["direction_accuracy_delta_pp"], 5.0)
    assert np.allclose(paired["balanced_accuracy_delta_pp"], 5.0)
    assert np.allclose(paired["mcc_delta_vmd_minus_full_ta"], 0.1)
    assert paired["sequence_window"].eq(3).all()


def test_summarize_final_results_uses_fold_means_for_paper_metrics():
    performance, paired_summary = track_a.summarize_final_results(_final_metrics())

    assert len(performance) == 2
    vmd = performance[performance["feature_set"] == "full_ta_vmd"].iloc[0]
    comparison = paired_summary.iloc[0]
    assert vmd["direction_accuracy_mean"] == pytest.approx(0.55)
    assert vmd["rmse_mean"] == pytest.approx(11.5)
    assert comparison["rmse_delta_mean"] == pytest.approx(-2.0)
    assert comparison["direction_accuracy_delta_pp_mean"] == pytest.approx(5.0)
    assert comparison["paired_outer_folds"] == 2


def test_compact_paper_table_keeps_only_paper_facing_columns():
    performance, paired_summary = track_a.summarize_final_results(_final_metrics())

    compact = track_a.build_compact_paper_table(
        performance,
        paired_summary,
    )

    assert compact.columns.tolist() == [
        "model",
        "selected_sequence_window",
        "full_ta_rmse_mean",
        "full_ta_rmse_std",
        "vmd_rmse_mean",
        "vmd_rmse_std",
        "rmse_delta_vmd_minus_full_ta",
        "rmse_delta_ci95_lower",
        "rmse_delta_ci95_upper",
        "rmse_exact_sign_flip_pvalue",
        "full_ta_direction_accuracy_mean",
        "full_ta_direction_accuracy_std",
        "vmd_direction_accuracy_mean",
        "vmd_direction_accuracy_std",
        "direction_accuracy_delta_pp",
        "direction_accuracy_delta_pp_ci95_lower",
        "direction_accuracy_delta_pp_ci95_upper",
        "direction_exact_sign_flip_pvalue",
        "full_ta_balanced_accuracy_mean",
        "full_ta_balanced_accuracy_std",
        "vmd_balanced_accuracy_mean",
        "vmd_balanced_accuracy_std",
        "balanced_accuracy_delta_pp",
        "balanced_accuracy_delta_pp_ci95_lower",
        "balanced_accuracy_delta_pp_ci95_upper",
        "balanced_accuracy_exact_sign_flip_pvalue",
        "full_ta_mcc_mean",
        "full_ta_mcc_std",
        "vmd_mcc_mean",
        "vmd_mcc_std",
        "mcc_delta_vmd_minus_full_ta",
        "full_ta_direction_coverage_mean",
        "vmd_direction_coverage_mean",
        "full_ta_runtime_seconds_mean",
        "vmd_runtime_seconds_mean",
        "paired_outer_folds",
        "seeds_per_fold",
    ]
    assert compact.loc[0, "selected_sequence_window"] == 3


def test_runtime_summary_separates_selection_and_outer_test():
    selection = _selection_metrics()
    final = _final_metrics()

    runtime = track_a.build_runtime_summary(selection, final)

    assert set(runtime["stage"]) == {"window_selection", "outer_test"}
    assert runtime["runtime_seconds_total"].sum() == pytest.approx(
        selection["runtime_seconds"].sum() + final["runtime_seconds"].sum()
    )
    assert (runtime["completed_fits"] > 0).all()


def test_model_predictor_forwards_requested_seed(monkeypatch):
    recorded = {}

    def fake_predictor(fold, sequence_length, random_seed):
        recorded.update(
            {
                "fold": fold,
                "sequence_length": sequence_length,
                "random_seed": random_seed,
            }
        )
        return np.array([1.0])

    monkeypatch.setattr(track_a, "predict_cnn_fold", fake_predictor)

    prediction = track_a.predict_model(
        "cnn",
        object(),
        sequence_window=5,
        seed=123,
    )

    assert prediction.tolist() == [1.0]
    assert recorded["sequence_length"] == 5
    assert recorded["random_seed"] == 123


def test_completed_track_a_outputs_cover_selection_and_paired_outer_test():
    selection = pd.read_csv(track_a.SELECTION_METRICS_FILE)
    locked = pd.read_csv(track_a.LOCKED_WINDOWS_FILE)
    final = pd.read_csv(track_a.FINAL_METRICS_FILE)
    paired = pd.read_csv(track_a.PAIRED_DELTAS_FILE)
    performance = pd.read_csv(track_a.FINAL_PERFORMANCE_FILE)
    paired_summary = pd.read_csv(track_a.PAIRED_SUMMARY_FILE)
    compact = pd.read_csv(track_a.PAPER_COMPACT_TABLE_FILE)
    runtime = pd.read_csv(track_a.RUNTIME_SUMMARY_FILE)

    assert len(selection) == 5 * 2 * 5 * 4
    assert len(locked) == 5
    assert len(final) == 5 * 2 * 5 * 4
    assert len(paired) == 5 * 5 * 4
    assert len(performance) == 5 * 2
    assert len(paired_summary) == 5
    assert len(compact) == 5
    assert len(runtime) == (5 * 2 * 5) + (5 * 2)
    assert selection["test_year"].max() < track_a.FIRST_OUTER_TEST_YEAR
    assert set(final["test_year"]) == {2022, 2023, 2024, 2025}
    assert set(final["seed"]) == set(track_a.FINAL_SEEDS)
    assert np.isfinite(
        selection.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    ).all()
    assert np.isfinite(
        final.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    ).all()
    assert (selection["runtime_seconds"] > 0.0).all()
    assert (final["runtime_seconds"] > 0.0).all()
    assert runtime["completed_fits"].sum() == len(selection) + len(final)
