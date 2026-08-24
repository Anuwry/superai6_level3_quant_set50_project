import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import models.track_b_fusion as fusion
from models.baseline_common import (
    CLOSE_COLUMN,
    DATE_COLUMN,
    TARGET_COLUMN,
    FoldData,
    FoldSpec,
)
from models.point_in_time_data import LABEL_DATE_COLUMN
from models.track_b_data import DAILY_FEATURE_COLUMNS
from models.track_b_fusion import (
    FUSION_TEST_YEARS,
    NEWS_START_YEAR,
    build_fusion_paper_table,
    build_paired_fusion_summary,
    join_news_features,
    restrict_to_news_era,
)


def _market_frame(dates: list[str]) -> pd.DataFrame:
    values = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            DATE_COLUMN: pd.to_datetime(dates),
            CLOSE_COLUMN: 100.0 + values,
            "technical_feature": values,
            TARGET_COLUMN: 101.0 + values,
        }
    )


def _daily_news(dates: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"date": pd.to_datetime(dates)})
    for index, column in enumerate(DAILY_FEATURE_COLUMNS, start=1):
        frame[column] = float(index)
    frame["article_count"] = frame["article_count"].astype(int)
    frame["ticker_mention_count"] = frame["ticker_mention_count"].astype(int)
    frame["news_available"] = 1
    return frame


def test_track_b_fusion_contract_is_locked_to_common_oos_news_period():
    assert NEWS_START_YEAR == 2019
    assert FUSION_TEST_YEARS == (2022, 2023)


def test_restrict_to_news_era_keeps_same_rows_for_paired_ablation():
    train = _market_frame(["2018-12-28", "2019-01-02", "2021-12-30"])
    test = _market_frame(["2022-01-04", "2022-01-05"])

    restricted_train, restricted_test = restrict_to_news_era(train, test)

    assert restricted_train[DATE_COLUMN].dt.year.tolist() == [2019, 2021]
    assert restricted_test[DATE_COLUMN].tolist() == test[DATE_COLUMN].tolist()
    assert restricted_train[DATE_COLUMN].max() < restricted_test[DATE_COLUMN].min()


def test_join_news_features_adds_only_locked_features_without_missing_values():
    market = _market_frame(["2019-01-02", "2019-01-03"])
    news = _daily_news(["2019-01-02", "2019-01-03"])

    fused = join_news_features(market, news)

    assert fused[DATE_COLUMN].tolist() == market[DATE_COLUMN].tolist()
    assert fused.columns.tolist() == [
        DATE_COLUMN,
        CLOSE_COLUMN,
        "technical_feature",
        *DAILY_FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]
    assert not fused.isna().any().any()
    assert fused.loc[0, "news_sentiment_mean"] == pytest.approx(1.0)


def test_join_news_features_rejects_incomplete_or_duplicate_daily_calendar():
    market = _market_frame(["2019-01-02", "2019-01-03"])
    incomplete = _daily_news(["2019-01-02"])
    duplicate = pd.concat([_daily_news(["2019-01-02"]), _daily_news(["2019-01-02"])])

    with pytest.raises(ValueError, match="cover every market date"):
        join_news_features(market, incomplete)
    with pytest.raises(ValueError, match="duplicate"):
        join_news_features(market.iloc[:1], duplicate)


def _fusion_metrics() -> pd.DataFrame:
    rows = []
    for model in ("lstm", "cnn"):
        for seed in (42, 123):
            for fold, year in (("fold_1", 2022), ("fold_2", 2023)):
                for feature_set, rmse, direction in (
                    ("technical_vmd", 10.0, 0.50),
                    ("technical_vmd_news", 9.0, 0.55),
                ):
                    rows.append(
                        {
                            "model": model,
                            "feature_set": feature_set,
                            "sequence_window": 5,
                            "seed": seed,
                            "fold": fold,
                            "test_year": year,
                            "rmse": rmse,
                            "mae": rmse - 1.0,
                            "mape": 1.0,
                            "r2": 0.8,
                            "direction_accuracy": direction,
                            "balanced_accuracy": direction - 0.01,
                            "mcc": 2.0 * direction - 1.0,
                            "direction_coverage": 1.0,
                            "runtime_seconds": 2.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_build_paired_fusion_summary_matches_model_seed_and_fold():
    performance, paired, summary = build_paired_fusion_summary(_fusion_metrics())

    assert len(performance) == 4
    assert len(paired) == 8
    assert len(summary) == 2
    assert paired["rmse_delta_news_minus_technical"].eq(-1.0).all()
    assert np.allclose(paired["direction_accuracy_delta_pp"], 5.0)
    assert np.allclose(paired["balanced_accuracy_delta_pp"], 5.0)
    assert np.allclose(paired["mcc_delta_news_minus_technical"], 0.1)
    assert summary["paired_runs"].eq(4).all()
    assert summary["rmse_delta_mean"].eq(-1.0).all()
    assert np.allclose(summary["direction_accuracy_delta_pp_mean"], 5.0)


def test_fusion_paper_table_keeps_only_paper_facing_columns():
    performance, _, summary = build_paired_fusion_summary(_fusion_metrics())

    paper = build_fusion_paper_table(performance, summary)

    assert paper.columns.tolist() == [
        "model",
        "selected_sequence_window",
        "technical_rmse_mean",
        "news_rmse_mean",
        "rmse_delta_news_minus_technical",
        "rmse_delta_ci95_lower",
        "rmse_delta_ci95_upper",
        "rmse_exact_sign_flip_pvalue",
        "technical_direction_accuracy_mean",
        "news_direction_accuracy_mean",
        "direction_accuracy_delta_pp",
        "direction_accuracy_delta_pp_ci95_lower",
        "direction_accuracy_delta_pp_ci95_upper",
        "direction_exact_sign_flip_pvalue",
        "technical_balanced_accuracy_mean",
        "news_balanced_accuracy_mean",
        "balanced_accuracy_delta_pp",
        "balanced_accuracy_delta_pp_ci95_lower",
        "balanced_accuracy_delta_pp_ci95_upper",
        "balanced_accuracy_exact_sign_flip_pvalue",
        "technical_mcc_mean",
        "news_mcc_mean",
        "mcc_delta_news_minus_technical",
        "technical_direction_coverage_mean",
        "news_direction_coverage_mean",
        "technical_runtime_seconds_mean",
        "news_runtime_seconds_mean",
        "paired_outer_folds",
        "seeds_per_fold",
    ]
    assert len(paper) == 2


def test_restrict_and_join_reject_invalid_market_and_news_values():
    valid = _market_frame(["2019-01-02"])
    missing_column = valid.drop(columns=CLOSE_COLUMN)
    duplicate_market = pd.concat([valid, valid], ignore_index=True)
    non_finite_news = _daily_news(["2019-01-02"])
    non_finite_news.loc[0, "news_sentiment_mean"] = np.inf

    with pytest.raises(ValueError, match="missing columns"):
        restrict_to_news_era(missing_column, valid)
    with pytest.raises(ValueError, match="duplicate dates"):
        restrict_to_news_era(duplicate_market, valid)
    with pytest.raises(ValueError, match="non-finite"):
        join_news_features(valid, non_finite_news)


def test_create_daily_news_features_writes_complete_trading_calendar(
    tmp_path, monkeypatch
):
    prediction_path = tmp_path / "predictions.csv"
    output_path = tmp_path / "daily.csv"
    pd.DataFrame(
        {
            "article_id": ["a1"],
            "date": ["2022-01-03"],
            "ticker": ["AAA"],
            "sentiment_score": [0.8],
            "predicted_label": ["positive"],
        }
    ).to_csv(prediction_path, index=False)
    monkeypatch.setattr(
        fusion,
        "_market_dates_for_fusion",
        lambda source_dir: pd.DatetimeIndex(["2022-01-03", "2022-01-04"]),
    )

    daily = fusion.create_daily_news_features(
        prediction_path,
        output_path,
        source_dir=tmp_path,
    )

    assert output_path.is_file()
    assert daily["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2022-01-03",
        "2022-01-04",
    ]
    assert daily["news_available"].tolist() == [0, 1]


def _fold_spec(tmp_path: Path) -> FoldSpec:
    train_path = tmp_path / "train_2018_2021.csv"
    test_path = tmp_path / "test_2022.csv"
    _market_frame(["2018-12-28", "2019-01-02", "2021-12-30"]).to_csv(
        train_path, index=False
    )
    _market_frame(["2022-01-04", "2022-01-05"]).to_csv(test_path, index=False)
    return FoldSpec(
        fold="fold_1",
        train_path=train_path,
        test_path=test_path,
        train_start_year=2018,
        train_end_year=2021,
        test_year=2022,
    )


def test_prepare_fold_pair_keeps_identical_dates_and_adds_news_only_to_fusion(
    tmp_path,
):
    spec = _fold_spec(tmp_path)
    daily = _daily_news(["2019-01-02", "2021-12-30", "2022-01-04", "2022-01-05"])

    folds = fusion.prepare_fold_pair(spec, daily)

    technical = folds["technical_vmd"]
    news = folds["technical_vmd_news"]
    assert technical.train[DATE_COLUMN].tolist() == news.train[DATE_COLUMN].tolist()
    assert technical.test[DATE_COLUMN].tolist() == news.test[DATE_COLUMN].tolist()
    assert set(DAILY_FEATURE_COLUMNS).isdisjoint(technical.feature_columns)
    assert set(DAILY_FEATURE_COLUMNS) <= set(news.feature_columns)
    assert technical.spec.train_start_year == 2019


def test_scaled_fold_casts_integer_context_features_before_fractional_scaling(
    tmp_path,
):
    spec = _fold_spec(tmp_path)
    daily = _daily_news(
        ["2019-01-02", "2020-01-02", "2021-12-30", "2022-01-04"]
    )
    daily["article_count"] = [0, 7, 1, 2]
    train = join_news_features(
        _market_frame(["2019-01-02", "2020-01-02"]),
        daily,
    )
    test = join_news_features(_market_frame(["2022-01-04"]), daily)
    context = join_news_features(_market_frame(["2021-12-30"]), daily)
    fold = fusion._fold_data(spec, train, test, context)

    scaled, _ = fusion._scaled_fold(fold)

    assert scaled.context is not None
    assert scaled.context["article_count"].iloc[0] == pytest.approx(1.0 / 7.0)
    assert pd.api.types.is_float_dtype(scaled.context["article_count"])


def test_restrict_to_news_era_rejects_label_observed_on_test_start():
    train = _market_frame(["2019-01-02", "2021-12-30"])
    train.insert(
        1,
        LABEL_DATE_COLUMN,
        pd.to_datetime(["2019-01-03", "2022-01-04"]),
    )
    test = _market_frame(["2022-01-04"])
    test.insert(
        1,
        LABEL_DATE_COLUMN,
        pd.to_datetime(["2022-01-05"]),
    )

    with pytest.raises(ValueError, match="labels observed"):
        restrict_to_news_era(train, test)


def test_run_configuration_is_resumable_and_writes_metrics(tmp_path, monkeypatch):
    spec = _fold_spec(tmp_path)
    train, test = restrict_to_news_era(
        pd.read_csv(spec.train_path, parse_dates=[DATE_COLUMN]),
        pd.read_csv(spec.test_path, parse_dates=[DATE_COLUMN]),
    )
    fold = FoldData(
        spec=FoldSpec(
            fold="fold_1",
            train_path=spec.train_path,
            test_path=spec.test_path,
            train_start_year=2019,
            train_end_year=2021,
            test_year=2022,
        ),
        train=train,
        test=test,
        feature_columns=[CLOSE_COLUMN, "technical_feature"],
    )
    monkeypatch.setattr(fusion, "FUSION_OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(fusion, "_set_tensorflow_runtime", lambda seed: None)
    calls = []

    def fake_predict(model, scaled_fold, *, sequence_window, seed):
        calls.append((model, sequence_window, seed))
        return scaled_fold.test[TARGET_COLUMN].to_numpy(dtype=float)

    monkeypatch.setattr(fusion, "predict_model", fake_predict)

    first = fusion.run_configuration("cnn", "technical_vmd", 5, 42, fold)
    second = fusion.run_configuration("cnn", "technical_vmd", 5, 42, fold)

    assert first == second
    assert len(calls) == 1
    assert first["rmse"] == pytest.approx(0.0)
    assert first["runtime_seconds"] > 0
    assert list((tmp_path / "outputs").glob("**/metrics.json"))


def test_run_fusion_experiment_iterates_paired_arms_and_saves_metrics(
    tmp_path, monkeypatch
):
    daily_path = tmp_path / "daily.csv"
    _daily_news(["2022-01-04"]).to_csv(daily_path, index=False)
    monkeypatch.setattr(fusion, "DAILY_NEWS_FILE", daily_path)
    monkeypatch.setattr(fusion, "FUSION_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(fusion, "METRICS_FILE", tmp_path / "metrics.csv")
    monkeypatch.setattr(fusion, "_selected_windows", lambda: {"cnn": 5})
    spec = FoldSpec(
        "fold_1",
        tmp_path / "train.csv",
        tmp_path / "test.csv",
        2019,
        2021,
        2022,
    )
    monkeypatch.setattr(fusion, "_selected_specs", lambda test_years: [spec])
    monkeypatch.setattr(
        fusion,
        "prepare_fold_pair",
        lambda selected_spec, daily: {
            "technical_vmd": object(),
            "technical_vmd_news": object(),
        },
    )

    def fake_run(model, feature_set, window, seed, fold, force=False):
        return {
            "model": model,
            "feature_set": feature_set,
            "sequence_window": window,
            "seed": seed,
            "fold": "fold_1",
            "test_year": 2022,
            "rmse": 1.0,
            "mae": 1.0,
            "mape": 1.0,
            "r2": 0.0,
            "direction_accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "mcc": 0.0,
            "direction_coverage": 1.0,
            "runtime_seconds": 0.1,
        }

    monkeypatch.setattr(fusion, "run_configuration", fake_run)
    monkeypatch.setattr(fusion, "build_and_save_reports", lambda metrics: None)
    monkeypatch.setattr(
        fusion,
        "_write_experiment_metadata",
        lambda metrics, runtime_seconds: None,
    )

    result = fusion.run_fusion_experiment(
        models=("cnn",), seeds=(42,), test_years=(2022,)
    )

    assert len(result) == 2
    assert set(result["feature_set"]) == set(fusion.FEATURE_SETS)
    assert (tmp_path / "metrics.csv").is_file()


def test_collect_and_save_reports_reads_checkpoint_metrics(tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    for index, row in _fusion_metrics().iterrows():
        path = run_root / f"run_{index}" / "metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row.to_dict()), encoding="utf-8")
    monkeypatch.setattr(fusion, "FUSION_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(fusion, "METRICS_FILE", tmp_path / "metrics.csv")
    monkeypatch.setattr(fusion, "PERFORMANCE_FILE", tmp_path / "performance.csv")
    monkeypatch.setattr(fusion, "PAIRED_FILE", tmp_path / "paired.csv")
    monkeypatch.setattr(fusion, "PAIRED_SUMMARY_FILE", tmp_path / "summary.csv")
    monkeypatch.setattr(fusion, "PAPER_TABLE_FILE", tmp_path / "paper.csv")

    completed = fusion.collect_completed_metrics()
    performance, paired, summary, paper = fusion.build_and_save_reports()

    assert len(completed) == 16
    assert len(performance) == 4
    assert len(paired) == 8
    assert len(summary) == 2
    assert len(paper) == 2
    assert (tmp_path / "paper.csv").is_file()


def test_write_experiment_metadata_records_runtime_and_contract(tmp_path, monkeypatch):
    metadata_path = tmp_path / "metadata.json"
    monkeypatch.setattr(fusion, "RUN_METADATA_FILE", metadata_path)
    monkeypatch.setattr(
        fusion,
        "_selected_windows",
        lambda: {model: 5 for model in fusion.TRACK_A_MODELS},
    )
    metrics = _fusion_metrics().iloc[:4].copy()

    fusion._write_experiment_metadata(metrics, runtime_seconds=12.5)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert payload["news_start_year"] == 2019
    assert payload["command_runtime_seconds"] == pytest.approx(12.5)
    assert payload["completed_fits"] == 4
    assert payload["scaling"].endswith("train only")


def test_main_prepare_and_report_routes_without_running_models(
    monkeypatch,
):
    prepared = pd.DataFrame({"value": [1]})
    reported = (prepared,)
    monkeypatch.setattr(fusion, "create_daily_news_features", lambda: prepared)
    monkeypatch.setattr(fusion, "build_and_save_reports", lambda: reported)

    monkeypatch.setattr(sys, "argv", ["track_b_fusion.py", "prepare"])
    assert fusion.main() is prepared
    monkeypatch.setattr(sys, "argv", ["track_b_fusion.py", "report"])
    assert fusion.main() is reported
