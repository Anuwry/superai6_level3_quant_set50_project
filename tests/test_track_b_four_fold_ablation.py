from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models import track_b_four_fold_ablation as four_fold


def test_configure_fusion_module_sets_isolated_four_fold_contract(tmp_path):
    configured = four_fold.configure_fusion_module(output_dir=tmp_path)

    assert configured.FUSION_TEST_YEARS == (2022, 2023, 2024, 2025)
    assert configured.FUSION_OUTPUT_DIR == tmp_path
    assert configured.DAILY_NEWS_FILE == four_fold.FORWARD_DAILY_NEWS_FILE
    assert configured.METRICS_FILE == tmp_path / "metrics_by_seed_fold.csv"
    assert "frozen" in configured.SENTIMENT_PREDICTION_CONTRACT


def test_run_four_fold_ablation_passes_all_years_to_paired_runner(
    monkeypatch,
    tmp_path,
):
    calls: dict[str, object] = {}
    configured = four_fold.configure_fusion_module(output_dir=tmp_path)

    def fake_run_fusion_experiment(**kwargs):
        calls.update(kwargs)
        return "done"

    monkeypatch.setattr(configured, "run_fusion_experiment", fake_run_fusion_experiment)
    monkeypatch.setattr(
        four_fold,
        "configure_fusion_module",
        lambda **_: configured,
    )

    result = four_fold.run_four_fold_ablation(
        models=("cnn",),
        seeds=(42,),
        force=True,
        output_dir=tmp_path,
    )

    assert result == "done"
    assert calls == {
        "models": ("cnn",),
        "seeds": (42,),
        "test_years": (2022, 2023, 2024, 2025),
        "force": True,
    }


def test_required_daily_news_artifact_is_not_silently_regenerated(
    monkeypatch,
    tmp_path,
):
    missing = tmp_path / "missing.csv"
    monkeypatch.setattr(four_fold, "FORWARD_DAILY_NEWS_FILE", missing)

    try:
        four_fold.require_forward_daily_news()
    except FileNotFoundError as error:
        assert str(Path(missing)) in str(error)
    else:
        raise AssertionError("Missing forward daily news should fail")


def test_build_yearly_and_period_summaries_preserve_outer_fold_grain():
    paired = pd.DataFrame(
        {
            "model": ["cnn", "cnn", "cnn", "cnn"],
            "test_year": [2023, 2023, 2024, 2024],
            "seed": [42, 123, 42, 123],
            "rmse_delta_news_minus_technical": [1.0, 3.0, -4.0, -2.0],
            "direction_accuracy_delta_pp": [2.0, 4.0, -1.0, 1.0],
            "balanced_accuracy_delta_pp": [1.0, 3.0, -2.0, 0.0],
            "mcc_delta_news_minus_technical": [0.1, 0.3, -0.2, 0.0],
            "runtime_seconds_delta_news_minus_technical": [0.1, 0.3, 0.2, 0.4],
        }
    )

    yearly, period = four_fold.build_yearly_and_period_summaries(paired)

    assert yearly["test_year"].tolist() == [2023, 2024]
    assert yearly["rmse_delta_mean"].tolist() == pytest.approx([2.0, -3.0])
    assert yearly["period"].tolist() == [
        "labelled_validation",
        "frozen_forward",
    ]
    assert yearly["balanced_accuracy_delta_pp_mean"].tolist() == pytest.approx(
        [2.0, -1.0]
    )
    assert period["period"].tolist() == [
        "frozen_forward",
        "labelled_validation",
    ]
    assert period["outer_folds"].tolist() == [1, 1]
    assert period["seeds_per_fold"].tolist() == [2, 2]


def test_configured_fusion_context_restores_module_globals(monkeypatch, tmp_path):
    configured = four_fold.configure_fusion_module(output_dir=tmp_path)
    original_output = four_fold.fusion.FUSION_OUTPUT_DIR
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"ok": [1]})

    monkeypatch.setattr(four_fold.fusion, "run_fusion_experiment", fake_runner)

    result = configured.run_fusion_experiment(models=("cnn",))

    assert result["ok"].tolist() == [1]
    assert calls == [{"models": ("cnn",)}]
    assert four_fold.fusion.FUSION_OUTPUT_DIR == original_output


def test_configured_report_rebuilds_metadata_from_all_checkpoints(
    monkeypatch,
    tmp_path,
):
    configured = four_fold.configure_fusion_module(output_dir=tmp_path)
    completed = pd.DataFrame(
        {
            "model": ["lstm", "cnn"],
            "runtime_seconds": [1.5, 2.5],
        }
    )
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        four_fold.fusion,
        "collect_completed_metrics",
        lambda: completed,
    )
    def fake_reports(metrics):
        calls["reports_metrics"] = metrics
        return "reported"

    monkeypatch.setattr(
        four_fold.fusion,
        "build_and_save_reports",
        fake_reports,
    )

    def fake_metadata(metrics, *, runtime_seconds):
        calls["metadata_metrics"] = metrics
        calls["runtime_seconds"] = runtime_seconds

    monkeypatch.setattr(
        four_fold.fusion,
        "_write_experiment_metadata",
        fake_metadata,
    )

    configured.build_and_save_reports()

    assert calls["reports_metrics"] is completed
    assert calls["metadata_metrics"] is completed
    assert calls["runtime_seconds"] is None


def test_save_yearly_and_period_summaries_writes_both_tables(tmp_path):
    paired = pd.DataFrame(
        {
            "model": ["cnn", "cnn"],
            "test_year": [2023, 2024],
            "seed": [42, 42],
            "rmse_delta_news_minus_technical": [1.0, -1.0],
            "direction_accuracy_delta_pp": [2.0, 1.0],
            "balanced_accuracy_delta_pp": [1.0, 0.5],
            "mcc_delta_news_minus_technical": [0.1, 0.05],
            "runtime_seconds_delta_news_minus_technical": [0.1, 0.2],
        }
    )
    paired.to_csv(tmp_path / "paired_deltas_by_seed_fold.csv", index=False)

    yearly, period = four_fold.save_yearly_and_period_summaries(
        output_dir=tmp_path
    )

    assert len(yearly) == 2
    assert len(period) == 2
    assert (tmp_path / four_fold.YEARLY_SUMMARY_NAME).is_file()
    assert (tmp_path / four_fold.PERIOD_SUMMARY_NAME).is_file()


def test_report_stage_rebuilds_reports_and_period_summaries(monkeypatch, tmp_path):
    calls = []

    class StubConfigured:
        def build_and_save_reports(self):
            calls.append("report")
            return "reported"

    monkeypatch.setattr(
        four_fold,
        "configure_fusion_module",
        lambda: StubConfigured(),
    )
    monkeypatch.setattr(
        four_fold,
        "save_yearly_and_period_summaries",
        lambda: calls.append("summaries"),
    )

    result = four_fold.main(["report"])

    assert result == "reported"
    assert calls == ["report", "summaries"]
