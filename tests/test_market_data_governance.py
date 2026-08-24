from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import models.market_data_governance as governance
from models.market_data_governance import (
    LicensePolicy,
    align_causal_frequencies,
    audit_market_frame,
    audit_repository_distribution,
    load_vendor_market_csv,
    parse_volume,
    prepare_frequency,
    restrict_to_reference_dates,
    run_market_data_governance,
    validate_access_policy,
)


def _market_frame(
    dates: list[str],
    closes: list[float],
    *,
    change_pct: list[float] | None = None,
) -> pd.DataFrame:
    values = pd.Series(closes, dtype=float)
    calculated_change = values.pct_change().mul(100).fillna(0.0)
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Close": values,
            "Open": values,
            "High": values + 1.0,
            "Low": values - 1.0,
            "Volume": 1_000_000.0,
            "Change_pct": (
                change_pct if change_pct is not None else calculated_change
            ),
        }
    )


def _write_vendor_csv(
    path: Path,
    rows: list[list[object]],
) -> None:
    pd.DataFrame(
        rows,
        columns=["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"],
    ).to_csv(path, index=False)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.25K", 1_250.0),
        ("2.5M", 2_500_000.0),
        ("3B", 3_000_000_000.0),
        ("1,234", 1_234.0),
    ],
)
def test_parse_volume_expands_vendor_suffixes(raw: str, expected: float) -> None:
    assert parse_volume(raw) == expected


def test_parse_volume_rejects_missing_vendor_volume() -> None:
    with pytest.raises(ValueError, match="missing volume"):
        parse_volume("-")


def test_load_vendor_market_csv_uses_explicit_month_day_year(tmp_path: Path) -> None:
    source = tmp_path / "set100_daily.csv"
    _write_vendor_csv(
        source,
        [
            ["12/31/2024", "2,100.00", "2,090", "2,110", "2,080", "2B", "1.00%"],
            ["01/04/2012", "1,000.00", "995", "1,005", "990", "1.2M", "0.50%"],
        ],
    )

    result = load_vendor_market_csv(source)

    assert result["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2012-01-04",
        "2024-12-31",
    ]
    assert result.loc[0, "Close"] == 1_000.0
    assert result.loc[1, "Volume"] == 2_000_000_000.0


def test_load_vendor_market_csv_supports_thai_set50_headers(tmp_path: Path) -> None:
    source = tmp_path / "set50_daily.csv"
    pd.DataFrame(
        [["01/04/2012", "726.99", "727.22", "731.54", "725.16", "684.66K", "1.20%"]],
        columns=[
            "วันเดือนปี",
            "ล่าสุด",
            "ราคาเปิด",
            "สูงสุด",
            "ต่ำสุด",
            "ปริมาณ",
            "% เปลี่ยน",
        ],
    ).to_csv(source, index=False)

    result = load_vendor_market_csv(source)

    assert result.loc[0, "Date"] == pd.Timestamp("2012-01-04")
    assert result.loc[0, "Volume"] == 684_660.0


def test_prepare_frequency_uses_next_observed_close_as_target() -> None:
    raw = _market_frame(
        ["2021-12-30", "2022-01-04", "2022-01-05"],
        [100.0, 103.0, 101.0],
    )

    result = prepare_frequency(raw)

    assert result["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2021-12-30",
        "2022-01-04",
    ]
    assert result["Target_Next_Close"].tolist() == [103.0, 101.0]


def test_align_causal_frequencies_uses_only_completed_periods() -> None:
    daily = _market_frame(
        ["2024-01-08", "2024-01-09", "2024-02-01", "2024-02-02"],
        [100.0, 101.0, 102.0, 103.0],
    )
    weekly = _market_frame(
        ["2023-12-31", "2024-01-07", "2024-01-14"],
        [80.0, 90.0, 95.0],
    )
    monthly = _market_frame(
        ["2023-12-01", "2024-01-01", "2024-02-01"],
        [60.0, 70.0, 75.0],
    )

    result = align_causal_frequencies(daily, weekly, monthly)

    january = result.loc[result["Date"] == pd.Timestamp("2024-01-08")].iloc[0]
    february = result.loc[result["Date"] == pd.Timestamp("2024-02-01")].iloc[0]
    assert january["Close_W"] == 80.0
    assert january["Close_M"] == 60.0
    assert february["Close_W"] == 90.0
    assert february["Close_M"] == 70.0
    assert result.iloc[-1]["Target_Next_Close"] == 103.0


def test_restrict_to_reference_dates_uses_intersection_and_reports_coverage(
    tmp_path: Path,
) -> None:
    aligned = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "Close": [101.0, 102.0, 103.0],
        }
    )
    reference_path = tmp_path / "set50_reference.csv"
    pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            )
        }
    ).to_csv(reference_path, index=False)

    common, coverage = restrict_to_reference_dates(aligned, reference_path)

    assert common["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]
    assert coverage["required_dates"] == 4
    assert coverage["matched_dates"] == 3
    assert coverage["missing_dates"] == 1
    assert coverage["coverage"] == 0.75
    assert coverage["missing_date_sample"] == ["2024-01-02"]


def test_audit_repository_distribution_reports_tracked_restricted_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_git = "C:/Program Files/Git/cmd/git.exe"
    monkeypatch.setattr(governance.shutil, "which", lambda executable: resolved_git)
    commands: list[list[str]] = []

    class GitResult:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(command: list[str], **kwargs: object) -> GitResult:
        del kwargs
        commands.append(command)
        paths = (
            "data-prepared/SET50_aligned.csv\n"
            "data-raw/SET50_days.csv\n"
        )
        if "ls-files" in command or "--name-only" in command:
            return GitResult(paths)
        return GitResult("abc123|2026-06-22T18:05:02+07:00\n")

    monkeypatch.setattr(governance.subprocess, "run", fake_run)

    audit, summary = audit_repository_distribution(tmp_path)

    assert audit["path"].tolist() == [
        "data-prepared/SET50_aligned.csv",
        "data-raw/SET50_days.csv",
    ]
    assert summary["tracked_restricted_file_count"] == 2
    assert (
        summary["repository_distribution_gate"]
        == "REVIEW_REQUIRED_TRACKED_RESTRICTED_DATA"
    )
    assert summary["history_restricted_file_count"] == 2
    assert (
        summary["git_history_audit"]
        == "REVIEW_REQUIRED_RESTRICTED_DATA_IN_HISTORY"
    )
    assert audit["history_commit_count"].tolist() == [1, 1]
    assert commands
    assert {command[0] for command in commands} == {resolved_git}


def test_audit_market_frame_flags_ohlc_anomaly_without_repair() -> None:
    frame = _market_frame(
        ["2025-12-10", "2025-12-11"],
        [837.62, 828.42],
        change_pct=[0.0, -1.10],
    )
    frame.loc[1, ["Open", "High", "Low"]] = [840.26, 839.88, 825.54]

    checks, anomalies = audit_market_frame(
        frame,
        dataset_id="set50_daily",
        frequency="daily",
    )

    ohlc_check = checks.set_index("check_id").loc["ohlc_containment"]
    assert ohlc_check["status"] == "WARN"
    assert anomalies["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2025-12-11"
    ]
    assert anomalies.loc[0, "action"] == "retain_raw_pending_verification"


def test_audit_market_frame_fails_inconsistent_change_percent() -> None:
    frame = _market_frame(
        ["2024-01-02", "2024-01-03"],
        [100.0, 110.0],
        change_pct=[0.0, 1.0],
    )

    checks, _ = audit_market_frame(
        frame,
        dataset_id="set100_daily",
        frequency="daily",
    )

    change_check = checks.set_index("check_id").loc[
        "percent_change_close_to_close"
    ]
    assert change_check["status"] == "FAIL"


def test_run_governance_prepares_set100_and_writes_release_gates(
    tmp_path: Path,
) -> None:
    raw_set50 = tmp_path / "set50"
    raw_set100 = tmp_path / "set100"
    output = tmp_path / "governance"
    prepared = tmp_path / "prepared"
    outer_folds = tmp_path / "outer"
    selection_folds = tmp_path / "selection"
    raw_set50.mkdir()
    raw_set100.mkdir()

    daily_dates = pd.bdate_range("2016-11-01", "2025-01-08")
    daily_rows = [
        [
            date.strftime("%m/%d/%Y"),
            f"{1_000 + index:.2f}",
            f"{999 + index:.2f}",
            f"{1_001 + index:.2f}",
            f"{998 + index:.2f}",
            "1M",
            f"{(100 / (999 + index)):.2f}%" if index else "0.00%",
        ]
        for index, date in enumerate(daily_dates)
    ]
    weekly_dates = pd.date_range("2016-10-30", "2025-01-05", freq="W-SUN")
    weekly_rows = [
        [
            date.strftime("%m/%d/%Y"),
            f"{900 + index:.2f}",
            f"{899 + index:.2f}",
            f"{901 + index:.2f}",
            f"{898 + index:.2f}",
            "5M",
            f"{(100 / (899 + index)):.2f}%" if index else "0.00%",
        ]
        for index, date in enumerate(weekly_dates)
    ]
    monthly_dates = pd.date_range("2016-10-01", "2025-01-01", freq="MS")
    monthly_rows = [
        [
            date.strftime("%m/%d/%Y"),
            f"{800 + index:.2f}",
            f"{799 + index:.2f}",
            f"{801 + index:.2f}",
            f"{798 + index:.2f}",
            "20M",
            f"{(100 / (799 + index)):.2f}%" if index else "0.00%",
        ]
        for index, date in enumerate(monthly_dates)
    ]
    for directory in (raw_set50, raw_set100):
        _write_vendor_csv(directory / "daily.csv", daily_rows)
        _write_vendor_csv(directory / "weekly.csv", weekly_rows)
        _write_vendor_csv(directory / "monthly.csv", monthly_rows)

    policy = LicensePolicy(
        status="publicly_accessible_provider_download",
        evidence_status="verified_public_provider_pages",
        raw_redistribution_allowed=False,
        derived_results_publication_allowed=True,
    )
    result = run_market_data_governance(
        set50_paths={
            "daily": raw_set50 / "daily.csv",
            "weekly": raw_set50 / "weekly.csv",
            "monthly": raw_set50 / "monthly.csv",
        },
        set100_paths={
            "daily": raw_set100 / "daily.csv",
            "weekly": raw_set100 / "weekly.csv",
            "monthly": raw_set100 / "monthly.csv",
        },
        output_dir=output,
        set100_prepared_dir=prepared,
        set100_outer_folds_dir=outer_folds,
        set100_selection_folds_dir=selection_folds,
        license_policy=policy,
        set50_reference_aligned=None,
    )

    assert result["internal_research_gate"] == "PASS"
    assert result["public_release_gate"] == "PASS"
    assert result["set100_data_gate"] == "PASS"
    assert (prepared / "SET100_aligned.csv").is_file()
    assert (outer_folds / "fold_1" / "point_in_time_contract.json").is_file()
    assert (selection_folds / "fold_4" / "point_in_time_contract.json").is_file()
    license_rows = pd.read_csv(output / "license_register.csv")
    assert license_rows.loc[0, "license_id"] == (
        "public_provider_historical_download_access_v1"
    )
    assert license_rows.loc[0, "evidence_status"] == (
        "verified_public_provider_pages"
    )
    assert not bool(license_rows.loc[0, "raw_redistribution_allowed"])
    public_anomalies = pd.read_csv(output / "anomaly_summary.csv")
    assert public_anomalies.columns.tolist() == [
        "dataset_id",
        "Date",
        "anomaly_type",
        "action",
    ]
    assert not {"Open", "High", "Low", "Close"}.intersection(
        public_anomalies.columns
    )
    assert (output / "private" / "anomaly_ledger.csv").is_file()
    assert (output / "private" / "correction_ledger.csv").is_file()
    release = json.loads((output / "release_gates.json").read_text())
    assert release["public_release_gate"] == "PASS"

def test_access_policy_accepts_only_recorded_public_provider_claim() -> None:
    public_policy = LicensePolicy(
        status="publicly_accessible_provider_download",
        evidence_status="verified_public_provider_pages",
        raw_redistribution_allowed=False,
        derived_results_publication_allowed=True,
    )
    validate_access_policy(public_policy)

    institutional_policy = LicensePolicy(
        status="user_attested_institutional_research_use",
        evidence_status="verified_private_register",
        raw_redistribution_allowed=False,
        derived_results_publication_allowed=True,
    )
    with pytest.raises(ValueError, match="public provider access"):
        validate_access_policy(institutional_policy)
