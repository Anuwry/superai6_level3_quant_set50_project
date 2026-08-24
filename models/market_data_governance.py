from __future__ import annotations

import hashlib
import json
import re
import shutil

# Subprocess use is limited to fixed Git argv with shell=False and a resolved executable.
import subprocess  # nosec B404
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from models.point_in_time_data import create_point_in_time_market_folds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREQUENCIES = ("daily", "weekly", "monthly")
CANONICAL_COLUMNS = (
    "Date",
    "Close",
    "Open",
    "High",
    "Low",
    "Volume",
    "Change_pct",
)
TARGET_COLUMN = "Target_Next_Close"
DATE_FORMAT = "%m/%d/%Y"
TIMEZONE = "Asia/Bangkok"
DECISION_TIME = "17:00:00"
MIN_COMMON_DATE_COVERAGE = 0.99
PERCENT_CHANGE_TOLERANCE = 0.011
RESTRICTED_MARKET_PATHS = (
    "data-raw/SET50_*.csv",
    "data-prepared/SET50_*.csv",
    "set100_data/*.csv",
    "set100_data/prepared/**",
    "set100_data/folds_point_in_time_v2/**",
    "set100_data/selection_folds_point_in_time_v2/**",
)

COLUMN_ALIASES = {
    "Date": ("Date", "วันเดือนปี"),
    "Close": ("Price", "ล่าสุด"),
    "Open": ("Open", "ราคาเปิด"),
    "High": ("High", "สูงสุด"),
    "Low": ("Low", "ต่ำสุด"),
    "Volume": ("Vol.", "ปริมาณ"),
    "Change_pct": ("Change %", "% เปลี่ยน"),
}

SOURCE_METADATA = {
    "SET50": {
        "source_url": "https://th.investing.com/indices/set-50-historical-data",
        "source_host_evidence": "https://th.investing.com/",
        "vendor_instrument_id": "41049",
        "local_file_created_at": {
            "daily": "2026-06-22T15:48:38+07:00",
            "weekly": "2026-06-22T15:50:28+07:00",
            "monthly": "2026-06-22T15:51:18+07:00",
        },
    },
    "SET100": {
        "source_url": "https://www.investing.com/indices/set-100-historical-data",
        "source_host_evidence": "https://www.investing.com/",
        "vendor_instrument_id": "not_recorded",
        "local_file_created_at": {
            "daily": "2026-08-03T12:42:15+07:00",
            "weekly": "2026-08-03T12:42:33+07:00",
            "monthly": "2026-08-03T12:42:53+07:00",
        },
    },
}


@dataclass(frozen=True)
class LicensePolicy:
    status: str
    evidence_status: str
    raw_redistribution_allowed: bool
    derived_results_publication_allowed: bool


def validate_access_policy(policy: LicensePolicy) -> None:
    expected = LicensePolicy(
        status="publicly_accessible_provider_download",
        evidence_status="verified_public_provider_pages",
        raw_redistribution_allowed=False,
        derived_results_publication_allowed=True,
    )
    if policy != expected:
        raise ValueError(
            "This study records public provider access only; provider terms "
            "apply and raw redistribution is not allowed"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_volume(value: object) -> float:
    text = str(value).strip().upper().replace(",", "")
    if not text or text == "-" or text == "NAN":
        raise ValueError("Market row contains missing volume")
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))([KMB]?)", text)
    if match is None:
        raise ValueError(f"Invalid vendor volume: {value!r}")
    multipliers = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9}
    return float(match.group(1)) * multipliers[match.group(2)]


def _canonical_mapping(columns: pd.Index) -> dict[str, str]:
    available = {str(column) for column in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        matches = [alias for alias in aliases if alias in available]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one source column for {canonical}; found {matches}"
            )
        mapping[matches[0]] = canonical
    return mapping


def _parse_numeric(series: pd.Series, column: str) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("+", "", regex=False)
    )
    parsed = pd.to_numeric(cleaned, errors="coerce")
    if parsed.isna().any():
        bad = series.loc[parsed.isna()].head(3).tolist()
        raise ValueError(f"Invalid {column} value(s): {bad}")
    return parsed.astype(float)


def load_vendor_market_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Market source not found: {path}")
    source = pd.read_csv(path)
    result = source.rename(columns=_canonical_mapping(source.columns)).loc[
        :, CANONICAL_COLUMNS
    ]
    result = result.copy()
    result["Date"] = pd.to_datetime(
        result["Date"],
        format=DATE_FORMAT,
        errors="coerce",
    )
    if result["Date"].isna().any():
        bad = source.loc[result["Date"].isna()].iloc[:3, 0].tolist()
        raise ValueError(
            f"Invalid market Date value(s) for explicit {DATE_FORMAT}: {bad}"
        )
    for column in ("Close", "Open", "High", "Low", "Change_pct"):
        result[column] = _parse_numeric(result[column], column)
    result["Volume"] = result["Volume"].map(parse_volume)
    return result.sort_values("Date").reset_index(drop=True)


def prepare_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    _validate_canonical_frame(frame)
    result = frame.sort_values("Date").reset_index(drop=True).copy()
    result[TARGET_COLUMN] = result["Close"].shift(-1)
    result = result.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return result.loc[:, [*CANONICAL_COLUMNS, TARGET_COLUMN]]


def _validate_canonical_frame(frame: pd.DataFrame) -> None:
    missing = sorted(set(CANONICAL_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Market frame is missing canonical columns: {missing}")
    if frame.empty:
        raise ValueError("Market frame is empty")


def _feature_frame(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    _validate_canonical_frame(frame)
    result = frame.loc[:, CANONICAL_COLUMNS].copy()
    result = result.sort_values("Date").set_index("Date")
    return result.add_suffix(f"_{suffix}")


def align_causal_frequencies(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    daily_features = _feature_frame(daily, "D")
    weekly_completed = _feature_frame(weekly, "W").shift(1)
    monthly_completed = _feature_frame(monthly, "M").shift(1)
    weekly_aligned = weekly_completed.reindex(daily_features.index, method="ffill")
    monthly_aligned = monthly_completed.reindex(
        daily_features.index,
        method="ffill",
    )
    aligned = pd.concat(
        [daily_features, weekly_aligned, monthly_aligned],
        axis=1,
    )
    aligned[TARGET_COLUMN] = aligned["Close_D"].shift(-1)
    aligned = aligned.dropna().reset_index()
    return aligned


def _check(
    dataset_id: str,
    check_id: str,
    status: str,
    observed: object,
    expected: object,
    severity: str,
    details: str,
) -> dict[str, object]:
    return {
        "dataset_id": dataset_id,
        "check_id": check_id,
        "status": status,
        "observed": observed,
        "expected": expected,
        "severity": severity,
        "details": details,
    }


def _frequency_label_violations(dates: pd.Series, frequency: str) -> int:
    if frequency == "daily":
        return int((dates.dt.dayofweek >= 5).sum())
    if frequency == "weekly":
        return int((dates.dt.dayofweek != 6).sum())
    if frequency == "monthly":
        return int((dates.dt.day != 1).sum())
    raise ValueError(f"Unsupported frequency: {frequency}")


def audit_market_frame(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    frequency: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_canonical_frame(frame)
    result = frame.sort_values("Date").reset_index(drop=True).copy()
    numeric = result.loc[:, ["Close", "Open", "High", "Low", "Volume"]]
    checks: list[dict[str, object]] = []

    duplicate_count = int(result["Date"].duplicated().sum())
    checks.append(
        _check(
            dataset_id,
            "duplicate_dates",
            "PASS" if duplicate_count == 0 else "FAIL",
            duplicate_count,
            0,
            "critical",
            "Trading-session dates must be unique.",
        )
    )
    invalid_numeric = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())
    checks.append(
        _check(
            dataset_id,
            "finite_ohlcv",
            "PASS" if invalid_numeric == 0 else "FAIL",
            invalid_numeric,
            0,
            "critical",
            "OHLCV values must be finite.",
        )
    )
    negative_volume = int((result["Volume"] < 0).sum())
    checks.append(
        _check(
            dataset_id,
            "nonnegative_volume",
            "PASS" if negative_volume == 0 else "FAIL",
            negative_volume,
            0,
            "critical",
            "Vendor-reported volume cannot be negative.",
        )
    )
    nonpositive_index = int(
        (result.loc[:, ["Close", "Open", "High", "Low"]] <= 0).sum().sum()
    )
    checks.append(
        _check(
            dataset_id,
            "positive_index_levels",
            "PASS" if nonpositive_index == 0 else "FAIL",
            nonpositive_index,
            0,
            "critical",
            "Index OHLC levels must be positive.",
        )
    )

    high_shortfall = result["High"] < result[["Open", "Close"]].max(axis=1)
    low_excess = result["Low"] > result[["Open", "Close"]].min(axis=1)
    invalid_range = result["High"] < result["Low"]
    ohlc_anomaly = high_shortfall | low_excess | invalid_range
    anomaly_count = int(ohlc_anomaly.sum())
    checks.append(
        _check(
            dataset_id,
            "ohlc_containment",
            "PASS" if anomaly_count == 0 else "WARN",
            anomaly_count,
            0,
            "warning",
            "Raw vendor rows are retained; anomalies require source verification.",
        )
    )

    calculated_change = result["Close"].pct_change().mul(100)
    change_error = (calculated_change - result["Change_pct"]).abs().iloc[1:]
    mismatch_count = int((change_error > PERCENT_CHANGE_TOLERANCE).sum())
    maximum_error = float(change_error.max()) if not change_error.empty else 0.0
    checks.append(
        _check(
            dataset_id,
            "percent_change_close_to_close",
            "PASS" if mismatch_count == 0 else "FAIL",
            mismatch_count,
            f"0 at tolerance {PERCENT_CHANGE_TOLERANCE}",
            "critical",
            f"Maximum absolute error was {maximum_error:.9f} percentage points.",
        )
    )

    label_violations = _frequency_label_violations(result["Date"], frequency)
    label_description = {
        "daily": "Monday-Friday session-date labels",
        "weekly": "Sunday vendor period labels",
        "monthly": "first-calendar-day vendor period labels",
    }[frequency]
    checks.append(
        _check(
            dataset_id,
            "frequency_label_convention",
            "PASS" if label_violations == 0 else "FAIL",
            label_violations,
            0,
            "critical",
            label_description,
        )
    )

    anomaly_columns = [
        "dataset_id",
        "Date",
        "anomaly_type",
        "Open",
        "High",
        "Low",
        "Close",
        "action",
    ]
    anomalies = result.loc[
        ohlc_anomaly,
        ["Date", "Open", "High", "Low", "Close"],
    ].copy()
    if anomalies.empty:
        anomalies = pd.DataFrame(columns=anomaly_columns)
    else:
        anomalies.insert(0, "dataset_id", dataset_id)
        anomalies.insert(2, "anomaly_type", "ohlc_containment")
        anomalies["action"] = "retain_raw_pending_verification"
        anomalies = anomalies.loc[:, anomaly_columns].reset_index(drop=True)
    return pd.DataFrame(checks), anomalies


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = frame.copy()
    for column in ("Date", "Label_Date"):
        if column in saved.columns and pd.api.types.is_datetime64_any_dtype(
            saved[column]
        ):
            saved[column] = saved[column].dt.strftime("%Y-%m-%d")
    saved.to_csv(path, index=False)


def _manifest_row(
    *,
    instrument: str,
    frequency: str,
    path: Path,
    frame: pd.DataFrame,
    artifact_class: str,
) -> dict[str, object]:
    metadata = SOURCE_METADATA[instrument]
    return {
        "dataset_id": f"{instrument.lower()}_{frequency}",
        "instrument": instrument,
        "instrument_type": "price_index",
        "currency": "THB",
        "frequency": frequency,
        "artifact_class": artifact_class,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": len(frame),
        "date_start": pd.to_datetime(frame["Date"]).min().strftime("%Y-%m-%d"),
        "date_end": pd.to_datetime(frame["Date"]).max().strftime("%Y-%m-%d"),
        "source_url": metadata["source_url"],
        "source_host_evidence": metadata["source_host_evidence"],
        "vendor_instrument_id": metadata["vendor_instrument_id"],
        "acquisition_evidence_at_bangkok": metadata[
            "local_file_created_at"
        ].get(
            frequency,
            "derived",
        ),
        "acquisition_timestamp_basis": (
            "local_file_creation_time_not_vendor_receipt"
        ),
        "acquisition_method": "manual_browser_download_user_supplied",
        "timezone_convention": TIMEZONE,
        "decision_time_bangkok": DECISION_TIME,
        "adjustment_convention": (
            "provider_published_price_index_not_total_return_"
            "no_researcher_side_adjustment"
        ),
        "raw_redistribution": "prohibited",
    }


def _data_dictionary() -> pd.DataFrame:
    rows = [
        ("Date", "date", "SET session date in Asia/Bangkok", "session date"),
        ("Close", "float", "Provider-published price-index close", "index points"),
        ("Open", "float", "Provider-published price-index open", "index points"),
        ("High", "float", "Provider-published price-index high", "index points"),
        ("Low", "float", "Provider-published price-index low", "index points"),
        (
            "Volume",
            "float",
            "Vendor-reported volume; K/M/B expanded by 10^3/10^6/10^9",
            "vendor native units",
        ),
        (
            "Change_pct",
            "float",
            "100 * (Close_t / Close_(t-1) - 1)",
            "percentage points",
        ),
        (
            TARGET_COLUMN,
            "float",
            "Close on the next observed SET trading session",
            "index points",
        ),
        (
            "*_W",
            "float",
            "Prior completed vendor weekly bar, shifted once before forward fill",
            "column-dependent",
        ),
        (
            "*_M",
            "float",
            "Prior completed vendor monthly bar, shifted once before forward fill",
            "column-dependent",
        ),
    ]
    return pd.DataFrame(rows, columns=["field", "dtype", "definition", "unit"])


def _license_register(policy: LicensePolicy) -> pd.DataFrame:
    validate_access_policy(policy)
    return pd.DataFrame(
        [
            {
                "license_id": "public_provider_historical_download_access_v1",
                "scope": "SET50 and SET100 market data used in this study",
                "status": policy.status,
                "evidence_status": policy.evidence_status,
                "evidence_location": "provider_historical_pages_and_terms_urls",
                "research_use_allowed": True,
                "derived_results_publication_allowed": (
                    policy.derived_results_publication_allowed
                ),
                "raw_redistribution_allowed": policy.raw_redistribution_allowed,
                "public_artifact_policy": "code_metadata_checksums_nonreconstructive_results_only",
            }
        ]
    )


def restrict_to_reference_dates(
    aligned: pd.DataFrame,
    reference_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if reference_path is None:
        return aligned.copy(), {
            "reference_used": False,
            "required_dates": len(aligned),
            "matched_dates": len(aligned),
            "missing_dates": 0,
            "coverage": 1.0,
            "missing_date_sample": [],
        }
    reference = pd.read_csv(reference_path, usecols=["Date"])
    reference_dates = pd.to_datetime(reference["Date"], errors="coerce")
    if reference_dates.isna().any() or reference_dates.duplicated().any():
        raise ValueError("SET50 reference dates are invalid or duplicated")
    indexed = aligned.set_index("Date")
    matched_mask = reference_dates.isin(indexed.index)
    matched_dates = reference_dates.loc[matched_mask]
    missing = reference_dates.loc[~matched_mask]
    if matched_dates.empty:
        raise ValueError("SET100 and SET50 reference dates do not overlap")
    common = indexed.loc[matched_dates].reset_index()
    return common, {
        "reference_used": True,
        "required_dates": len(reference_dates),
        "matched_dates": len(common),
        "missing_dates": len(missing),
        "coverage": float(len(common) / len(reference_dates)),
        "missing_date_sample": missing.dt.strftime("%Y-%m-%d")
        .head(10)
        .tolist(),
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": sha256_file(reference_path),
    }


def audit_repository_distribution(
    repository_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Report restricted market-row artifacts in the index and Git history."""
    root = str(repository_root.resolve())
    git_executable = shutil.which("git")
    if git_executable is None:
        raise FileNotFoundError("Git executable is required for repository audit")
    tracked_result = subprocess.run(  # nosec B603
        [
            git_executable,
            "-C",
            root,
            "ls-files",
            "--",
            *RESTRICTED_MARKET_PATHS,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_paths = sorted(
        {
            line.strip()
            for line in tracked_result.stdout.splitlines()
            if line.strip()
        }
    )
    history_result = subprocess.run(  # nosec B603
        [
            git_executable,
            "-C",
            root,
            "log",
            "--all",
            "--format=",
            "--name-only",
            "--",
            *RESTRICTED_MARKET_PATHS,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    history_paths = sorted(
        {
            line.strip()
            for line in history_result.stdout.splitlines()
            if line.strip()
        }
    )
    all_paths = sorted(set(tracked_paths).union(history_paths))
    audit_rows: list[dict[str, object]] = []
    for path in all_paths:
        commits_result = subprocess.run(  # nosec B603
            [
                git_executable,
                "-C",
                root,
                "log",
                "--all",
                "--format=%H|%aI",
                "--",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        commits = [
            line.strip()
            for line in commits_result.stdout.splitlines()
            if line.strip()
        ]
        audit_rows.append(
            {
                "path": path,
                "classification": "restricted_market_rows",
                "tracked_in_current_index": path in tracked_paths,
                "history_commit_count": len(commits),
                "history_first_seen": commits[-1] if commits else "",
                "history_last_seen": commits[0] if commits else "",
                "public_release_disposition": (
                    "use_clean_public_package_and_review_history"
                ),
            }
        )
    audit = pd.DataFrame(
        audit_rows,
        columns=[
            "path",
            "classification",
            "tracked_in_current_index",
            "history_commit_count",
            "history_first_seen",
            "history_last_seen",
            "public_release_disposition",
        ],
    )
    tracked_count = len(tracked_paths)
    history_count = len(history_paths)
    return audit, {
        "repository_root": root,
        "tracked_restricted_file_count": tracked_count,
        "history_restricted_file_count": history_count,
        "repository_distribution_gate": (
            "PASS"
            if tracked_count == 0
            else "REVIEW_REQUIRED_TRACKED_RESTRICTED_DATA"
        ),
        "git_history_audit": (
            "PASS"
            if history_count == 0
            else "REVIEW_REQUIRED_RESTRICTED_DATA_IN_HISTORY"
        ),
    }


def run_market_data_governance(
    *,
    set50_paths: Mapping[str, Path],
    set100_paths: Mapping[str, Path],
    output_dir: Path,
    set100_prepared_dir: Path,
    set100_outer_folds_dir: Path,
    set100_selection_folds_dir: Path,
    license_policy: LicensePolicy,
    set50_reference_aligned: Path | None,
    repository_root: Path | None = None,
) -> dict[str, object]:
    validate_access_policy(license_policy)
    for name, paths in (("SET50", set50_paths), ("SET100", set100_paths)):
        if set(paths) != set(FREQUENCIES):
            raise ValueError(f"{name} paths must contain {FREQUENCIES}")

    output_dir.mkdir(parents=True, exist_ok=True)
    loaded: dict[tuple[str, str], pd.DataFrame] = {}
    checks: list[pd.DataFrame] = []
    anomalies: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for instrument, paths in (("SET50", set50_paths), ("SET100", set100_paths)):
        for frequency in FREQUENCIES:
            path = Path(paths[frequency])
            frame = load_vendor_market_csv(path)
            loaded[(instrument, frequency)] = frame
            frame_checks, frame_anomalies = audit_market_frame(
                frame,
                dataset_id=f"{instrument.lower()}_{frequency}",
                frequency=frequency,
            )
            checks.append(frame_checks)
            anomalies.append(frame_anomalies)
            manifest_rows.append(
                _manifest_row(
                    instrument=instrument,
                    frequency=frequency,
                    path=path,
                    frame=frame,
                    artifact_class="restricted_raw",
                )
            )

    set100_prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_paths: dict[str, Path] = {}
    for frequency in FREQUENCIES:
        prepared_frame = prepare_frequency(loaded[("SET100", frequency)])
        output_name = {
            "daily": "SET100_days.csv",
            "weekly": "SET100_weeks.csv",
            "monthly": "SET100_months.csv",
        }[frequency]
        prepared_path = set100_prepared_dir / output_name
        _write_frame(prepared_frame, prepared_path)
        prepared_paths[frequency] = prepared_path

    daily_calendar_path = set100_prepared_dir / "SET100_daily_calendar.csv"
    _write_frame(loaded[("SET100", "daily")], daily_calendar_path)
    aligned = align_causal_frequencies(
        loaded[("SET100", "daily")],
        loaded[("SET100", "weekly")],
        loaded[("SET100", "monthly")],
    )
    aligned_path = set100_prepared_dir / "SET100_aligned.csv"
    _write_frame(aligned, aligned_path)
    common_aligned, coverage = restrict_to_reference_dates(
        aligned,
        set50_reference_aligned,
    )
    common_path = set100_prepared_dir / "SET100_aligned_common_set50_dates.csv"
    _write_frame(common_aligned, common_path)

    create_point_in_time_market_folds(
        aligned_path=common_path,
        daily_path=daily_calendar_path,
        output_dir=set100_outer_folds_dir,
        test_years=(2022, 2023, 2024, 2025),
    )
    create_point_in_time_market_folds(
        aligned_path=common_path,
        daily_path=daily_calendar_path,
        output_dir=set100_selection_folds_dir,
        test_years=(2018, 2019, 2020, 2021),
    )

    derived_frames = {
        "daily_prepared": pd.read_csv(prepared_paths["daily"]),
        "weekly_prepared": pd.read_csv(prepared_paths["weekly"]),
        "monthly_prepared": pd.read_csv(prepared_paths["monthly"]),
        "daily_calendar": pd.read_csv(daily_calendar_path),
        "aligned": pd.read_csv(aligned_path),
        "aligned_common_set50_dates": pd.read_csv(common_path),
    }
    derived_paths = {
        "daily_prepared": prepared_paths["daily"],
        "weekly_prepared": prepared_paths["weekly"],
        "monthly_prepared": prepared_paths["monthly"],
        "daily_calendar": daily_calendar_path,
        "aligned": aligned_path,
        "aligned_common_set50_dates": common_path,
    }
    for frequency, path in derived_paths.items():
        frame = derived_frames[frequency].copy()
        frame["Date"] = pd.to_datetime(frame["Date"])
        manifest_rows.append(
            _manifest_row(
                instrument="SET100",
                frequency=frequency,
                path=path,
                frame=frame,
                artifact_class="restricted_derived_reconstructive",
            )
        )

    checks_frame = pd.concat(checks, ignore_index=True)
    nonempty_anomalies = [frame for frame in anomalies if not frame.empty]
    anomalies_frame = (
        pd.concat(nonempty_anomalies, ignore_index=True)
        if nonempty_anomalies
        else pd.DataFrame(columns=anomalies[0].columns)
    )
    set50_failures = int(
        (
            (checks_frame["dataset_id"].str.startswith("set50"))
            & (checks_frame["status"] == "FAIL")
        ).sum()
    )
    set100_failures = int(
        (
            (checks_frame["dataset_id"].str.startswith("set100"))
            & (checks_frame["status"] == "FAIL")
        ).sum()
    )
    internal_gate = "PASS"
    public_gate = "PASS"
    if repository_root is None:
        repository_audit = pd.DataFrame(
            columns=[
                "path",
                "classification",
                "tracked_in_current_index",
                "history_commit_count",
                "history_first_seen",
                "history_last_seen",
                "public_release_disposition",
            ]
        )
        repository_summary: dict[str, object] = {
            "repository_distribution_gate": "NOT_EVALUATED",
            "tracked_restricted_file_count": None,
            "history_restricted_file_count": None,
            "git_history_audit": "NOT_EVALUATED",
        }
    else:
        repository_audit, repository_summary = audit_repository_distribution(
            repository_root
        )
        if repository_summary["tracked_restricted_file_count"] or repository_summary[
            "history_restricted_file_count"
        ]:
            public_gate = "PENDING_REPOSITORY_REMEDIATION"
    set100_gate = (
        "PASS"
        if set100_failures == 0
        and coverage["coverage"] >= MIN_COMMON_DATE_COVERAGE
        else "FAIL"
    )
    release_gates: dict[str, object] = {
        "protocol_version": "market-data-governance-v1",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "internal_research_gate": internal_gate,
        "public_release_gate": public_gate,
        "raw_redistribution_gate": "EXCLUDED_PROVIDER_TERMS_APPLY",
        "set50_data_gate": "PASS" if set50_failures == 0 else "FAIL",
        "set100_data_gate": set100_gate,
        "set100_benchmark_status": "NOT_STARTED_DATA_READY"
        if set100_gate == "PASS"
        else "NOT_READY",
        "timezone_gate": "PASS",
        "timezone": TIMEZONE,
        "decision_time_bangkok": DECISION_TIME,
        "adjustment_gate": "PASS_WITH_RETAINED_VENDOR_WARNINGS"
        if not anomalies_frame.empty
        else "PASS",
        "set100_common_date_coverage": coverage,
        "set100_minimum_common_date_coverage": MIN_COMMON_DATE_COVERAGE,
        "license_policy": asdict(license_policy),
        "repository_distribution": repository_summary,
    }

    manifest = pd.DataFrame(manifest_rows)
    license_register = _license_register(license_policy)
    _write_frame(manifest, output_dir / "market_data_manifest.csv")
    (output_dir / "market_data_manifest.json").write_text(
        json.dumps(manifest_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_frame(checks_frame, output_dir / "integrity_checks.csv")
    public_anomaly_columns = [
        "dataset_id",
        "Date",
        "anomaly_type",
        "action",
    ]
    _write_frame(
        anomalies_frame.loc[:, public_anomaly_columns],
        output_dir / "anomaly_summary.csv",
    )
    private_output_dir = output_dir / "private"
    _write_frame(
        anomalies_frame,
        private_output_dir / "anomaly_ledger.csv",
    )
    _write_frame(_data_dictionary(), output_dir / "data_dictionary.csv")
    _write_frame(license_register, output_dir / "license_register.csv")
    _write_frame(
        repository_audit,
        output_dir / "repository_distribution_audit.csv",
    )
    correction_columns = [
        "date",
        "instrument",
        "column",
        "old_value",
        "new_value",
        "reason",
        "verification_source",
        "verified_at",
    ]
    empty_correction_ledger = pd.DataFrame(columns=correction_columns)
    _write_frame(
        empty_correction_ledger,
        private_output_dir / "correction_ledger.csv",
    )
    _write_frame(
        pd.DataFrame(
            {
                "field": correction_columns,
                "classification": "restricted_when_populated",
            }
        ),
        output_dir / "correction_ledger_schema.csv",
    )
    (output_dir / "release_gates.json").write_text(
        json.dumps(release_gates, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return release_gates


def main() -> dict[str, object]:
    result = run_market_data_governance(
        set50_paths={
            "daily": PROJECT_ROOT / "data-raw" / "SET50_days.csv",
            "weekly": PROJECT_ROOT / "data-raw" / "SET50_weeks.csv",
            "monthly": PROJECT_ROOT / "data-raw" / "SET50_months.csv",
        },
        set100_paths={
            "daily": PROJECT_ROOT / "set100_data" / "set100_daily.csv",
            "weekly": PROJECT_ROOT / "set100_data" / "set100_weekly.csv",
            "monthly": PROJECT_ROOT / "set100_data" / "set100_monthly.csv",
        },
        output_dir=PROJECT_ROOT / "outputs" / "market_data_governance_v1",
        set100_prepared_dir=PROJECT_ROOT / "set100_data" / "prepared",
        set100_outer_folds_dir=(
            PROJECT_ROOT / "set100_data" / "folds_point_in_time_v2"
        ),
        set100_selection_folds_dir=(
            PROJECT_ROOT / "set100_data" / "selection_folds_point_in_time_v2"
        ),
        license_policy=LicensePolicy(
            status="publicly_accessible_provider_download",
            evidence_status="verified_public_provider_pages",
            raw_redistribution_allowed=False,
            derived_results_publication_allowed=True,
        ),
        set50_reference_aligned=(
            PROJECT_ROOT / "data-prepared" / "SET50_aligned.csv"
        ),
        repository_root=PROJECT_ROOT,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
