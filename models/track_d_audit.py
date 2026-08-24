from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_common import PROJECT_ROOT
from models.track_d_protocol import TrackDConfig, verify_freeze_manifest

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "track_d_q2"
DATA_MANIFEST = PROJECT_ROOT / "data-track-d" / "forward_data_manifest.json"


def expected_cardinalities(
    *,
    forward_rows: int,
    economic_rows: int,
    features: int,
    config: TrackDConfig,
) -> dict[str, int]:
    models = len(config.models)
    objectives = len(config.objectives)
    seeds = len(config.seeds)
    validation_years = len(config.selection_years)
    strategies = 2
    costs = len(config.cost_bps_grid)
    thresholds = len(config.confidence_thresholds)
    conditions = 3
    randomization_comparisons = 2
    return {
        "validation_metric_rows": models * objectives * validation_years,
        "selected_threshold_rows": models * objectives,
        "forward_prediction_seed_rows": forward_rows * models * objectives * seeds,
        "forward_prediction_average_rows": forward_rows * models * objectives,
        "forward_metric_rows": models * objectives,
        "selective_metric_rows": models * objectives * thresholds,
        "economic_summary_rows": (
            models * objectives * thresholds * strategies * costs
        ),
        "economic_daily_rows": (
            economic_rows * models * objectives * thresholds * strategies * costs
        ),
        "xai_attribution_rows": (
            models * config.xai_instances * conditions * features
        ),
        "xai_randomization_rows": (
            models * config.xai_instances * randomization_comparisons
        ),
        "xai_deletion_rows": models * config.xai_instances,
        "xai_random_deletion_rows": (
            models
            * config.xai_instances
            * config.xai_random_deletion_repeats
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required Track D artifact not found: {path}")
    return pd.read_csv(path)


def _finite(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return bool(np.isfinite(values.to_numpy(dtype=float)).all())


def run_integrity_audit(
    *,
    output_dir: Path = OUTPUT_DIR,
    data_manifest_path: Path = DATA_MANIFEST,
) -> dict[str, object]:
    config = TrackDConfig()
    freeze_path = output_dir / "freeze_manifest.json"
    freeze = verify_freeze_manifest(freeze_path)
    if not data_manifest_path.is_file():
        raise FileNotFoundError(f"Forward data manifest not found: {data_manifest_path}")
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    for raw_path, expected in data_manifest["artifact_sha256"].items():
        path = Path(raw_path)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Forward data artifact hash mismatch: {path}")
    frames = {
        "validation_metrics": _read(output_dir / "validation_metrics.csv"),
        "selected_thresholds": _read(output_dir / "selected_thresholds.csv"),
        "forward_seed": _read(output_dir / "forward_predictions_by_seed.csv"),
        "forward_average": _read(
            output_dir / "forward_predictions_seed_averaged.csv"
        ),
        "forward_metrics": _read(output_dir / "forward_metrics.csv"),
        "selective_metrics": _read(
            output_dir / "selective_prediction_metrics.csv"
        ),
        "economic_summary": _read(output_dir / "economic_summary.csv"),
        "economic_daily": _read(output_dir / "economic_daily.csv"),
        "xai_attributions": _read(output_dir / "xai_attributions.csv"),
        "xai_randomization": _read(
            output_dir / "xai_randomization_summary.csv"
        ),
        "xai_deletion": _read(output_dir / "xai_deletion_summary.csv"),
        "xai_random_deletion": _read(
            output_dir / "xai_random_deletion_effects.csv"
        ),
    }
    average = frames["forward_average"]
    group_sizes = average.groupby(["model", "objective"], sort=False).size()
    if group_sizes.nunique() != 1:
        raise ValueError("Forward average groups have unequal row counts")
    forward_rows = int(group_sizes.iloc[0])
    economic_eligible = average.groupby(
        ["model", "objective"], sort=False
    )["economic_eligible"].sum()
    if economic_eligible.nunique() != 1:
        raise ValueError("Economic eligibility differs across model cells")
    economic_rows = int(economic_eligible.iloc[0])
    expected = expected_cardinalities(
        forward_rows=forward_rows,
        economic_rows=economic_rows,
        features=int(data_manifest["features"]),
        config=config,
    )
    actual = {
        "validation_metric_rows": len(frames["validation_metrics"]),
        "selected_threshold_rows": len(frames["selected_thresholds"]),
        "forward_prediction_seed_rows": len(frames["forward_seed"]),
        "forward_prediction_average_rows": len(frames["forward_average"]),
        "forward_metric_rows": len(frames["forward_metrics"]),
        "selective_metric_rows": len(frames["selective_metrics"]),
        "economic_summary_rows": len(frames["economic_summary"]),
        "economic_daily_rows": len(frames["economic_daily"]),
        "xai_attribution_rows": len(frames["xai_attributions"]),
        "xai_randomization_rows": len(frames["xai_randomization"]),
        "xai_deletion_rows": len(frames["xai_deletion"]),
        "xai_random_deletion_rows": len(frames["xai_random_deletion"]),
    }
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    record("artifact_cardinalities", actual == expected, {"expected": expected, "actual": actual})
    records = {
        "models": set(frames["forward_metrics"]["model"]),
        "objectives": set(frames["forward_metrics"]["objective"]),
        "seeds": set(frames["forward_seed"]["seed"].astype(int)),
    }
    record("model_registry", records["models"] == set(config.models), sorted(records["models"]))
    record(
        "objective_registry",
        records["objectives"] == set(config.objectives),
        sorted(records["objectives"]),
    )
    record("seed_registry", records["seeds"] == set(config.seeds), sorted(records["seeds"]))
    selected = frames["selected_thresholds"]
    record(
        "threshold_registry",
        set(selected["selected_threshold"].astype(float)).issubset(
            config.confidence_thresholds
        ),
        sorted(set(selected["selected_threshold"].astype(float))),
    )
    forward_dates = pd.to_datetime(average["Date"], errors="coerce")
    record(
        "forward_year_only",
        bool(forward_dates.notna().all() and set(forward_dates.dt.year) == {2026}),
        [str(forward_dates.min().date()), str(forward_dates.max().date())],
    )
    key_specs = {
        "forward_seed_unique": ("forward_seed", ("model", "objective", "seed", "Date")),
        "forward_average_unique": ("forward_average", ("model", "objective", "Date")),
        "economic_summary_unique": (
            "economic_summary",
            ("model", "objective", "threshold", "strategy", "cost_bps"),
        ),
        "xai_attribution_unique": (
            "xai_attributions",
            ("model", "condition", "test_row_index", "feature"),
        ),
        "xai_random_deletion_unique": (
            "xai_random_deletion",
            ("model", "test_row_index", "repeat"),
        ),
    }
    for name, (frame_key, keys) in key_specs.items():
        duplicate_count = int(frames[frame_key].duplicated(list(keys)).sum())
        record(name, duplicate_count == 0, duplicate_count)
    finite_specs = {
        "forward_probability_finite": (
            "forward_average",
            ("probability", "y_true"),
        ),
        "forward_metrics_finite": (
            "forward_metrics",
            (
                "direction_accuracy",
                "balanced_accuracy",
                "mcc",
                "auc",
                "brier",
                "log_loss",
                "ece_10",
            ),
        ),
        "economic_summary_finite": (
            "economic_summary",
            (
                "coverage",
                "round_trip_units",
                "gross_cumulative_return",
                "net_cumulative_return",
                "net_annualized_sharpe",
                "maximum_drawdown",
            ),
        ),
        "xai_attribution_finite": (
            "xai_attributions",
            ("signed_attribution", "absolute_attribution"),
        ),
        "xai_sanity_finite": (
            "xai_randomization",
            ("absolute_rank_spearman",),
        ),
        "xai_deletion_finite": (
            "xai_deletion",
            ("top_deletion_effect", "faithfulness_percentile"),
        ),
    }
    for name, (frame_key, columns) in finite_specs.items():
        record(name, _finite(frames[frame_key], columns), list(columns))
    access = data_manifest["access_ledger"]
    frozen_at = datetime.fromisoformat(str(freeze["frozen_at"]))
    accessed_at = datetime.fromisoformat(str(access["accessed_at_utc"]))
    record(
        "freeze_precedes_forward_access",
        frozen_at < accessed_at,
        {"frozen_at": frozen_at.isoformat(), "accessed_at": accessed_at.isoformat()},
    )
    all_passed = all(bool(row["passed"]) for row in checks)
    payload = {
        "protocol_version": config.protocol_version,
        "created_at_utc": datetime.now().astimezone().isoformat(timespec="seconds"),
        "all_checks_passed": all_passed,
        "checks": checks,
    }
    output_path = output_dir / "integrity_audit.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not all_passed:
        failed = [row["check"] for row in checks if not row["passed"]]
        raise ValueError(f"Track D integrity audit failed: {failed}")
    return payload


if __name__ == "__main__":
    run_integrity_audit()
