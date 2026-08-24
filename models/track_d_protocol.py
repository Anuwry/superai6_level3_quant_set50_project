from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from models.track_a_final import FINAL_SEEDS

TRACK_D_PROTOCOL_VERSION = "track-d-direction-forward-v1"
REGISTERED_MODELS = (
    "lstm",
    "cnn",
    "lstm_cnn",
    "lstm_attention",
    "lstm_cnn_attention",
)
REGISTERED_WINDOWS = {
    "lstm": 5,
    "cnn": 20,
    "lstm_cnn": 20,
    "lstm_attention": 10,
    "lstm_cnn_attention": 20,
}


@dataclass(frozen=True)
class TrackDConfig:
    protocol_version: str = TRACK_D_PROTOCOL_VERSION
    models: tuple[str, ...] = REGISTERED_MODELS
    windows: Mapping[str, int] = field(
        default_factory=lambda: dict(REGISTERED_WINDOWS)
    )
    objectives: tuple[str, ...] = ("direct", "multitask")
    selection_years: tuple[int, ...] = (2019, 2020, 2021)
    forward_year: int = 2026
    seeds: tuple[int, ...] = tuple(FINAL_SEEDS)
    confidence_thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65)
    minimum_coverage: float = 0.20
    minimum_positive_years: int = 2
    cost_bps_grid: tuple[float, ...] = (5.0, 10.0, 20.0)
    primary_cost_bps: float = 10.0
    return_loss_weight: float = 0.25
    xai_instances: int = 30
    xai_background: int = 100
    xai_nsamples: int = 200
    xai_top_k: int = 10
    xai_random_deletion_repeats: int = 100
    tensorflow_deterministic_ops: bool = True
    tensorflow_enable_onednn_opts: bool = False

    def __post_init__(self) -> None:
        if self.models != REGISTERED_MODELS:
            raise ValueError("Track D models must match the five-model registry")
        if dict(self.windows) != REGISTERED_WINDOWS:
            raise ValueError("Track D windows must match the locked registry")
        if self.objectives != ("direct", "multitask"):
            raise ValueError("Track D objectives must be direct and multitask")
        thresholds = np.asarray(self.confidence_thresholds, dtype=float)
        if (
            not np.isfinite(thresholds).all()
            or np.any(thresholds < 0.50)
            or np.any(thresholds >= 1.0)
            or tuple(sorted(set(thresholds))) != tuple(thresholds)
        ):
            raise ValueError("Invalid confidence threshold registry")
        if not 0.0 < self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be in (0, 1]")
        if self.minimum_positive_years < 1:
            raise ValueError("minimum_positive_years must be positive")
        costs = np.asarray(self.cost_bps_grid, dtype=float)
        if not np.isfinite(costs).all() or np.any(costs < 0.0):
            raise ValueError("Transaction costs must be finite and non-negative")
        if self.primary_cost_bps not in self.cost_bps_grid:
            raise ValueError("Primary cost must be included in cost_bps_grid")
        if self.return_loss_weight <= 0.0:
            raise ValueError("return_loss_weight must be positive")


def direction_labels(
    next_close: np.ndarray,
    current_close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    future = np.asarray(next_close, dtype=float).reshape(-1)
    current = np.asarray(current_close, dtype=float).reshape(-1)
    if future.shape != current.shape:
        raise ValueError("next_close and current_close shapes differ")
    if not np.isfinite(future).all() or not np.isfinite(current).all():
        raise ValueError("Direction inputs must be finite")
    change = future - current
    eligible = change != 0.0
    labels = (change > 0.0).astype(np.float32)
    return labels, eligible


def select_confidence_threshold(
    rows: Sequence[Mapping[str, object]],
    *,
    minimum_coverage: float,
    minimum_positive_years: int,
) -> float:
    eligible = [
        row
        for row in rows
        if float(row["coverage"]) >= minimum_coverage
        and int(row["positive_years"]) >= minimum_positive_years
    ]
    if not eligible:
        return 0.50
    selected = max(
        eligible,
        key=lambda row: (
            float(row["mean_net_return"]),
            float(row["coverage"]),
            -float(row["threshold"]),
        ),
    )
    return float(selected["threshold"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_payload(config: TrackDConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["windows"] = dict(config.windows)
    return json.loads(json.dumps(payload))


def freeze_track_d_protocol(
    *,
    output_path: Path,
    protocol_path: Path,
    implementation_paths: Iterable[Path],
    selection_artifact_paths: Iterable[Path] = (),
    input_artifact_paths: Iterable[Path] = (),
    market_data_accessed: bool,
    config: TrackDConfig | None = None,
) -> dict[str, object]:
    if market_data_accessed:
        raise ValueError("2026 market data must not be accessed before freeze")
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    paths = tuple(implementation_paths)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Implementation file(s) not found: {missing}")
    selection_paths = tuple(selection_artifact_paths)
    missing_selection = [
        str(path) for path in selection_paths if not path.is_file()
    ]
    if missing_selection:
        raise FileNotFoundError(
            "Selection artifact file(s) not found: "
            f"{missing_selection}"
        )
    input_paths = tuple(input_artifact_paths)
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            f"Input artifact file(s) not found: {missing_inputs}"
        )
    registered = TrackDConfig() if config is None else config
    payload = {
        "protocol_version": registered.protocol_version,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "market_data_2026_accessed_before_freeze": False,
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "implementation_sha256": {
            str(path): _sha256(path) for path in paths
        },
        "selection_artifact_sha256": {
            str(path): _sha256(path) for path in selection_paths
        },
        "input_artifact_sha256": {
            str(path): _sha256(path) for path in input_paths
        },
        "config": config_payload(registered),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def verify_freeze_manifest(manifest_path: Path) -> dict[str, object]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Freeze manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    registered: dict[str, str] = {
        str(payload["protocol_path"]): str(payload["protocol_sha256"]),
    }
    for field_name in (
        "implementation_sha256",
        "selection_artifact_sha256",
        "input_artifact_sha256",
    ):
        values = payload.get(field_name, {})
        if not isinstance(values, dict):
            raise TypeError(f"Freeze manifest field {field_name} is invalid")
        registered.update(
            {str(path): str(digest) for path, digest in values.items()}
        )
    for raw_path, expected in registered.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen file not found: {path}")
        if _sha256(path) != expected:
            raise ValueError(f"Frozen file hash mismatch: {path}")
    return payload
