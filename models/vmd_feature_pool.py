from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from models.baseline_common import (
    CLOSE_COLUMN,
    DATA_FOLDS_DIR,
    DATE_COLUMN,
    PROJECT_ROOT,
    TARGET_COLUMN,
    discover_folds,
)
from models.full_non_ta_feature_pool import SCALER_METADATA_NAME
from models.full_ta_feature_pool import FULL_TA_FEATURES, build_full_ta_features
from models.point_in_time_data import (
    CONTEXT_FILE_NAME,
    LABEL_DATE_COLUMN,
    MARKET_MASTER_NAME,
    POINT_IN_TIME_DATA_FOLDS_DIR,
)
from models.neural_network_folds import scale_train_test_frames

FULL_TA_VMD_DATA_FOLDS_DIR = PROJECT_ROOT / "data-folds-full-ta-vmd"
FULL_TA_VMD_NN_DATA_FOLDS_DIR = PROJECT_ROOT / "data-folds-full-ta-vmd-nn"
FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR = (
    PROJECT_ROOT / "data-folds-full-ta-vmd-point-in-time-v2"
)
FULL_TA_VMD_POINT_IN_TIME_NN_DATA_FOLDS_DIR = (
    PROJECT_ROOT / "data-folds-full-ta-vmd-point-in-time-v2-nn"
)
VMD_CONFIG_NAME = "vmd_config.json"


@dataclass(frozen=True)
class VMDConfig:
    """Configuration for leakage-safe rolling Variational Mode Decomposition."""

    window_size: int = 60
    num_modes: int = 5
    penalty: float = 1000.0
    dual_step: float = 0.0
    dc_mode: bool = True
    tolerance: float = 1e-7
    max_iterations: int = 500

    def __post_init__(self) -> None:
        if self.window_size < 4:
            raise ValueError("VMD window_size must be at least 4")
        if self.num_modes < 2:
            raise ValueError("VMD num_modes must be at least 2")
        if self.penalty <= 0.0:
            raise ValueError("VMD penalty must be positive")
        if self.dual_step < 0.0:
            raise ValueError("VMD dual_step cannot be negative")
        if self.tolerance <= 0.0:
            raise ValueError("VMD tolerance must be positive")
        if self.max_iterations < 1:
            raise ValueError("VMD max_iterations must be positive")


@dataclass(frozen=True)
class VMDResult:
    modes: np.ndarray
    center_frequencies: np.ndarray
    iterations: int
    converged: bool


def vmd_feature_names(num_modes: int) -> list[str]:
    if num_modes < 2:
        raise ValueError("At least two VMD modes are required for denoising")
    return [
        *[f"VMD_IMF_{index}" for index in range(1, num_modes)],
        "VMD_Denoised_Close",
        "VMD_Noise_Energy_Ratio",
    ]


VMD_FEATURES = vmd_feature_names(VMDConfig().num_modes)
VMDDecomposer = Callable[[np.ndarray, VMDConfig], VMDResult]


def _validate_signal(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("VMD input must be one-dimensional")
    if len(values) < 4:
        raise ValueError("VMD input must contain at least four observations")
    if not np.isfinite(values).all():
        raise ValueError("VMD input contains non-finite values")
    return values


def variational_mode_decomposition(
    signal: np.ndarray,
    config: VMDConfig = VMDConfig(),
) -> VMDResult:
    """Decompose a signal into frequency-ordered modes using the VMD ADMM updates."""

    original = _validate_signal(signal)
    original_length = len(original)
    working = original if original_length % 2 == 0 else np.pad(original, (0, 1), mode="edge")

    half_length = len(working) // 2
    mirrored = np.concatenate(
        [
            working[:half_length][::-1],
            working,
            working[-half_length:][::-1],
        ]
    )
    spectrum_length = len(mirrored)
    frequencies = (
        np.arange(1, spectrum_length + 1, dtype=float) / spectrum_length
        - 0.5
        - 1.0 / spectrum_length
    )
    positive_start = spectrum_length // 2
    signal_spectrum = np.fft.fftshift(np.fft.fft(mirrored))
    positive_spectrum = signal_spectrum.copy()
    positive_spectrum[:positive_start] = 0.0

    modes_spectrum = np.zeros(
        (config.num_modes, spectrum_length),
        dtype=np.complex128,
    )
    center_frequencies = np.arange(config.num_modes, dtype=float) / (
        2.0 * config.num_modes
    )
    if config.dc_mode:
        center_frequencies[0] = 0.0
    lagrange_multiplier = np.zeros(spectrum_length, dtype=np.complex128)
    converged = False
    iterations = 0

    for iteration in range(1, config.max_iterations + 1):
        previous_modes = modes_spectrum.copy()
        previous_centers = center_frequencies.copy()
        updated_sum = np.zeros(spectrum_length, dtype=np.complex128)
        remaining_sum = previous_modes.sum(axis=0)

        for mode_index in range(config.num_modes):
            remaining_sum -= previous_modes[mode_index]
            residual = (
                positive_spectrum
                - updated_sum
                - remaining_sum
                - lagrange_multiplier / 2.0
            )
            denominator = 1.0 + config.penalty * (
                frequencies - previous_centers[mode_index]
            ) ** 2
            modes_spectrum[mode_index] = residual / denominator

            if not (config.dc_mode and mode_index == 0):
                positive_mode = modes_spectrum[mode_index, positive_start:]
                mode_power = np.abs(positive_mode) ** 2
                power_sum = float(mode_power.sum())
                if power_sum > np.finfo(float).eps:
                    center_frequencies[mode_index] = float(
                        np.dot(
                            frequencies[positive_start:],
                            mode_power,
                        )
                        / power_sum
                    )
            updated_sum += modes_spectrum[mode_index]

        lagrange_multiplier += config.dual_step * (
            modes_spectrum.sum(axis=0) - positive_spectrum
        )
        change = float(
            np.sum(np.abs(modes_spectrum - previous_modes) ** 2)
            / spectrum_length
        )
        iterations = iteration
        if change <= config.tolerance:
            converged = True
            break

    full_spectrum = np.zeros(
        (config.num_modes, spectrum_length),
        dtype=np.complex128,
    )
    full_spectrum[:, positive_start:] = modes_spectrum[:, positive_start:]
    mirrored_indexes = np.arange(1, positive_start + 1)[::-1]
    full_spectrum[:, mirrored_indexes] = np.conj(
        modes_spectrum[:, positive_start:]
    )
    full_spectrum[:, 0] = np.conj(full_spectrum[:, -1])

    reconstructed = np.real(
        np.fft.ifft(
            np.fft.ifftshift(full_spectrum, axes=1),
            axis=1,
        )
    )
    crop_start = spectrum_length // 4
    cropped_modes = reconstructed[
        :,
        crop_start : crop_start + len(working),
    ][:, :original_length]

    order = np.argsort(center_frequencies, kind="stable")
    sorted_modes = cropped_modes[order]
    sorted_centers = center_frequencies[order]
    if not np.isfinite(sorted_modes).all():
        raise ValueError("VMD produced non-finite modes")
    return VMDResult(
        modes=sorted_modes,
        center_frequencies=sorted_centers,
        iterations=iterations,
        converged=converged,
    )


def build_rolling_vmd_features(
    close: np.ndarray | pd.Series,
    config: VMDConfig = VMDConfig(),
    decomposer: VMDDecomposer = variational_mode_decomposition,
) -> pd.DataFrame:
    """Build past-only VMD features; each row uses [t-window+1, ..., t]."""

    values = np.asarray(close, dtype=float)
    if values.ndim != 1:
        raise ValueError("Close series must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("Close series contains non-finite values")

    names = vmd_feature_names(config.num_modes)
    feature_values = np.full((len(values), len(names)), np.nan, dtype=float)
    for end_index in range(config.window_size - 1, len(values)):
        start_index = end_index - config.window_size + 1
        window = values[start_index : end_index + 1]
        result = decomposer(window, config)
        modes = np.asarray(result.modes, dtype=float)
        centers = np.asarray(result.center_frequencies, dtype=float)
        expected_shape = (config.num_modes, config.window_size)
        if modes.shape != expected_shape:
            raise ValueError(
                f"VMD modes have shape {modes.shape}; expected {expected_shape}"
            )
        if centers.shape != (config.num_modes,):
            raise ValueError("VMD center frequencies have an invalid shape")
        order = np.argsort(centers, kind="stable")
        ordered_modes = modes[order]
        retained_modes = ordered_modes[:-1]
        noise_mode = ordered_modes[-1]
        signal_energy = max(
            float(np.mean(window**2)),
            np.finfo(float).eps,
        )
        row = [
            *retained_modes[:, -1].tolist(),
            float(retained_modes[:, -1].sum()),
            float(np.mean(noise_mode**2) / signal_energy),
        ]
        feature_values[end_index] = row

    return pd.DataFrame(feature_values, columns=names)


def build_full_ta_vmd_features(
    frame: pd.DataFrame,
    config: VMDConfig = VMDConfig(),
    decomposer: VMDDecomposer = variational_mode_decomposition,
) -> pd.DataFrame:
    ordered = frame.copy()
    ordered[DATE_COLUMN] = pd.to_datetime(ordered[DATE_COLUMN])
    ordered = ordered.sort_values(DATE_COLUMN).reset_index(drop=True)
    if ordered[DATE_COLUMN].duplicated().any():
        raise ValueError("VMD input contains duplicate dates")

    full_ta = build_full_ta_features(ordered)
    rolling_vmd = build_rolling_vmd_features(
        ordered[CLOSE_COLUMN],
        config=config,
        decomposer=decomposer,
    )
    result = pd.concat([full_ta, rolling_vmd], axis=1)
    metadata_columns = (
        [LABEL_DATE_COLUMN]
        if LABEL_DATE_COLUMN in result.columns
        else []
    )
    columns = [
        DATE_COLUMN,
        *metadata_columns,
        *FULL_TA_FEATURES,
        *vmd_feature_names(config.num_modes),
        TARGET_COLUMN,
    ]
    return result.loc[:, columns]


def prepare_full_ta_vmd_fold_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: VMDConfig = VMDConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dates = set(pd.to_datetime(train[DATE_COLUMN]))
    test_dates = set(pd.to_datetime(test[DATE_COLUMN]))
    combined = pd.concat([train, test], ignore_index=True)
    features = build_full_ta_vmd_features(combined, config=config)
    features[DATE_COLUMN] = pd.to_datetime(features[DATE_COLUMN])
    cleaned = features.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    train_features = cleaned[cleaned[DATE_COLUMN].isin(train_dates)].copy()
    test_features = cleaned[cleaned[DATE_COLUMN].isin(test_dates)].copy()
    if train_features.empty or test_features.empty:
        raise ValueError("VMD preprocessing removed an entire train or test split")
    train_features[DATE_COLUMN] = train_features[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    test_features[DATE_COLUMN] = test_features[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    if LABEL_DATE_COLUMN in train_features.columns:
        train_features[LABEL_DATE_COLUMN] = pd.to_datetime(
            train_features[LABEL_DATE_COLUMN]
        ).dt.strftime("%Y-%m-%d")
        test_features[LABEL_DATE_COLUMN] = pd.to_datetime(
            test_features[LABEL_DATE_COLUMN]
        ).dt.strftime("%Y-%m-%d")
    return train_features, test_features


def create_full_ta_vmd_folds(
    source_dir: Path = DATA_FOLDS_DIR,
    output_dir: Path = FULL_TA_VMD_DATA_FOLDS_DIR,
    config: VMDConfig = VMDConfig(),
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = discover_folds(source_dir)
    source_frames: list[pd.DataFrame] = []
    split_dates: dict[str, tuple[set[pd.Timestamp], set[pd.Timestamp]]] = {}
    for spec in specs:
        train = pd.read_csv(spec.train_path)
        test = pd.read_csv(spec.test_path)
        train_dates = set(pd.to_datetime(train[DATE_COLUMN]))
        test_dates = set(pd.to_datetime(test[DATE_COLUMN]))
        split_dates[spec.fold] = (train_dates, test_dates)
        source_frames.extend([train, test])

    master_path = source_dir / MARKET_MASTER_NAME
    if master_path.exists():
        master_source = pd.read_csv(master_path)
        master_source[DATE_COLUMN] = pd.to_datetime(
            master_source[DATE_COLUMN]
        )
        master_source = master_source.sort_values(DATE_COLUMN).reset_index(
            drop=True
        )
    else:
        combined_source = pd.concat(source_frames, ignore_index=True)
        combined_source[DATE_COLUMN] = pd.to_datetime(
            combined_source[DATE_COLUMN]
        )
        value_columns = [
            column
            for column in combined_source.columns
            if column != DATE_COLUMN
        ]
        conflicts = (
            combined_source.groupby(DATE_COLUMN, sort=False)[value_columns]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicts.any():
            conflict_dates = (
                conflicts[conflicts].index.strftime("%Y-%m-%d").tolist()
            )
            raise ValueError(
                f"Source folds disagree on shared dates: {conflict_dates[:5]}"
            )
        master_source = (
            combined_source.drop_duplicates(DATE_COLUMN)
            .sort_values(DATE_COLUMN)
            .reset_index(drop=True)
        )
    vmd_iterations: list[int] = []
    converged_windows = 0

    def tracked_decomposer(
        signal: np.ndarray,
        used_config: VMDConfig,
    ) -> VMDResult:
        nonlocal converged_windows
        result = variational_mode_decomposition(signal, used_config)
        vmd_iterations.append(result.iterations)
        converged_windows += int(result.converged)
        return result

    master_features = build_full_ta_vmd_features(
        master_source,
        config=config,
        decomposer=tracked_decomposer,
    )
    master_features[DATE_COLUMN] = pd.to_datetime(master_features[DATE_COLUMN])
    cleaned_features = (
        master_features.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    config_payload = {
        **asdict(config),
        "input_signal": CLOSE_COLUMN,
        "causal_scope": "rolling past-only including current row",
        "context_source": (
            str(master_path)
            if master_path.exists()
            else "union of source fold rows"
        ),
        "purged_boundary_rows_used_as_context_only": master_path.exists(),
        "denoising_rule": "remove the highest-center-frequency mode",
        "feature_columns": vmd_feature_names(config.num_modes),
        "generation_diagnostics": {
            "total_windows": len(vmd_iterations),
            "converged_windows": converged_windows,
            "non_converged_windows": len(vmd_iterations) - converged_windows,
            "convergence_rate": (
                converged_windows / len(vmd_iterations)
                if vmd_iterations
                else 0.0
            ),
            "mean_iterations": (
                float(np.mean(vmd_iterations))
                if vmd_iterations
                else 0.0
            ),
            "max_iterations_used": max(vmd_iterations, default=0),
        },
    }
    for spec in specs:
        train_dates, test_dates = split_dates[spec.fold]
        train_features = cleaned_features[
            cleaned_features[DATE_COLUMN].isin(train_dates)
        ].copy()
        test_features = cleaned_features[
            cleaned_features[DATE_COLUMN].isin(test_dates)
        ].copy()
        train_end = max(train_dates)
        test_start = min(test_dates)
        context_features = cleaned_features.loc[
            cleaned_features[DATE_COLUMN].gt(train_end)
            & cleaned_features[DATE_COLUMN].lt(test_start)
        ].copy()
        if train_features.empty or test_features.empty:
            raise ValueError(f"VMD preprocessing removed all rows from {spec.fold}")
        if master_path.exists() and context_features.empty:
            raw_dates = pd.to_datetime(master_source[DATE_COLUMN])
            if raw_dates.gt(train_end).lt(test_start).any():
                raise ValueError(
                    f"VMD preprocessing removed all context rows from {spec.fold}"
                )
        for frame in (train_features, context_features, test_features):
            frame[DATE_COLUMN] = frame[DATE_COLUMN].dt.strftime("%Y-%m-%d")
            if LABEL_DATE_COLUMN in frame.columns:
                frame[LABEL_DATE_COLUMN] = pd.to_datetime(
                    frame[LABEL_DATE_COLUMN]
                ).dt.strftime("%Y-%m-%d")
        fold_dir = output_dir / spec.fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_features.to_csv(fold_dir / spec.train_path.name, index=False)
        if not context_features.empty:
            context_features.to_csv(
                fold_dir / CONTEXT_FILE_NAME,
                index=False,
            )
        test_features.to_csv(fold_dir / spec.test_path.name, index=False)
        with (fold_dir / VMD_CONFIG_NAME).open("w", encoding="utf-8") as file:
            json.dump(config_payload, file, indent=2)
    return output_dir


def create_scaled_full_ta_vmd_nn_folds(
    source_dir: Path = FULL_TA_VMD_DATA_FOLDS_DIR,
    output_dir: Path = FULL_TA_VMD_NN_DATA_FOLDS_DIR,
) -> Path:
    if not source_dir.exists():
        create_full_ta_vmd_folds(output_dir=source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in discover_folds(source_dir):
        train = pd.read_csv(spec.train_path)
        test = pd.read_csv(spec.test_path)
        context_path = spec.train_path.parent / CONTEXT_FILE_NAME
        context = (
            pd.read_csv(context_path)
            if context_path.is_file()
            else None
        )
        scaled_train, scaled_test, metadata = scale_train_test_frames(train, test)
        scaled_context = None
        if context is not None:
            columns = list(metadata["columns"])
            scaled_context = context.copy()
            scale = np.asarray(metadata["scale"], dtype=float)
            offset = np.asarray(metadata["min"], dtype=float)
            scaled_context.loc[:, columns] = (
                context.loc[:, columns].to_numpy(dtype=float) * scale
                + offset
            )
        metadata["source_dir"] = str(source_dir)
        fold_dir = output_dir / spec.fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        scaled_train.to_csv(fold_dir / spec.train_path.name, index=False)
        if scaled_context is not None:
            scaled_context.to_csv(
                fold_dir / CONTEXT_FILE_NAME,
                index=False,
            )
        scaled_test.to_csv(fold_dir / spec.test_path.name, index=False)
        with (fold_dir / SCALER_METADATA_NAME).open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        config_source = spec.train_path.parent / VMD_CONFIG_NAME
        if config_source.exists():
            config_payload = json.loads(config_source.read_text(encoding="utf-8"))
            with (fold_dir / VMD_CONFIG_NAME).open("w", encoding="utf-8") as file:
                json.dump(config_payload, file, indent=2)
    return output_dir


def main() -> tuple[Path, Path]:
    original_dir = create_full_ta_vmd_folds(
        source_dir=POINT_IN_TIME_DATA_FOLDS_DIR,
        output_dir=FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
    )
    scaled_dir = create_scaled_full_ta_vmd_nn_folds(
        source_dir=FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
        output_dir=FULL_TA_VMD_POINT_IN_TIME_NN_DATA_FOLDS_DIR,
    )
    print(f"Created Full TA + VMD folds at {original_dir}")
    print(f"Created scaled Full TA + VMD folds at {scaled_dir}")
    return original_dir, scaled_dir


if __name__ == "__main__":
    main()
