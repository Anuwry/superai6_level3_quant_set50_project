from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from models.baseline_common import DATE_COLUMN, PROJECT_ROOT, discover_folds
from models.full_non_ta_feature_pool import SCALER_METADATA_NAME
from models.full_ta_feature_pool import (
    FULL_TA_DATA_FOLDS_DIR,
    FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR,
)
from models.neural_network_folds import scale_train_test_frames
from models.point_in_time_data import (
    CONTEXT_FILE_NAME,
    LABEL_DATE_COLUMN,
    POINT_IN_TIME_CONTRACT_NAME,
    purge_cross_boundary_training_labels,
)
from models.vmd_feature_pool import (
    FULL_TA_VMD_DATA_FOLDS_DIR,
    FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
    VMD_CONFIG_NAME,
)

TRACK_A_SELECTION_DATA_DIR = PROJECT_ROOT / "data-track-a-window-selection"
FULL_TA_SELECTION_DIR = TRACK_A_SELECTION_DATA_DIR / "full_ta"
FULL_TA_SELECTION_NN_DIR = TRACK_A_SELECTION_DATA_DIR / "full_ta_nn"
FULL_TA_VMD_SELECTION_DIR = TRACK_A_SELECTION_DATA_DIR / "full_ta_vmd"
FULL_TA_VMD_SELECTION_NN_DIR = TRACK_A_SELECTION_DATA_DIR / "full_ta_vmd_nn"
TRACK_A_POINT_IN_TIME_SELECTION_DATA_DIR = (
    PROJECT_ROOT / "data-track-a-window-selection-point-in-time-v2"
)
FULL_TA_POINT_IN_TIME_SELECTION_DIR = (
    TRACK_A_POINT_IN_TIME_SELECTION_DATA_DIR / "full_ta"
)
FULL_TA_POINT_IN_TIME_SELECTION_NN_DIR = (
    TRACK_A_POINT_IN_TIME_SELECTION_DATA_DIR / "full_ta_nn"
)
FULL_TA_VMD_POINT_IN_TIME_SELECTION_DIR = (
    TRACK_A_POINT_IN_TIME_SELECTION_DATA_DIR / "full_ta_vmd"
)
FULL_TA_VMD_POINT_IN_TIME_SELECTION_NN_DIR = (
    TRACK_A_POINT_IN_TIME_SELECTION_DATA_DIR / "full_ta_vmd_nn"
)


def _validate_selection_years(
    selection_years: Iterable[int],
    first_test_year: int,
) -> tuple[int, ...]:
    years = tuple(selection_years)
    if not years:
        raise ValueError("At least one selection year is required")
    if any(isinstance(year, bool) or not isinstance(year, int) for year in years):
        raise ValueError("Selection years must be integers")
    if len(set(years)) != len(years) or tuple(sorted(years)) != years:
        raise ValueError("Selection years must be unique and increasing")
    if max(years) >= first_test_year:
        raise ValueError("Selection years must precede the first outer test")
    return years


def _selection_source_spec(source_dir: Path, first_test_year: int):
    candidates = [
        spec
        for spec in discover_folds(source_dir)
        if spec.train_end_year == first_test_year - 1
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one source fold ending immediately before "
            f"{first_test_year}; found {len(candidates)}"
        )
    return candidates[0]


def create_pretest_selection_folds(
    source_dir: Path,
    output_dir: Path,
    *,
    selection_years: Iterable[int],
    first_test_year: int,
) -> Path:
    years = _validate_selection_years(selection_years, first_test_year)
    source_spec = _selection_source_spec(source_dir, first_test_year)
    source = pd.read_csv(
        source_spec.train_path,
        parse_dates=[DATE_COLUMN, LABEL_DATE_COLUMN],
    )
    source = source.sort_values(DATE_COLUMN).reset_index(drop=True)
    source_years = set(source[DATE_COLUMN].dt.year)
    missing_years = sorted(set(years).difference(source_years))
    if missing_years:
        raise ValueError(f"Source data is missing selection years: {missing_years}")

    output_dir.mkdir(parents=True, exist_ok=True)
    vmd_config_path = source_spec.train_path.parent / VMD_CONFIG_NAME
    vmd_config = (
        json.loads(vmd_config_path.read_text(encoding="utf-8"))
        if vmd_config_path.exists()
        else None
    )
    for fold_index, selection_year in enumerate(years, start=1):
        train = source[source[DATE_COLUMN].dt.year < selection_year].copy()
        validation = source[source[DATE_COLUMN].dt.year == selection_year].copy()
        if train.empty or validation.empty:
            raise ValueError(
                f"Cannot create pretest selection fold for {selection_year}"
            )
        if train[DATE_COLUMN].max() >= validation[DATE_COLUMN].min():
            raise ValueError(f"Selection fold {selection_year} is not temporal")
        candidate_train = train.copy()
        train, purge_audit = purge_cross_boundary_training_labels(
            train,
            validation,
        )
        retained_dates = set(train[DATE_COLUMN])
        boundary_context = candidate_train.loc[
            ~candidate_train[DATE_COLUMN].isin(retained_dates)
        ].copy()

        train_start_year = int(train[DATE_COLUMN].dt.year.min())
        train_end_year = int(train[DATE_COLUMN].dt.year.max())
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        saved_train = train.assign(
            **{
                DATE_COLUMN: train[DATE_COLUMN].dt.strftime("%Y-%m-%d"),
                LABEL_DATE_COLUMN: train[LABEL_DATE_COLUMN].dt.strftime(
                    "%Y-%m-%d"
                ),
            }
        )
        saved_validation = validation.assign(
            **{
                DATE_COLUMN: validation[DATE_COLUMN].dt.strftime("%Y-%m-%d"),
                LABEL_DATE_COLUMN: validation[
                    LABEL_DATE_COLUMN
                ].dt.strftime("%Y-%m-%d"),
            }
        )
        saved_context = boundary_context.copy()
        if not saved_context.empty:
            saved_context[DATE_COLUMN] = saved_context[
                DATE_COLUMN
            ].dt.strftime("%Y-%m-%d")
            saved_context[LABEL_DATE_COLUMN] = saved_context[
                LABEL_DATE_COLUMN
            ].dt.strftime("%Y-%m-%d")
        saved_train.to_csv(
            fold_dir / f"train_{train_start_year}_{train_end_year}.csv",
            index=False,
        )
        if not saved_context.empty:
            saved_context.to_csv(
                fold_dir / CONTEXT_FILE_NAME,
                index=False,
            )
        saved_validation.to_csv(
            fold_dir / f"test_{selection_year}.csv",
            index=False,
        )
        contract = {
            "protocol_version": "point_in_time_v2",
            "stage": "pretest_window_selection",
            "selection_year": selection_year,
            "purge_rule": "retain Label_Date < first validation Date",
            **asdict(purge_audit),
        }
        with (fold_dir / POINT_IN_TIME_CONTRACT_NAME).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(contract, file, indent=2)
        if vmd_config is not None:
            with (fold_dir / VMD_CONFIG_NAME).open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(vmd_config, file, indent=2)
    return output_dir


def create_scaled_selection_folds(
    source_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in discover_folds(source_dir):
        train = pd.read_csv(spec.train_path)
        validation = pd.read_csv(spec.test_path)
        context_path = spec.train_path.parent / CONTEXT_FILE_NAME
        context = (
            pd.read_csv(context_path)
            if context_path.is_file()
            else None
        )
        scaled_train, scaled_validation, metadata = scale_train_test_frames(
            train,
            validation,
        )
        columns = list(metadata["columns"])
        scale = pd.Series(metadata["scale"], index=columns, dtype=float)
        offset = pd.Series(metadata["min"], index=columns, dtype=float)
        scaled_context = None
        if context is not None:
            scaled_context = context.copy()
            scaled_context.loc[:, columns] = (
                context.loc[:, columns].astype(float) * scale + offset
            )
        selection_metadata = {
            **metadata,
            "fit_scope": "selection_train_only",
            "selection_year": spec.test_year,
            "source_dir": str(source_dir),
        }
        fold_dir = output_dir / spec.fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        scaled_train.to_csv(fold_dir / spec.train_path.name, index=False)
        if scaled_context is not None:
            scaled_context.to_csv(
                fold_dir / CONTEXT_FILE_NAME,
                index=False,
            )
        scaled_validation.to_csv(
            fold_dir / spec.test_path.name,
            index=False,
        )
        with (fold_dir / SCALER_METADATA_NAME).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(selection_metadata, file, indent=2)

        vmd_config_path = spec.train_path.parent / VMD_CONFIG_NAME
        if vmd_config_path.exists():
            vmd_config = json.loads(vmd_config_path.read_text(encoding="utf-8"))
            with (fold_dir / VMD_CONFIG_NAME).open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(vmd_config, file, indent=2)
    return output_dir


def prepare_track_a_selection_data(
    *,
    selection_years: Iterable[int],
    first_test_year: int,
    full_ta_source_dir: Path = FULL_TA_POINT_IN_TIME_DATA_FOLDS_DIR,
    full_ta_selection_dir: Path = FULL_TA_POINT_IN_TIME_SELECTION_DIR,
    full_ta_selection_nn_dir: Path = FULL_TA_POINT_IN_TIME_SELECTION_NN_DIR,
    vmd_source_dir: Path = FULL_TA_VMD_POINT_IN_TIME_DATA_FOLDS_DIR,
    vmd_selection_dir: Path = FULL_TA_VMD_POINT_IN_TIME_SELECTION_DIR,
    vmd_selection_nn_dir: Path = (
        FULL_TA_VMD_POINT_IN_TIME_SELECTION_NN_DIR
    ),
) -> dict[str, Path]:
    create_pretest_selection_folds(
        full_ta_source_dir,
        full_ta_selection_dir,
        selection_years=selection_years,
        first_test_year=first_test_year,
    )
    create_scaled_selection_folds(
        full_ta_selection_dir,
        full_ta_selection_nn_dir,
    )
    create_pretest_selection_folds(
        vmd_source_dir,
        vmd_selection_dir,
        selection_years=selection_years,
        first_test_year=first_test_year,
    )
    create_scaled_selection_folds(
        vmd_selection_dir,
        vmd_selection_nn_dir,
    )
    return {
        "full_ta": full_ta_selection_dir,
        "full_ta_nn": full_ta_selection_nn_dir,
        "full_ta_vmd": vmd_selection_dir,
        "full_ta_vmd_nn": vmd_selection_nn_dir,
    }
