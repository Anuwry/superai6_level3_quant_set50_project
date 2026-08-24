import json

import pandas as pd

from models.baseline_common import DATE_COLUMN
from models.point_in_time_data import LABEL_DATE_COLUMN
from models.time_based_validation import (
    FULL_TA_TUNING_DIR,
    FULL_TA_TUNING_NN_DIR,
    discover_validation_folds,
)


def test_full_ta_tuning_folds_have_strict_temporal_order():
    specs = discover_validation_folds(FULL_TA_TUNING_DIR)

    assert len(specs) == 4
    for spec in specs:
        train = pd.read_csv(
            spec.train_path,
            parse_dates=[DATE_COLUMN, LABEL_DATE_COLUMN],
        )
        validation = pd.read_csv(spec.validation_path, parse_dates=[DATE_COLUMN])
        test = pd.read_csv(spec.test_path, parse_dates=[DATE_COLUMN])

        assert train[DATE_COLUMN].max() < validation[DATE_COLUMN].min()
        assert train[LABEL_DATE_COLUMN].max() < validation[DATE_COLUMN].min()
        assert validation[DATE_COLUMN].max() < test[DATE_COLUMN].min()
        assert set(validation[DATE_COLUMN].dt.year) == {spec.validation_year}
        assert set(test[DATE_COLUMN].dt.year) == {spec.test_year}
        assert list(train.columns) == list(validation.columns) == list(test.columns)


def test_full_ta_nn_scaler_is_fit_on_inner_train_only():
    specs = discover_validation_folds(FULL_TA_TUNING_NN_DIR)

    assert len(specs) == 4
    for spec in specs:
        train = pd.read_csv(spec.train_path)
        numeric_train = train.drop(columns=[DATE_COLUMN, LABEL_DATE_COLUMN])
        metadata_path = spec.train_path.parent / "minmax_scaler.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        assert metadata["fit_scope"] == "inner_train_only"
        assert metadata["validation_year"] == spec.validation_year
        assert LABEL_DATE_COLUMN not in metadata["columns"]
        assert numeric_train.min().min() >= -1e-12
        assert numeric_train.max().max() <= 1.0 + 1e-12
