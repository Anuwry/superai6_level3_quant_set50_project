from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.pit_cdr_lstm_runner import (
    VARIANTS,
    build_six_model_table,
    validate_candidate_cohort,
)


def _frozen_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in ("lstm", "cnn", "lstm_cnn", "lstm_attention", "lstm_cnn_attention"):
        for date, close, actual, prediction in (
            ("2024-01-02", 100.0, 101.0, 100.5),
            ("2024-01-03", 101.0, 100.0, 100.5),
            ("2025-01-02", 100.0, 99.0, 99.5),
            ("2025-01-03", 99.0, 100.0, 99.5),
        ):
            rows.append(
                {
                    "model": model,
                    "test_year": int(date[:4]),
                    "Date": date,
                    "Close_D": close,
                    "y_true": actual,
                    "y_pred": prediction,
                }
            )
    return pd.DataFrame(rows)


def _candidate_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        for seed in (42, 123):
            for date, close, actual, probability in (
                ("2024-01-02", 100.0, 101.0, 0.8),
                ("2024-01-03", 101.0, 100.0, 0.2),
                ("2025-01-02", 100.0, 99.0, 0.2),
                ("2025-01-03", 99.0, 100.0, 0.8),
            ):
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "test_year": int(date[:4]),
                        "Date": date,
                        "Close_D": close,
                        "y_true": actual,
                        "probability": probability,
                    }
                )
    return pd.DataFrame(rows)


def test_candidate_must_match_frozen_dates_and_actuals() -> None:
    frozen = _frozen_predictions()
    candidate = _candidate_predictions()

    audit = validate_candidate_cohort(candidate, frozen, seeds=(42, 123))

    assert audit["passed"] is True
    assert audit["rows_per_variant_seed"] == 4
    changed = candidate.copy()
    changed.loc[0, "y_true"] = 999.0
    with pytest.raises(ValueError, match="actual values"):
        validate_candidate_cohort(changed, frozen, seeds=(42, 123))


def test_six_model_table_uses_seed_averaged_candidate_probability() -> None:
    table = build_six_model_table(
        _candidate_predictions(),
        _frozen_predictions(),
        seeds=(42, 123),
    )

    assert len(table) == 6
    assert set(table["model"]) == {
        "LSTM",
        "CNN",
        "LSTM-CNN",
        "LSTM-Attention",
        "LSTM-CNN-Attention",
        "PIT-CDR-LSTM",
    }
    ours = table.loc[table["model"].eq("PIT-CDR-LSTM")].iloc[0]
    assert ours["balanced_accuracy_mean"] == pytest.approx(1.0)
    assert ours["direction_accuracy_mean"] == pytest.approx(1.0)
    assert np.isnan(ours["rmse_mean"])
