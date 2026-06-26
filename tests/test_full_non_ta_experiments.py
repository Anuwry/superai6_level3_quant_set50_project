from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.full_non_ta_experiments import (
    FULL_NON_TA_OUTPUT_DIR,
    LSTM_WINDOWS,
    make_ridge_full_non_ta_model,
)


def test_ridge_full_non_ta_uses_standard_scaler():
    model = make_ridge_full_non_ta_model()

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)


def test_lstm_full_non_ta_windows_are_explicit():
    assert LSTM_WINDOWS == [5, 10, 20, 40, 60]


def test_full_non_ta_output_dir_is_separate_from_naive():
    assert FULL_NON_TA_OUTPUT_DIR.name == "full_non_ta_feature_pool"
