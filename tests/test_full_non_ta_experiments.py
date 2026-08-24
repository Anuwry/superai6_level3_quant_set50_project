from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.full_non_ta_experiments import (
    ATTENTION_LSTM_CNN_SEQUENCE_LENGTH,
    ATTENTION_LSTM_SEQUENCE_LENGTH,
    CNN_SEQUENCE_LENGTH,
    FULL_NON_TA_OUTPUT_DIR,
    LSTM_CNN_SEQUENCE_LENGTH,
    LSTM_SEQUENCE_LENGTH,
    make_ridge_full_non_ta_model,
)


def test_ridge_full_non_ta_uses_standard_scaler():
    model = make_ridge_full_non_ta_model()

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)


def test_lstm_full_non_ta_uses_naive_baseline_sequence_length():
    assert LSTM_SEQUENCE_LENGTH == 20


def test_cnn_full_non_ta_uses_same_comparison_window_as_lstm():
    assert CNN_SEQUENCE_LENGTH == LSTM_SEQUENCE_LENGTH


def test_lstm_cnn_full_non_ta_uses_same_comparison_window_as_lstm():
    assert LSTM_CNN_SEQUENCE_LENGTH == LSTM_SEQUENCE_LENGTH


def test_attention_models_use_same_comparison_window_as_lstm():
    assert ATTENTION_LSTM_SEQUENCE_LENGTH == LSTM_SEQUENCE_LENGTH
    assert ATTENTION_LSTM_CNN_SEQUENCE_LENGTH == LSTM_SEQUENCE_LENGTH


def test_full_non_ta_output_dir_is_separate_from_naive():
    assert FULL_NON_TA_OUTPUT_DIR.name == "full_non_ta_feature_pool"
