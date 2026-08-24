import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_common import TARGET_COLUMN, discover_folds
from models.vmd_feature_pool import (
    FULL_TA_VMD_DATA_FOLDS_DIR,
    FULL_TA_VMD_NN_DATA_FOLDS_DIR,
    VMD_FEATURES,
    VMDConfig,
    VMDResult,
    build_rolling_vmd_features,
    variational_mode_decomposition,
)
from models.vmd_experiments import (
    COMPARISON_BY_FOLD_FILE,
    COMPARISON_FILE,
    VMD_MODEL_RUNNERS,
    VMD_RESULT_DIRS,
)


def test_vmd_configuration_and_feature_contract():
    config = VMDConfig()

    assert config.window_size == 60
    assert config.num_modes == 5
    assert config.penalty == 1000.0
    assert config.dc_mode is True
    assert VMD_FEATURES == [
        "VMD_IMF_1",
        "VMD_IMF_2",
        "VMD_IMF_3",
        "VMD_IMF_4",
        "VMD_Denoised_Close",
        "VMD_Noise_Energy_Ratio",
    ]
    assert TARGET_COLUMN not in VMD_FEATURES


def test_vmd_reconstructs_a_synthetic_signal_and_sorts_center_frequencies():
    index = np.arange(128, dtype=float)
    signal = (
        10.0
        + 1.5 * np.sin(2.0 * np.pi * index / 32.0)
        + 0.3 * np.sin(2.0 * np.pi * index / 4.0)
    )

    result = variational_mode_decomposition(
        signal,
        VMDConfig(
            window_size=128,
            num_modes=3,
            penalty=1000.0,
            max_iterations=500,
        ),
    )

    assert result.modes.shape == (3, 128)
    assert np.all(np.diff(result.center_frequencies) >= 0.0)
    assert np.sqrt(np.mean((result.modes.sum(axis=0) - signal) ** 2)) < 0.5
    assert result.converged


def test_rolling_vmd_features_are_causal_when_future_values_change():
    index = np.arange(80, dtype=float)
    close = 100.0 + 0.2 * index + np.sin(index / 4.0)
    changed = close.copy()
    changed[60:] += 1000.0
    config = VMDConfig(
        window_size=32,
        num_modes=3,
        penalty=1000.0,
        max_iterations=300,
    )

    original_features = build_rolling_vmd_features(close, config=config)
    changed_features = build_rolling_vmd_features(changed, config=config)

    np.testing.assert_allclose(
        original_features.loc[:59].to_numpy(dtype=float),
        changed_features.loc[:59].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-10,
        equal_nan=True,
    )


def test_denoised_close_excludes_only_the_highest_frequency_mode():
    def fake_decomposer(signal: np.ndarray, config: VMDConfig) -> VMDResult:
        length = len(signal)
        modes = np.vstack(
            [
                np.full(length, 10.0),
                np.full(length, 2.0),
                np.full(length, 1.0),
            ]
        )
        return VMDResult(
            modes=modes,
            center_frequencies=np.array([0.0, 0.1, 0.4]),
            iterations=1,
            converged=True,
        )

    features = build_rolling_vmd_features(
        np.arange(1.0, 9.0),
        config=VMDConfig(window_size=4, num_modes=3),
        decomposer=fake_decomposer,
    )

    final = features.iloc[-1]
    assert final["VMD_IMF_1"] == 10.0
    assert final["VMD_IMF_2"] == 2.0
    assert final["VMD_Denoised_Close"] == 12.0
    assert final["VMD_Noise_Energy_Ratio"] == 1.0 / np.mean(np.arange(5.0, 9.0) ** 2)


def test_vmd_runner_contains_exactly_the_five_requested_models():
    assert list(VMD_MODEL_RUNNERS) == [
        "lstm",
        "cnn",
        "lstm_cnn",
        "lstm_attention",
        "lstm_cnn_attention",
    ]
    assert "attention_lstm_cnn" not in VMD_MODEL_RUNNERS


def test_generated_vmd_folds_are_finite_and_scaled_train_only():
    original_specs = discover_folds(FULL_TA_VMD_DATA_FOLDS_DIR)
    scaled_specs = discover_folds(FULL_TA_VMD_NN_DATA_FOLDS_DIR)

    assert len(original_specs) == len(scaled_specs) == 4
    for original_spec, scaled_spec in zip(original_specs, scaled_specs, strict=True):
        train = pd.read_csv(original_spec.train_path)
        test = pd.read_csv(original_spec.test_path)
        scaled_train = pd.read_csv(scaled_spec.train_path)
        metadata_path = scaled_spec.train_path.parent / "minmax_scaler.json"
        vmd_config_path = original_spec.train_path.parent / "vmd_config.json"

        assert list(train.columns) == list(test.columns)
        assert all(feature in train.columns for feature in VMD_FEATURES)
        assert np.isfinite(train.drop(columns=["Date"]).to_numpy(dtype=float)).all()
        assert np.isfinite(test.drop(columns=["Date"]).to_numpy(dtype=float)).all()
        assert metadata_path.exists()
        assert vmd_config_path.exists()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        vmd_metadata = json.loads(vmd_config_path.read_text(encoding="utf-8"))
        assert metadata["fit_scope"] == "train_only"
        assert metadata["source_dir"] == str(FULL_TA_VMD_DATA_FOLDS_DIR)
        diagnostics = vmd_metadata["generation_diagnostics"]
        assert diagnostics["total_windows"] > 0
        assert 0 <= diagnostics["converged_windows"] <= diagnostics["total_windows"]
        assert 0.0 <= diagnostics["convergence_rate"] <= 1.0
        assert diagnostics["max_iterations_used"] <= vmd_metadata["max_iterations"]
        assert scaled_train.drop(columns=["Date"]).min().min() >= -1e-12
        assert scaled_train.drop(columns=["Date"]).max().max() <= 1.0 + 1e-12


def test_vmd_benchmark_outputs_cover_five_models_and_four_folds():
    aggregate = pd.read_csv(COMPARISON_FILE)
    by_fold = pd.read_csv(COMPARISON_BY_FOLD_FILE)

    assert aggregate["model"].tolist() == list(VMD_MODEL_RUNNERS)
    assert set(by_fold["model"]) == set(VMD_MODEL_RUNNERS)
    assert len(by_fold) == 5 * 4
    assert by_fold.groupby("model")["fold"].nunique().eq(4).all()
    assert np.isfinite(aggregate.select_dtypes(include=[np.number])).all().all()
    assert np.isfinite(by_fold.select_dtypes(include=[np.number])).all().all()

    for result_dir in VMD_RESULT_DIRS.values():
        metrics = pd.read_csv(
            COMPARISON_FILE.parent / result_dir / "metrics_by_fold.csv"
        )
        assert len(metrics) == 4
        assert metrics["n_test"].tolist() == [241, 243, 244, 234]
