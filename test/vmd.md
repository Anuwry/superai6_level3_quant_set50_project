# Full TA + causal rolling VMD benchmark

> **Historical benchmark note:** This file records the initial fixed-window-20
> VMD experiment and is retained for traceability. For paper claims about input
> sequence length, use `test/vmd_window_sweep.md` and
> `outputs/full_ta_vmd_window_sweep`. The newer sweep resets the Keras session
> and seed for every fold, enables deterministic TensorFlow operations, records
> runtime, and reruns window 20 under the same reproducibility protocol as
> windows 1, 3, 5, and 10.

## Objective

Add Variational Mode Decomposition (VMD) features to the existing Full TA
feature pool and benchmark the following five neural-network models:

1. LSTM
2. CNN
3. LSTM-CNN
4. LSTM-Attention (`LSTM -> causal MultiHeadAttention`)
5. LSTM-CNN-Attention (`LSTM -> causal Conv1D -> causal MultiHeadAttention`)

The old Attention-LSTM-CNN model is intentionally excluded.

## Leakage-safe VMD design

- Input signal: `Close_D` only.
- Rolling window: 60 trading days.
- Each feature row at time `t` uses only observations from `t-59` through `t`.
- Number of modes: 5.
- Penalty factor (alpha): 1000.
- Dual ascent step (tau): 0.
- DC mode: enabled.
- Tolerance: `1e-7`.
- Maximum iterations: 500.
- Modes are ordered by final center frequency.
- The highest-frequency mode is treated as noise.
- Generation diagnostics: 3,297 of 3,322 rolling windows converged within
  500 iterations (99.25%); every window produced finite features.

Added features:

- `VMD_IMF_1` to `VMD_IMF_4`
- `VMD_Denoised_Close` (sum of the four retained modes at time `t`)
- `VMD_Noise_Energy_Ratio`

The 60-day window matches the longest existing Full TA lookback, so all Full
TA and Full TA + VMD folds contain exactly the same train/test dates and row
counts. MinMax scaling is fitted on each training fold only.

VMD follows the finite-bandwidth mode decomposition described by
Dragomiretskiy and Zosso. The implementation parameters are recorded in every
fold's `vmd_config.json`. See the
[MathWorks VMD reference](https://www.mathworks.com/help/signal/ref/vmd.html).

## Benchmark protocol

- Walk-forward folds: test years 2022, 2023, 2024, and 2025.
- Baseline: Full TA.
- Treatment: Full TA + causal rolling VMD.
- Fixed model architecture and hyperparameters from the existing benchmark.
- Random seed: 42.
- No Optuna and no hyperparameter retuning.
- 20 epochs, batch size 32, sequence length 20.

## Mean results across four folds

| Model | Full TA RMSE | +VMD RMSE | RMSE delta | Full TA DA | +VMD DA | DA delta |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 15.868 | 15.512 | -0.356 | 50.20% | 53.01% | +2.82 pp |
| CNN | 27.951 | 24.843 | -3.108 | 52.61% | 51.02% | -1.60 pp |
| LSTM-CNN | 27.636 | 30.894 | +3.258 | 51.25% | 51.16% | -0.09 pp |
| LSTM-Attention | 30.908 | 30.126 | -0.782 | 51.23% | 51.02% | -0.20 pp |
| LSTM-CNN-Attention | 48.227 | 40.927 | -7.300 | 51.07% | 50.84% | -0.23 pp |

Negative RMSE delta means VMD reduced the error. Positive DA delta means VMD
improved direction accuracy.

## Interpretation

- VMD reduced average RMSE for four of five models, but only LSTM improved
  average direction accuracy.
- LSTM is the strongest VMD result here: its direction accuracy improved in
  all four folds, although RMSE improved in only two.
- LSTM-Attention had a small RMSE reduction in all four folds, but direction
  accuracy improved in only two.
- CNN and LSTM-CNN-Attention average RMSE gains are not stable across folds;
  a large improvement in the 2025 fold drives much of the aggregate gain.
- VMD made LSTM-CNN worse on RMSE in every fold.
- These are single-seed results. They establish the pipeline and provide an
  initial comparison, but do not yet establish statistical robustness.

For the current direction-prediction objective, retain Full TA + VMD for LSTM
as the leading candidate. Treat VMD as optional for the other architectures
until a future multi-seed evaluation confirms a repeatable gain.

## Reproduction

Create the feature folds:

```powershell
python -m models.vmd_feature_pool
```

Run one model:

```powershell
python -m models.vmd_experiments --model lstm
```

Valid model keys are `lstm`, `cnn`, `lstm_cnn`, `lstm_attention`, and
`lstm_cnn_attention`. Use `--model all` to run all five and
`--model comparison` to rebuild the comparison CSV files from saved metrics.

Outputs:

- `data-folds-full-ta-vmd/`
- `data-folds-full-ta-vmd-nn/`
- `outputs/full_ta_vmd_feature_pool/benchmark_comparison.csv`
- `outputs/full_ta_vmd_feature_pool/benchmark_comparison_by_fold.csv`
- Per-model metrics, predictions, and run metadata under
  `outputs/full_ta_vmd_feature_pool/<model>/`
