# Full TA + causal VMD sequence-window sweep

## Purpose

This experiment tests whether shorter model input histories are more suitable
for next-day SET50 direction prediction. It benchmarks five models with
sequence windows of 1, 3, 5, 10, and 20 trading days:

1. LSTM
2. CNN
3. LSTM-CNN
4. LSTM-Attention
5. LSTM-CNN-Attention

The old Attention-LSTM-CNN model is intentionally excluded. Window 20 is
retained as the control.

There are two different windows in this experiment:

- The causal VMD calculation always uses the latest 60 trading days.
- The neural-network input sequence uses 1, 3, 5, 10, or 20 rows.

Test sequences advance one trading day at a time. The prediction target remains
the next trading day for every sequence-window setting.

## Experimental protocol

- Feature set: Full TA plus causal rolling VMD.
- Walk-forward test folds: 2022, 2023, 2024, and 2025.
- Number of completed model-window-fold runs: 100.
- Hyperparameters: fixed at the benchmark settings.
- Epochs: 20.
- Batch size: 32.
- Random seed: 42.
- No Optuna or model-specific retuning.
- VMD window: 60 trading days.
- Number of VMD modes: 5; the highest-frequency mode is removed as noise.
- Scaling: fitted independently on each training fold.

For reproducibility, every fold clears the Keras session and resets TensorFlow's
random seed to 42. TensorFlow deterministic operations are enabled for the
entire sweep. The deterministic window-20 runs in this directory are the
controls used in the comparisons below.

## Complete results

Values are the mean across four walk-forward folds. Direction accuracy (DA) is
shown as a percentage. Runtime is mean wall-clock seconds per fold.

| Model | Sequence window | RMSE | DA | Runtime (s/fold) |
|---|---:|---:|---:|---:|
| LSTM | 1 | 12.897 | 49.72% | 7.663 |
| LSTM | 3 | 12.816 | 52.35% | 8.201 |
| LSTM | 5 | 12.510 | 49.20% | 8.985 |
| LSTM | 10 | 15.031 | 53.23% | 10.739 |
| LSTM | 20 | 15.512 | 53.01% | 13.337 |
| CNN | 1 | 11.125 | 50.00% | 7.282 |
| CNN | 3 | 14.432 | 51.76% | 7.404 |
| CNN | 5 | 13.421 | 49.37% | 7.299 |
| CNN | 10 | 15.644 | 52.62% | 7.257 |
| CNN | 20 | 26.406 | 52.05% | 7.698 |
| LSTM-CNN | 1 | 17.680 | 52.10% | 9.177 |
| LSTM-CNN | 3 | 18.574 | 51.49% | 10.249 |
| LSTM-CNN | 5 | 23.314 | 51.27% | 10.892 |
| LSTM-CNN | 10 | 21.801 | 52.09% | 12.823 |
| LSTM-CNN | 20 | 30.050 | 51.68% | 16.884 |
| LSTM-Attention | 1 | 14.528 | 52.09% | 11.288 |
| LSTM-Attention | 3 | 17.110 | 53.34% | 14.562 |
| LSTM-Attention | 5 | 18.823 | 51.58% | 15.520 |
| LSTM-Attention | 10 | 21.549 | 50.83% | 17.386 |
| LSTM-Attention | 20 | 29.256 | 51.34% | 22.264 |
| LSTM-CNN-Attention | 1 | 13.931 | 52.84% | 14.805 |
| LSTM-CNN-Attention | 3 | 18.841 | 51.69% | 16.276 |
| LSTM-CNN-Attention | 5 | 28.481 | 52.00% | 17.158 |
| LSTM-CNN-Attention | 10 | 33.364 | 50.54% | 20.560 |
| LSTM-CNN-Attention | 20 | 40.927 | 50.84% | 26.053 |

The unrounded results, standard deviations, per-fold values, and deltas against
window 20 are stored in the CSV files under
`outputs/full_ta_vmd_window_sweep`.

## Direction-selected windows

For the direction-prediction objective, the selected window is the one with the
highest mean direction accuracy for each model. Ties would be resolved by lower
RMSE and then the shorter window.

| Model | Selected window | DA, mean +/- SD | RMSE, mean +/- SD | DA delta vs. window 20 | Runtime, mean +/- SD (s/fold) | Runtime reduction |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 10 | 53.23% +/- 4.89% | 15.031 +/- 3.024 | +0.22 pp | 10.739 +/- 0.948 | 19.48% |
| CNN | 10 | 52.62% +/- 2.30% | 15.644 +/- 1.932 | +0.56 pp | 7.257 +/- 0.487 | 5.74% |
| LSTM-CNN | 1 | 52.10% +/- 2.61% | 17.680 +/- 8.479 | +0.42 pp | 9.177 +/- 0.551 | 45.65% |
| LSTM-Attention | 3 | 53.34% +/- 4.41% | 17.110 +/- 4.262 | +2.00 pp | 14.562 +/- 0.670 | 34.59% |
| LSTM-CNN-Attention | 1 | 52.84% +/- 2.87% | 13.931 +/- 5.366 | +2.00 pp | 14.805 +/- 0.831 | 43.17% |

The highest mean direction accuracy is LSTM-Attention with window 3
(53.34%). LSTM-CNN-Attention with window 1 provides the lowest RMSE among the
direction-selected models (13.931), while CNN with window 10 has the shortest
mean runtime (7.257 seconds per fold).

If the primary objective were RMSE instead of direction accuracy, the selected
windows would be LSTM 5, CNN 1, LSTM-CNN 1, LSTM-Attention 1, and
LSTM-CNN-Attention 1.

## Interpretation and paper-use caution

- Short input sequences substantially reduce RMSE for all five models relative
  to their deterministic window-20 controls.
- The largest direction-accuracy gains occur in the two attention models:
  approximately 2.00 percentage points.
- The direction gains for LSTM, CNN, and LSTM-CNN are small: 0.22, 0.56, and
  0.42 percentage points. They should not be described as conclusive.
- A shorter window reduces LSTM-family runtime because the recurrent and
  attention layers process fewer timesteps.
- These results use one controlled random seed and four test folds. Before
  making a statistical superiority claim in the paper, repeat the selected
  configurations with multiple seeds and report confidence intervals or a
  paired fold/seed comparison.

## Runtime definition and environment

Runtime is measured using `time.perf_counter` around model construction,
training, and test inference for each fold. TensorFlow import and warm-up happen
before timing. Data loading, inverse scaling, metric calculation, CSV writing,
and VMD feature generation are excluded.

- Operating system: Windows 10, build 26200, AMD64.
- Processor: AMD64 Family 25 Model 116, AuthenticAMD.
- Python: 3.11.15.
- TensorFlow: 2.21.0.
- Device: CPU only; no GPU was available.
- Total recorded fit-plus-inference time for the final 100 runs:
  1,287.043 seconds (21.451 minutes).

The machine-readable environment record is in `runtime_environment.json`.

## Saved artifacts

- `metrics_by_model_window_fold.csv`: all 100 per-fold records, including
  runtime.
- `metrics_by_model_window.csv`: mean, standard deviation, ranks, and deltas for
  all 25 model-window combinations.
- `best_windows_by_model.csv`: direction-selected and RMSE-selected windows.
- `paper_best_direction_windows.csv`: compact paper-ready table.
- `runtime_by_model_window.csv`: runtime mean, standard deviation, and total.
- `<model>/window_<n>/metrics_by_fold.csv`: four fold metrics for one setting.
- `<model>/window_<n>/predictions_fold_<1-4>.csv`: row-level predictions.
- `<model>/window_<n>/run_metadata.json`: data paths, model settings, package
  versions, random seed, and runtime definition.

## Reproduction

Run all five models for the four new sequence windows:

```powershell
C:\Users\narak\anaconda3\python.exe -m models.vmd_window_sweep --model all --windows 1 3 5 10
```

Rerun the deterministic window-20 controls:

```powershell
C:\Users\narak\anaconda3\python.exe -m models.vmd_window_sweep --model all --windows 20 --force
```

Rebuild the aggregate CSV files from the saved per-fold artifacts:

```powershell
C:\Users\narak\anaconda3\python.exe -m models.vmd_window_sweep --model comparison
```
