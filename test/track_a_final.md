# Track A final experiment log

## Status

Completed on 2026-07-28. This document is the authoritative experiment log for
the final Track A paper analysis. Both the leakage-free window-selection stage
and the paired multi-seed outer test are complete.

Completed model fits:

```text
Window selection: 200
Confirmatory outer test: 200
Total: 400
```

## Research objective

Track A evaluates causal Variational Mode Decomposition (VMD) as a denoising
and feature-extraction treatment for next-day SET50 index prediction. The
central paper question is:

> Does adding causal rolling VMD features improve predictive performance over
> the same Full TA input under a paired, leakage-free, multi-seed evaluation?

## Why the earlier sequence-window sweep was not the final experiment

The exploratory sweep compared sequence windows 1, 3, 5, 10, and 20 using the
2022-2025 outer test folds. It was useful for hypothesis generation, but using
those same test results to select a window and report final performance would
introduce test-set selection bias.

The final protocol therefore discards the test-selected windows for
confirmatory evaluation. Windows are selected only from data ending in 2021,
then locked before the 2022-2025 outer test is evaluated.

## Final experimental design

### Feature-set conditions

1. Control: Full TA.
2. Treatment: Full TA plus causal rolling VMD.

The comparison is paired: model architecture, selected sequence window,
training period, outer test period, random seed, epochs, batch size, optimizer,
and evaluation code are identical. Only the presence of the VMD features
changes.

### Models

1. LSTM
2. CNN
3. LSTM-CNN
4. LSTM-Attention
5. LSTM-CNN-Attention

The old Attention-LSTM-CNN architecture is excluded.

Shared training settings:

- Epochs: 20.
- Batch size: 32.
- Optimizer: Adam.
- Loss: mean squared error.
- Training shuffle: disabled.
- Dense hidden units: 8 with ReLU.
- Output: one linear regression unit.

Architecture-specific settings:

| Model | Architecture |
|---|---|
| LSTM | LSTM(16) -> Dense(8) -> output |
| CNN | causal Conv1D(32, kernel 3) -> global average pooling -> Dense(8) -> output |
| LSTM-CNN | LSTM(16, sequences) -> causal Conv1D(32, kernel 3) -> global average pooling -> Dense(8) -> output |
| LSTM-Attention | LSTM(16, sequences) -> causal multi-head attention(2 heads, key dimension 8) -> global average pooling -> Dense(8) -> output |
| LSTM-CNN-Attention | LSTM(16, sequences) -> causal Conv1D(32, kernel 3) -> causal multi-head attention(2 heads, key dimension 8) -> global average pooling -> Dense(8) -> output |

Input dimensionality:

- Full TA control: 116 input features.
- Full TA plus VMD: 122 input features.
- VMD contribution: 6 features.

### VMD configuration

- Input signal: daily SET50 close (`Close_D`).
- Causal rolling window: 60 trading days.
- Modes: 5.
- Penalty (`alpha`): 1000.
- Dual ascent step (`tau`): 0.
- DC mode: enabled.
- Convergence tolerance: `1e-7`.
- Maximum iterations: 500.
- Denoising rule: remove the mode with the highest final center frequency.
- Retained features: `VMD_IMF_1` through `VMD_IMF_4`,
  `VMD_Denoised_Close`, and `VMD_Noise_Energy_Ratio`.

Each VMD row at date `t` uses only observations from `t-59` through `t`.

### Leakage-free window selection

Candidate model sequence windows:

```text
1, 3, 5, 10, 20 trading days
```

Pretest expanding-window selection folds:

| Selection fold | Training years | Validation year |
|---|---|---:|
| 1 | 2012-2017 | 2018 |
| 2 | 2012-2018 | 2019 |
| 3 | 2012-2019 | 2020 |
| 4 | 2012-2020 | 2021 |

The first outer test year is 2022. Therefore, no observation from any outer
test fold is used to select the sequence window.

For each model and candidate window, validation performance is averaged
symmetrically across:

- Four pretest validation years.
- Full TA and Full TA plus VMD.
- Fixed selection seed 42.

The shared window for a model is selected using:

1. Highest mean validation direction accuracy.
2. Lower mean validation RMSE as the first tie-break.
3. Shorter sequence window as the second tie-break.

A shared window across both feature sets prevents the selection rule from
favoring either the control or VMD treatment.

### Confirmatory outer test

After window selection, the selected window is locked for each model. Each
feature-set condition is retrained and evaluated with expanding outer folds:

| Outer fold | Training years | Test year |
|---|---|---:|
| 1 | 2012-2021 | 2022 |
| 2 | 2012-2022 | 2023 |
| 3 | 2012-2023 | 2024 |
| 4 | 2012-2024 | 2025 |

Final random seeds:

```text
42, 123, 456, 789, 2025
```

This produces 20 paired outer-test observations per model and 200 final model
fits in total:

```text
5 models x 2 feature sets x 4 outer folds x 5 seeds = 200
```

The window-selection stage contains another 200 fits:

```text
5 models x 2 feature sets x 5 windows x 4 validation folds = 200
```

### Preprocessing

- The scaler is fitted independently on the training portion of every fold.
- Validation and test values are transformed with the corresponding
  training-only scaler.
- The target is converted back to original SET50 index units before metrics
  are calculated.
- Test sequences advance by one trading day.
- The target is the next trading day's close.

### Reproducibility controls

- TensorFlow deterministic operations are enabled.
- The Keras session is cleared before every model fit.
- Python, NumPy, and TensorFlow seeds are reset for every fit.
- Model shuffling is disabled.
- Input-file SHA-256 hashes are saved in `input_manifest.json`.
- Python, TensorFlow, platform, processor, and device information are saved in
  `runtime_environment.json`.

### Runtime definition

Runtime is measured with `time.perf_counter` around:

```text
model construction + training + evaluation-period inference
```

TensorFlow import/warm-up, data loading, scaling, inverse scaling, metric
calculation, statistical aggregation, and file writing are excluded.

## Statistical reporting plan

Performance for each feature set is first averaged across the five seeds
within each outer fold. Paper mean and standard deviation are then calculated
across the four outer-fold means, preserving temporal folds as the independent
evaluation units.

VMD-minus-control differences are paired by:

```text
model + sequence window + seed + outer fold
```

For each model, seed-level paired differences are averaged within the outer
fold. The paper reports:

- Mean paired RMSE difference.
- Mean paired direction-accuracy difference in percentage points.
- Standard deviation across four fold-level differences.
- Fold-level 95% t confidence interval.
- Exact two-sided sign-flip permutation p-value across four outer folds.

With only four outer folds, the statistical analysis has low power. Effect
sizes, fold consistency, and confidence intervals will be emphasized instead
of relying only on p-values.

## Implementation and verification log

### 2026-07-28 - Protocol correction

- Classified the previous window sweep as exploratory.
- Defined four pre-2022 expanding validation folds.
- Defined symmetric shared-window selection across control and VMD.
- Defined a locked-window paired outer test.
- Defined five final random seeds.

### 2026-07-28 - Data preparation

- Generated Full TA selection folds for 2018-2021.
- Generated Full TA plus VMD selection folds for 2018-2021.
- Generated independently scaled neural-network versions.
- Verified equal row counts between paired feature sets.
- Verified strict temporal ordering.
- Verified scaler fit scope is selection training data only.

Selection row counts:

| Validation year | Training rows | Validation rows |
|---:|---:|---:|
| 2018 | 1,386 | 245 |
| 2019 | 1,631 | 244 |
| 2020 | 1,875 | 243 |
| 2021 | 2,118 | 241 |

### 2026-07-28 - Code and tests

- Added `models/track_a_data.py`.
- Added `models/track_a_final.py`.
- Added `tests/test_track_a_final.py`.
- Added explicit random-seed support to LSTM, CNN, LSTM-CNN, and
  LSTM-Attention predictors.
- Preserved default seed-42 behavior for existing benchmarks.
- Added tests for temporal selection folds, scaler fit scope, symmetric window
  selection, rejection of outer-test data, paired metric alignment, fold-level
  aggregation, exact sign-flip testing, and seed forwarding.
- Impacted test set result: 55 passed.
- Actual CNN/Full-TA/window-1 selection smoke run completed for all four
  pretest folds with finite metrics and runtime.

### 2026-07-28 - Window-selection execution

- Completed 50 configurations and 200 pretest validation fits.
- Confirmed selection rows contain only 2018-2021.
- Confirmed exactly eight validation results per model-window combination:
  four validation years times two feature sets.
- Confirmed all predictions and metrics are finite.
- Confirmed all runtime values are positive.
- Total recorded selection fit-plus-inference time: 1,933.647 seconds
  (32.227 minutes).
- The first selection process stopped without a Python traceback after 196 of
  200 fits. The resumable runner preserved all completed artifacts. A second
  invocation detected the 49 completed configurations, ran only the missing
  four-fold configuration, and then generated the aggregate files. No
  completed metrics were recomputed or altered.

### 2026-07-28 - Confirmatory outer-test execution

- Completed 50 configurations and 200 outer-test fits.
- Completed 100 exact Full-TA/VMD pairs.
- Confirmed five models, two feature sets, five seeds, and four outer folds.
- Confirmed every model-feature-fold group contains variation across seeds.
- Confirmed all predictions and metrics are finite.
- Confirmed all runtime values are positive.
- Total recorded outer-test fit-plus-inference time: 2,912.412 seconds
  (48.540 minutes).
- The outer-test process and final aggregation exited successfully.

### 2026-07-28 - Final verification

- Track A focused tests: 12 passed.
- Full project test suite: 102 passed, 4 non-blocking warnings.
- Selection metric files: 50.
- Selection metadata files: 50.
- Selection prediction files: 200.
- Outer-test metric files: 50.
- Outer-test metadata files: 50.
- Outer-test prediction files: 200.
- Aggregate selection rows: 200.
- Aggregate outer-test rows: 200.
- Paired Full-TA/VMD rows: 100.
- Input files in hash manifest: 32.
- SHA-256 manifest mismatches: 0.
- Python syntax/import compilation completed successfully for all changed Track
  A modules and seed-aware model modules.

The four test-suite warnings are unrelated to Track A results: one synthetic
unit test has too few samples for R-squared, and three warnings report the
existing use of deprecated `datetime.utcnow()` in shared legacy output code.

## Window-selection results

The following windows were selected exclusively from the pre-2022 validation
folds. Validation direction accuracy and RMSE are averaged across eight
conditions per model-window: four validation years and two feature sets.

| Model | Locked window | Validation DA, mean +/- SD | Validation RMSE, mean +/- SD |
|---|---:|---:|---:|
| LSTM | 5 | 49.33% +/- 1.66% | 29.193 +/- 9.063 |
| CNN | 5 | 49.95% +/- 3.07% | 27.231 +/- 16.764 |
| LSTM-CNN | 20 | 47.98% +/- 2.95% | 47.382 +/- 13.895 |
| LSTM-Attention | 20 | 49.17% +/- 3.22% | 59.293 +/- 19.634 |
| LSTM-CNN-Attention | 20 | 49.28% +/- 2.53% | 67.246 +/- 18.586 |

These validation values are used only for hyperparameter selection. They are
not combined with outer-test results or presented as final model performance.

## Final paired outer-test results

Means and standard deviations below are calculated across the four outer-fold
means after first averaging the five random seeds within each fold.

| Model | Window | Full TA RMSE | +VMD RMSE | RMSE delta | Full TA DA | +VMD DA | DA delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 5 | 13.861 +/- 2.226 | 13.244 +/- 1.568 | -0.617 | 52.56% +/- 2.25% | 52.04% +/- 1.72% | -0.52 pp |
| CNN | 5 | 19.268 +/- 5.711 | 16.305 +/- 2.263 | -2.963 | 52.56% +/- 0.93% | 51.66% +/- 1.62% | -0.90 pp |
| LSTM-CNN | 20 | 34.257 +/- 20.839 | 32.691 +/- 14.949 | -1.566 | 51.00% +/- 2.36% | 51.79% +/- 3.04% | +0.80 pp |
| LSTM-Attention | 20 | 31.884 +/- 4.350 | 39.236 +/- 8.720 | +7.352 | 50.77% +/- 2.87% | 50.42% +/- 1.83% | -0.35 pp |
| LSTM-CNN-Attention | 20 | 45.415 +/- 18.969 | 43.449 +/- 21.996 | -1.966 | 49.71% +/- 3.62% | 50.25% +/- 3.07% | +0.53 pp |

Negative RMSE delta favors VMD. Positive direction-accuracy delta favors VMD.

### Paired fold-level uncertainty

| Model | RMSE delta 95% CI | RMSE sign-flip p | DA delta 95% CI (pp) | DA sign-flip p |
|---|---:|---:|---:|---:|
| LSTM | [-1.848, 0.615] | 0.375 | [-1.480, 0.435] | 0.250 |
| CNN | [-8.965, 3.039] | 0.250 | [-2.708, 0.910] | 0.250 |
| LSTM-CNN | [-11.840, 8.708] | 0.750 | [-0.281, 1.875] | 0.125 |
| LSTM-Attention | [-3.594, 18.299] | 0.250 | [-2.823, 2.132] | 0.625 |
| LSTM-CNN-Attention | [-10.603, 6.671] | 0.625 | [-2.226, 3.288] | 0.625 |

Every confidence interval includes zero. With only four independent temporal
folds, none of the VMD effects should be described as statistically
conclusive.

### Fold consistency

- LSTM RMSE improved in three of four outer folds; direction accuracy improved
  in one of four.
- CNN RMSE improved in three of four folds; direction accuracy improved in one
  of four.
- LSTM-CNN RMSE improved in two of four folds; direction accuracy improved in
  all four folds.
- LSTM-Attention RMSE improved in one of four folds; direction accuracy
  improved in one of four.
- LSTM-CNN-Attention RMSE improved in three of four folds; direction accuracy
  improved in two of four.

LSTM-CNN has the most consistent direction effect, with a mean gain of
0.80 percentage points and positive direction deltas in all four folds.
However, its 95% interval still includes zero and its exact two-sided p-value
is 0.125, so this is suggestive rather than confirmatory evidence.

## Runtime results

Runtime is mean seconds per completed model-fold fit, calculated over 20 outer
runs per model-feature condition.

| Model | Window | Full TA runtime | +VMD runtime | VMD overhead |
|---|---:|---:|---:|---:|
| LSTM | 5 | 7.488 s | 7.632 s | +0.143 s |
| CNN | 5 | 6.055 s | 6.109 s | +0.054 s |
| LSTM-CNN | 20 | 13.835 s | 13.942 s | +0.107 s |
| LSTM-Attention | 20 | 18.852 s | 20.267 s | +1.415 s |
| LSTM-CNN-Attention | 20 | 25.186 s | 26.253 s | +1.067 s |

Recorded runtime totals:

| Stage | Fits | Total seconds | Total minutes |
|---|---:|---:|---:|
| Pretest window selection | 200 | 1,933.647 | 32.227 |
| Confirmatory outer test | 200 | 2,912.412 | 48.540 |
| Combined final protocol | 400 | 4,846.059 | 80.768 |

These totals exclude TensorFlow startup, preprocessing, metric calculation,
VMD feature generation, aggregation, file writing, diagnostic runs, and the
interrupted partial invocation's uncompleted work.

## Track A conclusion

Track A implementation and confirmatory evaluation are complete.

The evidence does not support a claim that causal rolling VMD universally
improves next-day SET50 direction prediction:

- VMD reduced mean RMSE for four of five models.
- VMD improved mean direction accuracy for only LSTM-CNN and
  LSTM-CNN-Attention.
- The largest and most fold-consistent direction gain is LSTM-CNN at
  +0.80 percentage points.
- LSTM-Attention became materially worse on RMSE after adding VMD.
- All fold-level 95% confidence intervals include zero.

The paper should therefore describe VMD as a model-dependent auxiliary
denoising feature rather than a universally beneficial preprocessing step.
The VMD feature stream can proceed to the later regime-specific SHAP stage,
where feature selection may retain or remove individual VMD components by
market regime. It should not be claimed that Track A alone establishes
statistical superiority.

For the strongest final predictive results in this experiment:

- Full TA CNN has the highest control direction accuracy: 52.56%.
- Full TA LSTM is essentially tied on direction accuracy at 52.56% and has
  substantially lower RMSE: 13.861.
- Full TA plus VMD LSTM has the lowest overall RMSE: 13.244, but its direction
  accuracy is 0.52 percentage points below its paired Full TA control.

## Paper table recommendation

Use `paper_track_a_compact.csv` as the main Track A ablation table. Include the
complete per-window selection table in an appendix or supplementary file.
Report both error and direction metrics because VMD often reduces RMSE without
improving direction accuracy.

## Artifact locations

- `data-track-a-window-selection/`: pretest selection datasets.
- `outputs/track_a_final/window_selection/`: per-configuration validation
  metrics, predictions, runtime, and metadata.
- `outputs/track_a_final/outer_test/`: paired multi-seed test artifacts.
- `outputs/track_a_final/input_manifest.json`: input hashes.
- `outputs/track_a_final/runtime_environment.json`: runtime environment.
- `outputs/track_a_final/locked_windows.csv`: locked model windows.
- `outputs/track_a_final/final_metrics_by_seed_fold.csv`: all final metrics.
- `outputs/track_a_final/paired_ablation_summary.csv`: paired VMD effects.
- `outputs/track_a_final/paper_track_a_compact.csv`: compact main-paper table.
- `outputs/track_a_final/paper_track_a_table.csv`: complete wide paper table.
- `outputs/track_a_final/runtime_summary.csv`: runtime for every stage, model,
  feature set, and sequence window.

## Reproduction commands

Prepare selection data:

```powershell
D:\conda_envs\my_env\python.exe -m models.track_a_final data
```

Run leakage-free window selection:

```powershell
D:\conda_envs\my_env\python.exe -m models.track_a_final selection
```

Run the locked-window, paired, multi-seed outer test:

```powershell
D:\conda_envs\my_env\python.exe -m models.track_a_final outer-test
```

Run or resume the complete Track A workflow:

```powershell
D:\conda_envs\my_env\python.exe -m models.track_a_final all
```
