# Point-in-Time Protocol v2 Correction Log

Status: **COMPLETE — PRE-SHAP CORRECTIONS VERIFIED**  
Correction date: 2026-07-31 (Asia/Bangkok)  
Protocol family: `point_in_time_v2` / `track_a_point_in_time_v2`

## Why this correction was required

The pre-SHAP reviewer audit found that the last training row in every
expanding fold used a next-day target whose price was observed on the first
evaluation date. The feature timestamps did not overlap, but the label
timestamps did. That makes the former folds unsuitable for model selection,
outer evaluation, or SHAP.

The previous artifacts have not been deleted or overwritten. Corrected data
and outputs use new `*-point-in-time-v2` directories.

## Frozen point-in-time rule

For every train/evaluation split:

```text
retain a training observation only if
Label_Date < min(Date in evaluation split)
```

`Label_Date` is the next observed SET50 trading date. Each attached date is
verified by checking that `Target_Next_Close` equals the SET50 close observed
on that date.

## Outer-fold corrections

| Fold | Test year | Removed feature date | Removed label date | Corrected base train rows | Test rows |
|---|---:|---|---|---:|---:|
| fold_1 | 2022 | 2021-12-30 | 2022-01-04 | 2,418 | 241 |
| fold_2 | 2023 | 2022-12-30 | 2023-01-03 | 2,659 | 243 |
| fold_3 | 2024 | 2023-12-28 | 2024-01-02 | 2,902 | 244 |
| fold_4 | 2025 | 2024-12-30 | 2025-01-02 | 3,146 | 234 |

Generated source:

- `data-folds-point-in-time-v2/`
- Per-fold `point_in_time_contract.json`
- Root `run_metadata.json` with source hashes and fold hashes

## Derived feature artifacts

| Artifact | Features passed to model | Train rows by outer fold | Status |
|---|---:|---|---|
| Full TA v2 | 116 | 2,358 / 2,599 / 2,842 / 3,086 | generated and validated |
| Full TA + causal VMD v2 | 122 | 2,358 / 2,599 / 2,842 / 3,086 | generated and validated |
| Min-max scaled copies | same | same | train-only scaling; `Label_Date` excluded |

Directories:

- `data-folds-full-ta-point-in-time-v2/`
- `data-folds-full-ta-point-in-time-v2-nn/`
- `data-folds-full-ta-vmd-point-in-time-v2/`
- `data-folds-full-ta-vmd-point-in-time-v2-nn/`

Validation performed:

- every derived fold retains `Label_Date`;
- `Label_Date` is excluded from feature matrices and scalers;
- every fold satisfies `max(Label_Date_train) < min(Date_test)`;
- Full TA and VMD folds retain their expected 116 and 122 model features;
- VMD remains rolling, past-only, and computed before fold extraction from the
  point-in-time source.

## Sequence-boundary context correction

The purged row cannot be used as a supervised sample because its label is
observed on the first evaluation date. Its features, however, were observed
before that date and are required to keep a multi-day sliding window
consecutive. Each fold now stores this row separately as
`context_before_test.csv`.

Contract:

- the row is excluded from all training targets and `model.fit`;
- the scaler is fit on the supervised train partition only;
- the fitted scaler transforms the context row without refitting;
- the context feature row is inserted between train history and the first
  evaluation row for all five neural architectures;
- the context target and `Label_Date` are never model features.

| Evaluation stage | Year | Context-only feature date |
|---|---:|---|
| Window selection | 2018 | 2017-12-29 |
| Window selection | 2019 | 2018-12-28 |
| Window selection | 2020 | 2019-12-30 |
| Window selection | 2021 | 2020-12-30 |
| Outer test | 2022 | 2021-12-30 |
| Outer test | 2023 | 2022-12-30 |
| Outer test | 2024 | 2023-12-28 |
| Outer test | 2025 | 2024-12-30 |

All dates were verified in Full TA, Full TA + VMD, and their train-only scaled
copies. The first evaluation sequence therefore no longer skips a trading
day. Recomputing every scaled context value from the saved train-fitted
`scale` and `min` parameters produced a maximum absolute error of
`2.22e-16`.

## Window-selection corrections

Selection years remain 2018–2021 and outer years remain 2022–2025. The last
training observation crossing into each validation year is now purged.

| Validation year | Corrected Full-TA/VMD train rows | Validation rows | Max retained label date | First validation date |
|---:|---:|---:|---|---|
| 2018 | 1,385 | 245 | 2017-12-29 | 2018-01-03 |
| 2019 | 1,630 | 244 | 2018-12-28 | 2019-01-02 |
| 2020 | 1,874 | 243 | 2019-12-30 | 2020-01-02 |
| 2021 | 2,117 | 240 | 2020-12-30 | 2021-01-04 |

Directories:

- `data-track-a-window-selection-point-in-time-v2/full_ta/`
- `data-track-a-window-selection-point-in-time-v2/full_ta_nn/`
- `data-track-a-window-selection-point-in-time-v2/full_ta_vmd/`
- `data-track-a-window-selection-point-in-time-v2/full_ta_vmd_nn/`

## Direction-metric contract

The task is now explicitly binary Up versus Down:

- actual next-day no-change observations are retained for regression but
  excluded from binary direction metrics;
- a predicted exact no-change is an abstention;
- Direction Accuracy, Balanced Accuracy, and MCC are computed only where the
  actual direction is non-tied and the forecast is non-abstaining;
- direction coverage is
  `evaluated binary observations / all non-tied actual observations`;
- confusion counts and predicted-up share are saved for diagnostics.

Balanced Accuracy is the primary window-selection metric in v2. Ties are
resolved by Direction Accuracy, RMSE, and then the shorter window.

## Tests and verification

New/updated test coverage includes:

- label-date attachment and target-price verification;
- boundary purge, missing label date, and empty-train rejection;
- exclusion of `Label_Date` from features and scaling;
- label metadata preservation through Full TA;
- binary tie/abstention metrics and coverage;
- selection-boundary purge;
- Balanced Accuracy as the primary selection criterion;
- paired VMD deltas for DA, Balanced Accuracy, and MCC.

Final focused verification:

```text
119 focused point-in-time, five-model, Track A, Track B, and SHAP protocol
tests passed; 0 failed.
GradientExplainer smoke passed for all five architectures with finite,
exactly repeatable outputs; no feature ranking was generated.
```

## Completion status

At the close of the correction:

1. all corrected data artifacts are complete;
2. Track A window selection for all five models is complete;
3. the corrected Track A outer test is complete;
4. the corrected four-fold Track B forecast fusion is complete;
5. the SHAP v2 explainer smoke test and final hash audit are complete;
6. no SHAP ranking has been generated or inspected;
7. the pipeline is ready for the registered SHAP experiment.

Corrected window selection completed with 200 fold-level rows
(5 models x 2 feature sets x 5 windows x 4 validation years). All numerical
metrics are finite.

| Model | Locked W | Mean selection BA | Mean selection DA | Mean selection RMSE |
|---|---:|---:|---:|---:|
| LSTM | 5 | 0.508510 | 0.498463 | 30.202 |
| CNN | 20 | 0.511445 | 0.502960 | 40.646 |
| LSTM-CNN | 20 | 0.492058 | 0.482321 | 45.896 |
| LSTM-Attention | 10 | 0.501935 | 0.492187 | 49.624 |
| LSTM-CNN-Attention | 20 | 0.510224 | 0.499461 | 67.305 |

Selection fit/inference runtime sum: 1,613.415 seconds. Locked-window SHA-256:
`47456d5827bcb6a4ca01ce56cff21fce67d065f89c953150552ada3ca9faa55f`.

Execution note: the first monolithic selection process stopped after 24 of 50
configurations because of TensorFlow process memory growth. Completed
configuration artifacts were validated by their creation timestamp and
metadata. Remaining configurations are executed in isolated child processes
and merged only after all 50 pass. The registered runtime remains model
build + fit + evaluation inference; Python/TensorFlow process startup is
excluded consistently.

## Corrected Track A outer result

The outer evaluation contains 200 finite rows
(5 models × 2 feature sets × 5 seeds × 4 test years), 100 paired rows, and
full direction coverage. VMD effects are mixed and are not presented as a
universal improvement.

| Model | W | Full TA BA | +VMD BA | BA delta (pp) | Full TA RMSE | +VMD RMSE |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 5 | 0.534901 | 0.531555 | -0.335 | 13.751 | 13.260 |
| CNN | 20 | 0.521420 | 0.518429 | -0.299 | 26.961 | 22.475 |
| LSTM-CNN | 20 | 0.520754 | 0.524299 | +0.355 | 30.898 | 31.892 |
| LSTM-Attention | 10 | 0.525076 | 0.520122 | -0.495 | 24.582 | 41.828 |
| LSTM-CNN-Attention | 20 | 0.515832 | 0.509832 | -0.600 | 45.744 | 47.288 |

Track A selection plus outer fit/inference runtime sum: 4,158.397 seconds.
With four outer folds, exact sign-flip tests have limited resolution; the
paper reports fold-level effects and uncertainty rather than a superiority
claim.

| Model | Selection runtime (s) | Outer runtime (s) | Track A total (s) |
|---|---:|---:|---:|
| LSTM | 259.357 | 310.878 | 570.235 |
| CNN | 197.455 | 262.343 | 459.799 |
| LSTM-CNN | 303.365 | 576.192 | 879.557 |
| LSTM-Attention | 400.335 | 595.118 | 995.453 |
| LSTM-CNN-Attention | 452.903 | 800.451 | 1,253.353 |

## Author scope decision

On 2026-07-31 the author explicitly restricted every registered experiment
and paper-facing table to LSTM, CNN, LSTM-CNN, LSTM-Attention, and
LSTM-CNN-Attention.

## Track B article-cluster inference correction

The locked 2023 intrinsic benchmark contains 1,333 article-ticker pairs but
only 738 unique articles. Uncertainty is now computed at the article level:

- seeded cluster bootstrap samples `article_id` and retains all ticker rows
  belonging to each sampled article;
- the primary paired p-value uses article-level method swaps
  (50,000 seeded Monte Carlo sign flips);
- pair-level exact McNemar is retained only as a supplementary diagnostic.

Updated locked-test results:

| Comparison | Accuracy delta | Article-cluster 95% CI | Cluster p-value |
|---|---:|---:|---:|
| Terra leader minus Terra single | +6.752 pp | [4.306, 9.328] pp | 0.000020 |
| Terra leader minus local classifier | -6.227 pp | [-10.795, -1.572] pp | 0.00994 |

No API call or LLM relabeling was required. The result supports only a
budget-asymmetric three-role-system versus one-call comparison; it does not
isolate debate structure from additional inference compute.

## Corrected Track B forecasting result

Track B uses the same five locked models and windows as Track A. Each paired
comparison holds model, window, seed, test fold, technical inputs, and dates
fixed; the treatment adds only the eight daily news variables.

The completed artifact contains 200 finite fits
(5 models × 2 arms × 5 seeds × 4 years), 100 paired rows, and full direction
coverage.

| Model | W | Technical BA | +News BA | BA delta (pp) | Technical RMSE | +News RMSE |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 5 | 0.509553 | 0.521340 | +1.179 | 17.531 | 16.931 |
| CNN | 20 | 0.518188 | 0.517485 | -0.070 | 27.629 | 28.200 |
| LSTM-CNN | 20 | 0.527862 | 0.518889 | -0.897 | 30.868 | 33.436 |
| LSTM-Attention | 10 | 0.524127 | 0.527704 | +0.358 | 38.998 | 25.491 |
| LSTM-CNN-Attention | 20 | 0.528008 | 0.519133 | -0.887 | 40.562 | 40.441 |

Track B fit/inference runtime sum: 1,370.026 seconds. The evidence is mixed:
news improves average BA for LSTM and LSTM-Attention, but not the other three
architectures. None of the five four-fold BA contrasts reaches the minimum
two-sided exact sign-flip p-value of 0.125, so the paper must not claim a
universal or statistically established news benefit.

| Model | Technical runtime (s) | +News runtime (s) | Track B total (s) |
|---|---:|---:|---:|
| LSTM | 95.945 | 95.373 | 191.318 |
| CNN | 74.526 | 74.949 | 149.474 |
| LSTM-CNN | 149.505 | 149.463 | 298.968 |
| LSTM-Attention | 157.919 | 156.681 | 314.600 |
| LSTM-CNN-Attention | 207.948 | 207.717 | 415.666 |

Execution deviation: the first pass stopped with a native Windows access
violation during the final LSTM-CNN-Attention + news fit. The 199 completed
checkpoints were retained; a clean process reran only the missing
year-2025/seed-2025 cell. The final report was rebuilt from all 200 checkpoint
files and the consolidated metadata was verified.

## Pre-SHAP freeze

The machine-readable freeze record is
`test/pre_shap_freeze_manifest_v2.json`. It contains SHA-256 hashes for 27
registered source, environment, input, output, and smoke-test artifacts. All
27 hashes were recomputed successfully with no missing files or mismatches.
