# SET100 same-exchange robustness execution log v1

Execution date: 2026-08-03 (Asia/Bangkok)  
Protocol: `set100-same-exchange-robustness-v1`  
Freeze record: `test/set100_same_exchange_robustness_freeze_v1.json`  
Implementation: `models/set100_robustness.py`,
`models/set100_robustness_runner.py`, and
`models/set100_robustness_aggregate.py`

## 1. Outcome

The registered SET100 same-exchange robustness experiment completed all 100
model fits: five architectures, four outer test years, and five random seeds.
All pre-fit data gates, feature checks, cell-level integrity checks, paired-date
checks, and post-run cardinality checks passed.

SET100 did not improve the primary metric relative to SET50. Mean
seed-averaged balanced accuracy was lower on SET100 for all five architectures
by 0.95 to 2.17 percentage points. No cross-index difference was significant
after Holm correction at alpha 0.05. The result supports a reliability finding:
the weak direction-prediction signal observed on SET50 does not become stronger
when the frozen pipeline is transferred to the broader, overlapping SET100
index.

This is a same-exchange robustness check, not an independent external-market
replication. SET50 and SET100 share the Thai market, period, macroeconomic
conditions, and constituents.

## 2. Pre-result protocol freeze

The protocol was frozen at `2026-08-03T14:50:13Z`, before any SET100 model fit
or result was observed. The freeze stores file sizes and SHA-256 digests for the
governance gates, SET100 point-in-time fold metadata, SET50 locked windows and
results, feature code, and model code. All frozen inputs still matched at
aggregation time.

Registered choices were:

| Model | Window | Seeds | Outer test years |
|---|---:|---|---|
| LSTM | 5 | 42, 123, 456, 789, 2025 | 2022-2025 |
| CNN | 20 | 42, 123, 456, 789, 2025 | 2022-2025 |
| LSTM-CNN | 20 | 42, 123, 456, 789, 2025 | 2022-2025 |
| LSTM-Attention | 10 | 42, 123, 456, 789, 2025 | 2022-2025 |
| LSTM-CNN-Attention | 20 | 42, 123, 456, 789, 2025 | 2022-2025 |

No SET100 window selection, hyperparameter tuning, Optuna search, early
stopping, or result-dependent model exclusion was performed. Every model used
the same inherited training budget: 20 epochs, batch size 32, and
`shuffle=False`.

## 3. Data and feature controls

The market-data governance stage passed the SET100 internal-research gate. The
market files were supplied from publicly accessible provider historical-data
pages that offer a download option. This is not represented as an open licence;
provider terms apply, and raw row-level observations are excluded from public
release. This paragraph supersedes the earlier institutional-entitlement
description under the dated access amendment.

SET100 was aligned to the registered SET50 session-date reference. It matched
3,360 of 3,381 required dates (99.3789%), above the frozen 99% gate. Twenty-one
leading 2012 dates were excluded rather than backfilled because completed
monthly information was not yet causally available. Partial 2026 observations
were excluded from evaluation.

The final input contained 122 features: 116 full technical-analysis features
and six causal rolling VMD features. VMD and scaling were rebuilt within the
registered point-in-time folds. Min-max scalers were fit on training data only,
and a training row was retained only when its `Label_Date` preceded the first
test date. The difference between governance-stage and final training counts is
caused by the 60-row technical-indicator warm-up, not by test-dependent row
selection.

| Fold | Test year | Final train rows | Test rows | Last train label | First test date |
|---|---:|---:|---:|---|---|
| fold_1 | 2022 | 2,337 | 241 | 2021-12-30 | 2022-01-04 |
| fold_2 | 2023 | 2,578 | 243 | 2022-12-30 | 2023-01-03 |
| fold_3 | 2024 | 2,821 | 244 | 2023-12-28 | 2024-01-02 |
| fold_4 | 2025 | 3,065 | 234 | 2024-12-30 | 2025-01-02 |

SET100 news was not used. A historically point-in-time SET100 constituent/news
universe was unavailable, so adding SET50-oriented news to SET100 would have
created an asymmetric multimodal comparison. This experiment therefore tests
the frozen final numeric `Full TA + causal VMD` pipeline only.

## 4. Evaluation and inference

For each model and fold, predictions from the five registered seeds were
averaged before metric calculation. Seeds were not treated as independent
statistical replicates. The inferential unit was the outer test year, giving
four paired SET50-SET100 differences per model.

Balanced accuracy was the pre-specified primary metric. Direction accuracy,
MCC, ROC AUC, direction coverage, MAPE, NRMSE, and NMAE were secondary metrics.
Raw RMSE and MAE were retained within each index but were not used for
cross-index claims because SET50 and SET100 have different numerical scales.
Cross-index tests used exact two-sided sign-flip p-values over four folds.
Holm correction was applied across the five models separately within each
metric family.

With only four outer folds, the smallest attainable two-sided exact sign-flip
p-value is 0.125. The t-based 95% intervals in the output are descriptive; the
registered exact sign-flip and Holm-adjusted p-values govern significance
claims.

## 5. Primary results

Values below are means across the four seed-averaged outer folds. Standard
deviations are across folds. Delta is SET100 minus SET50 in percentage points.

| Model | SET50 BAcc | SET100 BAcc | Delta (pp) | 95% descriptive CI | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 0.5312 ± 0.0259 | 0.5163 ± 0.0324 | -1.497 | [-6.139, 3.145] | 0.375 | 1.000 |
| CNN | 0.5308 ± 0.0234 | 0.5189 ± 0.0238 | -1.194 | [-2.080, -0.308] | 0.125 | 0.625 |
| LSTM-CNN | 0.5268 ± 0.0302 | 0.5051 ± 0.0147 | -2.166 | [-6.560, 2.229] | 0.250 | 1.000 |
| LSTM-Attention | 0.5237 ± 0.0244 | 0.5142 ± 0.0141 | -0.955 | [-2.987, 1.077] | 0.250 | 1.000 |
| LSTM-CNN-Attention | 0.5110 ± 0.0377 | 0.5003 ± 0.0241 | -1.070 | [-6.852, 4.712] | 0.625 | 1.000 |

CNN had the highest mean SET100 balanced accuracy (0.5189), followed by LSTM
(0.5163) and LSTM-Attention (0.5142). These differences are descriptive model
rankings, not evidence that CNN is significantly superior.

The SET100 mean direction accuracies were 0.5179 (CNN), 0.5015 (LSTM), 0.4970
(LSTM-Attention), 0.4940 (LSTM-CNN), and 0.4868
(LSTM-CNN-Attention). Mean ROC AUC was highest for LSTM-Attention at 0.5564,
which shows that ranking quality and thresholded direction accuracy need not
select the same architecture.

Across all eight registered cross-index metric families and five models, zero
of 40 Holm-adjusted comparisons reached 0.05. For balanced accuracy, CNN was
lower on SET100 in all four folds, but its exact p-value was 0.125 and its Holm
p-value was 0.625. This must not be reported as a statistically significant
degradation.

## 6. Runtime results

TensorFlow 2.21 ran on native Windows CPU with deterministic operations enabled,
`TF_ENABLE_ONEDNN_OPTS=0`, and no GPU. SET100 timing used separate fit and
inference timers for every cell.

| Model | Parameters | Mean fit (s) | Mean inference (s) | Mean total/cell (s) | Total, 20 cells (s) |
|---|---:|---:|---:|---:|---:|
| LSTM | 9,041 | 5.834 | 0.321 | 6.155 | 123.106 |
| CNN | 12,017 | 5.842 | 0.166 | 6.007 | 120.147 |
| LSTM-CNN | 10,737 | 10.522 | 0.353 | 10.875 | 217.492 |
| LSTM-Attention | 10,129 | 9.395 | 0.421 | 9.815 | 196.308 |
| LSTM-CNN-Attention | 12,865 | 13.886 | 0.463 | 14.349 | 286.984 |

The sum of instrumented SET100 model time was 944.037 seconds. The resumable
full orchestration took 1,207.413 seconds while reusing the four LSTM/seed-42
smoke cells. The separate smoke process took 56.2 seconds, and causal feature
preparation took 65.5 seconds in this execution. Orchestration time includes
fresh Python/TensorFlow process startup, input validation, serialization, and
aggregation, so it should not be substituted for model fit time.

Legacy SET50 runtime values are retained in `runtime_summary.csv`, but the old
runner recorded a single total runtime rather than separate fit and inference
times. Direct component-level runtime comparisons between SET50 and SET100 are
therefore not claimed.

## 7. Integrity and disclosure controls

The completed audit records:

- 100/100 registered SET100 cells;
- 100/100 registered SET50 comparison cells;
- 40 seed-averaged market/fold metric rows;
- 20 paired SET50-SET100 model/fold rows;
- 20/20 exact cross-index date checks;
- zero SET100 row-level observations in public aggregate outputs; and
- frozen-input and feature-integrity gates passed.

Final verification produced 21/21 passing governance and SET100 tests. Coverage
for the three new SET100 modules was 85% overall: 89% for statistical protocol
logic, 91% for aggregation, and 80% for the operational runner. Ruff reported
no lint violations. All nine files registered inside `output_manifest.json`
matched their stored byte sizes and SHA-256 digests, and no public CSV contained
the row-level columns `Date`, `Close_D`, `y_true`, or `y_pred`.

Coverage verification repeated LSTM seed 42 in the isolated ignored path
`runtime_cache/coverage_set100_job`; it reproduced the official predictions and
did not overwrite the registered output. Automated cleanup of that temporary
folder was blocked by the execution safety policy before deletion, so it remains
an ignored runtime-cache artifact and is not part of the benchmark manifest.

Row-level predictions, model-job logs, generated feature folds, and scaled folds
remain under ignored private paths. Public artifacts contain only code,
protocols, hashes, non-reconstructive aggregate metrics, and audit summaries.
The current repository still contains seven legacy restricted SET50 files in
Git history; a clean publication package remains required before public release.

## 8. Paper interpretation

Defensible claim:

> Under a pre-frozen, leakage-controlled, compute-consistent protocol, the weak
> SET50 direction-prediction signal did not strengthen on the broader SET100
> index. All five architectures produced SET100 mean balanced accuracy between
> 0.500 and 0.519, and none of the paired cross-index differences remained
> significant after Holm correction.

Claims not supported by this experiment:

- SET100 improves predictive accuracy;
- any architecture is statistically superior on SET100;
- the result is an independent external-market replication;
- news or multimodal fusion generalises from SET50 to SET100; or
- a negative CNN interval alone establishes significant degradation.

The useful contribution is the transparent transfer audit: the pipeline was
frozen before transfer, all five architectures were retained, seeds were
averaged before inference, dates were paired exactly, weak/null findings were
reported without post-hoc tuning, and provider-hosted row-level records were
excluded from public release.

## 9. Reproduction

Run from the repository root after independently obtaining the provider data
under the terms applicable at the time of access:

```powershell
$env:PYTHONPATH=(Get-Location).Path
$env:KERAS_HOME='D:\SET50_direction_prediction_paper\runtime_cache\keras'
$env:MPLCONFIGDIR='D:\SET50_direction_prediction_paper\runtime_cache\mpl'
$env:NUMBA_CACHE_DIR='D:\SET50_direction_prediction_paper\runtime_cache\numba'
$env:TF_ENABLE_ONEDNN_OPTS='0'
$env:TF_DETERMINISTIC_OPS='1'
py -3.12 -m models.set100_robustness_runner prepare
py -3.12 -m models.set100_robustness_runner run
```

Primary machine-readable outputs are in
`outputs/set100_same_exchange_robustness_v1/`, especially
`paper_table.csv`, `seed_averaged_fold_metrics.csv`,
`market_deltas_by_fold.csv`, `market_inference_holm.csv`,
`runtime_summary.csv`, `integrity_audit.json`, and `output_manifest.json`.
