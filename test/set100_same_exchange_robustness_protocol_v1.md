# SET100 Same-Exchange Robustness Protocol v1

Freeze time (UTC): `2026-08-03T14:50:13Z`

Status: **FROZEN BEFORE FITTING ANY SET100 MODEL**

Protocol ID: `set100-same-exchange-robustness-v1`

## Evidence status and claim boundary

This experiment is a post-hoc, same-exchange robustness check designed after
the SET50 outcomes were known. It is not an untouched confirmatory experiment
and is not an external-market replication. SET50 and SET100 share the Thai
market, evaluation period, macroeconomic regimes, and overlapping constituents.
Results may support or weaken within-exchange robustness only.

The SET100 data gate passed before this protocol was frozen. No SET100 model
prediction, metric, window search, Optuna run, or model-selection result existed
at freeze time. Mixed, null, and negative results will be retained.

## Fixed research question

How stable are the five frozen SET50 numerical models when the identical model,
feature, temporal, seed, and evaluation contracts are transferred to SET100?

## Frozen data and temporal contract

- Source: causally aligned SET100 daily, weekly, and monthly price-index data.
- Common cohort: SET100 rows restricted to observed SET50 reference dates.
- Feature availability: Asia/Bangkok, conservative 17:00 decision cutoff.
- Target: close on the next observed SET trading session.
- Outer tests: 2022, 2023, 2024, and 2025.
- Partial 2026 data are excluded from the primary evaluation.
- SET100 selection folds for 2018--2021 are not fitted or used for model/window
  selection. All model choices come from the previously frozen SET50 protocol.
- Every training fold retains only rows with `Label_Date` earlier than the first
  evaluation date. Purged boundary rows may supply feature-only sequence context
  but are excluded from scaler fitting and `model.fit`.
- SET50 and SET100 evaluation dates must match exactly within every paired fold
  after feature construction; otherwise cross-index inference fails closed.

## Frozen feature contract

The experiment uses only the numerical `Full TA + causal rolling VMD` feature
set. No SET100 news feature is included because a complete historical SET100
constituent-news universe has not been established.

- Full numerical/technical-analysis columns: 116.
- Rolling VMD columns: 6.
- Total model features: 122, in the existing generator order.
- VMD window: 60 observed sessions.
- VMD modes: 5.
- Penalty: 1,000.
- Highest-center-frequency mode is treated as noise.
- Every VMD row uses only the trailing window ending at that row.
- MinMax scaling is fitted separately on each supervised training fold and is
  applied to boundary context and test rows without refitting.

The feature generator, VMD configuration, feature order, missing-value rule,
and scaler implementation are inherited unchanged from SET50 point-in-time v2.

## Frozen models, windows, seeds, and training

| Model | Sequence window |
|---|---:|
| LSTM | 5 |
| CNN | 20 |
| LSTM-CNN | 20 |
| LSTM-Attention | 10 |
| LSTM-CNN-Attention | 20 |

Seeds: `42, 123, 456, 789, 2025`.

Architecture, layer sizes, optimizer, loss, fixed 20 epochs, batch size 32,
chronological ordering, and `shuffle=False` are inherited unchanged from the
frozen SET50 Track A point-in-time-v2 model contract. There is no Optuna search,
early-stopping selection, threshold tuning, window reselection, feature
selection, or SET100-result-dependent fallback.

The required design contains 100 model fits: 5 models x 4 outer folds x 5
seeds. Execution is resumable at the model/fold/seed cell level. Each model/seed
job runs in a fresh process and clears the TensorFlow session before each fold,
containing memory growth while preserving fold-level runtime measurement.

## Frozen endpoints and aggregation

Primary endpoint: Balanced Accuracy (BAcc).

Secondary directional endpoints:

- Direction Accuracy (DA);
- Matthews correlation coefficient (MCC);
- ROC AUC using `y_pred - current_close` as the continuous direction score;
- evaluated direction coverage and predicted-up share.

Secondary level endpoints:

- RMSE, MAE, and MAPE;
- NRMSE and NMAE, each normalised by the fold mean absolute target level.

Raw RMSE and MAE are reported within each index but are not used for direct
SET50-versus-SET100 superiority claims because the indices have different point
scales. Scale-normalised errors may be compared across indices.

Per-seed metrics are retained. For primary fold-level reporting, predictions
are averaged across the five seeds within each model and fold before metrics are
recomputed. Seeds are not treated as independent inferential replicates.

## Frozen paired comparison and inference

The registered contrast is `SET100 - SET50` for each model and fold. Positive
deltas favour SET100 for BAcc, DA, MCC, and AUC. Negative deltas favour SET100
for MAPE, NRMSE, and NMAE.

- Exact paired sign-flip inference uses the four outer-year deltas.
- A 95% t interval over the four temporal folds is reported descriptively.
- Holm adjustment is applied across the five architectures separately within
  each metric family.
- The minimum attainable two-sided exact sign-flip p-value with four folds is
  0.125; failure to reach significance is not treated as evidence of equality.

No performance threshold is used to discard a model or repeat the experiment.

## Required outputs and fail-closed audit

The run must save:

- per-cell predictions and per-seed metrics;
- fit and inference runtime;
- training/test sequence counts and model parameter counts;
- seed-averaged fold predictions and metrics;
- model-level SET100 summaries;
- paired SET100-minus-SET50 fold deltas;
- exact sign-flip and Holm-adjusted inference;
- runtime and environment summaries;
- frozen and generated input hashes;
- a machine-readable integrity audit and an execution log for the paper.

The audit fails on an input-hash mismatch, unregistered window/seed/year,
missing or duplicate cell, non-finite prediction, feature-count drift, date or
target misalignment, scaler leakage, label-boundary violation, prediction-shape
error, incomplete seed averaging, or unexpected output cardinality.

The row-level SET100 feature folds and predictions are restricted,
reconstructive artifacts and must remain outside the public replication
package. Only code, hashes, schemas, and non-reconstructive summaries may be
released.
