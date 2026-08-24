# PIT-TLDN execution log v1

## Decision summary

The frozen PIT-TLDN inner-development run completed successfully. All ten
year-seed cells passed the leakage and integrity audit. The registered
promotion gate passed. Nevertheless, PIT-TLDN is not promoted into the frozen
five-model headline table at this stage because its full debate output did not
beat its strongest component worker and the CNN worker did not establish the
intended trend expertise.

These are deliberately reported as two different decisions:

- **Registered gate:** passed, without changing any condition after seeing the
  results.
- **Reviewer-facing scientific decision:** keep PIT-TLDN as a development
  candidate; do not rank it against the 2022--2025 frozen benchmark until a
  scientifically justified version beats its strongest worker in development
  and is evaluated under a comparable outer protocol.

The second decision is an explicitly post-result, more conservative judgment;
it is not presented as a preregistered condition.

## Files and implementation

- Pre-result protocol: `test/pit_tldn_protocol_v1.md`
- Frozen manifest: `test/pit_tldn_freeze_v1.json`
- Architecture: `models/pit_tldn.py`
- Runner: `models/pit_tldn_runner.py`
- Contract tests: `tests/test_pit_tldn.py`
- Results: `outputs/pit_tldn_inner_development_v1`

The architecture uses a W=20 causal multi-scale CNN Trend Worker, a W=5
multi-task LSTM Price Worker, worker-specific point-in-time top-30 SHAP masks,
and a 50-parameter constrained debate leader. The leader consumes the two
worker logits, explicit disagreement, worker confidence and four soft regime
context variables. It learns only from temporal out-of-fold worker claims.

## Frozen experiment contract

- Development validation years: 2020 and 2021.
- Seeds: 42, 123, 456, 789 and 2025.
- Temporal cross-fit splits: three expanding splits.
- Purge gap: 20 rows.
- Worker epochs: 15; leader epochs: 30; batch size: 32.
- SHAP selection seed: 31415; background cap: 48; explanation cap: 64;
  nsamples: 100.
- Primary metric: balanced accuracy at threshold 0.5.
- Outer years accessed: none.
- Incremental API calls and cost: zero.

The protocol was frozen at `2026-08-06T14:03:05Z`, before the first PIT-TLDN
cell was run. The freeze audit verified five input files by SHA-256.

## Data and leakage audit

Each inner-year scaler was fitted only on the inner training period. Each SHAP
mask was estimated from the training prefix belonging to its temporal split.
The leader training claims came from three purged out-of-fold blocks, never
from the final in-sample worker fits. All cross-fit blocks had a 20-row purge,
OOF endpoint indices were unique, training preceded validation, and all
probabilities were finite.

Audit totals:

- completed cells: 10/10;
- metric rows: 60;
- raw prediction rows: 14,490;
- seed-averaged prediction rows: 2,898;
- integrity status: passed;
- outer years accessed: none.

## Main inner-development results

Metrics below are calculated after averaging probabilities across five seeds
within each date. The reported standard deviation is across the two inner
validation years, not across seeds.

| Variant | Mean BAcc (%) | BAcc SD (pp) | Mean DA (%) | Mean MCC | BCE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| CNN Trend Worker + SHAP | 50.052 | 0.074 | 48.238 | 0.002 | 2.230 | 0.365 |
| LSTM Price Worker + SHAP | **52.557** | 3.084 | **51.361** | **0.067** | **0.720** | **0.263** |
| Simple average + SHAP | 50.178 | 0.251 | 48.655 | 0.004 | 0.816 | 0.298 |
| Leader without disagreement | 51.955 | 2.764 | 51.155 | 0.039 | 0.925 | 0.318 |
| PIT-TLDN, all 122 features | 49.880 | 0.170 | 48.030 | -0.004 | 0.874 | 0.310 |
| **PIT-TLDN, branch SHAP masks** | 51.998 | 2.826 | 50.530 | 0.046 | 1.197 | 0.342 |

PIT-TLDN was 1.946 percentage points above the CNN worker but 0.559 points
below the LSTM Price Worker. It also had binary cross-entropy 0.477 worse than
the LSTM worker, showing that the arbitration did not improve probability
calibration.

### Per-year seed-averaged results

| Year | CNN BAcc (%) | LSTM BAcc (%) | Average BAcc (%) | No-disagreement BAcc (%) | All-feature PIT-TLDN BAcc (%) | PIT-TLDN BAcc (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 50.000 | 50.376 | 50.000 | 50.000 | 50.000 | 50.000 |
| 2021 | 50.105 | **54.738** | 50.355 | 53.909 | 49.760 | 53.996 |

The development gain was concentrated in 2021. In 2020 the full model
collapsed to a single-class-like BAcc of 50%, so the result is not temporally
robust enough for a headline architecture claim.

## Debate diagnostics

- Mean absolute worker disagreement: 0.2014.
- Mean CNN leader weight: 0.4666 in 2020 and 0.4555 in 2021.
- Mean absolute bounded correction: 0.0678 in 2020 and 0.0349 in 2021.
- Final CNN/LSTM top-30 mask overlap: 5 features in each year; Jaccard 0.0909.

The low feature-mask overlap confirms that branch-specific SHAP produced
different evidence sets, and the non-collapsed leader weights show that the
leader used both workers. However, semantic differentiation alone was not
enough: the CNN branch remained near chance and poorly calibrated.

## Registered promotion result

The frozen gate passed all registered conditions:

1. integrity audit passed;
2. PIT-TLDN beat the simple average, no-disagreement leader and all-feature
   leader in mean inner BAcc;
3. its BAcc delta over the simple average was non-negative in both years
   (0.000 and +3.641 percentage points);
4. mean disagreement exceeded 0.02;
5. leader weight remained within 0.05--0.95 in both years.

The gate did not require the full model to beat each individual worker. The
observed underperformance versus the LSTM worker exposes this as a limitation
of the v1 promotion rule. The rule is not retroactively edited; the limitation
is disclosed and informs any separately frozen v2 protocol.

## Runtime and environment

- Python 3.11.15 on Windows 10 build 26200.
- NumPy 2.1.3; pandas 2.3.3; scikit-learn 1.7.2; TensorFlow 2.21.0;
  SHAP 0.51.0.
- TensorFlow ran on CPU because native Windows TensorFlow >=2.11 does not use
  CUDA GPU support in this environment.
- Sum of cell wall times: 1,044.88 seconds (17.41 minutes).
- Mean cell wall time: 104.49 seconds.
- Accounted component time: 719.30 seconds fitting, 30.50 seconds inference,
  and 16.83 seconds SHAP explanation.

One first attempt at 2020/seed 123 ended with native TensorFlow access violation
code `3221225477` before any partial cell was written. The cell was rerun in a
fresh process with TensorFlow inter-op, intra-op and OMP thread counts limited
to one to reduce memory pressure. No data, seed, model, epoch, batch or metric
setting changed. The rerun and all subsequent cells completed successfully.

## Protocol deviation note

The protocol text said the all-feature ablation would reuse the preliminary
full-feature workers fitted for SHAP. The implementation instead retrained
seed-matched full-feature workers for every temporal split. This increased
compute but gave the all-feature and selected-feature arms equal seed-specific
training. It did not change data access or model hyperparameters. The mismatch
is retained here rather than silently rewriting the frozen protocol.

## Scientific conclusion and next gate

PIT-TLDN v1 demonstrates three mechanisms correctly: point-in-time
branch-specific attribution, explicit disagreement, and leakage-safe temporal
leader training. It does not yet demonstrate that the complete architecture is
better than its strongest component. Therefore:

- do not add PIT-TLDN v1 to the frozen five-model headline ranking;
- do not claim a forecasting improvement from v1;
- retain it as a transparent negative/mixed development experiment;
- if a v2 is attempted, pre-register that the full model must beat the best
  individual worker in both mean BAcc and calibration before outer evaluation.

The file `outputs/pit_tldn_inner_development_v1/five_model_plus_candidate_status.csv`
shows the requested five-model-plus-candidate view while explicitly marking the
2020--2021 candidate result as not directly comparable with the frozen
2022--2025 outer results.
