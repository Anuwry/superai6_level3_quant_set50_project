# PIT-CMM-LSTM Exploratory Extension: Execution Log v1

Status: **complete**  
Protocol: `pit-cmm-lstm-exploratory-v1`  
Completed: 2026-08-04 (Asia/Bangkok)  
Evidence status: `post_freeze_exploratory_architecture_extension`

## 1. Research question

Test whether a point-in-time Competitive Matrix Memory LSTM
(`PIT-CMM-LSTM`) improves next-trading-day SET50 direction prediction over
the five registered neural architectures while retaining the same data,
temporal folds, seeds, training budget, prediction target, and integrated
pipeline arms.

This experiment is a post-freeze exploratory architecture extension. The
existing five-model results were not rerun or altered. Their frozen result
tables were joined to the newly executed PIT-CMM-LSTM results only during
aggregation.

## 2. Pre-result experiment contract

- Primary metric: balanced accuracy (BAcc).
- Primary arm: `Regime-SHAP-Numeric-News`.
- Primary comparator: LSTM.
- Secondary metrics: direction accuracy (DA), MCC, RMSE, and MAE.
- Point-in-time folds: four held-out years, 2022, 2023, 2024, and 2025.
- Seeds: 42, 123, 456, 789, and 2025.
- Sliding window: 5 trading days.
- Training: 20 epochs, batch size 32, Adam/MSE, `shuffle=False`.
- PIT-CMM configuration: 12 hidden units, rank-4 bullish memory, rank-4
  bearish memory, and an 8-unit dense head.
- No Optuna, threshold tuning, early stopping, post-result feature
  reselection, or automatic architecture refinement was used.
- Incremental API cost: USD 0.

The promotion rule required all of the following:

1. Mean final-arm BAcc improvement of at least 1.0 percentage point over
   LSTM.
2. Positive final-arm BAcc delta in at least three of four temporal folds.
3. Trainable-parameter increase no greater than 15% at each audited input
   dimension.
4. Complete finite predictions.

The frozen contract is stored in `test/pit_cmm_lstm_freeze_v1.json`; its
integrity check passed before aggregation.

## 3. Architecture implemented

PIT-CMM-LSTM retains the standard LSTM hidden state and cell state, and adds
two separate low-rank matrix memories inside the recurrent cell:

- a bullish memory updated from bullish evidence;
- a bearish memory updated from bearish evidence.

At each time step, the cell computes a two-way softmax evidence allocation,
applies independently learned memory decay, writes rank-one key/value updates
to the two memories, reads both memories with a query, and combines bullish
readout, sign-inverted bearish readout, their difference, and the signed
evidence margin. A trainable debate-scale parameter injects this competitive
readout into the LSTM candidate state. The final recurrent output is passed
through an 8-unit dense layer and a linear next-close output.

The model is point-in-time with respect to the supplied features: no held-out
targets are used to update weights, scalers, features, memories, or thresholds.

## 4. Executed design

The new model was executed on all four integrated arms:

1. `Global-Numeric`
2. `Global-Numeric-News`
3. `Regime-SHAP-Numeric`
4. `Regime-SHAP-Numeric-News`

There were 20 isolated fold-seed cells (4 folds x 5 seeds). Each cell produced
four arm-level metrics and eight actual model fits because regime-aware arms
fit separate regime-specific models. The completed run therefore contains:

- 20/20 completed cells;
- 80 seed-fold-arm metric rows;
- 160 executed model fits;
- 16 fold-arm rows after averaging predictions across five seeds;
- minimum regime-specific training size of 216 sequences;
- finite predictions for every test observation.

## 5. Six-model comparison

All percentage values below are temporal-fold means after seed-level
prediction averaging. The final arm is `Regime-SHAP-Numeric-News`.

| Model | Global Numeric BAcc (%) | Global + News BAcc (%) | Regime-SHAP Numeric BAcc (%) | Final BAcc (%) | Final DA (%) | Final MCC | Final RMSE | Final rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 51.847 | 50.265 | 51.879 | 52.007 | 51.983 | 0.0443 | 15.972 | 4 |
| CNN | 51.798 | 51.789 | 53.284 | 51.495 | 51.756 | 0.0308 | 22.741 | 6 |
| LSTM-CNN | 53.775 | 53.603 | 54.072 | 52.810 | 52.111 | 0.0657 | 29.794 | 2 |
| LSTM-Attention | 53.614 | 51.719 | 51.586 | 52.620 | 52.398 | 0.0618 | 20.405 | 3 |
| LSTM-CNN-Attention | 53.109 | 51.885 | 54.047 | **53.642** | **53.428** | **0.0893** | 31.768 | **1** |
| PIT-CMM-LSTM | **53.666** | **53.391** | 50.584 | 51.572 | 51.272 | 0.0371 | 24.239 | 5 |

PIT-CMM-LSTM ranked second of six in the Global-Numeric arm, 0.109 percentage
point below LSTM-CNN and 1.819 percentage points above LSTM. It also ranked
second in Global-Numeric-News. It did not retain this gain after regime/SHAP
routing and ranked fifth in the final integrated arm.

## 6. Primary PIT-CMM-LSTM versus LSTM contrast

| Temporal fold | LSTM final BAcc (%) | PIT-CMM final BAcc (%) | Delta (pp) |
|---|---:|---:|---:|
| Fold 1 | 49.631 | 49.993 | +0.362 |
| Fold 2 | 52.234 | 51.590 | -0.644 |
| Fold 3 | 52.961 | 49.791 | -3.170 |
| Fold 4 | 53.202 | 54.912 | +1.711 |
| Mean | 52.007 | 51.572 | -0.435 |

The fold-level exact sign-flip p-value was 0.75. This is descriptive because
four temporal folds provide low inferential resolution.

## 7. Promotion decision

PIT-CMM-LSTM **did not pass** the predeclared promotion rule:

- Mean BAcc delta gate: failed (-0.435 pp; required at least +1.0 pp).
- Temporal-consistency gate: failed (2/4 positive folds; required at least
  3/4).
- Parameter-budget gate: passed.
- Complete-finite-prediction gate: passed.

Parameter increases versus the matched LSTM were 2.32%, 4.15%, and 4.74% at
38, 88, and 130 input features, respectively, all below the 15% ceiling.

## 8. Prediction and failure diagnostics

- Prediction coverage was 100% in every fold and arm.
- No fold-arm produced a single direction class only.
- No prediction was an exact unchanged-close abstention.
- Some folds exhibited substantial directional imbalance. For example,
  Global-Numeric predicted upward movement for 7.9% of Fold 1 observations
  and 84.8% of Fold 3 observations.
- The architecture therefore did not collapse numerically, but its learned
  directional prior shifted considerably across temporal folds.
- The strongest degradation occurred when regime-specific SHAP feature
  routing was introduced: BAcc fell from 53.666% in Global-Numeric to 50.584%
  in Regime-SHAP-Numeric. This localizes the main failure to the interaction
  between the new recurrent memory and regime-specific fitting/feature
  selection, not to the global numerical backbone alone.

No negative-control experiment was included in v1, so no negative-control
claim may be made from this extension.

## 9. Runtime and compute

PIT-CMM-LSTM used native Windows TensorFlow 2.21 on CPU. TensorFlow GPU is not
available natively on Windows for this version; the installed RTX 4060 was
therefore not used.

- Total wall time across 20 isolated cells: 1,637.16 seconds (27.29 minutes).
- Mean/median cell wall time: 81.86/82.20 seconds.
- Cell range: 66.28 to 94.52 seconds.
- Final-arm PIT-CMM-LSTM: 60 fits, 477.22 fit-seconds total, 7.954 seconds per
  fit on average, and 56.54 inference-seconds total.
- Final-arm LSTM: 3.187 seconds per fit on average.
- Final-arm LSTM-CNN-Attention: 7.224 seconds per fit on average.

PIT-CMM-LSTM was the slowest model by mean final-arm fit time in this local
comparison, while remaining within the predeclared parameter budget.

## 10. Verification

- Targeted unit/integration tests: 19 passed.
- Core model and comparison-helper test coverage: 92% total
  (`pit_cmm_lstm.py` 97%; `pit_cmm_extension.py` 85%).
- Ruff lint: passed.
- Bandit scan: passed with no reported issue after documenting the internal,
  shell-disabled subprocess contract.
- Full-run integrity audit: passed.
- Static typing was diagnostic-only: Pyright's available TensorFlow and
  pandas stubs do not type the Keras/custom-cell and pandas operations used by
  the runner completely. Runtime tests and the 20-cell integration run are the
  authoritative verification for these paths.

## 11. Defensible paper interpretation

The v1 result does not support claiming that PIT-CMM-LSTM is the best final
SET50 model. It supports a narrower exploratory finding: competitive matrix
memory is promising as a global numerical recurrent backbone, where it ranked
second of six, but its current interaction with regime-specific feature
routing is unstable and removes that gain. If retained in the paper, it should
be reported as a pre-specified post-freeze exploratory extension and negative
integration result, not silently substituted for the frozen main model.

Any further architectural revision or regime-routing ablation must receive a
new protocol identifier and evidence label; v1 results must remain immutable.

## 12. Authoritative artifacts

- `outputs/pit_cmm_lstm_extension_v1/six_model_compact_comparison.csv`
- `outputs/pit_cmm_lstm_extension_v1/six_model_all_arms_comparison.csv`
- `outputs/pit_cmm_lstm_extension_v1/six_model_final_arm_comparison.csv`
- `outputs/pit_cmm_lstm_extension_v1/six_model_runtime_comparison.csv`
- `outputs/pit_cmm_lstm_extension_v1/pit_cmm_vs_lstm_fold_contrast.csv`
- `outputs/pit_cmm_lstm_extension_v1/prediction_distribution_diagnostics.csv`
- `outputs/pit_cmm_lstm_extension_v1/parameter_budget.csv`
- `outputs/pit_cmm_lstm_extension_v1/promotion_decision.json`
- `outputs/pit_cmm_lstm_extension_v1/integrity_audit.json`
- `outputs/pit_cmm_lstm_extension_v1/run_metadata.json`
