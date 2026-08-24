# PIT-CDR-LSTM direct 2024--2025 execution log v1

Date: 2026-08-07  
Protocol ID: `pit-cdr-lstm-direct-2024-2025-v1`  
Final status: **COMPLETED; INTEGRITY PASS WITH ONE DECLARED COUNT DEVIATION;
CANDIDATE REJECTED**

## 1. Why the evaluation was changed

The initial recommendation proposed an inner-fold promotion screen. The user
correctly noted that an inner result cannot be directly compared with the five
frozen models' 2024--2025 results. No inner result was therefore used as a
headline, promotion gate, or substitute benchmark. PIT-CDR-LSTM was frozen and
run directly on the identical final-arm test cohort used by the frozen five.

This is a retrospective architecture extension because the 2024--2025 outcomes
had already been observed elsewhere in the project. It is not an untouched
confirmatory test and no hyperparameter was changed after opening the result.

## 2. Frozen comparison contract

- Final input arm: `Regime-SHAP-Numeric-News`.
- Numerical features: 122 Full-TA plus causal rolling-VMD columns.
- News features: eight frozen expanding/local-NLP daily columns.
- Causal endpoint-regime masks: Bull=38, Sideway=130, Bear=88 active columns.
- Tests: 2024 (244 dates) and 2025 (234 dates), 478 dates total.
- Seeds: 42, 123, 456, 789, and 2025.
- Window: 5; LSTM(16), Dense(8), single direction logit.
- Training: 20 epochs, batch size 32, Adam 0.001, `shuffle=False`.
- Fixed direction threshold: 0.5.
- Seed probabilities averaged before fold metrics.
- Inference parameter count: 9,553 for every PIT-CDR variant.
- No Optuna, test-set stopping, threshold tuning, or second attempt.

The shared twin form exists only during training. Inference uses one LSTM tower
and has the same parameter count for the direct and relational variants.

## 3. Registered variants

1. `direct_lstm`;
2. `random_relations`;
3. `counter_direction_only`;
4. `cross_state_only`;
5. `pit_cdr_lstm`; and
6. `permuted_regime_cdr`.

The frozen mechanism gate required full PIT-CDR-LSTM to exceed all five controls
in equal-weight mean 2024--2025 Balanced Accuracy. One failed comparison was a
no-go condition.

## 4. Direct six-model result

All percentages below are equal-weight means of the 2024 and 2025 fold metrics.

| Rank | Model | BAcc (%) | Direction Accuracy (%) | MCC | RMSE | MAE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | LSTM-Attention | 53.703 | 52.725 | 0.0926 | 21.481 | 16.540 |
| 2 | LSTM-CNN-Attention | 53.122 | 51.888 | 0.0923 | 35.121 | 29.000 |
| 3 | LSTM | 53.082 | 52.307 | 0.0698 | 18.313 | 14.558 |
| 4 | CNN | 52.632 | 52.263 | 0.0543 | 24.060 | 18.459 |
| 5 | LSTM-CNN | 52.488 | 51.317 | 0.0681 | 29.014 | 23.363 |
| 6 | **PIT-CDR-LSTM** | **48.634** | **48.753** | **-0.0276** | N/A | N/A |

PIT-CDR-LSTM is a direction classifier, so assigning level RMSE/MAE to it would
be invalid. Its mean BAcc was 5.069 percentage points below LSTM-Attention and
4.448 points below the frozen LSTM.

### By year

| Model | 2024 BAcc (%) | 2025 BAcc (%) |
|---|---:|---:|
| CNN | 54.189 | 51.075 |
| LSTM | 52.961 | 53.202 |
| LSTM-CNN | 49.822 | 55.154 |
| LSTM-Attention | 53.590 | 53.816 |
| LSTM-CNN-Attention | 53.173 | 53.070 |
| **PIT-CDR-LSTM** | **48.408** | **48.860** |

The full candidate was below 50% BAcc in both test years.

## 5. Mechanism ablation

| Variant | Mean BAcc (%) | Mean DA (%) | Mean MCC | Mean Brier |
|---|---:|---:|---:|---:|
| Permuted-regime CDR | 53.540 | 53.188 | 0.0782 | 0.2605 |
| Cross-state only | 51.134 | 51.308 | 0.0224 | 0.2619 |
| Direct LSTM | 49.405 | 49.215 | -0.0120 | 0.2650 |
| Counter-direction only | 48.967 | 48.984 | -0.0207 | 0.2616 |
| **PIT-CDR-LSTM** | **48.634** | **48.753** | **-0.0276** | **0.2632** |
| Random relations | 47.684 | 47.319 | -0.0468 | 0.2661 |

Full PIT-CDR-LSTM beat only the random-relation control. It lost to direct LSTM
by 0.771 BAcc percentage points, counter-only by 0.333 points, cross-state-only
by 2.500 points, and permuted-regime CDR by 4.906 points. The surprisingly
strong permuted control and the full model's two-year sub-50% result reject the
claimed state-controlled relational mechanism.

Decision: **do not promote, tune, rerun, or add PIT-CDR-LSTM as Ours in the main
paper.** Any subsequent tuning on 2024--2025 would be test-set adaptation.

## 6. Runtime and environment

- Complete cells: 60 = 2 years x 5 seeds x 6 variants.
- End-to-end wall time: 1,333.835 seconds (22.23 minutes).
- Sum of recorded cell wall time: approximately 1,267.0 seconds.
- Sum of model fit time: 1,141.513 seconds.
- Sum of inference time: 33.641 seconds.
- Mean full PIT-CDR cell wall time: 20.806 seconds.
- Device recorded by TensorFlow: `/physical_device:CPU:0`.
- TensorFlow 2.21 on native Windows did not expose CUDA; this is a runtime fact,
  not a model-selection choice.
- Incremental API cost: USD 0.

## 7. Integrity and declared deviation

The input freeze verified 14 SHA-256 hashes. Candidate and frozen predictions
matched on date, current close, and actual next close. All 60 cells wrote finite
probabilities, weights, metrics, metadata, and runtime artifacts.

The freeze correctly registered 244 + 234 = 478 rows but incorrectly stated
that one was an actual zero-return tie, yielding an expected evaluable count of
477. Recalculation from the frozen actual values found no ties: all 478 rows are
direction-evaluable for every model, with 100% coverage. No row was added,
removed, or reclassified. The immutable freeze retains the original count and
the correction is declared in `protocol_deviation.json`.

Verification completed after execution:

- targeted tests: 8 passed;
- Ruff static checks: passed; and
- changed-module coverage: 87% overall (model 87%, runner 88% in the final
  verification pass).

## 8. Main artifacts

- Protocol: `test/pit_cdr_lstm_direct_protocol_v1.md`
- Immutable freeze: `test/pit_cdr_lstm_direct_freeze_v1.json`
- Model: `models/pit_cdr_lstm.py`
- Runner: `models/pit_cdr_lstm_runner.py`
- Six-model table: `outputs/pit_cdr_lstm_direct_2024_2025_v1/six_model_comparison_2024_2025.csv`
- Per-year six-model metrics: `outputs/pit_cdr_lstm_direct_2024_2025_v1/six_model_by_year_2024_2025.csv`
- Ablation summary: `outputs/pit_cdr_lstm_direct_2024_2025_v1/ablation_summary_2024_2025.csv`
- Per-year ablations: `outputs/pit_cdr_lstm_direct_2024_2025_v1/ablation_by_year_2024_2025.csv`
- Seed-averaged predictions: `outputs/pit_cdr_lstm_direct_2024_2025_v1/predictions_seed_averaged.csv`
- Runtime: `outputs/pit_cdr_lstm_direct_2024_2025_v1/runtime_summary.csv`
- Mechanism decision: `outputs/pit_cdr_lstm_direct_2024_2025_v1/mechanism_decision.json`
- Declared deviation: `outputs/pit_cdr_lstm_direct_2024_2025_v1/protocol_deviation.json`
- Integrity audit: `outputs/pit_cdr_lstm_direct_2024_2025_v1/integrity_audit.json`

