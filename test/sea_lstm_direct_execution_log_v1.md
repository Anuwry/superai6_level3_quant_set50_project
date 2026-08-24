# SEA-LSTM direct 2024--2025 execution and closure log v1

Date: 2026-08-07  
Protocol ID: `sea-lstm-direct-2024-2025-v1`  
Final status: **COMPLETED; INTEGRITY PASS; CANDIDATE REJECTED AND CLOSED**

## Decision summary

SEA-LSTM v1 is not eligible as the paper's proposed `Ours` model. Its frozen
equal-year mean balanced accuracy was **50.950%**, compared with **53.703%** for
the strongest frozen model, LSTM-Attention. SEA-LSTM also failed the internal
mechanism gate because `positive_memory_only` achieved **52.586%** BAcc.

The failure action registered before the run is now active: SEA-LSTM v1 is
closed without hyperparameter tuning, threshold tuning, feature changes, or a
second attempt. The artifacts are retained as an auditable negative
architecture screen and should not be presented as a successful headline
model.

## Frozen experiment contract

- Protocol: `test/sea_lstm_direct_protocol_v1.md`
- Machine-readable freeze: `test/sea_lstm_direct_freeze_v1.json`
- Tests: 2024 and 2025 only
- Test observations: 244 in 2024 and 234 in 2025; 478 total
- Direction-evaluable observations: 478; no actual zero-return ties
- Seeds: 42, 123, 456, 789, and 2025
- Window: 5
- Input: frozen 130-column `Regime-SHAP-Numeric-News` arm
- Training: 20 epochs, batch size 32, Adam 0.001, `shuffle=False`
- Loss: binary cross-entropy plus 0.05 Brier component
- Threshold: fixed at 0.5
- Aggregation: average five seed probabilities before annual metrics
- Primary metric: equal-weight mean of annual balanced accuracy
- Incremental API cost: USD 0

Thirteen input hashes were verified before fitting. Scaling, feature masks,
dates, labels, current closes, and test cohorts were inherited from the frozen
pipeline and independently checked against the five-model prediction artifact.

## Architecture tested

SEA-LSTM is a single recurrent model and contains no Leader, worker, ensemble,
router, or debate module. The cell maintains non-negative up-evidence and
down-evidence memories. A shared forget/input/output gating computation updates
both memories from a signed proposal. The final logit is the non-negative
weighted up evidence minus the non-negative weighted down evidence.

SEA used 15 evidence units and **9,691 trainable parameters**. The internal
standard LSTM control used **9,553 parameters**, a difference of 138 parameters
or approximately 1.44%. All three SEA variants had identical parameter counts.

## Registered ablation result

| Variant | Mean BAcc (%) | Mean DA (%) | Mean MCC | Mean Brier | Predicted-up share (%) |
|---|---:|---:|---:|---:|---:|
| Positive memory only | 52.586 | 51.291 | 0.0864 | 0.2533 | 89.152 |
| SEA-LSTM | 50.950 | 50.222 | 0.0214 | 0.2553 | 72.212 |
| Negative memory only | 50.615 | 50.606 | 0.0123 | 0.2513 | 50.436 |
| Standard LSTM control | 49.856 | 48.557 | 0.0040 | 0.2531 | 90.281 |

SEA-LSTM exceeded the internally trained standard LSTM by 1.094 percentage
points and exceeded negative-memory-only by 0.335 points. It lost to
positive-memory-only by **1.636 points**, so the result does not support the
claim that jointly accumulating both signs is beneficial.

## Annual result

| Variant | 2024 BAcc (%) | 2025 BAcc (%) |
|---|---:|---:|
| Positive memory only | 51.137 | 54.035 |
| SEA-LSTM | 50.431 | 51.469 |
| Negative memory only | 51.625 | 49.605 |
| Standard LSTM control | 48.967 | 50.746 |

SEA-LSTM did beat the standard direction-classification LSTM in both years, but
that control was itself weak. This local improvement therefore does not imply
competitiveness with the five frozen price-output architectures.

## Direct six-model comparison

| Rank | Model | Mean BAcc (%) | Mean DA (%) | Mean MCC |
|---:|---|---:|---:|---:|
| 1 | LSTM-Attention | 53.703 | 52.725 | 0.0926 |
| 2 | LSTM-CNN-Attention | 53.122 | 51.888 | 0.0923 |
| 3 | LSTM | 53.081 | 52.307 | 0.0698 |
| 4 | CNN | 52.632 | 52.263 | 0.0543 |
| 5 | LSTM-CNN | 52.488 | 51.317 | 0.0681 |
| 6 | SEA-LSTM | 50.950 | 50.222 | 0.0214 |

SEA-LSTM trailed LSTM-Attention by **2.753 percentage points** and trailed even
the lowest frozen model, LSTM-CNN, by **1.538 points**.

## Frozen promotion decision

| Condition | Result |
|---|---|
| Beat standard LSTM mean | Pass |
| Beat positive-memory-only mean | **Fail** |
| Beat negative-memory-only mean | Pass |
| Beat standard LSTM in 2024 | Pass |
| Beat standard LSTM in 2025 | Pass |
| Beat strongest frozen model mean | **Fail** |

Overall promotion: **FAIL**.

## Interpretation

The dual signed-memory restriction did not add useful complementary evidence.
The positive-only ablation was materially better than the full cell, while the
full cell's mean Brier score was the worst of the four registered variants and
its binary cross-entropy was 0.7041. The most defensible interpretation is that
the down-memory path introduced conflicting or poorly calibrated evidence on
this small, noisy next-day dataset. This is a negative result, not a basis for
post-result removal of the down path and relabelling the positive-only control
as the proposed architecture.

## Runtime and environment

| Variant | Cells | Total cell wall time (s) | Mean per cell (s) | Parameters |
|---|---:|---:|---:|---:|
| Standard LSTM | 10 | 64.505 | 6.450 | 9,553 |
| Positive memory only | 10 | 74.434 | 7.443 | 9,691 |
| Negative memory only | 10 | 75.370 | 7.537 | 9,691 |
| SEA-LSTM | 10 | 75.114 | 7.511 | 9,691 |

Total experiment wall time was **311.358 seconds** on
`/physical_device:CPU:0`. TensorFlow emitted non-fatal native-Windows graph
compatibility and retracing warnings; the process exited successfully, all 40
cells were complete, all probabilities and metrics were finite, and the cohort
and artifact integrity audits passed.

## Verification evidence

- Contract and integration tests: 13 passed
- Coverage across the two new implementation modules: 88%
- Ruff lint: passed
- Frozen input hashes checked: 13
- Candidate training cells: 40/40
- Candidate variant-seed cohorts: 20/20, each containing all 478 dates
- API cost: USD 0

## Artifact index

- `outputs/sea_lstm_direct_2024_2025_v1/ablation_summary_2024_2025.csv`
- `outputs/sea_lstm_direct_2024_2025_v1/ablation_by_year_2024_2025.csv`
- `outputs/sea_lstm_direct_2024_2025_v1/six_model_comparison_2024_2025.csv`
- `outputs/sea_lstm_direct_2024_2025_v1/six_model_by_year_2024_2025.csv`
- `outputs/sea_lstm_direct_2024_2025_v1/runtime_summary.csv`
- `outputs/sea_lstm_direct_2024_2025_v1/promotion_decision.json`
- `outputs/sea_lstm_direct_2024_2025_v1/integrity_audit.json`
- `outputs/sea_lstm_direct_2024_2025_v1/run_metadata.json`
- `outputs/sea_lstm_direct_2024_2025_v1/all_seed_predictions.csv`
- `outputs/sea_lstm_direct_2024_2025_v1/predictions_seed_averaged.csv`
- per-cell predictions, metrics, metadata, and weights under
  `outputs/sea_lstm_direct_2024_2025_v1/cells/`
