# FCTA-LSTM retrospective execution log v1

Protocol: `fcta-lstm-retrospective-2024-2025-v1`

Freeze: `2026-08-07T14:55:06Z`, before access to any FCTA-LSTM outer result

Execution completed: `2026-08-07T15:09:26Z`

Decision: **NOT PROMOTED; FCTA-LSTM v1 CLOSED WITHOUT TUNING OR RERUN**

## Evidence status and purpose

This was a frozen one-shot retrospective architecture-development screen, not
an untouched confirmatory test. Earlier project work had already exposed the
2024--2025 outcomes. The experiment therefore evaluates whether the frozen
mechanism was promising enough to retain; it cannot establish prospective
generalization.

The tested model was Faithful Counterfactual Temporal Attention LSTM
(FCTA-LSTM). It retained the successful next-close regression formulation and
attempted to supervise temporal attention using shared-weight leave-one-day-out
prediction changes. It contained no worker ensemble, Leader, debate module,
Optuna search, threshold tuning, or API call.

## Integrity

- Input hash audit: passed for all 13 frozen inputs.
- Cohort: identical to the five frozen models.
- Test rows: 244 in 2024 and 234 in 2025; 478 total.
- Features: the same 130-column Regime-SHAP-Numeric-News arm.
- Window: 5.
- Seeds: 42, 123, 456, 789, and 2025.
- Grid: 2 years x 5 seeds x 4 variants = 40 completed cells.
- Parameters: 10,705 for every variant.
- Missing or non-finite prediction outputs: none.
- Hardware used by TensorFlow: CPU. TensorFlow 2.21 on native Windows did not
  expose the installed GPU.
- Incremental API cost: USD 0.

## Internal ablation

Seed probabilities/predictions were averaged before each annual metric. The
reported mean gives equal weight to 2024 and 2025.

| Variant | Mean BAcc | Mean DA | Mean MCC | RMSE | MAE | Predicted-up share |
|---|---:|---:|---:|---:|---:|---:|
| Mask augmentation | 51.935% | 52.084% | 0.0400 | 16.567 | 13.071 | 48.38% |
| Direction consistency | 50.870% | 50.846% | 0.0182 | 16.632 | 13.045 | 53.38% |
| FCTA-LSTM (full) | 49.868% | 49.982% | -0.0028 | 16.829 | 13.196 | 50.57% |
| Attention control | 49.121% | 48.557% | -0.0194 | 15.998 | 13.248 | 65.33% |

The best internal predictive variant was mask augmentation, not the full
architecture. Adding the faithfulness-alignment term reduced mean BAcc by
2.067 percentage points relative to mask augmentation.

## Full FCTA annual result

| Test year | BAcc | DA | MCC | Predicted-up share |
|---:|---:|---:|---:|---:|
| 2024 | 50.131% | 50.820% | 0.0028 | 33.20% |
| 2025 | 49.605% | 49.145% | -0.0085 | 67.95% |
| Equal-year mean | 49.868% | 49.982% | -0.0028 | 50.57% |

The aggregate predicted-up share appears balanced only because the two years
have opposite biases. This is not stable directional behavior.

## Comparison with the five frozen models

| Rank | Model | Mean BAcc | Mean DA | Mean MCC | RMSE | MAE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | LSTM-Attention | 53.703% | 52.725% | 0.0926 | 21.481 | 16.540 |
| 2 | LSTM-CNN-Attention | 53.122% | 51.888% | 0.0923 | 35.121 | 29.000 |
| 3 | LSTM | 53.081% | 52.307% | 0.0698 | 18.313 | 14.558 |
| 4 | CNN | 52.632% | 52.263% | 0.0543 | 24.060 | 18.459 |
| 5 | LSTM-CNN | 52.488% | 51.317% | 0.0681 | 29.014 | 23.363 |
| 6 | FCTA-LSTM | 49.868% | 49.982% | -0.0028 | 16.829 | 13.196 |

FCTA-LSTM was 3.835 percentage points below LSTM-Attention in BAcc. Its lower
RMSE than LSTM-Attention did not translate into correct next-day direction,
which again demonstrates that level accuracy and directional discrimination
are different objectives in this dataset.

## Fidelity result and positional-bias audit

| Variant | Attention/counterfactual JSD (lower better) | Top-1 agreement |
|---|---:|---:|
| Attention control | 0.01738 | 95.98% |
| Direction consistency | 0.01843 | 95.16% |
| Mask augmentation | 0.02162 | 96.15% |
| FCTA-LSTM | 0.02199 | 95.51% |

Full FCTA was worse than the attention control on both registered fidelity
conditions. A post-execution descriptive audit also found that the maximum
attention weight occurred at `t-4`, the oldest day in the five-day window, in
100% of observations for every variant. The deletion target selected the same
day in approximately 95--96% of observations.

This apparent agreement is a structural positional artifact. In the frozen
implementation, causal self-attention scores were averaged across all query
positions, so the oldest key was available to every query while newer keys
were available to fewer queries. The oldest observation also propagates
through the greatest number of LSTM recurrent transitions. Consequently, both
attention and deletion sensitivity were biased toward `t-4`; the high top-1
agreement cannot be interpreted as independent explanation faithfulness.

The post-execution positional summary is descriptive only and was not used to
change or rerun the candidate.

## Promotion decision

Passed:

- full FCTA beat the attention control in mean BAcc;
- full FCTA beat the attention control in 2024.

Failed:

- did not beat direction consistency in mean BAcc;
- did not beat mask augmentation in mean BAcc;
- did not beat the attention control in 2025;
- did not beat the best frozen model;
- did not lower JSD relative to the attention control; and
- did not improve top-1 fidelity relative to the attention control.

Because all registered conditions were conjunctive, the final decision is
`promoted = false`. No FCTA v1 hyperparameter tuning, alternate seed subset,
threshold optimization, or second outer run is permitted.

## Runtime

| Variant | Cells | Mean wall seconds/cell | Total wall seconds |
|---|---:|---:|---:|
| Attention control | 10 | 11.06 | 110.60 |
| Direction consistency | 10 | 11.30 | 113.02 |
| Mask augmentation | 10 | 18.93 | 189.29 |
| FCTA-LSTM | 10 | 19.56 | 195.55 |

End-to-end wall time including data preparation and aggregation was 680.11
seconds (11.34 minutes). Full FCTA cost about 1.77 times the attention-control
runtime during this experiment, while ordinary inference retained one full
pass; the additional leave-one-day-out passes were used for training and the
fidelity audit.

## Paper-use decision

FCTA-LSTM must not be presented as the proposed or superior model. If retained
at all, it belongs in a supplementary negative architecture-screen table or a
limitations/design-lessons paragraph. The main benchmark remains the frozen
five-model table, led by LSTM-Attention.

The next candidate must avoid averaging causal self-attention across query
positions. A fair temporal attribution mechanism should give all five lags an
equal opportunity to receive attention, for example by using one terminal
query over independently embedded lag tokens and by testing positional-null
controls before any outer evaluation.

## Artifact index

- Protocol: `test/fcta_lstm_protocol_v1.md`
- Freeze manifest: `test/fcta_lstm_freeze_v1.json`
- Machine-readable decision: `outputs/fcta_lstm_2024_2025_v1/promotion_decision.json`
- Integrity audit: `outputs/fcta_lstm_2024_2025_v1/integrity_audit.json`
- Internal ablation: `outputs/fcta_lstm_2024_2025_v1/ablation_summary_2024_2025.csv`
- Annual results: `outputs/fcta_lstm_2024_2025_v1/ablation_by_year_2024_2025.csv`
- Six-model comparison: `outputs/fcta_lstm_2024_2025_v1/six_model_comparison_2024_2025.csv`
- Fidelity: `outputs/fcta_lstm_2024_2025_v1/fidelity_diagnostics.csv`
- Positional diagnostic: `outputs/fcta_lstm_2024_2025_v1/temporal_position_bias_summary.csv`
- Runtime: `outputs/fcta_lstm_2024_2025_v1/runtime_summary.csv`
- Reproducibility metadata: `outputs/fcta_lstm_2024_2025_v1/run_metadata.json`

