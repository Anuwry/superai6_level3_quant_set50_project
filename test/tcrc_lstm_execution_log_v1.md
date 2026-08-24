# TCRC-LSTM execution and closure log v1

## Final status

**Closed -- not promoted.** TCRC-LSTM failed the frozen inner-development gate.
No 2022--2025 outer run was executed, no post-result hyperparameter tuning was
performed, and the candidate is not eligible to be reported as the proposed
`Ours` model or as a primary paper claim.

## Why the experiment was run

The existing five-model diagnostics suggested three different empirical
strengths: LSTM produced the closest next-day index levels, CNN was relatively
sensitive to local reversals, and LSTM-CNN-Attention had the strongest overall
direction metric. TCRC-LSTM tested whether an end-to-end LSTM anchor plus a
CNN residual activated by a predicted turning-point gate could combine these
strengths without the dispersion seen in the existing hybrid models.

This architecture was motivated after the existing 2022--2025 results had been
viewed. It was therefore registered as a post-hoc method-development screen,
not an untouched confirmatory experiment.

## Frozen protocol

- Freeze: `test/tcrc_lstm_freeze_v1.json`
- Protocol: `test/tcrc_lstm_protocol_v1.md`
- Development years: 2020 and 2021
- Training histories: expanding 2019 -> 2020 and 2019--2020 -> 2021
- Seeds: 42, 123, 456, 789, 2025
- Five variants: LSTM anchor, unconditional CNN residual, latent turn gate,
  supervised turn gate, and full TCRC-LSTM
- CNN window: 20; LSTM sub-window: 5
- Fixed training: 20 epochs, batch size 32, Adam 0.001, chronological batches
- Primary metric: seed-averaged Balanced Accuracy derived from the sign of the
  reconstructed return
- Promotion: full model must beat every ablation in each year, exceed the LSTM
  anchor mean by at least 0.50 percentage point, and pass every integrity check

## Point-in-time correction made before fitting

The first smoke attempt detected that the last nominal training row had a
`Label_Date` on the first validation date. The runner was changed to purge all
training rows whose labels were not known strictly before validation start.
The purged rows remain usable only as feature history for boundary sequences.
A regression test was added before rerunning. This was a leakage-contract fix,
not a result-driven model change.

## Final seed-averaged results

| Variant | Mean BAcc | 2020 BAcc | 2021 BAcc | Mean Accuracy | Mean MCC | Mean RMSE | Mean MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM anchor | **50.90%** | 51.53% | 50.26% | 49.89% | 0.0412 | **13.96** | **9.72** |
| CNN residual | 50.76% | 48.84% | **52.68%** | **50.12%** | 0.0140 | 15.72 | 11.42 |
| Latent turn gate | 49.15% | 46.65% | 51.64% | 48.67% | -0.0198 | 15.26 | 10.95 |
| Supervised turn gate | 50.46% | 49.38% | 51.55% | 49.90% | 0.0080 | 15.01 | 10.64 |
| Full TCRC-LSTM | 50.21% | 48.14% | 52.28% | 49.50% | 0.0007 | 14.49 | 10.20 |

The full model was **0.69 percentage point below** its LSTM anchor on mean
Balanced Accuracy. It also failed to beat all ablations in either inner year.

## Mechanistic diagnostics

- True valid turning-point prevalence was 53.94% in 2020 and 52.50% in 2021.
- Full-model turning accuracy averaged only 50.53%, so the gate did not learn a
  reliable reversal signal.
- Its mean gate was 0.218 in 2020 and 0.507 in 2021, showing substantial temporal
  instability rather than consistent conditional correction.
- The unconditional CNN residual increased forecast displacement and produced
  worse RMSE/MAE than the LSTM anchor.
- Turn-conditioned attention did not recover the loss: full-model BAcc was below
  both the anchor and the simpler supervised-gate ablation.

These findings falsify the proposed mechanism on the frozen development cohort:
the CNN residual and turning gate did not provide stable incremental information
beyond the LSTM anchor.

## Metric audit

An aggregation regression test exposed that averaged-seed direction must be
computed from `y_pred > Close_D`, rather than thresholding the mean probability.
The final aggregate was rebuilt using the registered reconstructed-return rule.
The values in this log are the corrected final values.

## Runtime and environment

- Hardware detected: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB
- Execution device: CPU
- Reason: the installed native-Windows TensorFlow 2.21 environment cannot use
  NVIDIA GPU, and the installed PyTorch builds were CPU-only. An attempted CUDA
  wheel installation was stopped when it failed to provide download progress;
  no CUDA experiment result was generated.
- Trainable parameters: 21,420 for every capacity-matched variant
- Total model-fit time: 108.506 seconds
- Total inference time: 0.352 seconds
- Total cell wall time: 121.990 seconds across ten cells

## Verification

- Integrity: passed for all ten cells
- Candidate metric rows: 50
- Prediction rows: 12,075
- Focused tests: 16 passed
- Coverage for the two new modules: 86% overall
- No outer 2022--2025 result exists for TCRC-LSTM

## Retention rule

Retain the code, frozen protocol, predictions, metrics, and closure decision as a
negative architecture screen for auditability. Do not include TCRC-LSTM in the
main benchmark table and do not describe it as the proposed model. It may be
mentioned only as an exploratory negative result if manuscript space and the
research narrative justify it.

