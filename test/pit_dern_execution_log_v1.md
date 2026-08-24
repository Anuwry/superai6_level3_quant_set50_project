# PIT-DERN exploratory extension: execution log v1

## Status

- Protocol: `pit-dern-exploratory-v1`
- Evidence class: post-freeze exploratory architecture extension
- Executed: 2026-08-04 (Asia/Bangkok workstation; timestamps stored in UTC)
- Result: technically valid run; promotion criteria not met
- Incremental API cost: USD 0
- Hardware/runtime: native Windows TensorFlow 2.21 CPU; native-Windows GPU was unavailable

This extension was registered and hashed before its result files were read. It is not a replacement for the frozen five-model benchmark and must be reported as exploratory evidence.

## Registered design

PIT-DERN is a point-in-time dual-evidence retrieval network. A lightweight dual-axis mixer encodes a five-day sequence into a normalized embedding and jointly estimates direction probability and next-close delta. The retrieval memory contains training sequences only. Every memory label date must be strictly earlier than the first query date.

The retrieval layer evaluates two historical evidence pools separately: the five nearest Up cases and five nearest Down cases under cosine similarity. A transferability gate blends dual retrieval with the encoder according to similarity quality and retrieval evidence margin. Regime arms use the frozen per-regime SHAP selections as residual feature weights (selected = 1.00; unselected retained = 0.25), append causal Bull/Sideway/Bear probabilities, and retain the frozen news block unchanged.

No Optuna search, threshold search, early stopping, post-result hyperparameter editing, or selective seed removal was used.

## Evaluation contract

- Folds: 4 walk-forward test years, 2022-2025
- Seeds: 42, 123, 456, 789, 2025
- Arms: Global-Numeric; Global-Numeric-News; Regime-SHAP-Numeric; Regime-SHAP-Numeric-News
- Window: 5
- Epochs: 20
- Batch size: 32
- Primary metric: balanced accuracy
- Primary arm: Regime-SHAP-Numeric-News
- Primary comparator: frozen LSTM-CNN-Attention
- Seed handling: predictions are averaged across the five seeds before fold metrics are calculated
- Registered ablations: encoder-only, standard retrieval, dual evidence, PIT-DERN blend, shuffled-retrieval control

Promotion required all of the following: mean BAcc improvement of at least 1 percentage point; positive improvement in at least three of four folds; PIT-DERN better than encoder-only; PIT-DERN better than shuffled retrieval; parameter increase no greater than 15%; complete finite predictions.

## Integrity and reproducibility

- Frozen input hashes verified: 12/12
- Completed cells: 20/20
- Final model/seed/fold/arm metric rows: 80
- Ablation rows: 400
- Fit records: 80
- Seed-averaged fold/arm rows: 16
- Minimum training sequences per fit: 723
- All predictions finite: yes
- Unit/contract tests: 14 passed
- Focused source coverage: 81% total (`pit_dern.py` and `pit_dern_extension.py`)
- Ruff and Bandit checks: passed after removing one unused test import

## Primary six-model comparison

All values below use the final Regime-SHAP-Numeric-News arm and the same seed-averaged, four-fold reporting contract.

| Model | BAcc (%) | Direction accuracy (%) | MCC | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| LSTM | 52.007 | 51.983 | 0.0443 | 15.972 | 12.731 |
| CNN | 51.495 | 51.756 | 0.0308 | 22.741 | 17.729 |
| LSTM-CNN | 52.810 | 52.111 | 0.0657 | 29.794 | 23.445 |
| LSTM-Attention | 52.620 | 52.398 | 0.0618 | 20.405 | 15.897 |
| LSTM-CNN-Attention | **53.642** | **53.428** | **0.0893** | 31.768 | 25.342 |
| PIT-DERN | 50.159 | 49.497 | 0.0037 | **13.688** | **11.013** |

PIT-DERN did not improve the primary direction estimand. It did, however, produce the lowest final-arm RMSE and MAE in this table. This is a secondary level-error result and must not be reframed as success on next-day direction.

## Primary fold contrast

| Fold/test year | Frozen LSTM-CNN-Attention BAcc (%) | PIT-DERN BAcc (%) | Delta (pp) |
|---|---:|---:|---:|
| fold_1 / 2022 | 51.286 | 47.871 | -3.415 |
| fold_2 / 2023 | 57.038 | 50.329 | -6.709 |
| fold_3 / 2024 | 53.173 | 49.936 | -3.237 |
| fold_4 / 2025 | 53.070 | 52.500 | -0.570 |
| Mean | 53.642 | 50.159 | -3.483 |

The delta was negative in all four folds. The exact four-fold sign-flip p-value was 0.125; this is not conventional statistical significance and the small number of folds limits inferential power.

## Mechanism ablation in the primary arm

| Variant | BAcc (%) | Direction accuracy (%) | MCC | RMSE |
|---|---:|---:|---:|---:|
| Encoder only | 50.340 | 49.708 | 0.0075 | 14.016 |
| Standard top-10 retrieval | 48.525 | 48.200 | -0.0305 | 8.169 |
| Dual Up/Down evidence | 48.430 | 48.313 | -0.0326 | **7.682** |
| PIT-DERN blend | 50.159 | 49.497 | 0.0037 | 13.688 |
| Shuffled-retrieval control | 50.340 | 49.708 | 0.0075 | 13.936 |

The retrieval mechanisms sharply reduced close-level RMSE while degrading direction BAcc. This demonstrates an objective mismatch: retrieving historically similar close-level movements can improve magnitude/level proximity without learning a transferable Up/Down boundary. The final blend remained close to encoder-only because its mean retrieval gate was small (approximately 1.9%-3.8% across folds in the primary arm), despite high nearest-neighbor cosine similarity. High similarity alone therefore did not imply directional transferability.

The shuffled control tied encoder-only in mean primary-arm BAcc, and PIT-DERN did not beat either. Consequently, the run supplies no evidence that the proposed retrieval mechanism adds directional information.

## Promotion decision

| Gate | Result |
|---|---|
| Mean BAcc improvement >= 1 pp | Fail (-3.483 pp) |
| Positive temporal folds >= 3/4 | Fail (0/4) |
| Better than encoder-only | Fail |
| Better than shuffled control | Fail |
| Parameter increase <= 15% | Pass (-12.724%) |
| Complete finite predictions | Pass |
| Overall promotion | **Fail** |

PIT-DERN must not be labeled as the proposed winning model. The defensible paper use is a registered negative architecture extension or reliability-audit case study showing that strong representation similarity and low regression error do not guarantee next-day directional skill.

## Runtime

- Total wall time across 20 isolated cells: 607.73 seconds (10.13 minutes)
- Mean wall time per cell: 30.39 seconds
- Total model fit time across all 80 fits: 517.82 seconds
- Total embedding inference time: 40.75 seconds
- Total retrieval time: 4.58 seconds
- Final-arm trainable parameters: 9,180 versus 10,518.33 mean for the frozen regime-expert LSTM-CNN-Attention comparison (-12.724%)
- Final-arm PIT-DERN fit time: 130.60 seconds across 20 fits

Frozen regime-aware models executed three expert fits per seed (60 final-arm fits), whereas PIT-DERN executed one conditioned fit per seed (20 final-arm fits). Total runtime is therefore descriptive, not a compute-matched causal comparison.

## Paper reporting decision

Keep the frozen LSTM-CNN-Attention as the strongest direction model. PIT-DERN may be included in an exploratory architecture subsection or appendix with its freeze, ablations, negative controls, all-fold consistency, and objective-mismatch finding. Do not tune PIT-DERN on the same 2022-2025 test results and then present the tuned result as confirmatory. Any redesign must receive a new versioned protocol and be evaluated on a new untouched outer test period or be explicitly labeled post hoc.

## Artifact map

- Protocol: `test/pit_dern_protocol_v1.md`
- Freeze: `test/pit_dern_freeze_v1.json`
- Implementation: `models/pit_dern.py`
- Comparison/promotion logic: `models/pit_dern_extension.py`
- Runner: `models/pit_dern_runner.py`
- Output root: `outputs/pit_dern_extension_v1/`
- Six-model final table: `six_model_final_arm_comparison.csv`
- All-arm table: `six_model_all_arms_comparison.csv`
- Mechanism ablation: `ablation_summary.csv`
- Fold contrast: `pit_dern_vs_lstm_cnn_attention_fold_contrast.csv`
- Runtime: `runtime_summary.csv`, `runtime_by_cell.csv`, `six_model_runtime_comparison.csv`
- Promotion decision: `promotion_decision.json`
- Integrity audit: `integrity_audit.json`

