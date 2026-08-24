# Moving-block bootstrap audit v1

Audit time (UTC): `2026-08-03T16:57:00Z`  
Source protocol: `integrated-multimodal-posthoc-v1`  
Status: **VERIFIED; NO RERUN OR POST-HOC RETUNING**

## Purpose and evidence role

The daily circular moving-block bootstrap is a serial-dependence sensitivity
analysis for the integrated five-model SET50 experiment. The four temporal
outer folds and exact fold-level sign-flip tests remain the primary inference
units. The bootstrap is not used to replace the registered four-fold analysis
or to manufacture significance.

## Verified design

| Item | Verified value |
|---|---:|
| Architectures | 5 |
| Registered contrasts | 5 |
| Metrics | 2 |
| Output rows | 50 |
| Block construction | Circular moving blocks within each fold |
| Block length | 10 trading days |
| Bootstrap replicates | 10,000 |
| Outer folds | 4 |
| Daily rows per contrast | 960 or 962 |
| Multiplicity control | Holm within each registered model family |

The implementation resamples circular blocks independently within each fold,
computes a fold mean, and then averages the four fold means for each bootstrap
replicate. Seeds had already been averaged before this temporal inference.

## Directional result audit

For the primary directional endpoint (`balanced_accuracy_delta_pp`), the file
contains 25 rows: five models by five registered contrasts.

| Contrast family | Rows | Minimum Holm p-value | Largest absolute effect (pp) |
|---|---:|---:|---:|
| Final integrated effect | 5 | 1.000 | 0.994 |
| Global news effect | 5 | 0.412 | 1.895 |
| Regime-pipeline news effect | 5 | 0.998 | 1.790 |
| Regime-SHAP effect without news | 5 | 1.000 | 2.028 |
| Routing-news interaction | 5 | 0.543 | 2.929 |

No Balanced Accuracy contrast passed Holm adjustment at 0.05. The smallest
adjusted directional p-value was 0.412 for the LSTM-CNN-Attention global-news
contrast (point estimate -1.224 pp; 95% interval [-2.738, 0.139]).

The remaining 25 rows concern squared-error loss. Twelve of those rows have
Holm-adjusted p-values at or below 0.05. They are regression-level sensitivity
results and must not be described as evidence of improved next-day direction.

## Integrity evidence

- `integrity_audit.json` reports `passed = true` and exactly 50 bootstrap rows.
- All five registered model names and all five registered contrasts are
  present.
- The authoritative bootstrap artifact SHA-256 is
  `132322d639f74e22e6f0a6852dbbc68fa97df11478db98d08ead9bd4b7e3f42d`.
- The audit did not change block length, bootstrap count, seeds, contrasts,
  endpoints, or multiplicity families after inspecting results.

## Paper wording constraint

Permitted wording: daily circular moving-block-bootstrap sensitivity did not
confirm any Balanced Accuracy improvement after Holm correction, consistent
with the weak and architecture-dependent directional signal.

Blocked wording: the bootstrap proves that all multimodal components are
ineffective, or that significant squared-error results imply significant
directional forecasting gains.

## Authoritative artifacts

- `test/integrated_multimodal_protocol_v1.md`
- `test/integrated_multimodal_execution_log_v1.md`
- `outputs/integrated_multimodal_posthoc_v1/daily_block_bootstrap_holm.csv`
- `outputs/integrated_multimodal_posthoc_v1/integrity_audit.json`
- `models/track_c_inference.py`
- `models/integrated_multimodal_runner.py`
