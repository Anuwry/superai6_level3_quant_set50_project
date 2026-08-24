# Pre-SHAP Consistency Audit

> **SUPERSEDED ON 2026-07-31:** This earlier audit verified artifact schemas,
> counts, fold dates, and declared fit scopes, but did not test the timestamp
> of each training label. The deeper audit found that the last training
> `Target_Next_Close` in every fold equals the first test close. Do not use the
> PASS decision below as authorization to run SHAP. See
> `test/paper_reviewer_audit_pre_shap.md`.

Audit date: 2026-07-31 (Asia/Bangkok)  
Result: **PASS WITH DECLARED SCOPE LIMITS**

This audit checks the frozen protocol against saved experiment artifacts. It
does not evaluate SHAP results because SHAP has not yet been run.

## Artifact checks

| Contract item | Saved evidence | Verified value |
|---|---|---|
| Five architectures | `outputs/track_a_final/locked_windows.csv` | LSTM, CNN, LSTM-CNN, LSTM-Attention, LSTM-CNN-Attention |
| Locked windows | same file | 5, 5, 20, 20, 20 respectively |
| Numerical feature pool | fold CSV headers | Full TA = 116; Full TA + VMD = 122 |
| VMD increment | fold CSV header difference | 6 features |
| Outer test boundary | `outputs/track_a_final/input_manifest.json` | 2022, 2023, 2024, 2025; selection excludes these tests |
| Outer test rows | `outputs/track_c/daily_regime_v2/fold_summary.csv` | 241, 243, 244, 234 |
| Last available date | same file | 2025-12-18 |
| Regime timing | same file | current regime at day t for direction at t+1 |
| Regime fit scope | same file | fold-training-only |
| Regime target leakage | same file | target columns used = false |
| Pipeline figure | `output/pdf/first_pipeline_revised.pdf` | one landscape A4 page; visually verified |

## Cross-document checks

The following controlling documents agree on the model list, model-specific
windows, next-close regression target, four SHAP ablation arms, forecasting
metrics, and exclusion of profitability/live-trading claims:

- `test/pre_shap_experiment_manifest.md`
- `test/pre_shap_reviewer_risk_resolution.md`
- `pipeline/pipeline8.md`
- `output/pdf/first_pipeline_revised.pdf`

The revised PDF contains no BiLSTM stage, no SHAP-regression claim, and no
claim that historical trading or live execution has been completed.

## Declared limits that remain

1. Track C is post-hoc robustness evidence because daily regime v2 was
   designed after inspecting the HMM v1 outer diagnostic.
2. The 2025 fold is partial and ends on 2025-12-18.
3. The target is the SET50 index level. Economic claims require a separately
   frozen, tradeable-instrument backtest.
4. Unequal windows are valid only as locked model-pipeline comparisons.
   SHAP effects must be paired within each unchanged model/window pipeline.

These limits do not block the registered SHAP experiment. They do restrict
the claims that may be made from it.
