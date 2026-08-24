# Pre-SHAP Reviewer-Risk Resolution Log

> **STATUS UPDATE (2026-07-31):** The readiness decision below was superseded
> by the deeper audit in `test/paper_reviewer_audit_pre_shap.md`. SHAP is on
> hold pending boundary-label, direction-contract, attribution-target,
> explainer-contract, and capacity-control corrections.

Date: 2026-07-31 (Asia/Bangkok)

## Resolved before SHAP

| Reviewer concern | Resolution |
|---|---|
| Model/target ambiguity | Locked as next-close regression with MSE and derived direction evaluation. |
| Incorrect model list | Locked to LSTM, CNN, LSTM-CNN, LSTM-Attention, and LSTM-CNN-Attention. BiLSTM and legacy Attention-LSTM-CNN are excluded. |
| Unequal model windows | Declared as model-pipeline comparison; SHAP ablation is paired within each model/window. |
| Base-model SHAP bias | Replaced two-anchor selection with normalized consensus ranks from all five locked model-window pipelines. |
| SHAP data snooping | Added three-stage temporal train/rank/validate splits ending in 2021. |
| Arbitrary SHAP stopping | Locked fixed top-k grid, one-standard-error rule, metric guardrails, and Jaccard stability. |
| Regime leakage | Inner split must refit daily v2 threshold on inner training only. |
| Small regime samples | Locked 200/40/40 minimum sequence counts and global fallback. |
| Seeds treated as independent | Locked seed averaging within fold and four temporal folds as primary units. |
| Multiple comparisons | Locked five-model families and Holm correction. |
| News history truncating SHAP | Primary SHAP pool is the 122 numerical features; eight news features remain a paired add-on. |
| Persistence DA misinterpretation | No-change is retained for RMSE only; majority-direction and previous-direction baselines are required. |
| 2025 described as a full year | Locked disclosure that the file ends on 2025-12-18. |
| HMM Sideway mismatch | HMM v1 is an ablation; daily semantic v2 is the primary router. |
| Unsupported trading claim | Historical profitability and live execution removed from the registered forecasting claim. |

## Cannot be repaired retrospectively

Track C v2 was designed after the HMM v1 2022-2025 semantic diagnostic was
viewed. No documentation can restore those years as a pristine Track C
holdout. The required remedy is transparent scope:

- Track A retains its pre-2022 window-selection boundary.
- Track C is labelled post-hoc robustness/exploratory evidence.
- No SOTA, universal-improvement, or confirmatory-regime claim is permitted.
- A truly confirmatory regime result requires new future data or an external
  market fixed before analysis.

## Deferred until after SHAP

These are not current claims and do not block starting SHAP:

- Tradeable-instrument economic backtest.
- Transaction costs, slippage, futures roll rules, and turnover.
- Frozen forward/live execution.
- External-market validation.

They do block any later claim of proven profitability or live-market utility.

## Required outputs from the SHAP stage

- Daily sample counts by model, regime, split, and window.
- Every fallback event and reason.
- Per-model/per-fold SHAP rank files.
- Consensus rank and rank-stability files.
- Complete top-k validation curve, including rejected candidates.
- Frozen selected list for Bull, Sideway, and Bear.
- Runtime for explanation, selection, and model refits.
- Four-arm paired outer predictions.
- Derived direction confusion matrices, Balanced Accuracy, and MCC.
- RMSE/MAE and paired daily loss files.
- Deviation log, even if empty.

## Submission wording controls

Allowed:

> The study evaluates a leakage-aware multimodal and regime-conditioned
> forecasting workflow and reports mixed as well as negative ablation
> outcomes.

Not allowed without new evidence:

- "All components improve accuracy."
- "The proposed method is universally superior."
- "The regimes are ground truth."
- "The framework is profitable after costs."
- "The system was validated through live execution."
- "The 2022-2025 Track C result is an untouched confirmatory test."

## Readiness decision

The project is ready to start the registered SHAP experiment only after the
corrected pipeline figure and this manifest are used as the controlling
documents. Economic backtesting remains a separate future protocol.
