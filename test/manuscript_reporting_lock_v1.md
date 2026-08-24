# Manuscript reporting lock v1

Status: **CONTROLLING PRE-MANUSCRIPT REPORTING SPECIFICATION**  
Lock date: 2026-08-04 (Asia/Bangkok)  
Target journal family: computational economics / forecasting / applied AI  
Central framing: reliability audit, not state-of-the-art forecasting

This document consolidates the completed experiment logs into one writing
contract. It does not change an observed result, promote an evidence tier, or
replace a frozen experiment protocol. Where a historical log records an
earlier state, the dated execution log remains part of the audit trail and this
document controls only manuscript wording and placement.

## 1. Single paper-level question

> Under a leakage-controlled point-in-time protocol, do numerical denoising,
> predicted news, role-based LLM sentiment, regime-aware feature selection,
> and added neural complexity yield improvements that remain reliable across
> architectures, time, and a broader index on the same exchange?

The answer is mixed. Some components improve selected architectures or
intrinsic endpoints, but no primary forecasting balanced-accuracy contrast
survives the registered multiplicity controls. This is the central audit
finding, not a failed attempt to manufacture a winning model.

## 2. Five main points and evidence class

| Main point | Paper role | Evidence class | Required boundary |
|---|---|---|---|
| 1. Point-in-time data reliability | Foundation for every result | Primary | Control observed feature/label availability; do not claim immunity from every possible bias |
| 2. Numerical and denoising reliability | Full TA, causal VMD, frozen window panel | Corrected primary/secondary ablation | Effects are architecture- and metric-dependent; four outer years imply low fold-level power |
| 3. Multimodal and LLM reliability | Predicted-news fusion, four frozen falsification controls, plus a separate intrinsic text benchmark | Pre-frozen retrospective falsification and robustness; earlier integrated arm is post-hoc | No BAcc control family survived Holm; the Leader result is intrinsic and is not the downstream news-feature source |
| 4. Regime-aware explainability reliability | Capacity-aware routing, progressive SHAP, diagnostic LIME | Post-hoc robustness and diagnostic | No universal regime/SHAP gain; LIME does not independently validate SHAP |
| 5. Forward and transfer reliability | Partial-2026 stress, SET100 transfer, serial-dependence and release audit | Robustness/exploratory | Partial-2026 is source-contingent; SET100 is same-exchange, not external-market replication |

The five registered architectures (LSTM, CNN, LSTM-CNN, LSTM-Attention, and
LSTM-CNN-Attention) form a common horizontal benchmark panel. They are not
presented as five paper contributions or as an exhaustive model tournament.

## 3. Endpoint and inference lock

### Market forecasting

- Primary directional endpoint: balanced accuracy.
- Secondary endpoints: direction accuracy, MCC, coverage, RMSE, and MAE as
  applicable to the relevant model objective.
- Seeds measure fitting variability and are averaged before temporal
  inference; they are not independent market samples.
- Outer years are the primary temporal units for the five-model comparisons.
- Registered exact sign-flip and Holm-adjusted families remain authoritative.
- The 10-day circular moving-block bootstrap is a serial-dependence
  sensitivity analysis. Its squared-error findings cannot be converted into
  directional significance claims.

### Intrinsic sentiment benchmark

- Accuracy is the principal compact endpoint; Macro-F1 is supporting evidence.
- Article ID, not article-ticker pair, is the clustering unit for uncertainty
  because one article may generate several labelled pairs.
- The equal-call and near-cost self-consistency controls support a system-level
  comparison only. They do not isolate debate reasoning as the causal
  mechanism and do not establish a downstream forecasting gain.

## 4. Main paper content lock

The main manuscript retains five compact result blocks:

1. **Protocol and cohort table:** date ranges, outer folds, feature counts,
   label-date purge, architectures, frozen windows, seeds, and primary metric.
2. **Numerical ablation table:** five-model Full TA versus causal VMD result,
   with fold uncertainty and the absence of multiplicity-adjusted superiority.
3. **Multimodal table:** Observed-News versus Market-Only and shuffled-news
   controls for all five architectures, with news-only, lagged-news, and
   random-feature controls retained in the aggregate evidence; a visibly
   separate panel reports the intrinsic compute-matched LLM benchmark.
4. **Regime-SHAP table:** capacity-aware registered contrasts and model-wise
   mixed/null outcomes; one sentence discloses the LIME diagnostic failure.
5. **Forward/transfer table:** partial-2026 predictive robustness and frozen
   SET100 transfer, with their different evidence classes stated in the title
   or footnote.

The main figures are limited to:

- the five-pillar reliability pipeline;
- the point-in-time temporal design and information cutoffs; and
- one compact effect/uncertainty view across the five architectures.

A compact computational-cost paragraph may report model/runtime scale in the
main text. Complete per-cell runtime and API ledgers remain supplementary.

## 5. Supplement lock

The Supplement retains, rather than deletes:

- every fold, seed, window, and secondary-metric table;
- complete grouped-LIME fidelity, stability, agreement, and failed rows;
- economic proxy, transaction-cost, turnover, drawdown, and DSR sensitivity;
- source-deviation and correction ledgers;
- full runtime, token, API-cost, and checkpoint summaries;
- temporal moving-block-bootstrap tables;
- benchmark/sanity results, including persistence and simple direction
  controls; and
- data-integrity warnings and the non-reconstructive provenance manifest.

LIME and the economic proxy are Supplement-only because their evidential roles
are diagnostic and exploratory. This placement must be disclosed in the main
text and is not permission to suppress an unfavourable result.

## 6. Terminology lock

| Use | Do not use |
|---|---|
| reliability audit | state-of-the-art forecasting model |
| architecture-dependent mixed evidence | universal improvement |
| source-contingent partial-2026 forward evaluation | pristine/full-year/live holdout |
| frozen same-exchange SET100 transfer audit | independent external-market replication |
| intrinsic compute-matched sentiment benchmark | proof that debate improves forecasting |
| SHAP association/attribution under the fitted model | causal market driver |
| LIME diagnostic stress test with limited fidelity | LIME validation of SHAP |
| exploratory economic proxy | profitable or deployable trading strategy |
| executed 2019--2025 integrated extension | fully integrated live trading pipeline |

## 7. Paper-ready data and availability position

- Daily, weekly, and monthly source files were obtained from publicly accessible
  provider historical-data pages; access and reuse remain subject to the
  provider terms and are not represented as an open-data licence.
- Session dates use Asia/Bangkok and a conservative 17:00 information cutoff.
- Weekly and monthly inputs are shifted to the previous completed period before
  daily alignment.
- The analysis uses provider-published price-index levels, not total-return
  indices, and applies no additional corporate-action or dividend adjustment.
- Raw SET50/SET100 observations cannot be redistributed. Code, contracts,
  checksums, aggregate evidence, and split specifications are included in the
  clean public replication package.
- The SET100 benchmark is complete at 100/100 fits and is negative
  same-exchange transfer evidence. Any earlier paper-ready text saying that it
  is still pending is superseded.

Authoritative wording is maintained in
`outputs/market_data_governance_v1/paper_data_statements.md`.

## 8. Submission declarations

Do not insert a generic declaration that the selected journal does not ask for.
At submission, check the actual target journal's mandatory declarations and
answer each required field accurately and narrowly. This reporting lock does
not authorise a false response or omission in a required field.

## 9. Remaining gates

### Public-access and redistribution gate

The provider pages and applicable terms URLs are recorded in the governance
manifest. The paper may state that the pages were publicly accessible at the
recorded acquisition time, but must not describe the data as openly licensed.
Raw provider rows remain excluded from the clean replication bundle. No private
institutional-entitlement evidence is required or claimed.

### Writing and submission gates

- draft the manuscript around the five main points above;
- reconcile every reported number to its authoritative CSV/JSON artifact;
- update the related-work review for the selected journal and submission date;
- prepare journal-specific Data Availability, Code Availability, limitations,
  title page, cover letter, and required declarations; and
- perform final language, citation, cross-reference, and table-footnote checks.

No additional model search, Optuna study, paid LLM experiment, or
accuracy-directed rerun is authorised by this reporting lock.

## 10. Controlling evidence

- `test/strong_q2_claims_register_v1.md`
- `test/strong_q2_claims_register_v2.md`
- `test/strong_q2_claims_register_v3.md`
- `test/strong_q2_hardening_execution_log_v1.md`
- `test/reliability_hardening_execution_log_v2.md`
- `test/primary_estimand_and_confirmatory_protocol_v1.md`
- `test/reliability_extension_protocol_v1.md`
- `test/q2_evidence_consolidation_log.md`
- `test/point_in_time_v2_correction_log.md`
- `test/integrated_multimodal_execution_log_v1.md`
- `test/track_b_compute_matched_execution_log_v1.md`
- `test/track_c_dual_xai_execution_log.md`
- `test/track_d_q2_execution_log.md`
- `test/set100_same_exchange_robustness_execution_log_v1.md`
- `test/moving_block_bootstrap_audit_v1.md`
- `test/market_data_governance_v1.md`
- `PUBLIC_REPLICATION_PACKAGE.md`
