# Strong-Q2 paper claims register v1

Frozen: `2026-08-03T15:53:07Z`  
Target population: the Stock Exchange of Thailand (SET) equity-index setting  
Primary index: SET50  
Within-market transfer index: SET100  
Central framing: reliability audit, not state-of-the-art forecasting

## 1. Central research claim

The paper evaluates whether apparent improvements from denoising, news
sentiment, regime conditioning, neural explanation, and additional model
complexity remain credible under point-in-time feature availability,
label-date purging, repeated seeds, temporal evaluation, multiple-testing
control, forward stress, and within-SET transfer.

The target of inference is deliberately limited to the SET setting. SET100 is
a same-exchange index-breadth transfer audit; it is not an independent
external-market replication.

## 2. Evidence tiers

| Tier | Meaning | Permitted role in paper |
|---|---|---|
| Primary | Frozen endpoint and comparison with the strongest available temporal control | Abstract, contribution list, conclusion |
| Secondary | Predefined supporting metric or ablation | Main results with multiplicity/uncertainty |
| Robustness | Transfer, sensitivity, capacity, or alternative-protocol check | Main results or compact robustness table |
| Diagnostic | Method-quality or failure-mode audit | Short main-text disclosure; full Supplement |
| Exploratory | Post-hoc or underpowered economic/forward analysis | Supplement or limitations; no confirmatory wording |

## 3. Claim register

| ID | Component | Tier | Allowed claim | Required evidence | Blocked claim | Current status |
|---|---|---|---|---|---|---|
| C1 | Point-in-time protocol | Primary | Feature and label availability were controlled at every temporal boundary | Fold contracts, label-date purge, train-only transforms, hashes | The study is leakage-free in every imaginable sense | Supported |
| C2 | Five-model comparison | Primary | The same five registered architectures were retained across the framework | Model/window registry, five seeds, complete-cell audits | The best-looking architecture was universally superior | Supported |
| C3 | Causal VMD | Secondary | VMD effects were architecture- and metric-dependent | Corrected Track A paired ablation and fold uncertainty | VMD universally improves forecasting or removes noise | Supported as mixed evidence |
| C4 | Predicted news fusion | Secondary/robustness | News effects depended on model, comparator, and period | Corrected five-model Track B and integrated 2 x 2 contrasts | News improves all models or all regimes | Supported as mixed evidence |
| C5 | LLM Leader versus one-call single | Robustness, budget-asymmetric | The Leader system outperformed one Terra call on the locked 2023 labelled cohort | Article-cluster comparison, exact checkpoints, cost/runtime | Debate structure alone caused the gain | Supported, but incomplete control |
| C6 | LLM compute-matched control | Robustness | To be determined only after the frozen equal-call and near-cost self-consistency control | Locked cohort, repeated single calls, cluster inference, tokens/cost/runtime | Leader is structurally superior before the control is run | Pending new experiment |
| C7 | Regime-SHAP | Robustness/post-hoc | Response to regime routing and SHAP-selected subsets differed by architecture and comparator | Capacity-aware seven-arm Track C evaluation and Holm families | SHAP or regime routing caused a universal gain | Supported as mixed evidence |
| C8 | SHAP explanations | Diagnostic/robustness | SHAP was the sole feature-ranking explainer and was subjected to stability/deletion/sanity checks | Frozen selector, deletion tests, trained/random/permuted controls | SHAP identifies causal market drivers | Supported with limitations |
| C9 | Grouped LIME | Diagnostic only | LIME exposed limitations of local surrogate fidelity and explanation agreement | Fidelity audit retaining every failed row | LIME validates SHAP or supplies reliable feature effects | Main-text claim blocked; Supplement only |
| C10 | Partial-2026 forward test | Robustness/exploratory | The source-contingent forward test exposed weak discrimination and collapse | Frozen model protocol, source deviation log, predictive/calibration metrics | Pristine confirmatory holdout or live-market validation | Supported with source limitation |
| C11 | Economic proxy | Exploratory only | A cost-sensitive proxy was examined and did not establish deployable profitability | Predefined 10-bps table, turnover/drawdown, DSR sensitivity | Profitable or market-ready trading strategy | Main-text headline blocked; Supplement only |
| C12 | SET100 transfer | Robustness | The frozen numeric pipeline did not strengthen on the broader SET100 index | 100 fits, exact date pairing, seed-averaged folds, Holm correction | Independent external-market validation or SET100 improvement | Supported as negative transfer evidence |
| C13 | Framework conclusion | Primary | A transparent audit can distinguish fragile gains, null results, method failures, and transfer limits | Complete evidence matrix and correction history | The framework guarantees accurate forecasts | Supported |

## 4. Main-text and Supplement boundary

Main text retains:

1. point-in-time protocol and five-model registry;
2. compact VMD ablation;
3. corrected news/integrated comparison;
4. capacity-aware regime-SHAP result;
5. compact forward robustness result;
6. SET50-to-SET100 within-market transfer; and
7. the LLM comparison only after its compute-matched limitation is resolved or
   explicitly retained as budget-asymmetric.

Supplement contains:

- all window/seed/fold rows;
- complete LIME fidelity, stability, and agreement diagnostics;
- economic-proxy and DSR tables;
- failed gates, collapse cases, source deviations, and correction history;
- expanded runtime and cost ledgers; and
- temporal bootstrap sensitivity tables.

LIME and economic evidence remain in the audit trail. Moving them out of the
headline is evidence tiering, not deletion or selective suppression.

## 5. Statistical rules

- Balanced accuracy is the primary directional metric for market models.
- Seeds quantify training variability and are averaged before temporal
  inference; they are not independent market samples.
- Outer years are the primary temporal units for the five-model experiments.
- Article ID is the primary inference unit for intrinsic news sentiment because
  one article can produce several ticker-labelled pairs.
- Exact fold sign-flip and registered Holm families remain authoritative.
- Moving-block bootstrap is a temporal sensitivity analysis and must not be
  described as a device for manufacturing significance.
- No result may be promoted to a higher evidence tier because its observed
  p-value or effect direction is attractive.

## 6. Updating this register

Every later result must append one of: `supported`, `not supported`,
`inconclusive`, `diagnostic failure`, or `protocol deviation`. Existing rows
must not be silently rewritten after viewing a new result. A revised version
must retain this file and state the reason, timestamp, affected claims, and
source artifacts.
