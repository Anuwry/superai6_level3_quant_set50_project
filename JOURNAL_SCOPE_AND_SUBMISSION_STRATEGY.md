# Journal scope and submission strategy

Assessment date: 4 August 2026  
Project: SET50/SET100 next-day direction forecasting

## 1. Readiness decision

The experimental evidence is sufficiently complete to start manuscript preparation. No additional model search, Optuna run, paid LLM call, or accuracy-oriented experiment is required for a well-matched Q2 submission.

The project should not be presented as a new state-of-the-art forecasting model. Its defensible contribution is a **point-in-time, leakage-audited reliability study of multimodal and regime-aware financial AI**, covering numerical denoising, news/LLM evidence, regime-dependent explainability, temporal robustness, and same-exchange transfer from SET50 to SET100.

No institutional market-data entitlement is claimed. The market series are
described as publicly accessible provider data obtained from historical-data
pages that offer a download option. Public accessibility is not described as
an open licence, and row-level provider data must not be included in the public
replication package. The source URLs, acquisition basis, hashes, timezone,
adjustment convention, and applicable provider terms are retained as evidence.

## 2. Recommended article scope

### Working title

> A Point-in-Time Reliability Audit of Multimodal and Regime-Aware Deep Learning for Next-Day SET Index Direction Forecasting

An alternative title that foregrounds the main finding is:

> When More Signals Do Not Mean More Skill: A Reliability Audit of Deep Learning for Emerging-Market Direction Forecasting

### Article type

Empirical methodological study / applied machine-learning reliability audit.

### Central research question

Under a leakage-controlled point-in-time protocol, do numerical denoising, news sentiment, LLM role-based deliberation, regime-aware feature selection, and explainability produce reliable and temporally transferable improvements in next-day SET index direction forecasting?

### Five manuscript pillars

The paper is organised around five reliability dimensions rather than five
disconnected experiments:

1. **Point-in-time data reliability:** feature availability, label-date purge,
   expanding temporal folds, frozen windows, repeated seeds, and audit trails.
2. **Numerical and denoising reliability:** Full TA versus causal rolling VMD
   across the five registered architectures.
3. **Multimodal and LLM reliability:** predicted-news fusion, intrinsic
   sentiment evaluation, and the compute-matched Bull/Bear/Leader control.
4. **Regime-aware explainability reliability:** daily Bull/Sideway/Bear
   routing, capacity controls, progressive SHAP, and diagnostic LIME stress.
5. **Forward and transfer reliability:** source-contingent partial-2026
   evaluation, frozen SET100 same-exchange transfer, temporal sensitivity, and
   release-package verification.

The five neural architectures are the common benchmark panel used across the
pillars; they are not five separate contributions.

### Claims that should lead the paper

1. A reproducible point-in-time evaluation protocol exposes performance inflation and failure modes that can be obscured by ordinary random splits or selective reporting.
2. VMD, news, and regime-aware SHAP provide architecture-dependent rather than universal benefits; no primary forecasting contrast survives multiplicity correction.
3. A compute-matched multi-role LLM system improves intrinsic sentiment classification over repeated identical-prompt inference, but that result must not be converted into a downstream forecasting claim.
4. Partial-2026 forward evaluation and frozen SET100 transfer show that apparent in-sample gains do not automatically transfer across time or index scope.
5. Negative and mixed results are the substantive audit finding, not an experimental failure.

### Results to de-emphasize or move to Supplementary Information

- Grouped LIME because local surrogate fidelity is weak for many rows.
- Economic/trading proxy because the best deflated-Sharpe probability is below 0.50 and it is exploratory.
- Full per-seed/per-fold tables, diagnostic failures, and secondary RMSE findings.
- Detailed prompt transcripts and LLM ledgers; retain the frozen protocol, aggregate results, and audit metadata in the main paper.

## 3. Journal shortlist

Quartiles vary by database, subject category, and release year. The labels below distinguish recent JCR/JIF and Scopus/CiteScore indications where available; the institution should verify the exact category in its subscribed JCR/Scopus source before submission.

| Priority | Journal | Current level indication | Scope fit | Relative chance for this manuscript | Cost model | Recommended use |
|---:|---|---|---|---|---|---|
| 1 | **Computational Economics** | 2025 JCR indication: Q2; Scopus category may appear Q1/Q2 | High if economic interpretation and SET market context lead | Medium | Hybrid | Best realistic Q2 route when APC budget is limited |
| 2 | **Journal of Forecasting** | Recent current indication: Q2; official page reports JIF 2.7 and 12% overall acceptance | Very high for temporal forecasting protocol; novelty bar is meaningful | Low-medium | Hybrid | Best forecasting-specific alternative |
| 3 | **Applied Artificial Intelligence** | Official 2024 JCR best quartile Q2; Scopus best quartile Q1 | Very high for evaluation/comparative applied-AI framing | Low; publisher reports 5% overall acceptance | Fully OA; APC/waiver must be checked | Best all-track scope fit, but costly and selective |
| 4 | **Intelligent Systems with Applications** | Recent public indication: JCR Q2 / Scopus Q1; official metrics JIF 4.3, CiteScore 8.3 | Very high: ML, NLP, multi-agent systems, and business/finance are explicit | Low-medium | Fully OA, listed APC USD 2,150 before taxes/discounts | Strong Q2/Q1-database option if funding is available |
| 5 | **Machine Learning with Applications** | Recent JCR/Scopus indication: Q1 | High only when framed as a general improvement to ML evaluation practice | Low / stretch | Fully OA, listed APC USD 2,460 before taxes/discounts | One-shot Q1 stretch; not the default target |
| 6 | **Computational Management Science** | Recent JCR indication Q3 and Scopus indication Q2 | Moderate; emphasize computational forecasting and decision reliability | Medium-high relative to the list | Hybrid | Sensible fallback if Q2 journals decline |

Official scope and metric pages consulted:

- [Computational Economics: aims and scope](https://link.springer.com/journal/10614/aims-and-scope)
- [Computational Economics: journal page and metrics](https://link.springer.com/journal/10614)
- [Journal of Forecasting: aims and scope](https://onlinelibrary.wiley.com/page/journal/1099131x/homepage/productinformation.html)
- [Journal of Forecasting: journal metrics](https://onlinelibrary.wiley.com/journal/1099131x)
- [Applied Artificial Intelligence: aims, metrics, and publishing model](https://www.tandfonline.com/journals/uaai20/about-this-journal)
- [Intelligent Systems with Applications: scope, metrics, and APC](https://www.sciencedirect.com/journal/intelligent-systems-with-applications/about/insights)
- [Machine Learning with Applications: author guide and scope](https://www.sciencedirect.com/journal/machine-learning-with-applications/publish/guide-for-authors)
- [Machine Learning with Applications: metrics and APC](https://www.sciencedirect.com/journal/machine-learning-with-applications/about/insights)
- [Computational Management Science: aims and scope](https://link.springer.com/journal/10287/aims-and-scope)

## 4. Recommended submission ladders

### Low-budget route

1. Computational Economics — target Q2.
2. Journal of Forecasting — target Q2.
3. Computational Management Science — Q3 JCR / Q2 Scopus fallback, subject to the institution's required database.

### APC-funded route

1. Applied Artificial Intelligence — Q2 JCR, Q1 Scopus best quartile.
2. Intelligent Systems with Applications — Q2/Q1 depending database/category.
3. Computational Economics — Q2 realistic fallback.

### Q1-stretch route

1. Machine Learning with Applications — submit once with a methodology/reliability narrative.
2. If rejected, do not chase another high-novelty Q1 AI journal with the unchanged manuscript; retarget immediately to Computational Economics or Applied Artificial Intelligence.

Expert Systems with Applications is not recommended as the first submission. Although its metrics are high, the present paper has no new algorithm and no consistently significant forecasting gain, creating substantial desk-rejection risk.

## 5. Honest quartile assessment

- **Q1:** possible only as a stretch submission; acceptance is unlikely without a new general method, independent external-market replication, or consistently stronger forecasting evidence.
- **Q2:** defensible and realistic when the journal explicitly values rigorous applied evaluation, forecasting methodology, reproducibility, and negative/mixed evidence.
- **Q3:** strong fallback. The experimental discipline and audit artifacts exceed what would normally justify targeting only Q4.

The manuscript itself does not possess a quartile; quartile is a property of the journal in a specified database, category, and year. The appropriate current description of the work is **a defensible Q2 candidate with a strong Q3 fallback**.

## 6. Work remaining before submission

1. Freeze the reliability-audit title, contribution list, primary endpoints, and confirmatory/exploratory labels.
2. Freeze the public-access provenance statement and retain dated provider-page
   and terms-of-use evidence; do not redistribute row-level provider data.
3. Write the manuscript around one narrative rather than four disconnected tracks.
4. Keep four main result tables: protocol/data; forecasting ablations; intrinsic LLM control; forward/transfer robustness.
5. Move LIME, trading proxy, full seed/fold outputs, and detailed failure diagnostics to Supplementary Information.
6. Write explicit Data Availability, Code Availability, limitations, and
   non-redistribution statements. Check all journal-mandated submission
   declarations at submission time and answer any required declaration
   accurately; do not add generic declarations that the selected journal does
   not request.
7. Tailor the abstract, cover letter, keywords, word count, and reference style to the first-choice journal.

## 7. Final recommendation

If publication cost is a real constraint, prepare the paper first for **Computational Economics**. If the institution can cover an OA charge and the priority is the best match to all four audit dimensions, prepare it for **Applied Artificial Intelligence**. A single Q1 attempt at **Machine Learning with Applications** is reasonable only if the team accepts a high rejection risk and does not delay the Q2 route after rejection.

## Evidence update: 4 August 2026

The journal strategy above remains valid after the pre-frozen multimodal
falsification run. The new 100-cell/400-fit experiment found no Holm-adjusted
BAcc benefit for observed news over market-only or shuffled-news controls.
This weakens any accuracy-improvement pitch but strengthens the reliability-
audit narrative by closing the missing information-content-control objection.
The current position is therefore a **reasonably solid, fit-dependent Q2
candidate with a strong Q3 fallback**, not a Q1-level forecasting advance.

For submission, use the generated tables under
`outputs/manuscript_tables_v1/`, lead with reliability and point-in-time
evaluation, place LIME/economics in the Supplement, and keep the intrinsic LLM
panel visibly separate from the downstream Local-NLP feature source. Do not
describe the future 252-session prospective protocol as a completed result.
