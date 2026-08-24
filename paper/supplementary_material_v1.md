# Supplementary material

## A. Reporting hierarchy and evidence boundaries

Balanced accuracy (BAcc) is the primary market-direction endpoint. Direction
accuracy and MCC are secondary directional endpoints; RMSE and MAE describe
level forecasts and cannot substitute for directional evidence. SHAP is a
main-text, model-conditional explanation analysis. LIME is a diagnostic stress
test, and the economic proxy is exploratory. The locked 2023 LLM experiment is
an intrinsic sentiment benchmark; its accuracy difference is not a market-
forecasting effect. The added Debate-Leader forecasting arm is reported as a
separate downstream comparison.

The historical SET50 experiments use four independent outer test years. Seeds
are repeated fits within a temporal fold and are averaged before fold-level
inference. Exact paired sign-flip tests over four years therefore have coarse
resolution. Ten-day circular moving-block bootstrap results are serial-
dependence sensitivities, not replacements for the registered fold-level
analysis. Holm correction is applied within each registered model/contrast/
metric family.

## B. Data governance and point-in-time contract

### Table S1. Market-data inventory and conventions

| Index/frequency | Rows | First label | Last label | Treatment |
|---|---:|---:|---:|---|
| SET50 daily | 3,403 | 2012-01-04 | 2025-12-22 | Session-date observations |
| SET50 weekly | 730 | 2012-01-01 | 2025-12-21 | Lag one completed provider period |
| SET50 monthly | 168 | 2012-01-01 | 2025-12-01 | Lag one completed provider period |
| SET100 daily | 3,549 | 2012-01-04 | 2026-08-03 | Session-date observations |
| SET100 weekly | 762 | 2012-01-01 | 2026-08-02 | Lag one completed provider period |
| SET100 monthly | 175 | 2012-02-01 | 2026-08-01 | Lag one completed provider period |

Date-only observations are interpreted in `Asia/Bangkok`. Features are treated
as available at 17:00 local time. The target is the close of the next observed
trading session. A supervised row enters training only if its label-observation
date is earlier than the first evaluation date. Weekly and monthly provider
labels are never interpreted as intraday timestamps and are shifted by one
completed period before forward filling.

SET50 and SET100 are provider-published price indices, not total-return
indices. No additional split, dividend, constituent, or corporate-action
adjustment is applied by the researchers. Two source OHLC-containment warnings
on 2025-12-11 are retained rather than silently altered. Reported percentage
changes otherwise agreed with close-to-close calculations within 0.011
percentage points. Forty of 42 governance checks passed without qualification;
the two retained warnings were noncritical.

The exact provider URLs, byte sizes, row counts, full SHA-256 values, local-file
time evidence, and source evidence are stored in
`outputs/market_data_governance_v1/`. Public accessibility is not represented
as an open-data licence. Raw or reconstructive provider rows are excluded from
the public replication package.

### Table S2. Outer point-in-time SET100 fold preparation

| Test year | Purged training rows | Test rows | Boundary labels purged |
|---:|---:|---:|---:|
| 2022 | 2,397 | 241 | 1 |
| 2023 | 2,638 | 243 | 1 |
| 2024 | 2,881 | 244 | 1 |
| 2025 | 3,125 | 234 | 1 |

The common SET50-date SET100 cohort contains 3,360 rows from 2012-03-01 through
2025-12-18, corresponding to 99.3789% of eligible SET50 reference dates. The
first 21 otherwise eligible SET50 dates were not backfilled because SET100
monthly data require one completed period.

## C. Numerical features, VMD, and training protocol

The numerical pool contains 116 technical-analysis variables and six causal
rolling-VMD variables. Every rolling transform is computed from information
available at or before date \(t\). VMD uses five modes; the highest-frequency
component is excluded and four retained modes produce the locked reconstructed
features. VMD and Full-TA arms use identical targets, dates, architecture,
window, optimizer, epoch budget, batch size, and seeds.

### Table S3. Frozen windows and numerical-treatment runtime

| Model | Window | Full-TA mean runtime (s) | VMD mean runtime (s) | BAcc delta, VMD − Full TA (pp) |
|---|---:|---:|---:|---:|
| LSTM | 5 | 7.835 | 7.709 | −0.335 |
| CNN | 20 | 6.547 | 6.570 | −0.299 |
| LSTM–CNN | 20 | 14.262 | 14.547 | +0.355 |
| LSTM–Attention | 10 | 14.206 | 15.549 | −0.495 |
| LSTM–CNN–Attention | 20 | 20.379 | 19.643 | −0.600 |

No numerical-treatment interval excluded zero. Windows were selected on
pre-2022 folds and then locked.

## D. News data, local NLP, and domain shift

The labelled Bilingual StockTBSA file spans 2018-01-03 through 2023-12-28 and
contains 10,295 articles, 15,949 article-ticker labels, and 12,706 valid
positive/neutral/negative pairs. Labels `exclude`, `not stock`, and `ambiguous`
are not converted to neutral. The primary local model is character TF–IDF plus
balanced logistic regression. Expanding evaluation trains through year
\(y-1\) and predicts year \(y\).

Unlabelled Kaohoon text from 2015–2017 contains 68,514 rows and is not mixed
into the labelled benchmark. No defensible homogeneous free source was found
for 2012–2014. The 2024–2025 official SET collection contains 69,824
deduplicated headlines; 4,619 article-symbol pairs pass point-in-time SET50
membership filtering. Models frozen at 2023 score those pairs. Because
publication time is unavailable, each article is mapped strictly to the next
trading day, including weekend news.

### Table S4. Expanding local NLP performance

| Task | Test year | Pairs | Accuracy | Macro-F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Sentiment | 2019 | 2,740 | 0.8124 | 0.7876 | 0.6929 |
| Sentiment | 2020 | 2,314 | 0.8155 | 0.7863 | 0.6898 |
| Sentiment | 2021 | 2,154 | 0.8343 | 0.7880 | 0.6830 |
| Sentiment | 2022 | 1,325 | 0.7985 | 0.7679 | 0.6370 |
| Sentiment | 2023 | 1,333 | 0.8282 | 0.7844 | 0.6966 |
| Relevance | 2023 | 1,811 | 0.8885 | 0.8591 | 0.7188 |

The complete expanding local experiment took 338.47 seconds. Official
2024–2025 headlines were substantially shorter than the labelled 2023 proxies
(approximately 65 versus 260 characters), and mean sentiment confidence fell
from 0.756 to approximately 0.495. This source shift limits interpretation of
forward news effects.

The locked daily block contains sentiment mean and standard deviation,
positive/negative/neutral ratios, article count, ticker-mention count, and a
news-availability flag. All eight fields are mandatory when news is included;
they are not selected using outer outcomes.

## E. Bull/Bear/Leader intrinsic benchmark

All roles use `gpt-5.6-terra` with distinct frozen prompts, structured outputs,
low reasoning effort, and response storage disabled. A stratified 60-pair 2022
sample was used once for prompt validation. The prompt was frozen before the
1,333-pair/738-article 2023 cohort was opened. The Leader receives Bull and Bear
analyses and emits positive/neutral/negative probabilities plus a scalar
sentiment score.

### Table S5. Locked intrinsic sentiment performance

| Method | Accuracy | Macro-F1 | Weighted-F1 | MCC |
|---|---:|---:|---:|---:|
| Local character TF–IDF | 0.8282 | 0.7844 | 0.8265 | 0.6966 |
| One LLM call | 0.6984 | 0.6178 | 0.6739 | 0.5484 |
| Three-call self-consistency | 0.7067 | — | — | — |
| Four-call self-consistency | 0.7059 | — | — | — |
| Bull/Bear/Leader | 0.7659 | 0.7025 | 0.7697 | 0.6190 |

Leader minus SC3 was +5.926 accuracy points (article-cluster 95% CI 3.491 to
8.487), and Leader minus SC4 was +6.002 points (95% CI 3.613 to 8.477); both
Holm-adjusted p-values were 0.000040. Leader minus the local classifier was
−6.227 points in the earlier paired comparison. These contrasts answer
different questions and are retained together.

### Table S6. Intrinsic LLM API accounting

| Component | Calls | Mean latency (s) | Input tokens | Output tokens | Cost (USD) |
|---|---:|---:|---:|---:|---:|
| Bear | 1,333 | 4.617 | 997,189 | 314,865 | 7.216 |
| Bull | 1,333 | 4.152 | 995,856 | 279,834 | 6.687 |
| Leader | 1,333 | 2.494 | 1,522,136 | 151,352 | 6.076 |
| One-call ablation | 1,333 | 2.433 | 1,050,509 | 134,264 | 4.640 |

The three-call self-consistency control cost USD 13.93, the four-call control
USD 18.58, and the full role system USD 19.98 under the compute-matched ledger.
The broader tracked intrinsic-development ledger was approximately USD 38.56.
All costs are reconstructed experimental accounting rather than a provider
billing invoice.

## F. Debate-Leader downstream SET50 extension

The Leader output is an intended downstream news source in the revised
multimodal pipeline. It must be aggregated by eligible information date and
mapped to the next trading session under the same 17:00 Bangkok cutoff. The
comparison must use identical target dates for Market-Only, Local-NLP, and
Debate-Leader arms.

### Table S7. Downstream Leader audit fields

| Field | Value |
|---|---|
| Result artifact | Pending author path |
| Test period and eligible rows | Pending |
| Architecture/window/seeds | Pending |
| Market-Only BAcc | Pending |
| Local-NLP BAcc | Pending |
| Debate-Leader BAcc | Pending |
| Leader − Market effect and paired interval | Pending |
| Multiplicity-adjusted p-value | Pending |
| Coverage/missing-news handling | Pending |
| API model, prompt hash, cost, and latency | Pending |

These fields are deliberately not populated from the Local-NLP run or the
intrinsic 2023 comparison. Reconciliation requirements are recorded in
`test/leader_downstream_manuscript_addendum_v1.md`.

## G. Causal daily market regimes

The selected router uses six risk-adjusted return horizons (1, 3, 5, 10, 20,
and 60 days), ADX directional strength, and a span-3 EWMA. Horizon weights are
0.05, 0.10, 0.15, 0.20, 0.25, and 0.25. The Sideway deadband is the training-
only 35th percentile of the absolute smoothed score. This choice was screened
on 2012–2019 development and 2020–2021 validation data rather than on
2022–2025 forecasting accuracy.

### Table S8. Outer regime prevalence

| Year | Bull | Sideway | Bear |
|---:|---:|---:|---:|
| 2022 | 73 (30.29%) | 105 (43.57%) | 63 (26.14%) |
| 2023 | 40 (16.46%) | 84 (34.57%) | 119 (48.97%) |
| 2024 | 81 (33.20%) | 104 (42.62%) | 59 (24.18%) |
| 2025 | 74 (31.62%) | 58 (24.79%) | 102 (43.59%) |
| Pooled | 268 (27.86%) | 351 (36.49%) | 343 (35.65%) |

Training-only thresholds were 0.065984, 0.064101, 0.064143, and 0.063138 for
the four expanding folds (coefficient of variation 1.85%). Prefix-causality
recomputation produced zero feature differences, zero membership differences,
and zero label mismatches. Quantile 0.30/0.40 and EWMA span 2/5 sensitivities
retained all regimes and passed all four training semantic gates.

## H. SHAP and LIME explanation reliability

SHAP rankings are computed using training data only within each fold and model.
The frozen regime-specific feature counts are Bull=30, Sideway=122, and
Bear=80. Hard causal routing is primary. The primary BAcc contrasts are reported
in the main text; no model survived Holm correction.

LIME was repeated 120 times per model-regime cell. A repeat was considered
reliable only when the local surrogate passed its fidelity criterion. Across
1,800 repeats, 71.83% were low fidelity. Cell-specific low-fidelity fractions
ranged from 65.83% to 77.50%. Agreement measures are therefore summarized only
among reliable repeats and remain diagnostic rather than evidence that LIME
validates SHAP.

## I. Partial-2026 and SET100 robustness

The 2026 test is source-contingent and partial, with 138 prediction rows in the
classification summary (137 in the trading proxy after return alignment). It
is not a full-year test. Several objectives collapsed to an almost constant
majority-side prediction, yielding 58.70% direction accuracy but 50.00% BAcc
and MCC 0. The best partial-2026 BAcc was 54.42% for direct LSTM; this is a
descriptive source-contingent result.

Frozen SET100 transfer reduced mean BAcc for every architecture by 0.955 to
2.166 points relative to its common-cohort SET50 comparator. No Holm-adjusted
comparison was significant. SET100 is a same-exchange breadth check, not an
independent-market replication.

## J. Exploratory economic proxy

The economic analysis is intentionally non-headline. It uses 137 aligned
partial-2026 rows, validation-selected threshold 0.5, long-flat and long-short
rules, and 10-bp transaction cost. The largest net cumulative return was 8.55%
for direct LSTM–CNN long-flat, with annualized Sharpe 1.75, maximum drawdown
−3.54%, break-even cost 27.91 bp, and deflated-Sharpe probability 0.441. Because
that probability is below 0.5 and the exercise is short, source-contingent, and
selection-sensitive, it is reported only as an exploratory audit.

## K. Runtime, verification, and artifact availability

The integrated Local-NLP multimodal run completed 800 fits across 100
model/fold/seed cells. Median cell wall times were 31.53 s (LSTM), 33.98 s
(CNN), 58.55 s (LSTM–CNN), 56.85 s (LSTM–Attention), and 80.27 s
(LSTM–CNN–Attention). One retained LSTM–CNN–Attention fit took 4,447.50 s; its
finite aligned predictions passed integrity checks, so it was not removed or
rerun.

The clean public replication bundle contains code, tests, protocols, aggregate
tables, schemas, hashes, and non-reconstructive integrity evidence. It excludes
raw market rows, raw news text, row-level predictions capable of reconstructing
provider data, private LLM responses, checkpoints, credentials, and secrets.
The manuscript tables are indexed with SHA-256 hashes in
`outputs/manuscript_tables_v1/table_index.csv`.
