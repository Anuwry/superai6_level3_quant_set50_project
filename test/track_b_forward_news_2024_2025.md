# Track B Forward News Extension Protocol (2024-2025)

Status: protocol locked before forward sentiment scoring and before the four-fold
fusion ablation.

Protocol lock date: 2026-07-30 (Asia/Bangkok)

## Purpose

Extend the Track B daily news features through 2025 so the final ablation can use
the same four outer walk-forward folds as Track A:

| Fold | Training years | Test year |
|---|---:|---:|
| 1 | 2019-2021 | 2022 |
| 2 | 2019-2022 | 2023 |
| 3 | 2019-2023 | 2024 |
| 4 | 2019-2024 | 2025 |

The comparison within every fold is paired:
`technical + VMD` versus `technical + VMD + news`.

## Locked sources

Forward news consists of official SET company-news headline metadata:

- 2024: `data-raw/track_b/SET_company_news_2024_2025/set_company_news_2024_th.csv`
- 2025: `data-raw/track_b/SET_company_news_2024_2025/set_company_news_2025_th.csv`
- source page: <https://www.set.or.th/th/market/news-and-alert/news>
- source API: <https://www.set.or.th/api/cms/v1/news/set>
- immutable retrieval metadata and SHA-256 values:
  `data-raw/track_b/SET_company_news_2024_2025/manifest.json`

The source contains headlines and metadata, not full article bodies and not
sentiment labels.

Point-in-time SET50 membership is reconstructed from official SET constituent
documents:

- H1 2024:
  <https://media.set.or.th/set/Documents/2023/Dec/SET50_100_H1_2024.pdf>
- H2 2024:
  <https://media.set.or.th/set/Documents/2024/Jun/SET50_100_H2_2024.pdf>
- H1 2025:
  <https://media.set.or.th/set/Documents/2024/Dec/SET50_100_H1_2025.pdf>
- revised H1 2025:
  <https://media.set.or.th/set/Documents/2025/Feb/SET50_100_H1_2025_revise.pdf>
- H2 2025:
  <https://media.set.or.th/set/Documents/2025/Jun/SET50_100_H2_2025.pdf>

The 2025 mid-cycle transition is dated using the SET announcements:

- GULF inclusion, effective 2025-04-02:
  <https://www.set.or.th/en/market/news-and-alert/newsdetails?id=95454300&symbol=SET>
- VGI inclusion, effective 2025-04-02:
  <https://www.set.or.th/en/market/news-and-alert/newsdetails?id=95454400&symbol=VGI>
- INTUCH exclusion, effective 2025-04-02:
  <https://www.set.or.th/en/market/news-and-alert/newsdetails?id=95455900&symbol=SET>
- GULF-to-GULFI transition from 2025-03-21:
  <https://www.set.or.th/en/market/news-and-alert/newsdetails?id=94908600&symbol=GULFI>

## Locked population rule

One inference item is one official `set_news_id` and `symbol` pair. A forward item
is eligible only when its symbol belongs to SET50 on its Bangkok publication
date. Membership intervals are inclusive and contain exactly 50 symbols.

The 2025 H1 membership has three point-in-time versions:

1. 2025-01-01 through 2025-03-20: original H1 list with `GULF` and `INTUCH`.
2. 2025-03-21 through 2025-04-01: ticker transition from `GULF` to `GULFI`.
3. 2025-04-02 through 2025-06-30: `GULF` and reserve constituent `VGI`
   replace `GULFI` and `INTUCH`.

This rule prevents filtering 2024-2025 news with a present-day constituent list.

## Locked text contract

StockTBSA article text usually starts with a duplicated headline. The local
models therefore train on a deterministic headline proxy:

1. normalize whitespace;
2. when a leading sequence of at least five tokens is immediately repeated, use
   the first sequence;
3. otherwise use the first 24 tokens.

Official SET inference uses the normalized headline. This reduces, but does not
eliminate, the source-domain shift between StockTBSA and official SET news.

## Locked inference protocol

- Local relevance model: character TF-IDF plus class-balanced logistic
  regression.
- Relevance threshold: 0.50.
- Local sentiment model: character TF-IDF plus class-balanced multinomial
  logistic regression.
- Sentiment score: `P(positive) - P(negative)`.
- 2019-2023 labelled years: expanding training using only earlier labelled years.
- 2024 and 2025: one frozen fit using all labelled 2018-2023 data.
- The 2025 model is not updated with 2024 predictions.
- No pseudo-label training, no Optuna tuning, and no LLM labels are introduced.

## Locked temporal alignment

News published on calendar date `t` is assigned strictly to the next SET trading
date after `t`. Same-day assignment is prohibited, including when the publication
occurs before market close. This conservative rule is identical to the completed
Track B fusion protocol and is tested for temporal leakage.

## Outputs to retain for the paper

- official membership PDFs, SHA-256 values, and interval table;
- counts before and after membership and relevance filtering;
- yearly predicted class distribution and confidence;
- headline-length and vectorizer-coverage domain-shift audit;
- local fit and prediction runtime for relevance and sentiment;
- item-level predictions for reproducibility;
- complete 2019-2025 daily feature calendar;
- package versions, random seed, input checksums, start/end timestamps, and total
  runtime;
- four-fold paired fusion metrics and model runtime.

## Interpretation constraint

The 2024-2025 source is headline-only and has no human sentiment labels. Its
predictions are suitable as frozen out-of-sample features, but they are not an
intrinsic accuracy benchmark. The paper must report the source-domain shift and
evaluate their usefulness through the downstream paired ablation.

## Completed execution

Execution completed: 2026-07-30 (Asia/Bangkok)

The membership parser visually and programmatically verified five official SET
PDFs. Each document contained 50 unique SET50 symbols. The final interval
artifact contains 300 rows across six non-overlapping versions, including the
GULF-to-GULFI ticker transition and the GULF/VGI mid-cycle replacement.

Forward filtering and inference counts:

| Year | Raw official headlines | Point-in-time SET50 pairs | Relevance-selected pairs | Trading days with news |
|---|---:|---:|---:|---:|
| 2024 | 32,966 | 2,099 | 1,223 | 225 |
| 2025 | 36,858 | 2,520 | 1,569 | 224 |
| **Total** | **69,824** | **4,619** | **2,792** | - |

The complete daily artifact has 1,690 trading dates from 2019-01-02 through
2025-12-18. The end date is determined by the available Track A market fold,
not by the final news publication date.

The complete forward-feature pipeline runtime was 126.19 seconds on CPU. Across
the six expanding/frozen fits, relevance fit and prediction time summed to
41.44 and 7.65 seconds, while sentiment fit and prediction time summed to
46.63 and 7.86 seconds.

## Headline-proxy intrinsic audit

The headline-proxy local models were evaluated only on labelled 2019-2023
pairs. These results measure the local model under the matched headline-proxy
contract; they do not assign intrinsic accuracy to 2024-2025.

| Task | Test year | Test pairs | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Relevance | 2019 | 3,142 | 0.8393 | 0.7166 |
| Sentiment | 2019 | 2,740 | 0.8073 | 0.7900 |
| Relevance | 2020 | 3,004 | 0.8266 | 0.7659 |
| Sentiment | 2020 | 2,314 | 0.7956 | 0.7551 |
| Relevance | 2021 | 2,854 | 0.8395 | 0.7977 |
| Sentiment | 2021 | 2,154 | 0.8254 | 0.7702 |
| Relevance | 2022 | 1,787 | 0.8243 | 0.7668 |
| Sentiment | 2022 | 1,325 | 0.8136 | 0.7845 |
| Relevance | 2023 | 1,811 | 0.8741 | 0.8427 |
| Sentiment | 2023 | 1,333 | 0.8402 | 0.8308 |

The exact values are retained in
`outputs/track_b/forward_news/headline_proxy_intrinsic_metrics.csv`.

## Domain-shift audit

The forward source shift is material:

| Year | Mean headline characters | Relevant rate | Mean sentiment confidence | Mean sentiment char n-grams |
|---|---:|---:|---:|---:|
| 2023 | 259.95 | 0.7117 | 0.7557 | 468.46 |
| 2024 | 66.40 | 0.5827 | 0.4944 | 200.71 |
| 2025 | 64.80 | 0.6226 | 0.4975 | 200.71 |

All forward rows have at least one known character n-gram, so the model does
not receive empty vectors. However, the lower n-gram count and confidence show
that vocabulary coverage alone does not remove the source/text-length shift.
This result was recorded without changing the locked threshold or retraining
on predictions.

The historical and forward periods also differ in sampling population:
StockTBSA supplies labelled article-ticker pairs in 2019-2023, while the
official forward source is explicitly restricted to point-in-time SET50
constituents. The four-fold report therefore includes separate
`labelled_validation` and `frozen_forward` period summaries and must not imply a
homogeneous news source across all years.

## Four-fold downstream ablation

The final run completed 200 neural fits and 100 one-to-one paired comparisons:
five models, two feature arms, four outer folds, and five seeds. There were no
missing pairs, duplicate configurations, or non-finite metrics.

| Model | W | Technical RMSE | +News RMSE | RMSE delta | Technical DA | +News DA | DA delta (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 5 | 17.5560 | 17.2507 | -0.3053 | 0.5026 | 0.5236 | +2.0980 |
| CNN | 5 | 19.2466 | 18.6915 | -0.5551 | 0.5127 | 0.5152 | +0.2480 |
| LSTM-CNN | 20 | 30.3844 | 33.0355 | +2.6511 | 0.5278 | 0.5148 | -1.2951 |
| LSTM-Attention | 20 | 45.3184 | 38.4049 | -6.9136 | 0.5080 | 0.5006 | -0.7377 |
| LSTM-CNN-Attention | 20 | 43.0175 | 41.5679 | -1.4495 | 0.5212 | 0.5146 | -0.6601 |

Negative RMSE delta is better; positive direction-accuracy delta is better.
No model improves both metrics strongly and uniformly across all four years.
LSTM has the largest overall direction gain, while LSTM-Attention has the
largest overall RMSE reduction.

For the frozen forward years only:

| Model | 2024-2025 RMSE delta | 2024-2025 DA delta (pp) |
|---|---:|---:|
| LSTM | +1.6309 | +1.0116 |
| CNN | -1.7422 | +0.2956 |
| LSTM-CNN | +1.1882 | -1.2236 |
| LSTM-Attention | -1.9330 | -1.0638 |
| LSTM-CNN-Attention | -4.7224 | -0.6165 |

CNN is the only model with improvements in both forward-period aggregate
metrics, but its direction gain is small. This is evidence of a mixed,
model-dependent news contribution rather than a universal improvement.

The statistical unit is the outer year, after averaging seeds. There are only
four independent fold-level effects per model. All RMSE exact sign-flip
p-values are at least 0.125 and all 95% confidence intervals cross zero.
Accordingly, the paper may report paired effect sizes and consistency, but not
claim a statistically established overall improvement.

The 200 forecasting fits required 1,710.39 seconds of measured model
build/training/inference time and 1,983.46 seconds command wall time on native
Windows CPU.

## Engineering verification

Final verification was completed on 2026-07-30:

- all 68 Track A/Track B regression tests passed;
- the two newly added pipeline modules reached 83% combined statement
  coverage (`track_b_forward_news.py` 81% and
  `track_b_four_fold_ablation.py` 91%);
- Ruff reported no lint violations in the changed source and test files;
- the official membership PDF parser was tested against a retained SET PDF;
- synthetic integration tests cover the six-version membership transition,
  frozen 2024-2025 inference contract, next-trading-day aggregation, artifact
  metadata, and temporary fusion configuration restoration;
- the final artifact audit found 200 unique model-fit configurations, 100
  complete paired comparisons, zero duplicate configurations, and zero
  non-finite reported metrics.

The complete repository-wide test collection also includes optional
visualization and Optuna suites. Those two unrelated modules were not collected
in the lightweight Track B environment because `matplotlib` and `optuna` are
not installed there; this does not affect the 68 relevant Track A/Track B
tests above.

## Completed artifacts

- `data-raw/track_b/SET50_membership_2024_2025/manifest.json`
- `data-raw/track_b/SET50_membership_2024_2025/set50_membership_intervals.csv`
- `outputs/track_b/forward_news/run_metadata.json`
- `outputs/track_b/forward_news/domain_shift_audit.csv`
- `outputs/track_b/forward_news/runtime_by_fit.csv`
- `outputs/track_b/forward_news/daily_news_features_2019_2025.csv`
- `outputs/track_b/four_fold_ablation/paper_track_b_four_fold_table.csv`
- `outputs/track_b/four_fold_ablation/paired_summary_by_year.csv`
- `outputs/track_b/four_fold_ablation/paired_summary_by_source_period.csv`
- `outputs/track_b/four_fold_ablation/run_metadata.json`
