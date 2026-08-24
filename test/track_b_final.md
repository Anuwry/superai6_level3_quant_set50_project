# Track B Final Experiment Log

Run completed: 2026-07-30 (Asia/Bangkok)

## Final status

Track B uses the labelled Bilingual StockTBSA common period rather than
claiming homogeneous news coverage from 2012 through 2025. The completed
experiment has four separately reported components:

1. an expanding-year local relevance and sentiment baseline;
2. a locked 2023 Terra single-pass versus Bull/Bear debate-and-Leader
   benchmark;
3. a frozen local headline extension using point-in-time SET50 news in
   2024-2025;
4. a paired technical-only versus technical-plus-predicted-news fusion
   experiment on four outer test years, 2022-2025.

The five fusion models and their sequence windows are inherited from Track A.
No window, threshold, or hyperparameter was selected using the 2022-2025
fusion test results.

## Data and reporting boundary

Primary labelled dataset:

- file: `data-raw/track_b/Bilingual_StockTBSA/Thai_Financial_TBSA_dataset_Updated.json`
- SHA-256:
  `1e87d6780e210a2d4f8da44680789d79fa07f7644afa791983592d0cdecd83ac`
- period: 2018-01-03 through 2023-12-28
- articles: 10,295
- article-ticker labels: 15,949
- valid positive/neutral/negative pairs: 12,706
- 2023 locked test pairs: 1,333

Labels `exclude`, `not stock`, and `ambiguous` are not converted to neutral.
The relevance task uses polarity labels as relevant, `exclude` and `not stock`
as irrelevant, and omits `ambiguous`.

Additional local data boundaries:

- 2015-2017 CMDF-VISTEC Kaohoon: 68,514 full-text rows without sentiment or
  ticker labels; these rows are not mixed into the main benchmark;
- 2024-2025 official SET: 69,824 deduplicated headlines without sentiment
  labels; 4,619 article-symbol pairs pass the completed point-in-time SET50
  membership filter and are scored by models frozen at 2023;
- 2012-2014: no defensible free, consistently dated source was found.

Therefore, Track A remains the 2012-2025 numerical benchmark, while Track B
claims intrinsic sentiment accuracy only for the labelled 2018-2023 period.
The 2024-2025 predictions are evaluated only as downstream forecasting
features, not as an intrinsic sentiment benchmark.

## Leakage controls

The local classifier uses one row per article-ticker pair and prepends the
target ticker to the article text. It is a character TF-IDF plus balanced
logistic-regression pipeline, so it does not require Thai word segmentation.

Every annual prediction is out of sample:

- train 2018, predict 2019;
- train 2018-2019, predict 2020;
- train 2018-2020, predict 2021;
- train 2018-2021, predict 2022;
- train 2018-2022, predict 2023.

For the forward extension, article text is converted to a deterministic
headline proxy before fitting. The same local relevance and sentiment
classifiers are then fitted once on all labelled 2018-2023 data and frozen for
both 2024 and 2025. The 2025 model is not updated with 2024 predictions. No
pseudo-label retraining or Optuna tuning is used.

Official headlines are eligible only if their symbol belongs to SET50 on the
Bangkok publication date. Six non-overlapping membership versions cover
2024-2025, including the March-April 2025 GULF/GULFI/INTUCH/VGI transition.

For fusion, article sentiment is assigned to the trading day strictly after
the article date. This conservative mapping is required because publication
time is unavailable. Weekend news is assigned to the next trading day.

The eight locked daily features are:

```text
news_sentiment_mean
news_sentiment_std
positive_ratio
negative_ratio
neutral_ratio
article_count
ticker_mention_count
news_available
```

The fusion training period begins in 2019, the first year with out-of-sample
news predictions. Technical-only and technical-plus-news arms use identical
dates. MinMaxScaler is fitted separately on each fold's training rows only.

## Local expanding-year results

| Task | Test year | Test pairs | Accuracy | Macro-F1 | Weighted-F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Sentiment | 2019 | 2,740 | 0.8124 | 0.7876 | 0.8122 | 0.6929 |
| Sentiment | 2020 | 2,314 | 0.8155 | 0.7863 | 0.8149 | 0.6898 |
| Sentiment | 2021 | 2,154 | 0.8343 | 0.7880 | 0.8295 | 0.6830 |
| Sentiment | 2022 | 1,325 | 0.7985 | 0.7679 | 0.7968 | 0.6370 |
| Sentiment | 2023 | 1,333 | 0.8282 | 0.7844 | 0.8265 | 0.6966 |
| Relevance | 2023 | 1,811 | 0.8885 | 0.8591 | 0.8895 | 0.7188 |

The full expanding-year local experiment took 338.47 seconds, including all
five sentiment fits/predictions and all five relevance fits/predictions.

## Terra debate protocol

Model and API contract:

- model: `gpt-5.6-terra`;
- API: Responses API with Structured Outputs;
- reasoning effort: low;
- API response storage: disabled;
- prompt version: `track-b-terra-v1`;
- raw article text is not written to checkpoints; SHA-256 is stored instead;
- every API role is checkpointed independently for safe resume;
- the local credential file is ignored by Git and its value is never logged.

The budget-aware production design is:

1. local relevance filter;
2. Terra Bull worker and Terra Bear worker;
3. Terra Leader producing positive/neutral/negative probabilities and a
   sentiment score;
4. a Terra single-pass call retained only as the debate ablation.

The local relevance filter is evaluated separately. The locked LLM benchmark
uses only the dataset's valid polarity pairs so that sentiment accuracy is not
confounded with the relevance task.

### Prompt validation

A fixed class-stratified 60-pair sample from 2022 was used once for prompt
validation. The prompt was then frozen before opening the 2023 test set.

| Method | Accuracy | Macro-F1 | Weighted-F1 | MCC |
|---|---:|---:|---:|---:|
| Terra single | 0.7167 | 0.6900 | 0.6900 | 0.5998 |
| Terra debate Leader | 0.7333 | 0.7280 | 0.7280 | 0.6063 |

The validation run cost USD 1.124195.

### Locked 2023 intrinsic test

All methods below use the exact same 1,333 article-ticker pairs.

| Method | Accuracy | Macro-F1 | Weighted-F1 | MCC |
|---|---:|---:|---:|---:|
| Local character TF-IDF | 0.8282 | 0.7844 | 0.8265 | 0.6966 |
| Terra single | 0.6984 | 0.6178 | 0.6739 | 0.5484 |
| Terra debate Leader | 0.7659 | 0.7025 | 0.7697 | 0.6190 |

Against Terra single, debate improved accuracy by 6.7517 percentage points
(paired bootstrap 95% CI 4.5761 to 8.8522) and Macro-F1 by 0.0848
(95% CI 0.0630 to 0.1065). The exact McNemar p-value is
`9.6786e-10`; the Leader alone corrected 154 single-pass errors, while the
single pass alone corrected 64 Leader errors.

The local classifier remained stronger than the Leader: Leader-minus-local
accuracy was -6.2266 percentage points (95% CI -9.0023 to -3.3758) and
Macro-F1 was -0.0818 (95% CI -0.1209 to -0.0434). The exact McNemar p-value is
`3.0153e-05`. This negative result is retained rather than selecting only the
better-looking LLM comparison.

For probabilistic evaluation, Terra single had log loss 0.6444 and multiclass
Brier score 0.3970; the Leader improved these to 0.5933 and 0.3483.

### API cost and runtime

| Role | Calls | Mean latency (s) | Input tokens | Output tokens | Cost (USD) |
|---|---:|---:|---:|---:|---:|
| Bear | 1,333 | 4.6169 | 997,189 | 314,865 | 7.2159475 |
| Bull | 1,333 | 4.1515 | 995,856 | 279,834 | 6.6871500 |
| Leader | 1,333 | 2.4938 | 1,522,136 | 151,352 | 6.0756200 |
| Single ablation | 1,333 | 2.4330 | 1,050,509 | 134,264 | 4.6402325 |
| **Locked test total** | **5,332** | — | **4,565,690** | **880,315** | **24.6189500** |

The validation and three-item smoke runs bring total tracked development plus
locked-test API cost to approximately USD 25.8063. The mean per-item critical
path for the locked run was 7.4993 seconds; the sum of item critical paths was
9,996.63 seconds. Checkpoint completion spanned approximately 60.6 minutes
because calls were concurrent and the run included resumptions.

Checkpoint audit:

- 1,333 unique completed predictions;
- 5,332 successful role calls;
- no duplicate item-role calls or response IDs;
- no unresolved items or probability-range/sum violations;
- 21 historical structured-output errors affecting 20 items, all recovered by
  checkpoint resume;
- one non-blocking decision/probability-argmax disagreement was retained
  unchanged because the test was locked. The stated decision label is used for
  classification metrics; probabilities are used for probabilistic metrics.

## Fusion experiment

Fusion uses Full TA plus causal rolling VMD as the numerical feature base.
Locked Track A windows are LSTM=5, CNN=5, LSTM-CNN=20,
LSTM-Attention=20, and LSTM-CNN-Attention=20. Each arm is run on the four
outer test years 2022, 2023, 2024, and 2025 with seeds 42, 123, 456, 789, and
2025. This gives 200 neural fits and 100 one-to-one paired comparisons.

| Model | W | Technical RMSE | +News RMSE | RMSE delta | Technical DA | +News DA | DA delta (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 5 | 17.5560 | 17.2507 | -0.3053 | 0.5026 | 0.5236 | +2.0980 |
| CNN | 5 | 19.2466 | 18.6915 | -0.5551 | 0.5127 | 0.5152 | +0.2480 |
| LSTM-CNN | 20 | 30.3844 | 33.0355 | +2.6511 | 0.5278 | 0.5148 | -1.2951 |
| LSTM-Attention | 20 | 45.3184 | 38.4049 | -6.9136 | 0.5080 | 0.5006 | -0.7377 |
| LSTM-CNN-Attention | 20 | 43.0175 | 41.5679 | -1.4495 | 0.5212 | 0.5146 | -0.6601 |

`RMSE delta` is news minus technical, so a negative value is better.
`DA delta` is the direction-accuracy change in percentage points.

Across all four years, news improves RMSE for four models but improves
direction accuracy only for LSTM and CNN. LSTM has the largest direction gain;
LSTM-Attention has the largest RMSE reduction. LSTM-CNN worsens on both
metrics. Therefore, the fusion result is mixed rather than a universal gain.

The separate frozen-forward summary for 2024-2025 is:

| Model | Forward RMSE delta | Forward DA delta (pp) |
|---|---:|---:|
| LSTM | +1.6309 | +1.0116 |
| CNN | -1.7422 | +0.2956 |
| LSTM-CNN | +1.1882 | -1.2236 |
| LSTM-Attention | -1.9330 | -1.0638 |
| LSTM-CNN-Attention | -4.7224 | -0.6165 |

CNN is the only model that improves both aggregate metrics in the two frozen
forward years, although its direction gain is small. This source-period
interaction is reported because official 2024-2025 SET headlines are much
shorter than the labelled headline proxies: mean length is about 65 characters
instead of about 260 in 2023, and mean sentiment confidence falls from 0.756
to about 0.495.

Inference must also respect the small effective sample for statistical testing:
there are 20 paired seed-fold runs per model but only four independent outer
test folds. Significance tests first average seeds within each fold. All RMSE
exact sign-flip p-values are at least 0.125 and every RMSE 95% confidence
interval crosses zero. The paper may report the paired effect sizes, but it
must not claim statistically established improvement.

The 200 completed neural fits required 1,710.39 seconds of measured model
build/training/inference time and 1,983.46 seconds wall-clock command time on
CPU. The forward news feature pipeline separately required 126.19 seconds.

## Verification status

The final artifact audit found 200 unique fit configurations, 100 complete
technical-versus-news pairs, no duplicate configurations, and no non-finite
metrics. All 68 relevant Track A/Track B regression tests passed. The two new
forward-news/four-fold modules have 83% combined statement coverage, and Ruff
reported no violations in the changed Python source and tests.

The repository also contains optional visualization and Optuna tests that are
outside this locked Track B environment. Full repository collection requires
installing `matplotlib` and `optuna`; their absence did not affect the relevant
68-test regression run or any reported experiment.

## Runtime environment

- Python 3.12.10;
- pandas 3.0.1;
- scikit-learn 1.8.0;
- TensorFlow 2.21.0;
- native Windows TensorFlow CPU execution;
- deterministic TensorFlow operations and fixed seeds;
- model runtime is measured with `time.perf_counter` over model build,
  training, and test inference.

## Reproducibility commands

```powershell
py -3.12 -m pip install -r requirements-track-b.txt
py -3.12 -m models.track_b_experiment local
py -3.12 -m models.track_b_experiment report `
  --output-dir outputs\track_b\llm\locked_test_2023
py -3.12 -m models.track_b_analysis `
  --output-dir outputs\track_b\llm\locked_test_2023 `
  --expected-pairs 1333 `
  --bootstrap-iterations 5000 `
  --local-predictions outputs\track_b\local_baseline\sentiment_predictions_expanding.csv
py -3.12 -m models.track_b_forward_news
py -3.12 -m models.track_b_four_fold_ablation run
py -3.12 -m models.track_b_four_fold_ablation report
```

The LLM execution command is intentionally omitted from this paper-facing log
to avoid encouraging an accidental rerun of the already locked and billed test
set. The checkpoint directory contains the prompt manifest, schemas, role
calls, predictions, costs, runtimes, statistical comparisons, and integrity
audit.

## Paper interpretation

The defensible conclusions are:

1. Debate materially improves Terra over a single Terra call on the locked
   2023 set.
2. Debate does not beat the lower-cost local character TF-IDF classifier on
   this labelled dataset.
3. Frozen official SET headlines can be used in a leakage-controlled
   2024-2025 downstream test, but their shorter text and lower confidence must
   be reported as domain shift.
4. The four-fold fusion result is model- and metric-dependent. CNN is the only
   model that improves both aggregate metrics in the frozen forward period,
   and its direction gain is small.
5. No overall news improvement is statistically established with only four
   independent outer years.
6. Track B does not establish homogeneous 2012-2025 news coverage.
7. The LLM system is suitable as a bounded benchmark and qualitative
   reasoning layer; the local filter/classifier is the practical default for
   cost-sensitive repeated deployment.

## Primary artifacts

- `outputs/track_b/local_baseline/paper_local_intrinsic_table.csv`
- `outputs/track_b/llm/locked_test_2023/paper_track_b_intrinsic_comparison.csv`
- `outputs/track_b/llm/locked_test_2023/paper_llm_paired_comparison.csv`
- `outputs/track_b/llm/locked_test_2023/paper_local_leader_paired_comparison.csv`
- `outputs/track_b/llm/locked_test_2023/llm_checkpoint_audit.json`
- `data-raw/track_b/SET50_membership_2024_2025/manifest.json`
- `outputs/track_b/forward_news/domain_shift_audit.csv`
- `outputs/track_b/forward_news/daily_news_features_2019_2025.csv`
- `outputs/track_b/forward_news/run_metadata.json`
- `outputs/track_b/four_fold_ablation/paper_track_b_four_fold_table.csv`
- `outputs/track_b/four_fold_ablation/paired_deltas_by_seed_fold.csv`
- `outputs/track_b/four_fold_ablation/paired_summary_by_year.csv`
- `outputs/track_b/four_fold_ablation/paired_summary_by_source_period.csv`
- `outputs/track_b/four_fold_ablation/run_metadata.json`
