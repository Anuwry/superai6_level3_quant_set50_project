# Track B One-Day Experiment Contract

Status: proposed fast-track design

## Scope decision

Keep the completed Track A numerical experiment at 2012-2025. Evaluate Track B
and Track A+B fusion on the downloadable VISTEC Bilingual Stock TBSA common
period only.

- Text data: 2018-2023
- Sentiment-feature generation: expanding-year, out-of-sample predictions
- Fusion evaluation years: 2022 and 2023
- Numerical comparator: rerun on exactly the same dates and folds

This avoids representing unavailable 2012-2017 and 2024-2025 news as neutral
sentiment.

## Track B intrinsic evaluation

Data split follows the dataset card:

- train: 2018-2020
- validation: 2021
- test: 2022-2023

Primary fast baseline:

- one instance per article-ticker pair
- prepend the target ticker to article text
- character TF-IDF, which does not require Thai word segmentation
- logistic regression classifier
- classes: positive, neutral, negative
- remove `exclude`, `not stock`, and `ambiguous`

Metrics:

- macro-F1
- weighted-F1
- accuracy
- class precision and recall
- confusion matrix
- training and inference runtime

The multi-agent LLM path is a bounded secondary experiment on a fixed,
class-stratified test subset. It is not run over all 10,295 articles on the
current CPU-only machine.

## Leakage-free daily features

Map labels to numeric values:

- positive = +1
- neutral = 0
- negative = -1

Generate expanding-year predictions:

- train 2018, predict 2019
- train 2018-2019, predict 2020
- train 2018-2020, predict 2021
- train 2018-2021, predict 2022
- train 2018-2022, predict 2023

For every news date, assign sentiment to the next trading day strictly after the
article date. This conservative rule is required because the dataset has no
publication time. Weekend news is assigned to the following trading day.

Daily feature contract:

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

On days with no news:

- counts and ratios = 0
- sentiment mean/std = 0
- `news_available = 0`

The availability flag distinguishes missing news from genuinely neutral news.

## Fusion experiment

Use the five locked Track A model families and their already selected sequence
windows. Do not rerun Optuna or select windows on 2022-2023.

Compare on identical rows:

1. Numerical-only
2. Numerical + predicted news sentiment

Optional appendix upper bound:

3. Numerical + gold-label sentiment

Gold sentiment must be labelled as an oracle upper bound and cannot be reported
as the deployable Track B result.

Use five seeds and paired fold-seed comparisons if runtime permits. Existing
Track A measurements indicate that the five neural model families are fast
enough for this restricted two-fold experiment on the current CPU.

## Paper reporting

Report Track A and Track B in separate tables:

- Track A: numerical benchmark, 2012-2025
- Track B intrinsic: sentiment classification, 2018-2023
- Fusion: numerical-only versus numerical+sentiment, common period 2018-2023

Do not claim that Track B covers 2012-2025.
