# PIT-FCG-LSTM pre-2022 development protocol v1

Protocol ID: `pit-fcg-lstm-inner-development-v1`  
Protocol date: 2026-08-04  
Result access at freeze: **no PIT-FCG-LSTM fit or metric existed**  
Status: architecture-development protocol; no 2022--2025 outcome may be used
for selection

## Research question and evidence status

Does a point-in-time, matched-placebo falsification gate help a bounded news
residual improve next-trading-day SET50 direction over a direct numerical LSTM,
ordinary numerical-news concatenation, and the same residual architecture
without falsification calibration?

The experiment is a post-freeze exploratory architecture extension because the
original five-model 2022--2025 results are already known. Architecture
development is restricted to 2019--2021. Passing this screen may justify a
separately labelled exploratory outer run; it cannot turn the known years into
untouched confirmatory evidence.

## Target and point-in-time contract

- Endpoint date: trading day `t`.
- Target: `1[Close[t+1] > Close[t]]`.
- Decision threshold: `0.5`, with ties classified Down.
- Every market, news, and regime input must be available by the end of day `t`.
- Sequence window: five trading rows ending at `t`.
- Test/validation labels never enter scaling, control matching, feature
  construction, threshold selection, or training.

## Frozen inner walk-forward splits

The common source is the 727-row 2019--2021 training partition of integrated
fold 1. Two inner evaluations are fixed:

| Inner fold | Training endpoints | Validation endpoints |
|---|---|---|
| `inner_2020` | 2019 | 2020 |
| `inner_2021` | 2019--2020 | 2021 |

The preceding four training rows supply validation sequence context. All model
variants use identical endpoint dates inside each inner fold. Standardization is
fit on the inner training rows only and applied unchanged to validation rows.

## Frozen inputs

- Numerical block: the 122 corrected Full-TA plus rolling point-in-time VMD
  features in their registered order.
- News block: the eight frozen daily features in their registered order:
  `news_sentiment_mean`, `news_sentiment_std`, `positive_ratio`,
  `negative_ratio`, `neutral_ratio`, `article_count`,
  `ticker_mention_count`, and `news_available`.
- Gate context at endpoint `t`: causal `prob_bull`, `prob_sideway`, `prob_bear`,
  and `routing_entropy` from the frozen point-in-time regime pipeline.
- The gate context is not appended to the numerical LSTM. It is available only
  to the news correction/gate branch.

## Past-only matched-placebo construction

For a training anchor sequence ending at row `i`, an eligible placebo news
sequence must end at row `j <= i - 5`. The placebo window therefore ends before
the anchor window begins. It must be inside the same inner training partition
and have the same causal endpoint regime.

Coverage matching uses a fixed three-component endpoint-window descriptor:

1. `log1p(sum(article_count))`;
2. `log1p(sum(ticker_mention_count))`; and
3. `mean(news_available)`.

Distance is Manhattan distance with fixed component weights `1, 1, 2`. If at
least five eligible same-year candidates exist, matching is restricted to the
same calendar year. Otherwise, all eligible prior candidates in the same regime
are used. One source is sampled deterministically from the five nearest
candidates using the registered seed. Anchors without an eligible same-regime
past source are removed from the **common training cohort for every arm**.

The random-control arm samples from all past candidates satisfying the same
five-row temporal gap while ignoring regime and coverage. It uses the same
common anchor cohort as the matched arm. Matching never uses labels.

For the validation falsification diagnostic, placebo sources come only from the
inner training partition, use the same regime/coverage rule, and never use a
validation row as a source. Validation endpoints lacking a same-regime training
source fail the integrity audit rather than being selectively removed.

## Frozen architecture

`PIT-FCG-LSTM` has three functional parts:

1. A 16-unit numerical LSTM and 8-unit dense head produce anchor logit
   `z_num`.
2. The current news window is summarized by its last row and temporal mean. A
   4-unit branch proposes residual `r = tanh(.)`, bounded to one logit.
3. A separate 4-unit branch outputs gate `g in [0,1]` from the numerical state,
   news summary, and causal gate context.

The inference logit is

`z_final = z_num + g * r`,

and inference requires only aligned news. The placebo branch shares all news
branch weights and exists only for training loss and falsification diagnostics.

## Frozen training objective

For label `y`, let `BCE(z)` denote per-sample binary cross-entropy from a logit.
Let `z_a = z_num + r_aligned` and `z_p = z_num + r_placebo` be ungated candidate
logits. The total loss for FCG arms is:

`L = L_final + 0.25 L_anchor + 0.25 L_candidate + 0.25 L_rank + 0.10 L_gate + 0.05 L_placebo`.

- `L_final = mean(BCE(z_num + g_aligned*r_aligned))`.
- `L_anchor = mean(BCE(z_num))`.
- `L_candidate = mean(BCE(z_a))`.
- `L_rank = mean(relu(0.01 + BCE(z_a) - stop_gradient(BCE(z_p))))`.
- Gate target is one only when `BCE(z_a)` improves on both `BCE(z_num)` and
  `BCE(z_p)` by at least `0.01`; otherwise it is zero. `L_gate` is BCE between
  this stop-gradient target and `g_aligned`.
- `L_placebo = mean((g_placebo*r_placebo)^2)`, suppressing admitted placebo
  corrections without forcing their class prediction to be wrong.

This is predictive falsification, not causal-effect estimation.

## Frozen variants and optimization

| Variant | Purpose |
|---|---|
| `direct_numeric_lstm` | 16-unit numerical LSTM with BCE direction head |
| `concat_lstm` | Same LSTM trained on concatenated 122+8 features |
| `bounded_residual` | Anchor, bounded residual, and learned gate; no placebo/rank/gate-calibration loss |
| `random_control_fcg` | Full FCG loss with a past-only unmatched random control |
| `matched_control_fcg` | Full PIT-FCG-LSTM with registered matched controls |

Common optimizer settings: Adam learning rate `0.001`, 20 epochs, batch size 32,
chronological `shuffle=False`, seeds `42, 123, 456, 789, 2025`, no early
stopping, Optuna, threshold search, epoch selection, or seed removal.

## Metrics and diagnostics

Primary metric: Balanced Accuracy (BAcc). Secondary metrics: direction
accuracy, MCC, binary cross-entropy, and Brier score. Predictions are averaged
over the five seeds before each inner-fold headline metric is computed. Runtime
and parameter count are recorded per fit.

Required mechanism diagnostics for both FCG arms:

- aligned-news versus placebo-news BAcc under the same fitted model;
- aligned and placebo mean/median gate values;
- gate-target positive rate;
- control source/anchor date gap, source partition, regime agreement, and
  coverage distance; and
- prediction completeness and finite-value checks.

## Pre-registered promotion rule

The full matched-control architecture passes the inner screen only if all hold:

1. Its mean inner-fold BAcc is strictly above `direct_numeric_lstm`,
   `concat_lstm`, and `bounded_residual`.
2. Its seed-averaged aligned-news BAcc is strictly above matched-placebo BAcc
   in both inner folds.
3. Its gate median is greater than `0.01` and less than `0.99` in both inner
   folds.
4. Its trainable parameter count is no more than 15% above the direct numerical
   LSTM.
5. Every required integrity check passes and every registered seed is retained.

The original shortlist's `+1 percentage point` and `3/4 outer folds` gates
remain requirements for later outer promotion. They cannot be evaluated or
tuned during this inner screen.

If any inner condition fails, the architecture is retained as a negative or
mixed development result and is not tuned on 2022--2025.

## Required artifacts

- frozen manifest and SHA-256 input hashes;
- control-pair audit table;
- per-seed predictions, metrics, parameter counts, and runtime;
- seed-averaged inner-fold predictions and metrics;
- aligned-versus-placebo diagnostics;
- promotion decision and integrity audit; and
- a Markdown execution log suitable for manuscript provenance.
