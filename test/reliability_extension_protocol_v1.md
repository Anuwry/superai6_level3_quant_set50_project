# Multimodal falsification and temporal-origin robustness protocol v1

Freeze time (UTC): `2026-08-03T21:17:58Z`  
Protocol ID: `multimodal-falsification-v1`  
Evidence class: **pre-frozen retrospective falsification/robustness**

## Fixed question

Do the observed point-in-time news features contain model-relevant information
beyond feature-count, marginal-distribution, stale-news, or numerical-only
effects across all five frozen architectures?

## Cohort and model lock

- SET50 folds: 2022, 2023, 2024, and 2025.
- Training cohort starts in 2019 for every arm.
- Five architectures, their frozen windows, five seeds, model parameters,
  epochs, chronological fitting, and train-only MinMax scaling are unchanged.
- Numerical pool: 116 Full TA plus six causal rolling VMD columns.
- News block: the eight ordered daily local-NLP features already used by the
  integrated experiment.
- No API calls, Optuna search, feature selection, threshold tuning, or outcome-
  directed fallback is allowed.

## New arms

| Arm | Fixed construction | Purpose |
|---|---|---|
| `News-Only` | Eight observed news features only | Tests standalone news signal |
| `Global-Numeric-Shuffled-News` | Full numerical pool plus within-split deterministic row permutation of the joint news block | Primary information-content falsification |
| `Global-Numeric-Lagged-News` | Full numerical pool plus news delayed five trading rows | Tests timing/staleness sensitivity |
| `Global-Numeric-Random-Features` | Full numerical pool plus eight deterministic standard-normal placebo features | Tests dimensionality/capacity effect |

The `Market-Only` and `Observed-News` references are the exact persisted,
seed-averaged predictions from the frozen integrated experiment. They are not
refitted or selected after observing the controls.

Shuffling occurs independently within train, context, and test splits, retains
the eight-feature vector as a unit, preserves every split's empirical marginal
distribution, and uses no labels. The lag transform is causal across the
chronological train-context-test chain and fills only the first five unavailable
rows with zero. Random features are generated without labels from a
protocol/fold-specific seed and are scaled using training rows only.

## Estimands and inference

Primary endpoint: BAcc. Primary control contrast:
`Observed-News - Global-Numeric-Shuffled-News`.

Secondary contrasts:

1. `Observed-News - Market-Only`;
2. `Observed-News - Global-Numeric-Lagged-News`;
3. `Observed-News - Global-Numeric-Random-Features`;
4. `News-Only - Market-Only`; and
5. `Global-Numeric-Shuffled-News - Market-Only`.

Seed predictions are averaged before calculating fold metrics. Exact four-fold
sign-flip inference and Holm correction across the five architectures are
reported with the attainable-p-value limitation. A 10-day circular moving-
block bootstrap with 10,000 replicates is the serial-dependence sensitivity.
Quarterly origin effects describe temporal stability and are not treated as
independent confirmatory samples.

## Fail-closed requirements

The run fails on changed input hashes, missing arms/cells/seeds, fewer than 200
training sequences, duplicate dates or fits, non-finite predictions, target or
date misalignment, future-looking lag construction, or any departure from the
five-model/four-fold/five-seed grid. Runtime, package versions, control seeds,
input hashes, predictions, aggregate inference, and integrity audits are saved.

