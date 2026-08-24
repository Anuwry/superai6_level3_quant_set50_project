# Primary estimand and prospective confirmatory protocol v1

Freeze time (UTC): `2026-08-03T21:17:58Z`  
Status: **FROZEN BEFORE NEW FALSIFICATION RESULTS AND FUTURE MARKET OUTCOMES**

## Current-paper estimand hierarchy

The primary forecasting endpoint is next-trading-day Balanced Accuracy (BAcc).
The primary multimodal estimand is the paired BAcc difference between
`Observed-News` (the frozen Full TA + causal VMD numerical pool plus eight
point-in-time local-NLP news features) and `Market-Only` under identical model,
window, seed, train/test dates, label purge, and train-only scaling.

The five registered architectures form one multiplicity family. Seeds are
averaged before temporal inference and are not treated as independent market
samples. Exact fold-level sign-flip tests with Holm adjustment remain the
registered low-power inference. A 10-trading-day circular moving-block
bootstrap with 10,000 replicates is the serial-dependence sensitivity analysis.
Direction Accuracy, MCC, RMSE, MAE, regime slices, quarterly origins, and
economic metrics are secondary or exploratory.

The primary falsification estimand is `Observed-News -
Global-Numeric-Shuffled-News` in BAcc. Shuffling is deterministic, preserves the
joint eight-feature vector and each split's empirical distribution, occurs
separately inside train/context/test, and never uses labels. Lagged-news,
random-feature, news-only, and shuffled-versus-market contrasts are secondary
controls. Holm correction is applied across the five architectures separately
within each contrast and metric family.

No current-paper result may be relabelled as an untouched confirmatory result.
The 2022--2025 outcomes and partial-2026 outcomes were inspected before this
document existed.

## Prospective confirmatory evaluation

The untouched confirmatory cohort begins with the first SET trading session
after `2026-07-30`, the last outcome currently present in the frozen forward
artifact. The primary analysis occurs after 252 newly labelled trading
sessions. A 126-session analysis is an explicitly labelled interim reliability
check and cannot replace the final analysis.

The prospective primary model is LSTM-CNN with the already frozen 20-day
window. Model architecture, features, VMD settings, news aggregation, local-NLP
checkpoint, direction threshold, regime router, and training policy are locked
from the development evidence. There is no Optuna search, threshold tuning,
feature reselection, early look used for model selection, or substitution of
an LLM/debate output for the local-NLP news source.

For every prediction date, the system must persist before the next close:

- prediction and probability/value needed to reconstruct direction;
- exact model, feature, scaler, and data-manifest hashes;
- information cutoff and Asia/Bangkok timestamp;
- news availability and last included publication timestamp;
- runtime and execution status; and
- a missing-prediction reason when no forecast is issued.

The primary prospective comparison is paired BAcc of observed-news versus
market-only over the complete 252-session cohort. Moving-block inference uses
the frozen 10-day block length. Coverage, failures, and abstentions are retained
in the denominator policy defined before outcome access. Transaction-cost and
economic analyses remain exploratory even if statistically favourable.

## Interpretation lock

A favourable falsification result supports information content but does not
prove causal news effects. A null result is retained. Intrinsic LLM debate
accuracy is reported separately and is not represented as the downstream
feature generator. SHAP is the principal fitted-model attribution method; LIME
and economic proxies remain Supplement-only diagnostics.

