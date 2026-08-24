# Integrated Multimodal Post-hoc Protocol v1

Freeze time (UTC): `2026-07-31T17:14:43Z`

Status: **FROZEN BEFORE ACCESS TO ANY NEW INTEGRATED RESULT**

Protocol ID: `integrated-multimodal-posthoc-v1`

## Evidence status and reporting rule

This is a post-hoc integrated extension because the 2022--2025 outcomes and the
earlier modular Track A--D results were already observed before this experiment
was designed. It must not be described as an untouched confirmatory test.

The news signal is produced by the frozen expanding/local NLP pipeline. No new
LLM API call is made. The LLM single-pass and worker-debate/leader experiment
remains a separate intrinsic ablation and is not presented as the source of the
features used here. This experiment therefore has USD 0 incremental API cost.

## Fixed research question

Does adding eight point-in-time daily news features improve the five frozen
next-trading-day SET50 models when the features are evaluated both globally and
inside the previously frozen regime-specific SHAP pipeline?

## Frozen common cohort

- Numerical source: corrected causal Full TA + rolling VMD point-in-time-v2
  folds.
- News source: `daily_news_features_2019_2025.csv`.
- Training starts on the first trading row in 2019 for every arm so that no arm
  receives a longer history than another.
- Outer tests remain 2022, 2023, 2024, and the available 2025 test fold.
- The train/test boundary, label purge, target, context rows, and test rows are
  identical within every model/fold/seed comparison.
- 2026 is excluded because no matching frozen 2026 news feature file exists.
  Existing numerical Track D evidence remains separate.

## Frozen models, windows, and seeds

| Model | Window |
|---|---:|
| LSTM | 5 |
| CNN | 20 |
| LSTM-CNN | 20 |
| LSTM-Attention | 10 |
| LSTM-CNN-Attention | 20 |

Seeds: `42, 123, 456, 789, 2025`.

Architecture, optimizer, loss, epoch count, batch size, chronological ordering,
and train-only scaling are inherited unchanged from the frozen Track A model
contract. There is no Optuna search, threshold tuning, window reselection, or
test-set-based stopping.

## Frozen feature contract

The numerical pool has 122 columns: 116 Full TA columns plus six causal rolling
VMD columns. The mandatory news block has these eight columns:

1. `news_sentiment_mean`
2. `news_sentiment_std`
3. `positive_ratio`
4. `negative_ratio`
5. `neutral_ratio`
6. `article_count`
7. `ticker_mention_count`
8. `news_available`

The news block is appended in the listed order and is not re-ranked or selected.
The previously frozen numerical SHAP selections remain Bull=30, Sideway=122,
and Bear=80. Therefore the regime multimodal experts contain 38, 130, and 88
features respectively. Keeping the full news block fixed prevents selection on
already-observed outer outcomes.

## Frozen 2 x 2 arms

| Arm | Routing/selection | News | Feature count |
|---|---|---|---|
| `Global-Numeric` | Global / all numerical | No | 122 |
| `Global-Numeric-News` | Global / all numerical | Yes | 130 |
| `Regime-SHAP-Numeric` | Hard regime / frozen SHAP | No | 30, 122, 80 |
| `Regime-SHAP-Numeric-News` | Hard regime / frozen SHAP | Yes | 38, 130, 88 |

Hard next-day regime routing uses only the previously generated causal regime
label available for that prediction row. There is no oracle routing.

## Feasibility guardrail fixed before fitting

Each regime expert must have at least 200 chronological training sequences;
otherwise the cell fails closed. A design-only count audit conducted before
fitting found a minimum of 212 sequences (fold 1, 20-day-window Bull expert), so
no fallback or relaxed threshold is registered.

## Frozen estimands and inference

Primary endpoint: Balanced Accuracy (BAcc). Secondary endpoints: Direction
Accuracy (DA), MCC, RMSE, and MAE. Seed predictions are averaged within each
model/fold/arm before fold metrics are computed.

Registered paired contrasts:

1. News effect globally: `Global-Numeric-News - Global-Numeric`.
2. Regime/SHAP effect without news: `Regime-SHAP-Numeric - Global-Numeric`.
3. News effect inside the final regime pipeline:
   `Regime-SHAP-Numeric-News - Regime-SHAP-Numeric`.
4. Final integrated pipeline effect:
   `Regime-SHAP-Numeric-News - Global-Numeric`.
5. Interaction:
   `(Regime-SHAP-Numeric-News - Regime-SHAP-Numeric) -`
   `(Global-Numeric-News - Global-Numeric)`.

For error metrics, negative deltas favour the first/treatment arm. Exact paired
sign-flip inference uses the four temporal folds. Holm adjustment is applied
across the five models separately for every contrast and metric family. Moving
block bootstrap is a temporal sensitivity analysis, not a replacement for the
four-fold primary inference.

## Required outputs and fail-closed audit

The run is resumable at model/fold/seed cell level and writes predictions,
per-fit training counts, parameter counts, fit runtime, inference runtime,
per-seed metrics, seed-averaged fold metrics, paired contrasts, adjusted
inference, runtime summaries, hashes, and a machine-readable integrity audit.

The audit fails if it finds a missing arm, duplicate key, non-finite value,
misaligned date/target/regime, an unexpected feature count, fewer than 200
training sequences, an incorrect seed/fold/model cardinality, or an input hash
different from the frozen manifest.

## Frozen input hashes (SHA-256)

The complete machine-readable list is stored in
`test/integrated_multimodal_freeze_v1.json`. Headline artifacts:

- locked windows: `47456d5827bcb6a4ca01ce56cff21fce67d065f89c953150552ada3ca9faa55f`
- daily news features: `13d3fc66a94c58bed5d5b49992bb63eec8e15336ed4f37bce0f29143089ab6f0`
- selected features: `533e0ec56d5008c95c8d77b4e98bfb510b449544ef09fc0e565f5f78a1c0c441`
- selected top-k: `5ffabe9bba272270e10f321de9cae0775f00d366618c49f0c02562f5f23e2264`

## Interpretation constraint

The integrated arm is supported only if improvements are temporally consistent
and survive the registered multiplicity analysis. A null or negative result is
retained and reported; it will not trigger model/window/feature reselection.
