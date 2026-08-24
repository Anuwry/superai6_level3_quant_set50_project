# Frozen Pre-SHAP Experiment Manifest

> **EXECUTION HOLD (2026-07-31):** A deeper label-timestamp audit found that
> the final training row in every fold carries the first test day's close as
> its target. This v1 manifest is preserved as an audit record but is not
> authorized for execution. No SHAP result has been viewed. Resolve the P0
> items in `test/paper_reviewer_audit_pre_shap.md` and issue protocol v2 first.

Status: **FROZEN BEFORE SHAP EXECUTION**  
Freeze date: 2026-07-31 (Asia/Bangkok)  
Protocol version: `track-c-shap-v1`

This manifest fixes the prediction contract, SHAP selection procedure,
ablation arms, statistical unit, and fallback rules before any SHAP result is
opened. Any later change must be versioned and reported as exploratory.

## 1. Scope and evidence boundary

Completed before this freeze:

- Track A: Full TA versus causal rolling VMD, leakage-free window selection,
  five neural architectures, five seeds, and four expanding outer folds.
- Track B: local sentiment/relevance, Terra single/debate benchmark, eight
  out-of-sample daily news features, and paired technical versus +News fusion.
- Track C1: causal daily Bull/Sideway/Bear regime v2 and HMM v1 ablation.

Next registered experiment:

- Track C2: nested, window-aware consensus SHAP and regime-specific feature
  routing.

Not yet claimed:

- Economic profitability after costs.
- Live trading or real-market execution.
- External-market generalization.
- A pristine confirmatory Track C result.

Track C v2 was introduced after inspecting the semantic failure of HMM v1 in
the 2022-2025 outer diagnostics. Therefore all Track C outer results are
post-hoc robustness evidence. The original Track A window/VMD experiment
retains its pre-2022 selection boundary.

## 2. Prediction contract

Entity:

- One SET50 trading day \(t\).

Training target:

\[
y_t = Close_{t+1}
\]

All five neural models are regression models trained with mean squared error.
Direction is a derived evaluation quantity:

\[
d_t = sign(Close_{t+1} - Close_t)
\]

\[
\hat d_t = sign(\widehat{Close}_{t+1} - Close_t)
\]

The manuscript must use:

> next-day SET50 closing-level regression with derived direction evaluation

It must not describe the five models as pure binary classifiers unless a
separate classification loss/output is implemented and evaluated.

Prediction time:

- Features and regime are available through the close of day \(t\).
- The output forecasts the close of day \(t+1\).
- Sequences advance by one trading day.

Primary selection endpoint:

- Direction Accuracy (DA), because the research objective is direction
  prediction.

Secondary forecast endpoints:

- RMSE and MAE in original SET50 index units.
- Balanced Accuracy and MCC for derived Up/Down direction.

Guardrails:

- A selected subset must not reduce validation Balanced Accuracy by more than
  1 percentage point relative to all features.
- A selected subset must not increase validation RMSE by more than 5%.

MAPE and \(R^2\) may appear as supplementary regression diagnostics but are
not selection metrics.

## 3. Data and time boundary

Outer walk-forward folds:

| Fold | Train | Test | Test rows |
|---|---|---|---:|
| 1 | 2012-2021 | 2022 | 241 |
| 2 | 2012-2022 | 2023 | 243 |
| 3 | 2012-2023 | 2024 | 244 |
| 4 | 2012-2024 | 2025 | 234 |

The fourth test file ends on 2025-12-18 and must be described as a partial
2025 trading year.

Numerical feature pool:

- Full TA: 116 features.
- Causal rolling VMD: 6 features.
- SHAP numerical pool: 122 features.

News is not included in the primary 122-feature SHAP ranking because its
out-of-sample daily history begins in 2019. The eight Track B features remain
a separately locked `+News` treatment:

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

Technical-only and +News comparisons must use identical eligible dates.

## 4. Daily regime contract

Primary router:

- Daily multi-timescale semantic regime v2.
- Current regime \(R_t\), not a smoothed future/Viterbi label.
- Horizons: 1, 3, 5, 10, 20, and 60 trading days.
- Training-only Sideway threshold.
- Hard routing is primary.
- Distance-based soft memberships are a secondary ablation and are not called
  calibrated posterior probabilities.

HMM v1 is retained only as an ablation because its middle-return state behaved
as a high-volatility transition state rather than a range-bound Sideway state.

Every inner SHAP split must refit the v2 threshold on inner training only.
The saved outer-fold thresholds must never be used to label an earlier inner
validation period.

## 5. Locked models and sequence windows

| Model | Locked window |
|---|---:|
| LSTM | 5 |
| CNN | 5 |
| LSTM-CNN | 20 |
| LSTM-Attention | 20 |
| LSTM-CNN-Attention | 20 |

The excluded legacy architecture is Attention-LSTM-CNN. BiLSTM is not part of
the registered five-model benchmark.

Cross-model results compare complete model pipelines because windows differ.
The SHAP effect itself is evaluated only through paired All-versus-SHAP
comparisons within the same model, window, dates, fold, and seed.

## 6. Window-aware consensus SHAP

Selection folds use data before the first outer test:

| Selection fold | Model training | SHAP ranking year | Top-k validation year |
|---|---|---|---|
| 1 | 2012-2017 | 2018 | 2019 |
| 2 | 2012-2018 | 2019 | 2020 |
| 3 | 2012-2019 | 2020 | 2021 |

For each selection fold and each Bull/Sideway/Bear regime:

1. Fit all five models with their locked windows and selection seed 42.
2. Fit scalers on model-training rows only.
3. Compute SHAP on ranking-year sequences whose endpoint belongs to the
   relevant regime.
4. Aggregate a 3D time-series explanation into one value per feature:

\[
I_{m,j} =
mean_{sample,time}\left(|SHAP_{m,s,t,j}|\right)
\]

5. Convert each model's importance values to normalized ranks. This prevents
   explanation scale and window length from dominating the consensus.
6. Average normalized ranks across five models and temporal selection folds.

This produces one shared ranking per regime, not 15 separately optimized
model-regime lists.

## 7. Progressive top-k curve and stopping rule

Fixed candidate grid:

```text
10, 20, 30, 40, 60, 80, 100, 122
```

Every candidate is evaluated with all five models at their locked windows.
No recursive SHAP recomputation is allowed in the primary experiment.

For each model \(m\), regime \(r\), and candidate \(k\):

\[
\Delta DA_{m,r,k}
=
DA^{Top-k}_{m,r,k} - DA^{All}_{m,r}
\]

Select the smallest \(k\) satisfying all conditions:

1. Mean validation DA is within one standard error of the best candidate.
2. Median paired \(\Delta DA\) across models and selection folds is not
   negative.
3. At least three of five models are non-inferior or improved.
4. No model loses more than 1 DA percentage point on average.
5. No model's RMSE increases by more than 5%.
6. Median top-k Jaccard stability across temporal ranking folds is at least
   0.50.

If no reduced subset passes, retain all 122 features for that regime and
report that SHAP did not provide a stable reduction. The outer test can never
be used to change \(k\) or the feature list.

## 8. Minimum-sample and fallback rules

For a regime-specific inner split:

- Minimum model-training sequences: 200.
- Minimum SHAP ranking sequences: 40.
- Minimum top-k validation sequences: 40.

If a threshold is not met:

1. Do not resample across time.
2. Do not borrow future observations.
3. Use the corresponding global model/ranking fallback.
4. Record the regime, fold, available count, required count, and fallback.

For an outer prediction, a missing/failed regime expert falls back to the
same model's global predictor. Fallback behavior must be included in runtime
and performance summaries.

## 9. Registered ablation arms

Primary four-arm comparison:

| Arm | Router | Numerical features |
|---|---|---|
| Global-All | None | All 122 |
| Global-SHAP | None | Global consensus top-k |
| Regime-All | Daily v2 | All 122 in each expert |
| Regime-SHAP | Daily v2 | Bull/Sideway/Bear consensus top-k |

Additional isolated treatments:

- `+News`: append the eight locked daily news features to both members of a
  paired comparison on identical 2019-2025 eligible dates.
- `HMM v1 router`: statistical-router ablation only.
- `Soft routing`: secondary experiment only after hard routing is complete.

The primary SHAP claim is based on `Regime-All` versus `Regime-SHAP`.
The primary routing claim is based on `Global-All` versus `Regime-All`.

## 10. Seeds, statistical unit, and multiplicity

Selection:

- Seed 42 only.

Final outer evaluation:

```text
42, 123, 456, 789, 2025
```

Seeds are repeated fits, not independent market samples. For inference:

1. Average seeds within model-arm-fold.
2. Treat the four temporal folds as the primary independent units.
3. Report fold-level paired deltas and confidence intervals.
4. Report the limited resolution/power of four-fold exact sign-flip tests.
5. Use moving-block bootstrap on paired daily losses as a sensitivity
   analysis, with block length and resample count fixed before execution.
6. Apply Holm correction within each family of five model comparisons.

No p-value computed by treating seeds or individual autocorrelated trading
days as independent may be used as primary evidence.

## 11. Baselines

Regression baselines:

- Current-close random walk/no-change forecast.
- Ridge regression.
- XGBoost.
- LightGBM.

Directional baselines on identical dates:

- Training-majority Up/Down direction.
- Previous-day direction.

The no-change forecast is a valid next-close RMSE baseline but its zero-change
prediction is not a useful Up/Down classification baseline. Its near-zero DA
must not be presented as evidence that a neural model has strong directional
skill.

All final baseline rows must use the same dates as the corresponding neural
comparison.

## 12. Economic and live-trading claims

The current target is an index level, not a directly specified tradeable
instrument. Therefore:

- Historical profitability, Sharpe, drawdown, transaction cost, and slippage
  are outside the current registered forecasting experiment.
- Live execution is removed from the current pipeline claim.
- An economic backtest may be added only after specifying a tradeable vehicle
  (for example a SET50 futures contract or ETF), executable signal timing,
  roll rules, fees, spread, and slippage before viewing its results.

Forecasting performance must not be described as proven trading
profitability.

## 13. Freeze and deviation policy

The following are frozen before SHAP:

- Feature pool and Track B separation.
- Models and windows.
- Selection folds.
- Regime definition.
- SHAP aggregation and consensus.
- Candidate feature counts.
- Stopping and fallback rules.
- Primary/secondary metrics.
- Ablation arms.
- Seed handling and statistical unit.

Every deviation must be logged with:

- date;
- reason;
- whether any SHAP/outer result had already been viewed;
- affected tables;
- classification as confirmatory, robustness, or exploratory.
