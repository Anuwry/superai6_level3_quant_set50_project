# Corrected Pre-SHAP Experiment Manifest v2

Status: **FROZEN AND READY FOR SHAP; NO RANKING GENERATED**  
Draft date: 2026-07-31 (Asia/Bangkok)  
Protocol version: `track-c-shap-point-in-time-v2`

No SHAP ranking or outer Track C2 result had been viewed when this revision
was written. This document supersedes `pre_shap_experiment_manifest.md` for
future execution. The v1 file is retained as an audit record.

## 1. Scope and evidence boundary

This protocol governs:

- the 122-feature Full TA + causal rolling VMD pool;
- the five registered neural architectures;
- causal daily Bull/Sideway/Bear routing;
- window-aware consensus SHAP;
- size-matched alternative feature selection;
- capacity-matched routing controls;
- the four expanding 2022–2025 outer evaluations.

Track C remains post-hoc robustness evidence because the daily semantic
regime protocol was refined after inspecting shortcomings in the earlier HMM
router. It is not described as a pristine confirmatory holdout.

Forecasting accuracy is not evidence of tradeable profitability. No Sharpe,
PnL, or live-execution claim is registered here.

## 2. Point-in-time data contract

The target remains:

\[
y_t = Close_{t+1}.
\]

Every observation carries:

```text
Date       = timestamp of features available through close t
Label_Date = next observed SET50 trading date when y_t is known
```

For every train/rank, train/validation, and train/test boundary:

```text
retain training row iff Label_Date < first evaluation Date
```

The purged boundary row is never a supervised training example. Its features
remain available as a context-only row because they were observed before the
first evaluation feature date. Causal feature engineering and rolling VMD use
that history; train-fitted scalers transform it without refitting; sequence
models prepend it to the first evaluation sequence. Its target is never passed
to `model.fit`. `Label_Date` is metadata and is excluded from all model feature
matrices and scalers.

Corrected outer source:

| Fold | Train feature years | Test | Purged boundary feature date | Corrected Full-TA/VMD train rows | Test rows |
|---|---|---:|---|---:|---:|
| 1 | 2012–2021 | 2022 | 2021-12-30 | 2,358 | 241 |
| 2 | 2012–2022 | 2023 | 2022-12-30 | 2,599 | 243 |
| 3 | 2012–2023 | 2024 | 2023-12-28 | 2,842 | 244 |
| 4 | 2012–2024 | 2025 | 2024-12-30 | 3,086 | 234 |

The 2025 evaluation is partial and ends on 2025-12-18.

Context-only dates are 2021-12-30, 2022-12-30, 2023-12-28, and 2024-12-30
for the four outer folds. The equivalent dates for window selection are
2017-12-29, 2018-12-28, 2019-12-30, and 2020-12-30.

## 3. Prediction and direction contract

The five neural networks remain next-close regressors trained with MSE.
For direction evaluation:

\[
d_t = sign(Close_{t+1} - Close_t), \qquad
\hat d_t = sign(\widehat{Close}_{t+1} - Close_t).
\]

The binary estimand is Up versus Down:

- actual no-change rows remain in regression metrics but are excluded from
  binary direction metrics;
- a predicted exact no-change is an abstention;
- Direction Accuracy (DA), Balanced Accuracy (BA), and MCC use only non-tied,
  non-abstaining observations;
- direction coverage is evaluated observations divided by all non-tied actual
  observations;
- ties, abstentions, confusion counts, coverage, and predicted-up share are
  reported.

Primary model-selection endpoint:

- Balanced Accuracy.

Secondary endpoints:

- Direction Accuracy and MCC;
- RMSE and MAE in original SET50 index units;
- direction coverage.

MAPE and \(R^2\) are supplementary.

## 4. Feature pool and news boundary

Numerical pool:

- Full TA: 116 features;
- causal rolling VMD: 6 features;
- total: 122.

The eight Track B news variables are excluded from the primary SHAP ranking.
They remain a separately paired `+News` treatment on identical dates:

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

## 5. Models and corrected locked windows

Registered models:

1. LSTM
2. CNN
3. LSTM-CNN
4. LSTM-Attention
5. LSTM-CNN-Attention

Window candidates remain:

```text
1, 3, 5, 10, 20
```

Windows are selected on point-in-time-purged 2018–2021 folds by mean BA across
Full TA and Full TA + VMD. Tie-break order is higher DA, lower RMSE, then the
shorter window.

| Model | Corrected locked window |
|---|---:|
| LSTM | 5 |
| CNN | 20 |
| LSTM-CNN | 20 |
| LSTM-Attention | 10 |
| LSTM-CNN-Attention | 20 |

Corrected `locked_windows.csv` SHA-256:
`47456d5827bcb6a4ca01ce56cff21fce67d065f89c953150552ada3ca9faa55f`.

The corrected outer and Track B forecasting reruns and final input-hash audit
are complete. SHAP execution is authorized under this frozen protocol; no
ranking has yet been generated or inspected.

## 6. Daily regime router

Primary router:

- causal daily multi-timescale semantic regime v2;
- Bull, Sideway, Bear at close \(t\);
- horizons 1, 3, 5, 10, 20, 60 trading days;
- Sideway threshold fit on the admissible training partition only;
- hard routing primary;
- distance-based soft memberships supplementary and not called calibrated
  probabilities.

All regime labels and thresholds must be regenerated from corrected
point-in-time folds. HMM v1 remains an ablation only.

## 7. Nested temporal selection

Selection is complete before the first 2022 outer test:

| Selection fold | Model training | Explanation/ranking year | Top-k validation year |
|---|---|---|---|
| 1 | 2012–2017 | 2018 | 2019 |
| 2 | 2012–2018 | 2019 | 2020 |
| 3 | 2012–2019 | 2020 | 2021 |

At both boundaries, rows with `Label_Date` on or after the next partition's
first `Date` are purged from supervised fitting. Their already-observed
features may be passed only as sequence context. Scalers, regime thresholds,
model weights, SHAP backgrounds, and selector statistics are fit only within
the permitted partition.

## 8. SHAP attribution target and explainer

SHAP explains predicted next-day change in original index units:

\[
g(X_t) =
inverseTargetScale(f(X_t))
-
inverseCloseScale(X_{t,last,Close_D}).
\]

It does not explain the raw predicted closing level. This aligns feature
ranking with the direction objective and reduces price-level dominance.

Primary explainer:

- `shap.GradientExplainer`;
- expected-gradients formulation;
- one differentiable TensorFlow output \(g(X_t)\);
- interventional reference represented by empirical background sequences.

Deterministic numerical contract per model/selection-fold/regime:

- background: 100 eligible training sequences, selected at evenly spaced
  chronological indices;
- explained ranking set: at most 128 eligible sequences, selected at evenly
  spaced chronological indices;
- minimum ranking sequences: 40;
- `nsamples=200`;
- explicit `rseed` derived from protocol seed, model, fold, and regime;
- float32 tensors;
- aggregation:

\[
I_{m,j}=mean_{sample,time}(|\phi_{m,s,time,j}|).
\]

Compatibility path:

1. use the Keras change-output model directly;
2. if TensorFlow eager/model-object compatibility fails, retry the same
   `GradientExplainer` algorithm with an explicit `tf.function`;
3. if either output is non-finite, has the wrong shape, or the repeatability
   smoke test fails, stop the affected selection cell and the full protocol;
4. do not silently substitute a different explainer or mix explanation
   families in the primary consensus.

Before any ranking is opened, a synthetic/small-fold smoke test must verify:

- output shape is `(samples, window, features)`;
- all attributions are finite;
- two identical-seed calls have feature-rank Spearman correlation at least
  0.99;
- perturbing only future rows cannot alter earlier explanation inputs or
  predictions.

Smoke status on 2026-07-31:

- SHAP 0.52.0 with TensorFlow 2.21.0;
- all five registered architectures returned finite tensors with shape
  `(2, 3, 6)`;
- repeated calls with the same seed were exactly equal
  (`max_abs_diff = 0`), which is stronger than the registered rank-correlation
  threshold;
- causal Full TA and rolling-VMD future-perturbation tests passed separately;
- `ranking_generated=false`.

Machine-readable evidence:
`outputs/track_c/shap_protocol_v2/explainer_smoke.json`.

SHAP documentation basis:

- https://shap.readthedocs.io/en/latest/generated/shap.GradientExplainer.html
- https://shap.readthedocs.io/en/latest/generated/shap.DeepExplainer.html

## 9. Cross-model consensus

For every model, selection fold, and regime:

1. compute feature importance using the frozen change-output explainer;
2. rank all 122 features;
3. convert ranks to \([0,1]\) normalized ranks;
4. average normalized ranks across all five model-window pipelines and all
   eligible temporal selection folds.

This yields one global ranking and one ranking per Bull/Sideway/Bear regime.
Raw attribution magnitudes are never averaged across architectures.

Correlated features are interpreted as predictive groups. The manuscript
must not claim that SHAP uniquely identifies causal drivers.

## 10. Progressive top-k and stopping

Fixed grid:

```text
10, 20, 30, 40, 60, 80, 100, 122
```

No recursive SHAP recomputation is allowed in the primary analysis.

Choose the smallest \(k\) that:

1. is within one standard error of the best mean validation BA;
2. has non-negative median paired BA delta across models/folds;
3. improves or is non-inferior for at least three of five models;
4. loses no more than 1 BA percentage point for any model on average;
5. increases no model's RMSE by more than 5%;
6. has median temporal top-k Jaccard stability at least 0.50.

If no reduced set passes, retain all 122 and report that stable reduction was
not demonstrated.

## 11. Size-matched alternative-selector control

To determine whether any benefit is SHAP-specific rather than generic
dimensionality reduction, add a Spearman filter control:

- compute absolute Spearman correlation between each endpoint feature and
  next-day return on the same ranking-year observations available to SHAP;
- apply the identical label-time purge;
- average normalized ranks across the same temporal selection folds;
- for every SHAP-selected \(k\), retain exactly \(k\) Spearman-ranked features;
- do not tune a separate \(k\) for the control.

Registered selector comparisons:

- `Global-SHAP-k` versus `Global-Spearman-k`;
- `Regime-SHAP-k` versus `Regime-Spearman-k`;
- each selector versus its paired All-features arm.

A reduction benefit without superiority to the size-matched control is
reported as evidence for feature reduction, not evidence for SHAP-specific
selection.

## 12. Capacity-matched routing control

`Global-All` fits one model while `Regime-All` fits up to three experts. The
primary routing claim therefore cannot use that unmatched contrast.

Add `Global3-All`:

- fit three global replicas per architecture/fold using the same three
  deterministic sub-seeds assigned to Bull/Sideway/Bear experts;
- every replica sees the complete admissible training set;
- average their next-close predictions;
- total fitted model count and trainable parameter count match the three
  regime experts;
- save training and inference runtime separately.

Primary routing contrast:

```text
Regime-All versus Global3-All
```

`Global-All versus Regime-All` remains descriptive only. If Regime-All does
not beat the stricter capacity-matched control, no regime-routing benefit is
claimed.

## 13. Registered arms

| Arm | Router/capacity | Numerical features |
|---|---|---|
| Global-All | one global model | all 122 |
| Global3-All | three global replicas, averaged | all 122 |
| Global-SHAP | one global model | consensus SHAP top-k |
| Global-Spearman | one global model | size-matched Spearman top-k |
| Regime-All | three causal regime experts | all 122 |
| Regime-SHAP | three causal regime experts | regime SHAP top-k |
| Regime-Spearman | three causal regime experts | size-matched regime Spearman top-k |

Primary SHAP reduction contrast:

```text
Regime-SHAP versus Regime-All
```

Primary SHAP-specific contrast:

```text
Regime-SHAP versus Regime-Spearman
```

Primary capacity-matched routing contrast:

```text
Regime-All versus Global3-All
```

## 14. Minimum samples and fallbacks

Per regime-specific selection cell:

- minimum training sequences: 200;
- minimum explanation/ranking sequences: 40;
- minimum top-k validation sequences: 40.

If unavailable:

1. do not resample across time;
2. do not borrow future observations;
3. use the corresponding global ranking/model fallback;
4. record available and required counts, fold, regime, model, and fallback;
5. include fallback frequency in paper tables.

## 15. Seeds, inference, and multiplicity

Selection seed:

```text
42
```

Outer seeds:

```text
42, 123, 456, 789, 2025
```

Seeds are repeated fits, not independent market samples.

Inference procedure:

1. average seeds within model-arm-fold;
2. treat the four temporal folds as primary independent units;
3. report paired fold deltas and 95% intervals;
4. report the limited resolution of four-fold exact sign-flip tests;
5. use a predeclared moving-block bootstrap on paired daily losses as
   sensitivity analysis;
6. apply Holm correction within each five-model comparison family.

No primary p-value treats seeds, article pairs, or autocorrelated days as
independent observations.

Track B intrinsic uncertainty uses an `article_id` cluster bootstrap and
article-level paired method swaps; its article-ticker McNemar result is
supplementary only.

## 16. Five-model paper scope

Per the author scope decision on 2026-07-31, Track A–C paper-facing benchmark
tables contain only the five registered neural architectures:

```text
LSTM
CNN
LSTM-CNN
LSTM-Attention
LSTM-CNN-Attention
```

## 17. Freeze and deviation policy

Authorization completed on 2026-07-31 after:

- 119 focused tests passed with 0 failures;
- corrected Track A produced 200 outer rows;
- corrected Track B produced 200 fit rows and 100 paired rows;
- the five-model SHAP smoke test passed without generating a ranking;
- all 27 entries in `test/pre_shap_freeze_manifest_v2.json` were recomputed
  with no missing files or hash mismatches.

Frozen identifiers:

- corrected `locked_windows.csv` SHA-256:
  `47456d5827bcb6a4ca01ce56cff21fce67d065f89c953150552ada3ca9faa55f`;
- Track A input manifest SHA-256:
  `a434716d32a2c29ae6eb51027d49b6ab0d2cd56d070a89840f93486b5b173b51`;
- SHAP smoke evidence SHA-256:
  `e4b7601706b6c690fe6fb69636a23b2a0a7b8cabbfec8b1825dcb6a5718aa796`;
- Git HEAD: `a1a4845ccc7c2fa017d74811c915487c7dd9ef51`.

Because the shared worktree contains other author changes, registered files
are frozen individually in the machine-readable manifest rather than relying
on the Git commit alone.

Every deviation after authorization records:

- date and reason;
- whether a SHAP ranking or outer result had been viewed;
- affected arms/tables;
- confirmatory, robustness, or exploratory classification.
