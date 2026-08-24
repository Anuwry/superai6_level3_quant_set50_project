# Track C Outer Inference Addendum

Status: **FROZEN BEFORE TOP-K SELECTION AND OUTER EXECUTION**  
Frozen at: `2026-07-31T16:25:00+07:00`  
Protocol version: `track-c-outer-inference-v1`

At freeze time, the SHAP rankings had been inspected and top-k validation was
still running. No selected top-k value, 2022-2025 Track C outer prediction, or
outer comparison had been generated or inspected. These inference settings
therefore cannot have been chosen from an outer result.

This addendum operationalizes the inference requirements already registered
in `test/pre_shap_experiment_manifest_v2.md`. It does not change any model,
feature, routing, selection, or outer arm.

## Registered contrasts

The treatment-minus-control contrasts are:

1. `global_shap_reduction`: Global-SHAP minus Global-All;
2. `global_shap_specificity`: Global-SHAP minus Global-Spearman;
3. `regime_shap_reduction`: Regime-SHAP minus Regime-All;
4. `regime_shap_specificity`: Regime-SHAP minus Regime-Spearman;
5. `regime_routing`: Regime-All minus Global3-All.

The last three are the primary Track C claims. The two global contrasts are
registered supporting analyses.

## Fold-level inference

Predictions are averaged over the five registered model-fit seeds within each
model-arm-fold before metrics or contrasts are calculated. The four temporal
folds are the independent units.

For every model and contrast, report treatment-minus-control deltas for BA,
DA, MCC, RMSE, and MAE. Report the mean, sample standard deviation, Student-t
95% interval over four folds, fold signs, and the two-sided exact sign-flip
p-value. Four folds imply a minimum non-zero two-sided exact p-value of 0.125;
this limitation must be stated.

Holm correction is applied across the five architectures separately within
each contrast-metric family. Seeds and daily rows are never treated as
independent replicates in this primary inference.

## Moving-block bootstrap sensitivity

The daily sensitivity analysis uses paired seed-averaged predictions on
identical dates.

Locked settings:

```text
bootstrap type      = circular moving-block bootstrap within each fold
block length        = 10 trading days
replicates          = 10,000
random seed         = 20260731
fold aggregation    = equal-weight mean of the four fold effects
interval            = percentile 95%
two-sided p-value   = 2 * min(P(delta <= 0), P(delta >= 0)), capped at 1
```

Two additive daily effects are bootstrapped:

- squared-error loss difference, treatment minus control, where a negative
  value favors treatment;
- balanced-accuracy contribution difference in percentage points, treatment
  minus control, where a positive value favors treatment.

For the BA contribution, actual no-change rows and any row not evaluated by
both arms are excluded. Class weights are fixed from each original fold
before resampling. The number of eligible paired direction rows is saved.

The block bootstrap is a sensitivity analysis for serial dependence. It does
not replace the four-fold primary estimand and must not be used to overstate
evidence when fold signs are inconsistent.

Holm correction is also reported for bootstrap p-values across the five
architectures within each contrast-loss family.

## Required artifacts

```text
outputs/track_c/outer_v2/
  predictions_seed_averaged.csv
  paired_fold_contrasts.csv
  fold_inference.csv
  daily_block_bootstrap.csv
  inference_holm_adjusted.csv
  inference_protocol.json
```

Any deviation must be recorded before interpreting the affected comparison.
