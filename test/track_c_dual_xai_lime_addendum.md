# Track C Dual-XAI Addendum: Temporally Grouped LIME

Status: **FROZEN BEFORE SHAP RANKING**  
Frozen at: `2026-07-31T15:26:49+07:00`  
Protocol version: `track-c-dual-xai-lime-v1`

At the time this addendum was frozen, no empirical SHAP feature ranking,
top-k validation curve, or Track C outer result had been generated or
inspected. The only existing SHAP result was the pre-registered synthetic
compatibility smoke test at
`outputs/track_c/shap_protocol_v2/explainer_smoke.json`, whose metadata
explicitly records `ranking_generated=false`.

This addendum supplements, and does not replace,
`test/pre_shap_experiment_manifest_v2.md`.

## 1. Purpose and evidence boundary

SHAP remains the only feature-ranking and top-k selection method in the
primary Track C experiment. LIME is added only as a post-selection,
model-explanation robustness audit.

LIME must not be used to:

- select or reject a feature;
- choose top-k;
- choose a model, window, seed, regime definition, or ablation arm;
- tune a perturbation setting after inspecting 2022-2025 explanations;
- revise the frozen SHAP or Spearman selectors.

The primary dual-XAI audit is performed on the full 122-feature
`Global-All` model. This keeps the input space identical across the five
architectures and avoids circularly validating SHAP on a model that can only
see SHAP-selected features.

Track C and this XAI audit remain post-hoc robustness/exploratory evidence.

## 2. Registered models, folds, and timing

The audit covers exactly the five paper models and their corrected locked
windows:

| Model | Window |
|---|---:|
| LSTM | 5 |
| CNN | 20 |
| LSTM-CNN | 20 |
| LSTM-Attention | 10 |
| LSTM-CNN-Attention | 20 |

The audit uses the four expanding outer folds:

```text
2022, 2023, 2024, partial 2025 ending 2025-12-18
```

The `Global-All` model is refit with seed 42 in each fold. The explanation
audit is allowed to run only after SHAP consensus ranking, top-k selection,
and all outer ablation predictions have been frozen. No explanation result
may feed back into training or selection.

## 3. Explanation target and common reference

Both SHAP and LIME explain the same scalar:

\[
g(X_t)=\widehat{Close}_{t+1}-Close_t
\]

in original SET50 index units.

Both methods use the same 100 train-only background sequences selected at
evenly spaced chronological indices. No validation or test observation is
used as an explanation background.

Outer explanation dates are selected without looking at prediction
correctness or explanation values:

- six dates per Bull/Sideway/Bear regime, model, and outer fold;
- dates are chosen at evenly spaced chronological indices within the regime;
- correctness is attached only after the dates are fixed;
- a cell with fewer than six eligible dates fails closed and is reported.

## 4. Temporally grouped LIME neighborhood

Standard independent perturbation of flattened lag-feature cells can create
implausible time-series fragments. The registered LIME interpretable
components are therefore the 122 original features, not `window x feature`
cells.

For one explained sequence:

1. draw a binary presence mask over 122 features;
2. retain the complete observed lag trajectory for every present feature;
3. replace the complete lag trajectory of every absent feature with the
   corresponding trajectory from one train-only background sequence;
4. use the same background sequence for all absent features in that
   perturbation to retain their joint historical context;
5. score the perturbed sequence with the frozen neural model;
6. fit the local weighted linear surrogate to predicted change.

Locked LIME settings:

```text
perturbations per explanation = 1024
feature presence probability  = 0.50
repeat seeds                  = 42, 123, 456, 789, 2025
interpretable features        = all available original features
surrogate                     = weighted Ridge
Ridge alpha                   = 1.0
distance                      = Euclidean distance in binary mask space
kernel width                  = 0.75 * sqrt(number of features)
kernel                        = sqrt(exp(-(distance^2)/(kernel_width^2)))
```

The first neighborhood row is always the unmodified observation with weight
one. Inputs and attributions must be finite. Invalid shape, non-finite
values, insufficient samples, or non-deterministic repeated-seed generation
causes the cell to fail closed.

## 5. SHAP-LIME comparison

For a local SHAP tensor, signed lag contributions are summed by feature:

\[
\phi_j^{local}=\sum_{lag}\phi_{lag,j}.
\]

No mean-absolute aggregation is used for the local sign comparison.

For every explained date and LIME repeat, save:

- full signed SHAP feature contributions;
- full signed LIME coefficients;
- LIME weighted local \(R^2\);
- absolute-rank Spearman correlation;
- top-10 Jaccard overlap;
- non-zero sign agreement;
- SHAP runtime, LIME runtime, and model inference runtime.

LIME stability is measured across its five repeat seeds using:

- pairwise top-10 Jaccard overlap;
- pairwise absolute-rank Spearman correlation;
- coefficient variation summaries.

No result is silently discarded. A LIME explanation with weighted local
\(R^2<0.70\) is flagged as low fidelity and is retained in counts and tables,
but it is excluded from substantive SHAP-LIME agreement claims.

## 6. Aggregation and reporting

The paper reports:

- model-fold-regime medians and interquartile ranges;
- fold-level pooled summaries after model-level reporting;
- number and proportion of low-fidelity explanations;
- correct and incorrect prediction slices as descriptive error analysis;
- all five models, including negative or discordant results;
- runtime and explanation throughput.

Seeds are repeated perturbation audits and are not independent market
samples. No p-value may treat LIME repeats or daily observations as
independent. Fold-level uncertainty and descriptive intervals are used.

## 7. Interpretation controls

Allowed wording:

> Temporally grouped LIME was used as a post-selection local robustness audit
> of SHAP explanations under a common train-only reference distribution.

Not allowed:

- LIME or SHAP proves a causal market driver.
- Agreement between LIME and SHAP proves that an explanation is true.
- A low-fidelity LIME coefficient is a reliable feature effect.
- Outer-test explanations justify changing the selected feature set.
- Dual-XAI concordance demonstrates profitable trading.

## 8. Required artifacts

The audit must produce:

```text
outputs/track_c/dual_xai_lime_v1/
  protocol.json
  selected_instances.csv
  local_explanations.csv
  agreement_by_instance.csv
  lime_stability_by_instance.csv
  summary_by_model_fold_regime.csv
  low_fidelity_audit.csv
  runtime_summary.csv
  run_metadata.json
  deviation_log.csv
```

The paper log must record source hashes, model/window registry, package
versions, hardware, seeds, all fallbacks/failures, and whether any outer
explanation was viewed before selection was frozen.

