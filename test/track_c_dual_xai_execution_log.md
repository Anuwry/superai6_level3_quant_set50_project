# Track C SHAP, Regime Ablation, and Dual-XAI Execution Log

Status: **COMPLETE AND INTEGRITY-AUDITED**  
Execution date: 2026-07-31 (Asia/Bangkok)  
Final protocol versions: `track-c-shap-point-in-time-v2`,
`track-c-outer-inference-v1`, and `track-c-dual-xai-lime-v1`

This is the paper-facing audit trail for SHAP ranking, progressive top-k
selection, seven-arm temporal outer evaluation, and the post-selection
SHAP-LIME robustness audit. Raw row-level artifacts remain the authoritative
record; this file records the design, execution facts, results, limitations,
runtime, corrections, and verification outcome.

## 1. Evidence boundary and frozen decisions

The main design was frozen in `test/pre_shap_experiment_manifest_v2.md`
before empirical SHAP ranking. The grouped-LIME settings in
`test/track_c_dual_xai_lime_addendum.md` were also frozen before ranking.

Two clarifications were frozen after rankings were available but before any
2022-2025 outer result or LIME explanation was inspected:

- `test/track_c_outer_inference_addendum.md` fixed the four-fold inference,
  moving-block bootstrap, contrasts, and Holm families;
- `test/track_c_dual_xai_structural_close_sensitivity.md` fixed a secondary
  agreement analysis excluding structural `Close_D`.

Neither clarification changed a feature, model, window, seed, top-k value,
regime label, or outer arm. Track C remains post-hoc robustness evidence
because the semantic daily regime router was refined after weaknesses of an
earlier HMM router had been observed. No causal or trading-profit claim is
permitted.

## 2. Data, models, windows, and regime router

- Models: LSTM, CNN, LSTM-CNN, LSTM-Attention, and
  LSTM-CNN-Attention only.
- Locked windows: 5, 20, 20, 10, and 20, respectively.
- Numerical inputs: 116 Full-TA variables plus 6 causal rolling-VMD
  variables, total 122.
- Prediction target: next observed close; direction is the sign of next close
  minus current close.
- Every split uses `Label_Date` purging. Scalers, thresholds, selectors,
  weights, and explanation backgrounds are fit from admissible training data
  only.
- Selection periods: 2012-2017/2018/2019,
  2012-2018/2019/2020, and 2012-2019/2020/2021 for
  train/rank/validate.
- Outer tests: 2022, 2023, 2024, and partial 2025 ending 2025-12-18.

The point-in-time regime router uses current-time trend features and
fold-training-only thresholds. It does not use the next-day target. Every
fold passed the semantic-order and minimum-share gates. Training Sideway
shares were 34.99%, 35.01%, 35.01%, and 35.00%; outer-test Sideway shares
were 43.57%, 34.57%, 42.62%, and 24.79%. The outer test therefore contains
all three regimes in every fold; Sideway is not missing.

## 3. SHAP ranking and selector controls

SHAP explains predicted next-close minus current close in original SET50
index units. The registered implementation used GradientExplainer, 100
train-only background sequences, at most 128 chronologically spaced ranking
sequences, `nsamples=200`, deterministic cell seeds, and float32 tensors.

Execution facts:

- 5 models x 3 temporal selection folds x 4 scopes = 60 SHAP cells;
- 60/60 cells completed, with finite and shape-valid tensors;
- zero fallback cells;
- cross-model consensus used normalized ranks, not raw magnitudes;
- a size-matched absolute-Spearman selector was generated from the same
  purged ranking rows;
- wall time: 203.90 seconds.

`Close_D` has an algebraic role because the explained target subtracts the
current close. Its rank is therefore not described as a discovered causal
market driver.

## 4. Progressive top-k validation and frozen result

The fixed grid was 10, 20, 30, 40, 60, 80, 100, and 122. The run completed
120/120 cells: five models, three folds, and eight k values. One global model
and three regime experts were evaluated in each cell. The frozen one-SE,
paired-BA, per-model BA, RMSE, and temporal-Jaccard gates were applied without
outer data or LIME.

| Scope | Selected k | Result |
|---|---:|---|
| Global | 122 | stable reduction was not demonstrated |
| Bull | 30 | smallest reduced set passing every gate |
| Sideway | 122 | stable reduction was not demonstrated |
| Bear | 80 | smallest reduced set passing every gate |

The Spearman control uses the identical scope-specific k. Validation wall
time was 741.66 seconds. Freeze hashes:

- `selected_top_k.json`:
  `5ffabe9bba272270e10f321de9cae0775f00d366618c49f0c02562f5f23e2264`;
- `selected_features.csv`:
  `533e0ec56d5008c95c8d77b4e98bfb510b449544ef09fc0e565f5f78a1c0c441`;
- `top_k_gate_audit.csv`:
  `3b777b756bda78177590f08764d18a12f7a44c716f08f488a2d12ca26a6ae0c9`.

Because global k equals 122, `Global-SHAP`, `Global-Spearman`, and
`Global-All` are identical by design. This is a valid negative selection
result, not three independent model wins.

## 5. Seven-arm outer execution

The arms were `Global-All`, `Global3-All`, `Global-SHAP`,
`Global-Spearman`, `Regime-All`, `Regime-SHAP`, and `Regime-Spearman`.
Five registered seeds were averaged within model-arm-fold. Regime experts and
the capacity-matched Global3 control used identical deterministic sub-seeds.

Execution and integrity facts:

- 100/100 model-fold-base-seed cells completed;
- 700 seed-fold-arm metric rows and 700 cell prediction files;
- 33,670 seed-averaged daily prediction rows;
- 1,100 executed unique fits representing 1,500 conceptual arm-fit
  references;
- 140 seed-averaged fold metric rows and 35 model-arm summary rows;
- dates, targets, regimes, identifiers, arm alignment, finiteness, and fit
  registries passed the independent integrity audit;
- the root fit registry retains model, fold, and base-seed identifiers.

Mean outer-fold results are shown below. BA is balanced accuracy; lower RMSE
is better. Exact values for all seven arms and all metrics are in
`outputs/track_c/outer_v2/arm_summary.csv`.

| Model | Global-All BA | Global3-All BA | Regime-All BA | Regime-SHAP BA | Regime-Spearman BA | Global-All RMSE | Regime-All RMSE | Regime-SHAP RMSE | Regime-Spearman RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.5308 | 0.5244 | 0.5307 | 0.5453 | 0.5333 | 20.831 | 19.100 | 19.669 | 18.114 |
| LSTM | 0.5312 | 0.5322 | 0.5127 | 0.5117 | 0.5178 | 12.092 | 12.479 | 13.377 | 11.722 |
| LSTM-Attention | 0.5237 | 0.5215 | 0.5065 | 0.4961 | 0.5075 | 36.429 | 17.454 | 17.680 | 17.139 |
| LSTM-CNN | 0.5268 | 0.5266 | 0.5318 | 0.5323 | 0.5433 | 30.507 | 19.853 | 18.891 | 18.601 |
| LSTM-CNN-Attention | 0.5110 | 0.5099 | 0.5375 | 0.5301 | 0.5337 | 46.386 | 22.120 | 21.767 | 24.561 |

The effect is heterogeneous. Regime-SHAP has the highest mean BA for CNN,
but not for the other four models. Regime-Spearman has the highest mean BA
for LSTM-CNN. Regime-All has the highest mean BA for
LSTM-CNN-Attention. The LSTM and LSTM-Attention BA results do not improve
under routing. This excludes any claim of universal regime or SHAP benefit.

## 6. Registered inference result

The primary units are four seed-averaged temporal folds. The run produced
100 paired fold contrasts, 125 fold-inference rows, 50 moving-block bootstrap
rows, and 175 Holm-adjusted inference rows.

Primary result:

- no exact four-fold comparison had raw p < 0.05;
- no exact four-fold comparison passed Holm at 0.05;
- all primary BA contrast intervals included zero;
- the minimum attainable non-zero two-sided exact p-value with four folds is
  0.125, and this resolution limitation must be stated.

For `Regime-SHAP - Regime-All`, mean BA changes were +1.458 percentage
points for CNN, -0.101 for LSTM, -1.035 for LSTM-Attention, +0.053 for
LSTM-CNN, and -0.740 for LSTM-CNN-Attention. Corresponding Holm p-values
were 0.625, 1.000, 0.625, 1.000, and 1.000. These are descriptive estimates,
not statistically established improvements.

The registered 10-day circular moving-block sensitivity found seven
Holm-significant squared-error contrasts, but no Holm-significant BA
sensitivity. Negative squared-error differences favor treatment. The seven
were:

- LSTM `Regime-SHAP - Regime-All`: +23.85, worse for SHAP;
- LSTM `Regime-SHAP - Regime-Spearman`: +45.36, worse for SHAP;
- CNN `Regime-SHAP - Regime-Spearman`: +68.59, worse for SHAP;
- LSTM-CNN `Regime-All - Global3-All`: -860.91, favors routing;
- LSTM-Attention `Regime-All - Global3-All`: -393.16, favors routing;
- LSTM-CNN-Attention `Regime-SHAP - Regime-Spearman`: -142.36,
  favors SHAP;
- LSTM-CNN-Attention `Regime-All - Global3-All`: -2202.14,
  favors routing.

These block-bootstrap findings are serial-dependence sensitivities and do not
replace the four-fold primary estimand.

## 7. Post-selection grouped-LIME audit

LIME was not used to select a feature, k, model, window, seed, regime, or
arm. The audit refit the full 122-feature `Global-All` seed-42 model for every
model-fold cell and explained the same original-unit predicted change as
SHAP.

Execution facts:

- 20/20 model-fold cells;
- 360 deterministic instances: six dates per Bull/Sideway/Bear within each
  model-fold cell;
- 1,024 temporally grouped perturbations and five LIME seeds per instance;
- 263,520 attribution rows and 1,800 SHAP-LIME comparison rows;
- 100 common train-only background sequences per cell;
- no outcome was used to choose an explanation date;
- every low-fidelity row was retained and separately flagged.

The weighted local-surrogate fidelity threshold was R2 >= 0.70. Overall,
1,293/1,800 rows (71.83%) were below this threshold. This is a substantive
limitation: grouped LIME was often unable to approximate the high-dimensional
neural prediction surface locally under the frozen neighborhood.

Agreement below is reported only for the reliable rows, as preregistered.

| Model | Reliable / 360 | Low-fidelity % | Median fidelity R2 | Median abs-rank Spearman | Median top-10 Jaccard | Median sign agreement |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 103 | 71.39 | 0.759 | 0.336 | 0.429 | 0.631 |
| LSTM | 105 | 70.83 | 0.754 | 0.411 | 0.333 | 0.664 |
| LSTM-Attention | 102 | 71.67 | 0.752 | 0.323 | 0.381 | 0.648 |
| LSTM-CNN | 104 | 71.11 | 0.759 | 0.342 | 0.333 | 0.648 |
| LSTM-CNN-Attention | 93 | 74.17 | 0.753 | 0.259 | 0.333 | 0.648 |

Low-fidelity shares were 75.67% in Bull, 72.67% in Sideway, and 67.17% in
Bear. Median pairwise top-10 stability across LIME repeats was only
0.111-0.127 by model, and median repeat rank correlation was 0.057-0.107.
Therefore LIME does not strongly corroborate SHAP in this experiment; it
serves as an honest stress test that exposes local-surrogate instability.

The predeclared sensitivity excluding structural `Close_D` did not improve
agreement. Reliable-row median rank correlations fell slightly for every
model, for example 0.411 to 0.396 for LSTM and 0.336 to 0.320 for CNN. The
modest agreement is therefore not presented as an artifact-free validation,
nor as a causal result.

## 8. Runtime and compute environment

The outer registry records actual fit and inference time, including reused-arm
accounting through conceptual references.

| Model | Unique fits | Conceptual references | Fit seconds | Inference seconds |
|---|---:|---:|---:|---:|
| CNN | 220 | 300 | 961.11 | 46.97 |
| LSTM | 220 | 300 | 5120.09 | 98.82 |
| LSTM-Attention | 220 | 300 | 2825.75 | 176.14 |
| LSTM-CNN | 220 | 300 | 2605.15 | 145.06 |
| LSTM-CNN-Attention | 220 | 300 | 3238.47 | 183.20 |

LIME audit wall-time sums across the four cells were 45.94, 54.10, 88.05,
100.98, and 129.99 seconds in the same model order. The audit also retains
separate fit, SHAP, LIME, and inference components in `runtime_summary.csv`.

Environment: Windows 11, Python 3.12.10, TensorFlow 2.21.0, NumPy 2.4.3,
pandas 3.0.1, scikit-learn 1.8.0, SHAP 0.52.0, CPU execution. Native Windows
TensorFlow >= 2.11 did not use CUDA. TensorFlow emitted known oneDNN dataset
attribute and retracing warnings; no final shape, finiteness, or gate failure
was caused by them.

All task-controlled temporary files and caches were redirected to
`D:/SET50_direction_prediction_paper/runtime_tmp` and
`D:/SET50_direction_prediction_paper/runtime_cache` because drive C had
insufficient free space.

## 9. Execution corrections and operational incidents

One LIME validation defect was detected before the final audit. The first
implementation compared frozen Outer predictions produced by
`model.predict()` with direct differentiable-graph outputs. oneDNN batch-path
rounding made the maximum difference 0.00010396, marginally above the frozen
0.00010000 gate. The validation was corrected to reproduce the exact frozen
Outer inference path; the differentiable explanation-graph difference is now
recorded separately. All 20 cells were regenerated under a source-file
implementation hash. Final maximum refit-versus-Outer error was
2.27e-13; maximum explanation-graph path difference was 0.00012283. No data,
weight seed, model, selector, protocol, or result threshold changed.

During the final outer model, a redundant deterministic worker remained after
its wrapper was stopped. It was identified and terminated. The intended seed
partitions were disjoint. The structural audit subsequently verified exactly
100 unique cells, 700 aligned prediction artifacts, 1,100 fit-registry rows,
and complete aggregate uniqueness. No duplicate or incomplete final result
was retained.

## 10. Final verification and required interpretation

Final verification:

- 75/75 focused `test_track_c*.py` tests passed;
- Ruff passed after the final import-order correction;
- Outer integrity audit: passed;
- LIME integrity audit: passed;
- selection freeze hashes: matched;
- final LIME implementation hashes: matched in 20/20 cells;
- final artifacts contain no missing required cell, duplicate key, invalid
  p-value, or non-finite attribution/metric checked by the audits.

Hash manifests are stored in:

- `outputs/track_c/outer_v2/outer_integrity_audit.json`;
- `outputs/track_c/dual_xai_lime_v1/lime_integrity_audit.json`.

Paper-safe conclusion:

> Point-in-time regime routing and SHAP reduction produced heterogeneous
> architecture-dependent effects. Stable dimensionality reduction was found
> only for Bull and Bear scopes, no primary four-fold contrast survived Holm
> correction, and the grouped-LIME audit showed limited fidelity and repeat
> stability. The contribution is therefore the leakage-controlled,
> capacity-matched evaluation and transparent dual-XAI stress test, not a
> universal accuracy or causal-feature claim.

The full negative and mixed results should remain in the paper or supplement.
Reporting only the best model/window/arm would overstate the evidence.
