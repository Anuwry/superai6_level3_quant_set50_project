# Paper Reviewer Audit Before SHAP

Audit date: 2026-07-31 (Asia/Bangkok)  
Scope: executed Track A-Track C artifacts, code, logs, and registered
pre-SHAP protocol  
Decision: **HOLD SHAP EXECUTION UNTIL P0 ITEMS ARE CORRECTED**

No full manuscript (`.tex`, `.docx`, or manuscript PDF) is present in the
workspace. This audit therefore evaluates the scientific evidence and claim
contract, not the abstract, literature review, references, journal template,
or final prose.

## Executive finding

The project has several strong controls: temporal outer folds, pre-2022
window selection, multi-seed neural fits, causal rolling VMD, locked LLM
prompt/test separation, runtime logging, negative-result reporting, and an
explicit post-hoc label for Track C.

However, a deeper point-in-time audit found a boundary-label leak in every
model fold. The final training row's next-close label is the first test day's
close. The existing Track A, trainable-baseline, tuned-model, and Track B
forecasting results therefore cannot yet be called leakage-free. The
current-close persistence calculation itself does not fit training labels and
is unaffected. SHAP must not be run on the contaminated training folds.

## P0: blockers before SHAP

| ID | Concern | Artifact evidence | Required correction |
|---|---|---|---|
| P0-1 | Boundary-label leakage | In every selection and outer fold, the last training `Target_Next_Close` exactly equals the first test row's `Close_D`. Examples: 2021-12-30 target = 2022-01-04 close = 999.46; 2024-12-30 target = 2025-01-02 close = 891.72. | Add an explicit label timestamp and require `label_date < test_start` for training. Drop/purge the boundary row before fitting the target scaler or model. Regenerate hashes and rerun every affected forecasting result. |
| P0-2 | Direction contract is not strictly binary | `sign(next_close-current_close)` contains two zero-return test days: one in 2022 and one in 2023. Existing DA silently treats them as a third value while the protocol says Up/Down. | Freeze one rule before rerunning: recommended—retain zero moves for regression but exclude them from binary DA, Balanced Accuracy, and MCC, while reporting excluded count/coverage. |
| P0-3 | SHAP output is misaligned with the primary endpoint | Models predict the nonstationary next closing level, but DA is primary. Absolute SHAP on the level output is likely to rank price-level features rather than directional-change information. | Freeze whether SHAP explains predicted change `f(x)-Close_t`, next return, or a separately trained direction output. Do not leave this implicit. |
| P0-4 | SHAP implementation is under-specified | The manifest does not lock the explainer, background/reference distribution, background size, explained sample cap, feature-dependence assumption, numerical tolerance, or TensorFlow fallback. | Add these choices and a deterministic explainer smoke test to protocol v2 before viewing any SHAP ranking. |
| P0-5 | Regime benefit is confounded by model capacity | `Global-All` uses one model while `Regime-All` can use three separately fitted experts. An improvement could come from extra parameters/ensemble capacity rather than meaningful regimes. | Add a capacity-matched global ensemble or a size-matched random/pseudo-regime router control. Freeze it before outer results are opened. |

### Boundary evidence

The exact boundary matches are:

| Stage | Fold | Last training feature date | Leaked target date/value |
|---|---|---|---|
| Window selection | 1 | 2017-12-29 | first 2018 test close = 1,156.28 |
| Window selection | 2 | 2018-12-28 | first 2019 test close = 1,046.71 |
| Window selection | 3 | 2019-12-30 | first 2020 test close = 1,081.00 |
| Window selection | 4 | 2020-12-30 | first 2021 test close = 924.36 |
| Outer test | 1 | 2021-12-30 | first 2022 test close = 999.46 |
| Outer test | 2 | 2022-12-30 | first 2023 test close = 1,013.39 |
| Outer test | 3 | 2023-12-28 | first 2024 test close = 884.83 |
| Outer test | 4 | 2024-12-30 | first 2025 test close = 891.72 |

This is one contaminated training label per fold. It may have a small
numerical effect, but it violates the point-in-time contract and is readily
detectable by a reviewer. The remedy is rerunning affected fits, not merely
adding a limitation paragraph.

## P1: major concerns before submission

### 1. Simple baselines currently outperform the neural regressors

Existing fold-mean RMSE:

| Method | Mean RMSE |
|---|---:|
| Current-close persistence | 7.536 |
| Full-TA Ridge | 7.864 |
| Best reported neural condition, LSTM + VMD | 13.244 |
| Full-TA LSTM | 13.861 |

The persistence baseline is therefore substantially better on closing-level
RMSE than every final neural model. The paper must show this result in the
main benchmark table. Omitting it would invite rejection for a weak or
selective baseline comparison.

All these target-based results must be recomputed after P0-1 is fixed.

### 2. Directional skill is weak and directionally imbalanced

A diagnostic recomputation from saved Track A predictions, excluding the two
zero-return rows only for binary metrics, found:

| Condition | Binary DA | Balanced Accuracy | MCC | Predicted-Up share |
|---|---:|---:|---:|---:|
| Full-TA LSTM | 52.67% | 53.48% | 0.076 | 60.01% |
| Full-TA CNN | 52.67% | 52.84% | 0.071 | 69.75% |
| LSTM-CNN + VMD | 51.90% | 52.60% | 0.061 | 63.60% |

The best MCC is close to zero and several models predict Up much more often
than the observed approximately balanced direction distribution. DA alone is
not sufficient. Main tables need confusion counts, Balanced Accuracy, MCC,
and direction baselines on identical dates.

### 3. Locked windows are statistically fragile

The pretest DA gap between the selected and second-ranked window is:

| Model | Selected window | DA gap to runner-up |
|---|---:|---:|
| LSTM | 5 | 0.157 pp |
| CNN | 5 | 0.580 pp |
| LSTM-CNN | 20 | 0.260 pp |
| LSTM-Attention | 20 | 0.308 pp |
| LSTM-CNN-Attention | 20 | 0.0016 pp |

For LSTM-CNN-Attention, W20 and W3 are effectively tied on DA, while W20 has
far worse selection RMSE (67.25 versus 42.60). The locked rule was applied as
documented, so this is not hidden test tuning, but it demonstrates selection
instability. Do not claim the chosen window is intrinsically optimal. Add a
pretest sensitivity table and describe it as a locked operational choice.

### 4. Four outer years provide low inferential power

Seeds are repeated fits, not independent market samples. After seed averaging,
there are only four temporal units. A two-sided exact sign-flip test cannot
produce a p-value below 0.125 with four nonzero fold effects. Holm correction
further reduces power.

Report effect sizes, fold signs, intervals, and daily moving-block bootstrap
as sensitivity analysis. Avoid language such as statistically superior unless
the corrected analysis genuinely supports it.

### 5. Track C is post-hoc and its semantic validation is partly circular

Daily regime v2 was designed after inspecting HMM v1 on the outer period.
Track C therefore remains post-hoc robustness evidence.

In addition, Bull/Sideway/Bear labels are constructed from multi-horizon
returns and ADX. Showing that Bull has positive 20-day return, Bear has
negative return, and Sideway is near zero is mainly a construction check, not
independent validation. The four-arm forecasting ablation and capacity-matched
control must provide the utility evidence.

### 6. Regime-specific sample-size thresholds are not justified

The registered `200/40/40` train/rank/validate minima are operational
thresholds, not power calculations. With 122 correlated features and deep
models, 200 training sequences may still overfit. Report all effective counts,
fallback rates, and learning-curve or minimum-size sensitivity evidence.

### 7. SHAP rankings need correlation-aware interpretation

The 122 TA/VMD features contain many highly correlated transforms of price.
SHAP allocation among correlated features is not unique and must not be called
causal importance. Lock the feature-dependence convention and describe the
output as predictive attribution. Jaccard stability is useful but does not
solve causal-identification concerns.

### 8. SHAP needs an alternative-selection control

All-features versus SHAP tests whether reduction helps, but not whether SHAP
is better than a simpler selector. If SHAP-guided refinement is a central
contribution, add at least one predeclared size-matched control such as random
feature subsets, correlation filtering, permutation importance, or a simple
embedded selector.

### 9. Track B pair-level inference is clustered

The locked 2023 test contains 1,333 article-ticker pairs but only 738 unique
articles. Pair-level bootstrap and McNemar calculations treat correlated
ticker pairs from the same article as independent. Recompute uncertainty with
article-level cluster bootstrap or another cluster-aware paired procedure.
Point estimates may stay unchanged.

### 10. The LLM debate improvement is compute-confounded

Terra debate uses Bull, Bear, and Leader calls, while the single-pass
comparison uses one call. The observed improvement demonstrates that the
three-role procedure beats one call under a larger compute budget; it does not
isolate debate structure. Either:

- add a compute-matched repeated-single/self-consistency control; or
- explicitly limit the claim to a budget-asymmetric system comparison.

Also describe Terra as one proprietary model under three role prompts, not as
three independent models.

### 11. Track B domain shift remains material

Mean headline length falls from about 260 characters in 2023 to about 65 in
2024-2025, and mean confidence falls from 0.756 to about 0.495. The 2024-2025
source is also a different population. Forward sentiment accuracy is unknown.
Only downstream forecasting effects may be claimed for those years.

### 12. Market-data provenance is incomplete

Raw SET50 CSVs exist, but the repository does not identify the provider,
download URL/date, license, adjustment convention, timezone, or treatment of
market-calendar anomalies. The Thai column format resembles an exported data
vendor file, but provenance must not be inferred. Add a formal source and
data-dictionary record plus raw-file hashes.

### 13. End-to-end reproduction is not yet clean

- Market alignment and target creation still depend on a multi-cell notebook
  with historical warning outputs and duplicated preparation attempts.
- There is no Track A pinned requirements file or environment lock.
- The top-level `README.md` still mentions Optuna/backtesting and an outdated
  model list, while the registered pipeline excludes those claims.
- Archived pipeline drafts contain placeholder results and unsupported stages.

Convert the authoritative data preparation to a deterministic script, add a
one-command reproduction path, pin dependencies, and make the repository
landing page match `pipeline8.md`.

## P2: limitations that are acceptable if disclosed

- Track A shows mixed VMD effects; no universal VMD improvement.
- Track B shows mixed news effects; no universal sentiment improvement.
- Terra debate loses to the local TF-IDF classifier on the locked test.
- No profitability, transaction-cost, live-execution, or external-market
  claim.
- The 2025 feature-date test ends on 2025-12-18 and is partial.
- Cross-model comparisons are complete pipeline comparisons because windows
  differ; SHAP effects must remain paired within model/window.

Negative or mixed outcomes do not invalidate the paper. Selective reporting
or stronger claims than the evidence would.

## Required correction order

1. Stop SHAP and record that no SHAP ranking/result was viewed.
2. Add label timestamps and purge cross-boundary training labels.
3. Regenerate scaled/unscaled selection and outer folds plus hashes.
4. Freeze the zero-return direction rule.
5. Rerun Track A window selection and outer evaluation.
6. Rerun affected classical/tuned baselines and Track B forecasting fusion.
   The Track B intrinsic text/LLM benchmark does not depend on the market
   boundary label and need not be rebilled.
7. Lock the SHAP explainer contract and attribution target.
8. Add a capacity-matched routing control and an alternative feature-selector
   control.
9. Version the corrected manifest as `track-c-shap-v2`; preserve v1 as an
   audit trail.
10. Only then begin SHAP.

## Claim-safe paper position after correction

The strongest defensible positioning is a leakage-audited, multimodal,
regime-conditioned SET50 forecasting study that reports positive, mixed, and
negative ablations. It should not be positioned as a profitable trading
system, a universally superior deep model, or proof that VMD, news, debate,
regimes, and SHAP each improve performance.
