# Track C Dual-XAI Structural-Close Sensitivity

Status: **FROZEN BEFORE OUTER EXPLANATIONS**  
Frozen at: `2026-07-31T16:33:00+07:00`  
Protocol version: `track-c-dual-xai-lime-v1-close-sensitivity`

At freeze time, SHAP selection rankings had been inspected, top-k validation
was incomplete, and no 2022-2025 outer explanation had been generated.

The registered explanation target is predicted next-close minus current
close. Consequently, `Close_D` is both an input to the forecasting model and
an explicit algebraic subtraction term. Its attribution can therefore be
partly structural rather than a discovered predictive mechanism. Agreement
between SHAP and LIME could also be inflated if both methods recover this
same structural term.

The primary 122-feature audit remains unchanged. A predeclared sensitivity
recomputes only the local agreement statistics after removing `Close_D` from
both attribution vectors:

- absolute-rank Spearman correlation;
- top-10 Jaccard overlap;
- non-zero sign agreement;
- LIME repeat stability.

No model is refit, no perturbation is rerun, and no feature set, top-k,
window, regime, or outer result can change from this sensitivity. All
121 remaining features are retained. The paper must:

1. call `Close_D` a structural current-price anchor;
2. avoid interpreting its rank as causal discovery;
3. show both all-feature and structural-close-excluded agreement;
4. avoid a dual-XAI concordance claim if agreement materially disappears
   after exclusion.

Required additional artifacts:

```text
outputs/track_c/dual_xai_lime_v1/
  agreement_excluding_structural_close.csv
  summary_excluding_structural_close.csv
```
