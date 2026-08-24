# Track C Outer Execution Addendum

Status: **FROZEN BEFORE OUTER EXECUTION**  
Frozen at: `2026-07-31T16:04:22+07:00`

This addendum resolves implementation details that were not numerically
specified in `test/pre_shap_experiment_manifest_v2.md`. It was written while
the pre-2022 top-k validation grid was still running. No 2022-2025 Track C
outer arm had been executed or inspected.

## Capacity-matched subseeds

For each registered outer base seed and regime, both the regime expert and
the corresponding `Global3-All` replica use the identical deterministic
subseed:

```text
SHA256(base_seed | "capacity_matched_expert" | regime) mod 2^31
```

Regimes are ordered:

```text
bull, sideway, bear
```

This ensures:

- three distinct replicas per base seed;
- identical architecture and subseeds for `Global3-All` and `Regime-All`;
- no outer result is used to choose seeds;
- reproducibility without Python's process-randomized `hash()`.

Global single-model arms use the registered base seed directly.

## Fit reuse

Within one model-fold-base-seed cell, fits with identical:

```text
architecture + training rows + feature set + random seed
```

may be computed once and reused across arms. Every reused arm is explicitly
marked. Runtime tables report:

- actual executed wall time;
- conceptual per-arm fit/inference time;
- whether an identical fit was reused.

Fit reuse cannot alter a prediction and is a computational optimization, not
an experimental arm.

## Feature order

Selected features are passed to every network in their original dataset
column order, not SHAP/Spearman rank order. This prevents feature reordering
from becoming an unintended treatment. The saved selector rank is retained
separately for interpretation.

For `k=122`, original column order makes the selected input tensor identical
to the registered Full TA + VMD input tensor.

## Outer execution gate

Outer execution is prohibited until these files exist and their hashes are
saved:

```text
outputs/track_c/topk_validation_v2/selected_top_k.json
outputs/track_c/topk_validation_v2/selected_features.csv
outputs/track_c/topk_validation_v2/top_k_gate_audit.csv
outputs/track_c/topk_validation_v2/selection_frozen.json
```

The grouped-LIME outer audit remains prohibited until all outer predictions
are complete. LIME does not affect the outer model, selected feature set, or
ablation results.

