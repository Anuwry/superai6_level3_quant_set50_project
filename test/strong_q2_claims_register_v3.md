# Strong-Q2 paper claims register v3 addendum

Revision time (UTC): `2026-08-03T22:12:34Z`  
Supersedes affected claims in v1/v2 only. Earlier registers remain audit history.

## New multimodal falsification claim

Permitted:

> Under the frozen 2022--2025 five-architecture evaluation, observed point-in-
> time news did not establish an incremental Balanced Accuracy benefit over
> market-only inputs or the distribution-preserving shuffled-news control. No
> one of 30 registered BAcc rows survived Holm adjustment under either exact
> four-fold inference or the 10-day moving-block sensitivity.

Also permitted:

> The result is architecture- and time-dependent. It supports the paper's
> reliability-audit framing and shows why favourable single-model news-fusion
> results should not be treated as a universal multimodal gain.

Blocked:

- news has no economic or causal information;
- shuffled news is a deployable forecasting feature;
- the experiment is an untouched confirmatory holdout;
- quarterly origins are independent samples;
- an unadjusted p-value below 0.05 overrides the registered Holm family; or
- intrinsic LLM sentiment accuracy proves downstream forecasting value.

## Primary values

- Observed-News minus Market-Only BAcc effects ranged from -1.895 to -0.009
  percentage points; all exact Holm p-values were 1.000.
- Observed-News minus shuffled-news BAcc effects ranged from -3.239 to +1.020
  percentage points; exact Holm p-values ranged from 0.625 to 1.000.
- Moving-block Holm p-values for the primary shuffled control ranged from
  0.098 to 0.8528; none met 0.05.
- Integrity: 100/100 cells, 400 new fits, all predictions finite, USD 0
  incremental API cost.

Authoritative evidence:

- `test/reliability_extension_protocol_v1.md`
- `test/reliability_extension_freeze_v1.json`
- `test/reliability_hardening_execution_log_v2.md`
- `outputs/multimodal_falsification_v1/integrity_audit.json`
- `outputs/multimodal_falsification_v1/paper_falsification_table.csv`

## Placement lock

The falsification panel belongs in the main multimodal results table. LIME and
economic results remain Supplement-only. The intrinsic Bull/Bear/Leader result
may remain a separate main-text panel, explicitly labelled intrinsic and not
the downstream news-feature source. The prospective 252-session evaluation is
future work under a pre-frozen protocol and has no present result.
