# Strong-Q2 paper claims register v2 addendum

Revision time (UTC): `2026-08-03T17:00:00Z`  
Supersedes for affected claims only:
`test/strong_q2_claims_register_v1.md`  
Reason: the pre-frozen LLM compute-matched experiment completed, the existing
moving-block-bootstrap evidence was independently audited, and a clean public
replication package was built.

The v1 register is retained unchanged. All claims not listed below keep their
v1 evidence tier, wording boundary, and status.

## Revision ledger

| Claim | v1 status | v2 status | Reason and authoritative evidence |
|---|---|---|---|
| C5: Leader vs one call | Supported but incomplete control | Supported as historical budget-asymmetric baseline | The one-call comparison remains valid but is no longer the sole basis for the Leader claim. |
| C6: LLM compute-matched control | Pending new experiment | **Supported as robustness evidence** | 1,333 locked pairs; SC3 equal-call and SC4 near-cost controls; article-cluster inference; `test/track_b_compute_matched_execution_log_v1.md`. |
| C13: framework conclusion | Supported | Supported with strengthened cost/reliability audit | The framework now includes a completed compute-matched LLM control and clean public artifact gate. |

## Revised C6 claim boundary

Permitted:

> On the locked 2023 intrinsic sentiment cohort, the multi-role
> Bull/Bear/Leader configuration exceeded identical-prompt self-consistency
> under an equal-call control (+5.926 Accuracy pp, article-cluster 95% CI
> [3.491, 8.487]) and a near-cost control (+6.002 pp, 95% CI [3.613, 8.477]);
> both registered comparisons had Holm-adjusted p=0.000040.

Blocked:

- debate reasoning alone caused the gain;
- independent heterogeneous agents were tested;
- the LLM Leader produced the downstream news features;
- the intrinsic LLM result proves a forecasting or trading gain; or
- the LLM Leader outperformed the frozen local TF-IDF sentiment model.

## Moving-block-bootstrap clarification

The existing integrated experiment used 10-day circular moving blocks and
10,000 replicates. Among 25 Balanced Accuracy rows, no contrast survived Holm;
the minimum adjusted p-value was 0.412. Twelve significant squared-error rows
remain regression sensitivities and must not be converted into directional
claims. Evidence: `test/moving_block_bootstrap_audit_v1.md`.

## Public-release gate

A fail-closed public package v2 containing 247 code, test, protocol, and
aggregate evidence files passed exact path/hash verification. It contains no raw market
rows, row-level predictions, private LLM checkpoints, or secret-like material.
The authoritative package digest is recorded only in the generated
`PUBLIC_MANIFEST.json` to avoid a self-referential hash inside the bundle.

This closes repository-remediation risk for a separate clean release bundle.
The access claim is limited to publicly accessible provider pages, provider
terms apply, and raw provider rows are not redistributed. No private
institutional-entitlement claim or evidence gate remains.

## Main-text placement after revision

The compact LLM compute-matched result may now appear in the main robustness
section. Detailed class metrics, checkpoint/runtime distributions, cost
accounting, correction history, and both control rows should remain in the
Supplement. The result must remain separate from downstream news-fusion
forecasting.

No other evidence tier is promoted because of this statistically favourable
result. LIME remains diagnostic/Supplement only, and the economic proxy remains
exploratory/Supplement only.
