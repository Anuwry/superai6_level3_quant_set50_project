# Strong-Q2 hardening execution log v1

Execution date: `2026-08-03`  
Status: **IMPLEMENTATION COMPLETE; ACCESS CLAIM CORRECTED 2026-08-04**

The former institutional-entitlement statement is superseded by
`test/public_market_data_access_amendment_2026-08-04.md`. The current claim is
publicly accessible provider data with provider terms applying and no raw-data
redistribution.

## Completed workstreams

| Workstream | Outcome |
|---|---|
| Claim discipline | Evidence tiers and blocked claims frozen in v1; result-driven changes recorded separately in v2. |
| LLM compute matching | 3,999 new calls completed under a USD 18 guard; Leader remained superior to SC3 and SC4 controls. |
| Serial-dependence sensitivity | Existing 10-day, 10,000-replicate circular moving-block artifact verified; no BAcc contrast passed Holm. |
| Public reproducibility | Fail-closed public package v2 with 247 files built; every path, byte count, and SHA-256 verified; no restricted file found. |
| Access handling | Public provider pages and terms recorded; raw provider rows excluded; no open-data or institutional-entitlement claim made. |

## LLM result that closes the compute-budget objection

The Leader system achieved 76.594% intrinsic sentiment Accuracy. Equal-call
self-consistency 3 achieved 70.668%, producing a +5.926 pp difference with an
article-cluster 95% interval [3.491, 8.487]. Near-cost self-consistency 4 used
93.0% of the Leader-system cost and achieved 70.593%; the Leader difference was
+6.002 pp with interval [3.613, 8.477]. Both registered Accuracy comparisons
had Holm-adjusted p=0.000040.

This is a robustness result for sentiment annotation, not evidence that LLM
sentiment improves SET50 direction forecasts. The actual forecasting features
remain the frozen expanding Local NLP outputs.

## Cost control

The 3,999 new calls cost an estimated USD 13.9433 from recorded token counts,
below the USD 18 guard. The complete intrinsic-LLM ledger (existing one-call,
existing three-call Leader system, and new repeated-call extension) totals
approximately USD 38.5623. No further paid API work is required by this
hardening stage.

## Statistical clarification

The integrated daily block-bootstrap output contains 50 rows because it covers
five models, five contrasts, and two metrics. Directional and regression rows
must be reported separately:

- Balanced Accuracy: 25 rows, zero Holm-significant contrasts, minimum adjusted
  p=0.412.
- Squared-error loss: 25 rows, 12 Holm-significant sensitivities.

Regression significance must not be presented as directional significance.

## Reproducibility and release control

The current generated release bundle is located at
`release/public_replication_package_v2/` and is deliberately ignored by the
working repository. Its generated manifest records every included path, byte
count, file hash, and the non-self-referential package digest.
An independent post-build pass found exact agreement between manifest and disk,
zero hash/size mismatches, and zero restricted paths.

The public bundle excludes raw/prepared/fold market data, daily predictions,
raw news, private LLM checkpoints, keys, environment files, and confidential
entitlement evidence.

## Tests and QA added in this stage

- LLM compute-matched tests: 8 passed; metric-core coverage 87%; Ruff passed.
- Public-package tests: 16 passed; module coverage 94%; Ruff passed.
- A probability-column regression test caught and fixed a secondary log-loss
  class-order issue before final reporting.
- Runtime metadata now derives from persisted checkpoint completion times and
  per-call runtimes rather than a re-aggregation command's wall clock.

## Remaining submission gate

Before submission, rebuild and audit the final clean package after all
manuscript-facing files are frozen. The paper must retain the public-access
claim boundary, state that provider terms apply, and exclude raw provider rows.

## Effect on journal positioning

This hardening removes the compute-budget objection to the intrinsic Leader
comparison and materially improves reproducibility/provenance defensibility. It
does not create a strong predictive signal, independent external-market
replication, or Q1-level algorithmic novelty. The work is best positioned as a
SET-focused reliability-audit paper: a more defensible Q2 candidate when the
target journal values rigorous evaluation, negative results, explainable AI,
and emerging-market evidence; acceptance and quartile are not guaranteed.

## Primary references

- `test/strong_q2_claims_register_v1.md`
- `test/strong_q2_claims_register_v2.md`
- `test/track_b_compute_matched_protocol_v1.md`
- `test/track_b_compute_matched_execution_log_v1.md`
- `test/moving_block_bootstrap_audit_v1.md`
- `PUBLIC_REPLICATION_PACKAGE.md`
- `outputs/track_b/llm/compute_matched_v1/`
- `release/public_replication_package_v2/PUBLIC_MANIFEST.json`
