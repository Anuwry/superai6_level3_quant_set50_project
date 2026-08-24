# Track B LLM compute-matched execution log v1

Status: **COMPLETE**  
Protocol ID: `track-b-llm-compute-matched-v1`  
Protocol frozen (UTC): `2026-08-03T15:53:07Z`  
Execution completed (UTC): `2026-08-03T16:51:52Z`  
Final metadata regenerated after metric/runtime QA (UTC):
`2026-08-03T16:55:50Z`

## Question closed by this experiment

The earlier Leader result compared a three-call multi-role system with one
single-pass call. That comparison established a system-level performance
difference but could not separate the effect from additional inference budget.
This experiment reused the locked one-call output and generated three new
identical-prompt single-pass replicates per item. It created:

1. a three-call self-consistency control with the same call count as the
   Bull/Bear/Leader system; and
2. a four-call self-consistency sensitivity with cost within the frozen +/-15%
   near-cost band around the Leader system.

The permitted claim is about the complete multi-role Leader configuration
versus repeated identical single-pass inference. The experiment does not prove
that abstract "debate reasoning" alone caused the difference.

## Frozen cohort and inference

| Item | Value |
|---|---:|
| Article-ticker pairs | 1,333 |
| Unique articles | 738 |
| Test year | 2023 |
| Negative / neutral / positive pairs | 92 / 585 / 656 |
| Model | `gpt-5.6-terra` |
| Reasoning effort | Low |
| Maximum output tokens | 384 |
| New single-pass calls | 3,999 |
| Primary comparison | Leader - self-consistency 3 |
| Near-cost sensitivity | Leader - self-consistency 4 |
| Cluster bootstrap | 5,000 replicates; article ID |
| Cluster sign-flip | 50,000 Monte Carlo iterations; article ID |
| Multiplicity | Holm across the two registered Accuracy comparisons |

Cohort verification passed exact item-ID, text-hash, and gold-label agreement
against the frozen 2023 benchmark. New checkpoints contain hashes, structured
verdicts, response IDs, tokens, cost, and runtime, but no raw article text or
gold label. The checkpoint directory is private and excluded from Git and the
public replication package.

## Intrinsic sentiment results

| Arm | Calls represented | Accuracy | Macro-F1 | MCC | Log-loss | Brier |
|---|---:|---:|---:|---:|---:|---:|
| One-call single | 1,333 | 69.842% | 61.776% | 0.5484 | 0.6444 | 0.3970 |
| Self-consistency 3 | 3,999 | 70.668% | 62.733% | 0.5612 | 0.6337 | 0.3900 |
| Self-consistency 4 | 5,332 | 70.593% | 62.628% | 0.5603 | 0.6302 | 0.3885 |
| Bull/Bear/Leader system | 3,999 | **76.594%** | **70.253%** | **0.6190** | **0.5933** | **0.3483** |

Repeated single-pass inference improved only modestly over one call. The
Leader system retained a materially larger Accuracy, Macro-F1, MCC, log-loss,
and Brier advantage under both controls.

## Registered paired comparisons

| Comparison | Accuracy delta | Article-cluster 95% CI | Macro-F1 delta | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Leader - self-consistency 3 | **+5.926 pp** | **[3.491, 8.487]** | +0.0752 | **0.000040** |
| Leader - self-consistency 4 | **+6.002 pp** | **[3.613, 8.477]** | +0.0763 | **0.000040** |

For the primary comparison, the Leader system corrected 144 pairs missed by
self-consistency 3, while the control corrected 65 pairs missed by the Leader.
The supplementary pair-level exact McNemar p-value was 4.78e-08. Article ID,
not article-ticker pair, remains the primary inference unit because one article
can generate several labelled ticker pairs.

## Cost and runtime audit

Costs below are estimates reconstructed from recorded token counts and the
frozen pricing configuration; they are not a billing invoice.

| Arm/system | Token-accounted cost (USD) | Cost ratio to Leader | Runtime total (s) | Mean per call (s) |
|---|---:|---:|---:|---:|
| One-call single | 4.6402 | 0.232 | 3,243.15 | 2.433 |
| Self-consistency 3 | 13.9332 | 0.697 | 9,401.69 | 2.351 |
| Self-consistency 4 | 18.5836 | **0.930** | 12,445.81 | 2.334 |
| Bull/Bear/Leader system | 19.9787 | 1.000 | 15,012.50 | 3.754 |

The SC4/Leader cost ratio was 0.9302 and therefore passed the frozen near-cost
gate of 0.85--1.15. The new 3,999 calls cost USD 13.9433, below the USD 18 hard
guard. Existing one-call and Leader-system costs plus this extension give a
project-accounted intrinsic-LLM total of approximately USD 38.5623.

The new checkpoints span 2,671.25 seconds from first to last completion,
including the deliberate pause used to confirm the remaining budget. Summed
new-call runtime was 9,202.67 seconds; mean, median, and p95 runtime per call
were 2.301, 2.045, and 3.662 seconds. Final local aggregation required 83.09
seconds and made no API calls.

## Resumption and completeness

The first invocation was stopped after the user raised a budget concern. The
child runner was explicitly terminated and the checkpointed cost was audited.
Execution later resumed from the saved `(item_id, replicate)` keys. Completed
calls were not repeated. The final audit found:

- 3,999 expected and 3,999 observed new calls;
- exactly 1,333 items in each of replicates 2, 3, and 4;
- zero recorded API errors;
- no duplicate response IDs;
- no missing or non-finite metric values; and
- incremental cost USD 13.9433, below the USD 18 guard.

## Metric correction retained in the audit trail

The first aggregate emitted a scikit-learn warning because probability columns
for log-loss were ordered positive/neutral/negative while scikit-learn mapped
labels lexicographically as negative/neutral/positive. Accuracy, F1, MCC,
paired counts, confidence intervals, and p-values were unaffected. A failing
perfect-probability regression test reproduced the issue; the metric class
order was corrected, and the full aggregate was regenerated. The corrected
test suite passed 8/8, metric-core coverage was 87%, and Ruff passed.

## Claim decision

**Supported as robustness evidence:** on the locked intrinsic Thai financial
sentiment cohort, the multi-role Bull/Bear/Leader configuration outperformed
both an equal-call three-pass control and a near-cost four-pass control, with
article-cluster intervals excluding zero and Holm-adjusted p=0.000040.

**Not supported:** this intrinsic result does not establish that LLM outputs
improve next-day SET50 forecasts, that debate reasoning is the sole causal
mechanism, or that the Leader system is superior to the local TF-IDF sentiment
model. Downstream forecasting continues to use frozen expanding Local NLP
features, not these LLM verdicts.

## Authoritative artifacts and hashes

- Protocol: `test/track_b_compute_matched_protocol_v1.md`
  (`13530a24015294402e260f126aecd6765efa630279ca0998e69d676ef7c555fb`)
- Freeze manifest: `test/track_b_compute_matched_freeze_v1.json`
  (`bc82929682614549f337f933d7ed4fe5f77b6149c77eaca0ef342bc129919f28`)
- Metrics: `outputs/track_b/llm/compute_matched_v1/metrics_by_arm.csv`
- Paired/Holm inference:
  `outputs/track_b/llm/compute_matched_v1/paired_comparisons_holm.csv`
- Cost/runtime: `outputs/track_b/llm/compute_matched_v1/runtime_cost_summary.csv`
- Runtime/environment metadata:
  `outputs/track_b/llm/compute_matched_v1/run_metadata.json`
- Integrity audits and public output manifest:
  `outputs/track_b/llm/compute_matched_v1/`
