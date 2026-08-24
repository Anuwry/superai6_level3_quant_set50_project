# Track B equal-call and near-cost LLM control protocol v1

Frozen: `2026-08-03T15:53:07Z`  
Protocol ID: `track-b-llm-compute-matched-v1`  
Status: frozen before any repeated-single control call or result

## 1. Question and evidence boundary

The existing locked 2023 benchmark found that a three-role
Bull/Bear/Leader system outperformed one `gpt-5.6-terra` single call. That
comparison is budget-asymmetric and cannot isolate debate structure from
additional inference compute.

This control asks whether the existing Leader still outperforms repeated
independent calls to the exact same single-pass prompt when call count is held
equal. A four-call self-consistency arm is added as a near-cost sensitivity
because the three-role system uses longer worker/leader outputs and inputs than
three short single calls.

The control is an intrinsic sentiment benchmark. Its outputs are not used to
retrain or select the downstream forecasting models.

## 2. Locked cohort and inherited results

- dataset: Bilingual StockTBSA, polarity-labelled article-ticker pairs;
- test year: 2023 only;
- pairs: 1,333;
- unique articles: 738;
- labels: 656 positive, 585 neutral, and 92 negative;
- item key: `article_id::TICKER`;
- exact item IDs, text hashes, and gold labels must match the existing locked
  predictions before an API request is allowed;
- existing single call is replicate 1 and is never rerun; and
- existing Leader output is the fixed treatment comparator.

The dataset, existing calls/predictions/metrics, prompts, analysis code, and
claims register are hashed in
`test/track_b_compute_matched_freeze_v1.json`. At freeze time the new output
directory did not exist and no repeated-single result had been observed.

## 3. Locked model, prompt, and calls

Every new call uses:

```text
model                 = gpt-5.6-terra
reasoning effort       = low
single prompt          = exact track-b-terra-v1 single prompt
structured schema      = SentimentVerdict
max output tokens      = 384
API response storage   = false
new replicate roles    = single_rep_2, single_rep_3, single_rep_4
incremental cost guard = USD 18.00
maximum concurrency    = 4 calls
```

No gold label, previous prediction, other replicate, worker argument, Leader
output, or aggregate result may appear in a prompt. The API does not expose a
fixed sampling seed for this configuration; reproducibility therefore relies
on append-only response checkpoints, response IDs, prompt hashes, usage, and
cost/runtime ledgers rather than a claim of deterministic regeneration.

## 4. Registered controls

| Arm | Inputs | Calls per item | Aggregation | Role |
|---|---|---:|---|---|
| Single | Existing replicate 1 | 1 | none | inherited baseline |
| Self-consistency-3 | Existing replicate 1 + new replicates 2-3 | 3 | mean class probabilities | primary equal-call control |
| Self-consistency-4 | Existing replicate 1 + new replicates 2-4 | 4 | mean class probabilities | near-cost sensitivity |
| Leader system | Existing Bull + Bear + Leader | 3 | existing Leader verdict | fixed treatment |

Mean probabilities are renormalised and the predicted class is the maximum in
the fixed order `negative`, `neutral`, `positive`. The four-call arm may be
called `near-cost` only if its measured total cost is within 15% of the fixed
three-role system cost; otherwise it is reported as a four-call sensitivity.

The primary comparison is Leader minus Self-consistency-3 accuracy. The
Self-consistency-4 contrast is secondary. Original one-call results are
reported only as inherited context.

## 5. Metrics and inference

Primary metric:

- three-class accuracy.

Secondary metrics:

- macro-F1, weighted-F1, MCC;
- log loss and multiclass Brier score;
- disagreement rate;
- input/output/reasoning tokens, tracked cost, and runtime.

Article ID is the inference unit because one article may have multiple ticker
labels. For both Leader-control contrasts:

- cluster bootstrap: 5,000 resamples, seed 42;
- cluster sign-flip: 50,000 Monte Carlo draws when exact enumeration is too
  large, seed 42;
- pair-level McNemar: supplementary only; and
- Holm correction: the two registered accuracy contrasts form one family.

The primary structural claim is supported only if Leader has a positive
accuracy effect against Self-consistency-3 and its article-cluster result
remains below 0.05 after the registered two-comparison Holm correction.
Otherwise the evidence is reported as inconclusive or not supportive. Even a
positive result does not prove that debate is universally superior because
token content and serial Leader adjudication cannot be perfectly matched by
repeated single prompts.

## 6. Cost and runtime guard

The existing one-call arm cost USD 4.6402325 over 1,333 calls. Three new
replicates per pair are estimated to cost USD 13.9206975 under the inherited
accounting schedule, placing the recorded project LLM total near USD 38.54.
The new runner must stop before issuing a request if its conservative reserved
cost would exceed the USD 18.00 incremental guard.

Actual usage and runtime, rather than this estimate, govern the final report.
Tracked cost is an experimental accounting convention based on recorded token
usage and the inherited price schedule; it is not represented as a provider
invoice.

## 7. Required artifacts

Public/non-reconstructive artifacts:

```text
outputs/track_b/llm/compute_matched_v1/
  cohort_integrity_audit.json
  checkpoint_audit.json
  metrics_by_arm.csv
  paired_comparisons.csv
  paired_comparisons_holm.csv
  runtime_cost_summary.csv
  paper_compute_matched_table.csv
  run_metadata.json
  output_manifest.json
```

Ignored private checkpoints:

```text
outputs/track_b/llm/compute_matched_v1/private/
  calls.jsonl
  errors.jsonl
```

Raw article text must never be written to the output directory. Checkpoints may
contain item IDs, text hashes, structured verdicts, response IDs, token usage,
cost, and runtime only.

## 8. Interpretation rules

Allowed before execution:

> The original Leader-versus-single result is budget-asymmetric; an equal-call
> self-consistency control is pending.

Blocked before execution:

- Debate structure caused the original improvement.
- Repeated single calls are weaker than the Leader.
- The four-call arm is cost-matched before the measured cost ratio passes its
  gate.
- Any control result changes the downstream local-NLP feature source.
