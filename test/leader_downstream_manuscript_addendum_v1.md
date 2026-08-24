# Debate-Leader downstream manuscript addendum v1

Date: 2026-08-07  
Status: **DESIGN INCLUDED; QUANTITATIVE ARTIFACT PENDING**

## Author instruction

The author clarified that the intended Leader is the final LLM judge in the
Bull/Bear debate pipeline, not the neural reconciliation leader in the rejected
PIT-SET50-CRIN experiment. The author also reported that an additional test
showed improved SET50 performance when the Debate-Leader output was used.

## Repository reconciliation

A filename and content search was performed in the project workspace,
`D:\Downloads`, and `C:\Users\narak\Downloads`. No artifact containing the
reported downstream Debate-Leader-to-SET50 result was located. The available
`leader_diagnostics.csv` files belong to PIT-SET50-CRIN and must not be used or
relabelled as LLM-debate evidence. `D:\Downloads\submission_llm.csv` belongs to
an unrelated question-answer task.

## Manuscript action

`paper/manuscript_v1.md` now includes:

1. a Bull/Bear/Leader news route in the multimodal pipeline;
2. point-in-time requirements for converting Leader outputs into daily
   features;
3. a separate downstream SET50 estimand, distinct from the locked intrinsic
   2023 sentiment benchmark; and
4. Table 3C for an identical-cohort Market-Only, Local-NLP, and Debate-Leader
   comparison.

No Local-NLP result has been renamed as an LLM result, and no quantitative
Leader forecasting value has been fabricated. Pending cells are explicit.

## Required artifact fields

Before the pending result can be promoted to a submission claim, import the
author's result file and reconcile at least:

- exact test dates and eligible target-row count;
- architecture, frozen window, seeds, and training cohort;
- Market-Only, Local-NLP, and Debate-Leader BAcc on identical dates;
- paired fold/year or daily predictions for an interval and p-value;
- multiplicity family and Holm-adjusted p-value, if multiple models or
  contrasts were inspected;
- headline cutoff, timezone, next-session mapping, and missing-news handling;
- prompt/model identifier and prompt hash;
- API token cost, runtime, and failed/retried calls; and
- paths and hashes of the source artifacts.

## Claim boundary until reconciliation

Permitted in the working draft:

> The author reports that the additional Debate-Leader news arm improved SET50
> performance; exact common-cohort estimates and uncertainty are pending
> artifact reconciliation.

Blocked from a submission-ready version until evidence is imported:

- a numerical gain without a source artifact;
- a statistically reliable or general gain without paired uncertainty and
  multiplicity control;
- renaming Local-NLP predictions as Debate-Leader outputs; or
- describing CRIN leader diagnostics as LLM debate evidence.
