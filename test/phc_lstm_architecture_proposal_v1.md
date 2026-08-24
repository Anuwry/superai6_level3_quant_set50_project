# PHC-LSTM architecture proposal v1

Date: 2026-08-04

Status: **rejected after mechanism-level novelty audit on 2026-08-04; not frozen and not executed**

Decision record: the full audit is in `test/phc_lstm_novelty_audit_v1.md`. The
proposal remains here only as an audit trail. It must not be described as the
paper's proposed architecture or implemented as the next experiment without a
new rationale and protocol version.

Working name: **PHC-LSTM — Paired-Hypothesis Contrastive LSTM**

## Task

Given only information observed through trading day `t`, output one probability: `P(Close[t+1] > Close[t])`. PHC-LSTM is a binary classifier. It does not forecast the next close, retrieve historical outcomes, or infer direction from a regression output.

## Architecture

PHC-LSTM maintains two recurrent states for the same historical window:

- an Up-hypothesis state conditioned by a learned positive hypothesis token;
- a Down-hypothesis state conditioned by the corresponding negative token.

Both streams traverse the window chronologically and use the **same recurrent weights**. This is not a BiLSTM: neither stream reads beyond day `t`, and neither reverses time. Weight tying limits capacity inflation and makes the two states comparable.

At timestep `j`, each stream receives the same observed feature vector, its opposite hypothesis token, and the signed difference between the two previous hidden states. The state difference acts as recurrent counter-evidence:

`h_up[j], c_up[j] = SharedLSTM(x[j], +q, h_up[j-1] - h_down[j-1])`

`h_down[j], c_down[j] = SharedLSTM(x[j], -q, h_down[j-1] - h_up[j-1])`

The final classifier is constrained to use an antisymmetric evidence difference:

`logit_up = w^T LayerNorm(h_up[W] - h_down[W])`

`P(Up) = sigmoid(logit_up)`

Swapping the two hypotheses reverses the evidence difference and therefore reverses the logit. This embeds the binary Up/Down opposition in the architecture rather than asking an unconstrained dense head to discover it from limited samples.

Existing point-in-time numerical, VMD, regime-probability, and news features may be inputs, but they do not create extra expert models. The central contribution is the tied paired-hypothesis recurrent update plus antisymmetric readout.

## Why this is not a renamed existing model

- BiLSTM uses separate forward/backward traversal of a sequence; PHC-LSTM uses two forward causal hypothesis-conditioned states.
- A two-LSTM ensemble has independent weights and averages predictions; PHC-LSTM ties weights, exchanges signed state differences at every step, and produces one contrastive logit.
- A mixture of experts routes or weights expert outputs; PHC-LSTM does not route samples to experts.
- Bull/bear contrastive financial models exist, so the novelty claim cannot be “first bull/bear competition.” The narrower proposed mechanism is the antisymmetric tied recurrent hypothesis pair with recurrent counter-evidence for point-in-time direction classification.

## Required ablations

1. Existing frozen LSTM price-regression baseline.
2. LSTM-Direct: same approximate capacity, ordinary binary direction head.
3. Parameter-matched stacked LSTM-Direct: controls for depth/parameters.
4. PH-LSTM without recurrent counter-evidence: hypothesis tokens plus tied recurrence only.
5. PHC-LSTM without weight tying: controls whether gains come from simply using two LSTMs.
6. Full PHC-LSTM: tied weights, recurrent counter-evidence, antisymmetric readout.

BiLSTM-Direct and xLSTM-Direct can be included as modern reference baselines, but they are not Ours.

## Evaluation boundary

Architecture and hyperparameters must be selected only on point-in-time inner walk-forward data ending by 2021. Windows 1, 3, 5, and 10 may be screened inside that development period. After selection, the architecture is frozen before any new full 2022-2025 run. Because those outer years have already been accessed by earlier models, PHC-LSTM must remain a post-freeze exploratory extension unless confirmed on the untouched SET100 target or a future complete period.

## Promotion logic

- Final integrated BAcc improvement of at least 1 percentage point over frozen LSTM-CNN-Attention.
- Positive BAcc delta in at least three of four folds.
- Full PHC-LSTM must beat LSTM-Direct, parameter-matched stacked LSTM, and PH-LSTM without counter-evidence.
- Full PHC-LSTM must not be worse than any regime-specific LSTM-Direct result by more than 1 percentage point.
- Parameter count no more than 15% above the frozen final comparator.
- Five seeds, full coverage, no threshold search on outer labels, no selective seed/fold removal.

## Novelty caution (superseded by rejection)

The completed mechanism-level audit found material collisions for all central
ideas, including the B4 model's `[UP]`/`[DOWN]` representations and bull/bear
competition for the same next-day direction target. PHC-LSTM is therefore
rejected as the primary Ours model. The exact assembly not appearing verbatim
does not make the mechanism sufficiently distinct.
