# Next directional LSTM screen v1

Date: 2026-08-04

Status: **superseded after user required a genuinely new Ours architecture; not frozen and not executed**

Revision note: LSTM-Direct, BiLSTM-Direct, and xLSTM-Direct remain useful ablations but cannot be the proposed Ours model because their architectures already exist.

## Target

Predict only whether the SET index closes Up or Down on day `t+1`, conditional on information available through day `t`.

The model output is a single sigmoid probability `P(Up at t+1)`. It does not predict the next close, does not retrieve historical deltas, and does not convert a regression output into direction.

## Candidate ladder

1. **LSTM-Direct** — ordinary causal LSTM with a sigmoid binary direction head. This isolates the benefit of fixing the objective mismatch.
2. **BiLSTM-Direct** — forward and backward LSTM encoders over the same fully observed historical window, followed by a sigmoid head. The backward encoder reverses only rows inside `[t-W+1, ..., t]`; it never consumes `t+1`.
3. **xLSTM-Direct** — compact xLSTM/sLSTM block with exponential gating and stabilized memory, followed by a sigmoid head.
4. **RA-xLSTM-Direct candidate** — one minimal proposed modification in which the already causal Bull/Sideway/Bear probabilities adjust a small gate on the xLSTM representation before the direction head.

The first three are baselines/candidates, not claimed novelty. RA-xLSTM is only promoted to an Ours candidate if the xLSTM base first improves on LSTM-Direct in pre-2022 inner validation and the regime gate then adds a reproducible improvement.

## Development boundary

- Candidate selection uses point-in-time inner walk-forward validation ending no later than 2021.
- Screen windows: 1, 3, 5, and 10, matching the registered short-horizon question.
- Primary development metric: balanced accuracy.
- Secondary metrics: direction accuracy, MCC, Brier score, per-regime recall, runtime, and parameters.
- No price-regression loss, retrieval, threshold tuning on outer labels, seed removal, or selective-fold reporting.
- The same feature arms and news inputs are retained later; the architecture screen starts with Global-Numeric to avoid conflating architecture with fusion effects.

## Decision rule

First determine whether direct classification helps. Then determine whether bidirectional processing or xLSTM memory improves the direct LSTM. Only after one backbone wins on inner validation is a single regime-aware modification allowed. This prevents a large composite model from hiding which change produced the result.
