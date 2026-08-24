# PIT-SET50-CRIN 2024-2025 Protocol v1

## Status and scope

- Protocol ID: `pit-set50-crin-2024-2025-v1`.
- Market and target: SET50 price-index direction from close at day `t` to the
  registered next-session close.
- Reported outer years: 2024 and 2025 only.
- Comparison cohort: the intersection of dates having registered frozen
  five-model predictions and sufficient constituent coverage. Every reported
  model is rescored on exactly the same dates.
- Seeds: 42, 123, 456, 789, and 2025. Probabilities are averaged across seeds
  before the fixed 0.5 direction threshold is applied.
- Primary metric: mean annual balanced accuracy across 2024 and 2025.
  Direction accuracy and MCC are secondary.

This run is a provisional internal feasibility extension. Constituent rows
come from the Yahoo Finance chart endpoint at zero API cost. They must be
replaced by institution-authorized rows before the extension can be described
as paper-ready. Raw constituent rows must not enter a public replication
package.

## Point-in-time and leakage controls

1. SET50 membership is taken from the official SET H1/H2 documents already
   hashed in the project. The 2025 GULF/GULFI/INTUCH/VGI mid-cycle changes are
   effective-date masked.
2. Before an outer year, the H1 universe announced at that forecast origin is
   backcast over its own price histories. During the test year, membership is
   changed only on the official effective dates.
3. The backcast is a deployable-universe design, not a reconstruction of
   historical SET50 constituents. This limitation must be stated if results
   are reported.
4. Constituent raw Close is used because SET50 is a price index. Adjusted Close
   is retained only for a later sensitivity analysis. No close price is
   forward-filled. Return features are clipped at fixed, pre-registered bounds.
5. The constituent worker is trained through 2020, validated on 2021, and then
   frozen. Thus its bottom-up scores for 2022 onward are out of sample.
6. The reconciliation leader is walk-forward: for the 2024 test it trains on
   2022 and validates on 2023; for the 2025 test it trains on 2022-2023 and
   validates on 2024.
7. Top-down inputs are the already frozen, seed-averaged final-arm predictions
   of CNN, LSTM, LSTM-Attention, LSTM-CNN, and LSTM-CNN-Attention. Scaling is
   fit on the leader training rows only.
8. Test labels cannot change epochs, model dimensions, clipping, membership
   thresholds, or the 0.5 decision threshold.

## Frozen architecture

The bottom-up worker accepts 59 possible symbols but masks the input to the
50-name point-in-time SET50 universe and to names having adequate observations.
Each member uses the same eight-unit LSTM and eight-unit projection. Masked
attention pools member embeddings into one index representation. The worker
jointly predicts SET50 direction and next-session constituent breadth, with
breadth MSE weighted 0.20.

The leader receives five standardized top-down forecast returns, the frozen
bottom-up logit, constituent coverage, and top-down disagreement. It learns a
six-unit top branch, a soft reconciliation gate, and a bounded correction of
at most 0.25 logit units. The output is a direction probability.

Frozen settings:

| Item | Value |
|---|---:|
| Constituent window | 20 sessions |
| Constituent features | 1-day log return; scaled 5-day log return; intraday range; 20-day log-volume z-score |
| Minimum usable members | 35 of 50 |
| Bottom worker epochs | maximum 18 |
| Leader epochs | maximum 35 |
| Batch size | 32 |
| Early-stopping patience | 4 |
| Optimizer learning rate | 0.001 |
| Primary decision threshold | 0.5 |

## Required ablations

1. `majority_vote`: equal vote of the five frozen top-down models.
2. `top_only_stack`: learned top-down stack with no constituent worker.
3. `bottom_only`: frozen constituent worker with no top-down forecasts.
4. `pit_set50_crin`: the full reconciliation architecture.

The full model can be described as adding constituent information only if it
beats the top-only stack on the same common cohort. Beating an individual base
model without beating the top-only stack is evidence for ensembling, not for
the constituent-reconciliation contribution.

## Release gates

- All ten year-seed cells complete and contain finite outputs.
- Every test row has at least 35 usable members.
- Five seeds exist for every date and variant before averaging.
- Frozen five-model and CRIN metrics use identical dates.
- Runtime, coverage, attention, gate values, and source hashes are saved.
- No raw constituent rows are copied into public outputs.

