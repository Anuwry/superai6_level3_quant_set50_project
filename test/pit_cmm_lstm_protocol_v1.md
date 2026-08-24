# PIT-CMM-LSTM Exploratory Architecture Extension Protocol v1

Freeze time (UTC): `2026-08-04`

Status: **FROZEN BEFORE ACCESS TO ANY PIT-CMM-LSTM OUTER RESULT**

Protocol ID: `pit-cmm-lstm-exploratory-v1`

## Evidence status

PIT-CMM-LSTM was designed after the 2022--2025 outcomes and the frozen five-model
results had already been inspected. It is therefore a post-freeze exploratory
architecture extension. It must not be presented as an untouched confirmatory
result, and it does not replace or modify the registered five-model family.

## Fixed question

Under the identical point-in-time multimodal protocol, does a compact LSTM cell
with two competing low-rank matrix memories produce more reliable next-trading-
day SET50 direction forecasts than the frozen LSTM and the other four frozen
architectures?

## Frozen architecture

- Working name: Point-in-Time Competitive Matrix-Memory LSTM (PIT-CMM-LSTM).
- Standard recurrent state: hidden state 12 and LSTM cell state 12.
- Additional states: bullish and bearish `4 x 4` matrix memories.
- The two matrix memories receive separate learned evidence weights and decay
  factors at every step.
- A competitive readout combines bullish, bearish, and signed-difference reads
  with the standard LSTM cell state.
- Dense head: 8 ReLU units followed by one linear next-close output.
- Optimizer/loss: Adam/MSE.
- Epochs/batch: 20/32.
- Chronological training: `shuffle=False`.
- Fixed window: 5. The window is inherited from the frozen LSTM architecture
  because PIT-CMM-LSTM is an LSTM extension and the short-context design is part
  of the stated hypothesis. No PIT-CMM window sweep is permitted in v1.
- No Optuna, test-set early stopping, threshold tuning, or feature reselection.

The first experiment tests the core competitive-memory architecture. A
negative-control consistency loss is not included in v1 and must not be claimed
as an implemented component.

## Frozen data and evaluation

- Folds: `fold_1`--`fold_4`, testing 2022--2025.
- Seeds: `42, 123, 456, 789, 2025`.
- Cohort, scaling, target, context rows, news assignment, VMD construction,
  regime routing, and SHAP-selected numerical subsets are inherited unchanged
  from `integrated-multimodal-posthoc-v1`.
- Arms: Global-Numeric, Global-Numeric-News, Regime-SHAP-Numeric, and
  Regime-SHAP-Numeric-News.
- Primary metric: Balanced Accuracy after averaging the five seed predictions
  within each fold and arm.
- Secondary metrics: Direction Accuracy, MCC, RMSE, MAE, runtime, training
  sequences, and trainable parameters.
- Primary architectural contrast: PIT-CMM-LSTM versus LSTM in the final
  `Regime-SHAP-Numeric-News` arm. All other pairwise rankings are descriptive.
- The six-model table retains the five frozen rows byte-for-byte from the
  completed integrated experiment and appends the new exploratory row.

## Promotion rule fixed before fitting

PIT-CMM-LSTM is not described as an improved model unless all of the following
hold:

1. Mean BAcc exceeds the frozen LSTM by at least 1.0 percentage point in the
   final integrated arm.
2. Fold-level BAcc delta is positive in at least three of four temporal folds.
3. Direction coverage is non-degenerate and finite predictions are produced in
   every seed/fold/arm cell.
4. The architecture remains within a 15% trainable-parameter budget of the
   frozen LSTM for representative input widths 38, 88, and 130.

Failure to meet these gates is retained as a negative architecture result.

## Frozen input hashes (SHA-256)

- Integrated freeze manifest: `30400c1051dad84910bd20c39af4b0d5ab1d62257f19f5379cbb599af9c4b954`
- Locked windows: `47456d5827bcb6a4ca01ce56cff21fce67d065f89c953150552ada3ca9faa55f`
- Daily news features: `13d3fc66a94c58bed5d5b49992bb63eec8e15336ed4f37bce0f29143089ab6f0`
- Regime SHAP selections: `533e0ec56d5008c95c8d77b4e98bfb510b449544ef09fc0e565f5f78a1c0c441`

## Required outputs

- Per-cell metrics, fit registry, predictions, metadata, and integrity audit.
- Seed-averaged fold predictions and metrics.
- Four-arm six-model comparison table.
- Final-arm six-model comparison table.
- LSTM-versus-PIT-CMM fold deltas and the pre-specified promotion decision.
- Runtime and parameter summaries.
- Machine-readable run metadata and an execution log suitable for the paper.

