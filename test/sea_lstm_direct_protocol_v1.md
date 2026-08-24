# SEA-LSTM direct 2024--2025 retrospective protocol v1

Freeze time (UTC): `2026-08-07T12:45:49Z`

Status: **FROZEN BEFORE ACCESS TO ANY SEA-LSTM RESULT**

Protocol ID: `sea-lstm-direct-2024-2025-v1`

## Evidence status

This is a one-shot retrospective architecture-development experiment. The
2024--2025 outcomes have already been inspected elsewhere in the project, so
this run is not an untouched confirmatory test. No value from these years may
be used to alter the architecture, optimizer, loss, threshold, feature arm,
window, seed set, or reporting rule after this freeze.

## Question and common cohort

The experiment asks whether an LSTM cell that explicitly accumulates positive
and negative temporal evidence improves next-day SET50 direction prediction.
All comparisons use the same frozen `Regime-SHAP-Numeric-News` arm, folds,
dates, targets, and current closes as the final five-model visualization:

- test years: 2024 and 2025 only;
- expected rows: 244 and 234 respectively, with all 478 direction-evaluable;
- training start: 2019;
- window: 5;
- seeds: 42, 123, 456, 789, and 2025;
- target: `Target_Next_Close > Close_D`;
- primary metric: equal-year mean balanced accuracy;
- secondary metrics: direction accuracy, MCC, Brier score, and binary
  cross-entropy;
- fixed decision threshold: probability strictly greater than 0.5; and
- seed aggregation: average probabilities before computing annual metrics.

The 130-column ordered input contains 122 point-in-time Full-TA plus causal
rolling-VMD numerical columns and eight frozen daily news columns. The frozen
endpoint-regime SHAP masks retain 38, 130, or 88 columns in Bull, Sideway, or
Bear observations. Scaling is fitted on the current training fold only.

## Frozen SEA cell

SEA means **Signed-Evidence Accumulation**. It is a single recurrent cell, not
an ensemble, router, debate system, or Leader. With 15 evidence units, the cell
uses shared forget, input, and output gates and one signed proposal:

```text
[f_t, i_t, o_t, e_t] = split(W_x x_t + W_h [U_(t-1), D_(t-1)] + b)
f_t, i_t, o_t = sigmoid(f_t), sigmoid(i_t), sigmoid(o_t)
e_t = tanh(e_t)
U_t = f_t * U_(t-1) + i_t * relu(e_t)
D_t = f_t * D_(t-1) + i_t * relu(-e_t)
h_t = [o_t * tanh(U_t), o_t * tanh(D_t)]
```

The signed-evidence head uses non-negative learned weights:

```text
logit = sum(softplus(a) * h_up) - sum(softplus(b) * h_down) + bias
probability = sigmoid(logit)
```

This makes the contribution signs structural: the up memory can only raise the
logit and the down memory can only lower it. Both memories are updated inside
one recurrent computation; no downstream model selects between them.

## Registered variants

1. `standard_lstm`: LSTM(16), Dense(8, ReLU), one sigmoid direction output;
2. `positive_memory_only`: the same SEA parameterization with negative-memory
   proposals deterministically disabled;
3. `negative_memory_only`: the same SEA parameterization with positive-memory
   proposals deterministically disabled; and
4. `sea_lstm`: both signed memories active.

All SEA variants retain identical trainable parameter counts. The SEA evidence
width of 15 is fixed to keep inference capacity close to the 9,553-parameter
standard LSTM control. Exact counts are recorded after construction.

## Optimization

- Adam learning rate 0.001;
- 20 epochs;
- batch size 32;
- chronological order (`shuffle=False`);
- no validation-based stopping;
- binary cross-entropy plus a fixed 0.05 Brier component;
- no Optuna, class weighting, resampling, threshold tuning, or post-result
  rerun.

## Promotion and stopping rule

SEA-LSTM is eligible as the paper's proposed architecture only if every rule
below passes:

1. its equal-year mean BAcc is strictly greater than all three registered
   internal controls;
2. it strictly exceeds `standard_lstm` in each of 2024 and 2025; and
3. its equal-year mean BAcc is strictly greater than the strongest of the five
   frozen models on the identical cohort.

Failure of any condition closes SEA-LSTM v1. A failed candidate is logged as a
negative architecture screen; there is no SEA-LSTM v1 tuning or second run.

## Integrity outputs

The run must save per-cell predictions, metrics, weights, parameter counts,
fit/inference/wall runtime, package versions, device metadata, cohort audit,
annual and mean ablation tables, the six-model comparison, and a machine-
readable promotion decision.
