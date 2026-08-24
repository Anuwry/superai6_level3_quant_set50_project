# FCTA-LSTM retrospective architecture protocol v1

Freeze time (UTC): `2026-08-07T14:55:06Z`

Status: **FROZEN BEFORE ACCESS TO ANY FCTA-LSTM OUTER RESULT**

Protocol ID: `fcta-lstm-retrospective-2024-2025-v1`

## Evidence status

This is a one-shot retrospective architecture-development screen. Outcomes
from 2024--2025 have already been inspected for earlier candidate models, so
this experiment is not an untouched confirmatory test. No result from these
years may be used to change the FCTA architecture, losses, weights, optimizer,
threshold, feature arm, window, seed set, or reporting rule after this freeze.

## Question and common cohort

The experiment asks whether attention that is trained to agree with the
prediction change caused by deleting each historical day improves both
next-day SET50 direction prediction and explanation fidelity. All variants use
the same `Regime-SHAP-Numeric-News` arm and the identical cohort used by the
five frozen benchmarks:

- test years: 2024 and 2025;
- expected rows: 244 and 234, all direction-evaluable;
- training start: 2019;
- window: 5;
- seeds: 42, 123, 456, 789, and 2025;
- 130 ordered features: 122 point-in-time Full-TA plus causal rolling-VMD
  numerical features and eight daily news features;
- endpoint-regime SHAP masks: 38 Bull, 130 Sideway, and 88 Bear features;
- train-fold-only MinMax scaling;
- primary metric: equal-year mean balanced accuracy;
- secondary predictive metrics: direction accuracy, MCC, RMSE, MAE, Brier
  score, and binary cross-entropy; and
- seed aggregation: mean prediction/probability before annual metrics.

The neural output is scaled next-day close. Direction is obtained by comparing
the predicted scaled next-day close with the known scaled current close. The
sigmoid temperature is the median non-zero absolute one-day scaled movement in
the current training fold; it is a train-only scale normalization and is not
tuned on test data.

## Frozen architecture

Each input receives one additional internal deletion-indicator channel. It is
zero during ordinary inference and one only at the removed timestep during
training counterfactual passes. All variants have the same parameters:

```text
130 features + deletion indicator
  -> LSTM(16, return_sequences=True)
  -> causal MultiHeadAttention(heads=2, key_dim=8)
  -> GlobalAveragePooling1D
  -> Dense(8, ReLU)
  -> scaled next-day close
```

The reported temporal attention is the mean source attention across heads and
query positions. For every training sequence, five shared-weight
counterfactual passes set one standardized input row to zero and activate its
deletion indicator. Their normalized prediction changes form the
counterfactual target:

```text
delta_i = abs(y_hat_full - y_hat_without_i)
q = softmax(stop_gradient(delta / mean(delta)))
```

FCTA minimizes the KL divergence from `q` to the ordinary temporal attention.
Counterfactual passes are training-only; normal inference uses one full pass.

## Registered ablation

1. `attention_control`: scaled-close MSE only;
2. `direction_consistency`: MSE plus direction BCE;
3. `mask_augmentation`: direction-consistency loss plus supervised prediction
   losses on all five deletion passes; and
4. `fcta_lstm`: mask augmentation plus counterfactual-attention fidelity loss.

Fixed loss weights are 0.10 for direction consistency, 0.25 for deletion-mask
augmentation, and 0.10 for faithfulness. All variants use Adam at 0.001,
20 epochs, batch size 32, chronological order, and no early stopping.

## Fidelity outcomes

Explanation fidelity is assessed independently of predictive accuracy using:

- Jensen-Shannon divergence between attention and counterfactual importance
  (lower is better);
- top-1 agreement between the most attended and most influential day;
- influence lift at the most attended day; and
- within-window attention/influence correlation.

## Promotion and stopping rule

FCTA-LSTM is eligible for the proposed-model position only if every condition
passes:

1. mean BAcc is strictly greater than all three internal controls;
2. BAcc is greater than `attention_control` in both 2024 and 2025;
3. mean BAcc is strictly greater than the strongest frozen model;
4. mean attention/counterfactual JSD is lower than `attention_control`; and
5. mean top-1 deletion agreement is higher than `attention_control`.

Failure closes FCTA-LSTM v1 without Optuna, threshold tuning, loss-weight
tuning, seed removal, or a second 2024--2025 run. A promotion would remain a
development finding and would require a newly arriving temporal holdout before
being described as confirmatory evidence.

## Required artifacts

Every cell saves predictions, temporal explanation weights, metrics, weights,
parameter count, fit/inference/wall runtime, package versions, and device
metadata. Aggregation saves cohort integrity, annual and mean ablations, the
six-model comparison, fidelity diagnostics, runtime summary, and a
machine-readable promotion decision. Incremental API cost is fixed at USD 0.

