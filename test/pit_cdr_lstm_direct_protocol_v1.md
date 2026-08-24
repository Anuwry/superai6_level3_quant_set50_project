# PIT-CDR-LSTM direct 2024--2025 retrospective protocol v1

Freeze time (UTC): `2026-08-07T00:00:00Z`

Status: **FROZEN BEFORE ACCESS TO ANY PIT-CDR-LSTM 2024--2025 RESULT**

Protocol ID: `pit-cdr-lstm-direct-2024-2025-v1`

## Evidence status

This is a frozen retrospective architecture extension. The 2024--2025 outcomes
were already used elsewhere in the project, so this experiment is not described
as an untouched confirmatory test. Its purpose is a direct, same-date comparison
with the five frozen final-arm models. No inner-fold score is used as a headline
comparison or as a gate for opening the 2024--2025 results.

## Common cohort and target

- Tests: fold 3 (2024) and fold 4 (2025).
- Expected rows: 244 in 2024 and 234 in 2025 (478 total).
- One zero-return row is retained in prediction artifacts but excluded from the
  directional metric exactly as in `binary_direction_metrics`, leaving 477
  direction-evaluable observations.
- Target: sign of `Target_Next_Close - Close_D` at a fixed 0.5 probability
  threshold. Zero-return training endpoints are excluded.
- Seeds: `42, 123, 456, 789, 2025`; seed probabilities are averaged before fold
  metrics are calculated.
- Training begins in 2019, matching the frozen integrated multimodal cohort.
- Window: 5, inherited from the frozen LSTM benchmark; it is not reselected.

## Fixed input arm

The input is the frozen `Regime-SHAP-Numeric-News` arm:

- 122 point-in-time Full-TA plus causal rolling-VMD numerical features;
- eight frozen expanding/local-NLP daily news features;
- causal hard Bull/Sideway/Bear routing labels; and
- frozen SHAP numerical selections of Bull=30, Sideway=122, Bear=80.

The model receives the ordered 130-column pool. A deterministic endpoint-regime
mask retains 38, 130, or 88 columns for Bull, Sideway, or Bear respectively.
The same mask is applied across every timestep of a sequence. Scaling is fitted
on the current fold's training data only. Test labels never enter scaling,
masking, pair construction, fitting, stopping, or threshold selection.

## Fixed architecture and losses

Inference uses one shared tower only:

1. LSTM(16);
2. Dense(8, ReLU); and
3. one direction logit.

Training uses a shared-weight twin view of that tower. There is no test-time
retrieval, gate, expert vote, or Leader. Fixed optimization settings are Adam
with learning rate 0.001, 20 epochs, batch size 32, chronological pair order,
and no early stopping.

Pointwise loss is binary cross-entropy plus a fixed 0.05 Brier component. Each
twin contributes weight 0.5. The two registered relation terms are:

- counter-direction ranking: weight 0.20, using nearest train-only state match
  with the same regime, opposite label, and at least 20 trading rows of temporal
  separation; and
- cross-state transport: weight 0.05, using the nearest train-only state match
  with the same label, a different regime, and the same minimum separation.

Matching state is frozen to seven causal regime outputs:
`composite_trend_score`, `directional_strength`, `prob_bull`, `prob_sideway`,
`prob_bear`, `routing_confidence`, and `routing_entropy`. State standardization
is train-only. Candidate absence fails closed; there is no relaxed fallback.

## Registered variants

1. `direct_lstm`: identical paired pointwise exposure, relation weights zero;
2. `random_relations`: label-compatible random partners;
3. `counter_direction_only`;
4. `cross_state_only`;
5. `pit_cdr_lstm`: both registered relations; and
6. `permuted_regime_cdr`: train-regime labels permuted for pair construction
   only, while causal input masks remain unchanged.

All variants share the exact inference architecture and parameter count. This
prevents a gain from being attributed to extra deployment capacity.

## Reporting and stopping rule

The six-model headline table contains the five frozen final-arm models plus
`pit_cdr_lstm`, evaluated only on 2024--2025. BAcc is primary; Direction Accuracy
and MCC are secondary. RMSE and MAE are not assigned to PIT-CDR-LSTM because it
is a direction classifier rather than a level regressor.

Mechanism support requires full PIT-CDR-LSTM to exceed `direct_lstm`,
`random_relations`, both single-relation ablations, and
`permuted_regime_cdr` in equal-weight mean 2024--2025 BAcc. A favourable raw
score without those ablations is reported as an unsupported performance change,
not evidence for the proposed relation mechanism. There is no tuning or second
attempt after opening these outcomes.

## Post-execution declared bookkeeping deviation

The frozen count statement above registered 478 test rows but incorrectly
expected one actual zero-return row and therefore 477 direction-evaluable rows.
Post-execution recomputation from the already frozen `Close_D` and `y_true`
columns found no actual ties: all 244 rows in 2024 and all 234 rows in 2025 are
direction-evaluable (478 total) for every model. No observation, prediction,
threshold, feature, or hyperparameter was changed. The original machine-readable
freeze remains unmodified; the discrepancy is recorded in
`outputs/pit_cdr_lstm_direct_2024_2025_v1/protocol_deviation.json`.
