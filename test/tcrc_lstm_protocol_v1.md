# TCRC-LSTM post-hoc candidate protocol v1

Frozen before any TCRC-LSTM result was generated. The architecture was motivated
after inspecting the existing five-model 2022--2025 results, so this is explicitly
a post-hoc method-development study and not a new untouched confirmatory test.

## Candidate

Turning-Point-Conditioned Residual-Correction LSTM (TCRC-LSTM): a five-day LSTM
return anchor, a causal multi-scale CNN over twenty days, a bounded residual
correction, a supervised next-day turning-point gate, and turn-conditioned
attention. The predicted Up probability is analytically tied to the sign of the
predicted return.

## Frozen development design

- Development cohort: 2019--2021 point-in-time numerical, VMD, news, and causal
  regime-context features.
- Expanding inner folds: train 2019 -> validate 2020; train 2019--2020 -> validate
  2021.
- Seeds: 42, 123, 456, 789, 2025.
- CNN window: 20 trading days; LSTM sub-window: last 5 trading days.
- Fixed training: 20 epochs, batch size 32, Adam 0.001, no shuffling.
- Primary metric: seed-averaged validation balanced accuracy derived from the
  sign of the model's reconstructed next-day return.
- Secondary metrics: direction accuracy, MCC, BCE, Brier score, RMSE, MAE,
  turning-point accuracy, parameter count, fit time, and inference time.

## Frozen ablations

1. `lstm_anchor`
2. `cnn_residual`
3. `latent_turn_gate`
4. `supervised_turn_gate`
5. `tcrc_full`

## Promotion rule

The full model proceeds to the already-observed 2022--2025 retrospective outer
benchmark only if all integrity checks pass, it beats every ablation in both
inner years, and its mean balanced accuracy exceeds the LSTM anchor by at least
0.50 percentage point. No hyperparameter adjustment is permitted after opening
the inner results under this protocol.

