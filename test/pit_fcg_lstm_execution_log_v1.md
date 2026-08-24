# PIT-FCG-LSTM pre-2022 development: execution log v1

Completed: 2026-08-04  
Protocol ID: `pit-fcg-lstm-inner-development-v1`  
Evidence status: post-freeze exploratory architecture development  
Overall result: **technically valid; inner promotion failed; outer evaluation not run**

## Decision in one paragraph

The exact label and exact proposed implementation of PIT-FCG-LSTM were not
located in the open primary literature searched, but the general ingredients
are crowded. Permutation-based modality-utilization loss, stochastic modality
shuffling with a gate, primary-modality delta gating, and regime-aware
price/sentiment gating already exist. The prototype therefore tested only the
narrow residual claim: whether strictly past, same-regime,
coverage-nearest-neighbour placebo news can supervise a sample-level gate for a
bounded news correction to a numerical LSTM anchor. The mechanism did not pass
the pre-registered pre-2022 screen and was not promoted to the already observed
2022--2025 outer period.

## Closest-work conclusion

The full collision audit is in `test/pit_fcg_lstm_novelty_audit_v1.md`.
Important collisions include:

- Singh et al. (Sensors, 2024), permutation/shuffle loss for regulating
  modality utilization: https://doi.org/10.3390/s24186054
- Xu et al. (Interspeech, 2025), stochastic modality shuffling plus an
  entropy-aware gate: https://doi.org/10.21437/Interspeech.2025-1593
- He et al. (CVPR Workshops, 2026), primary-modality incremental delta gating:
  https://openaccess.thecvf.com/content/CVPR2026W/CV4Edu/html/He_Delta-Gated_Incremental_Multi-Forward-Pass_Modeling_for_Robust_Multimodal_Classroom_Video_Understanding_CVPRW_2026_paper.html
- Haroon (arXiv, 2026), regime-aware sentiment/technical gating for financial
  direction: https://arxiv.org/abs/2607.23370

The safe novelty estimate after this audit is moderate, approximately 3.2/5.
The work must not claim to be the first shuffled or reliability-gated
multimodal network.

## Frozen design

- Development source: the 727 integrated market/news rows from 2019--2021.
- Inner folds: train 2019 / validate 2020; train 2019--2020 / validate 2021.
- Validation observations: 243 in 2020 and 240 in 2021.
- Common matched-control training sequences: 225 and 468, respectively.
- Inputs: 122 numerical features, eight frozen daily news features, and four
  causal regime-context variables used only by the correction gate.
- Window: five trading rows.
- Seeds: 42, 123, 456, 789, 2025.
- Optimization: Adam 0.001, 20 epochs, batch size 32, chronological
  `shuffle=False`, no early stopping, Optuna, threshold tuning, or seed removal.
- Primary metric: seed-averaged prediction Balanced Accuracy, threshold 0.5
  with ties classified Down.
- Incremental API cost: USD 0.

The protocol and input hashes were frozen before any new fit in
`test/pit_fcg_lstm_freeze_v1.json`.

## Architecture implemented

1. A 16-unit numerical LSTM and 8-unit dense head produce anchor logit
   `z_num`.
2. The aligned news window is summarized by its last row and temporal mean.
3. A four-unit residual branch proposes `r` bounded to one logit.
4. A separate four-unit branch predicts `g in [0,1]` from the numerical state,
   news summary, and causal regime context.
5. Inference uses `sigmoid(z_num + g*r)` and does not require placebo news.
6. During training only, a shared placebo branch supplies the registered rank,
   gate-calibration, and suppression losses.

The full matched model has 9,347 parameters versus 9,041 for the direct
numerical LSTM, an increase of 3.385%, inside the registered 15% ceiling.

## Headline inner results

Metrics below are computed after averaging the five seed probabilities inside
each inner fold, then averaging the two fold metrics.

| Variant | BAcc (%) | Direction accuracy (%) | MCC | BCE | Brier |
|---|---:|---:|---:|---:|---:|
| Direct numerical LSTM | **50.551** | 49.069 | **0.0213** | **0.8392** | **0.3032** |
| Numerical-news concatenation LSTM | 49.705 | 48.243 | -0.0439 | 0.8462 | 0.3081 |
| Bounded residual without falsification | 49.406 | 48.030 | -0.0152 | 0.8566 | 0.3083 |
| Random past-control FCG | 50.465 | **49.072** | 0.0094 | 0.8448 | 0.3055 |
| **Matched-control PIT-FCG-LSTM** | **50.428** | **49.072** | 0.0082 | 0.8571 | 0.3087 |

Matched PIT-FCG-LSTM was 0.123 percentage point below the direct numerical LSTM.
It exceeded concatenation and the uncalibrated bounded residual, but the frozen
rule required it to beat all three baselines.

## Inner-fold results

| Validation year | Direct LSTM BAcc (%) | Concat BAcc (%) | Bounded residual BAcc (%) | Random FCG BAcc (%) | Matched FCG BAcc (%) |
|---:|---:|---:|---:|---:|---:|
| 2020 | **50.353** | 49.153 | 49.881 | 49.786 | 49.810 |
| 2021 | 50.749 | 50.258 | 48.930 | **51.143** | 51.045 |

The matched mechanism was useful only in the later inner fold and did not show
consistent temporal advantage.

## Falsification diagnostics

| Validation year | Variant | Aligned BAcc (%) | Placebo BAcc (%) | Aligned gate median | Placebo gate median |
|---:|---|---:|---:|---:|---:|
| 2020 | Random-control FCG | 49.786 | 50.186 | 0.5033 | 0.5019 |
| 2020 | Matched-control FCG | 49.810 | **50.610** | 0.4826 | 0.4698 |
| 2021 | Random-control FCG | **51.143** | 50.059 | 0.4199 | 0.4335 |
| 2021 | Matched-control FCG | **51.045** | 47.847 | 0.4233 | 0.4363 |

The matched aligned input beat its placebo by 3.198 percentage points in 2021,
but lost by 0.800 point in 2020. The gate was non-trivial in both folds and the
mean final-epoch positive gate-target rate was 0.3687, so failure was not caused
by a gate that was always closed or always open. It was a temporal consistency
failure.

## Registered promotion decision

| Condition | Result |
|---|---|
| Beat Direct LSTM, concat LSTM, and bounded residual | **Fail** |
| Aligned news beat matched placebo in both inner folds | **Fail** |
| Gate median strictly between 0.01 and 0.99 in both folds | Pass |
| Parameter increase no more than 15% | Pass (3.385%) |
| Integrity audit passed | Pass |
| Overall inner promotion | **Fail** |

No six-model 2022--2025 table was generated. Running the outer years after this
failure would violate the pre-registered no-go rule and encourage tuning on
already observed outcomes.

## Runtime and compute

| Variant | Fits | Total fit seconds | Mean fit seconds | Total inference seconds | Parameters |
|---|---:|---:|---:|---:|---:|
| Direct numerical LSTM | 10 | 30.02 | 3.00 | 3.42 | 9,041 |
| Concatenation LSTM | 10 | 28.44 | 2.84 | 3.54 | 9,553 |
| Bounded residual | 10 | 42.56 | 4.26 | 0.27 | 9,347 |
| Random-control FCG | 10 | 47.07 | 4.71 | 0.53 | 9,347 |
| Matched-control PIT-FCG-LSTM | 10 | 47.70 | 4.77 | 0.53 | 9,347 |

Total measured cell wall time was 292.85 seconds; the outer command completed
in approximately 306 seconds including framework overhead. Execution used
TensorFlow 2.21.0 on native Windows CPU. TensorFlow emitted native-Windows GPU
unavailability, graph-attribute compatibility, and retracing warnings; all 10
cells nevertheless completed with finite outputs and passed their audits.

## Integrity and verification evidence

- 10/10 cells passed.
- 50 metric rows, 12,075 per-seed prediction rows, and 2,415 seed-averaged
  prediction rows were retained.
- 11,760 control-pair audit rows were retained.
- Past-source violations: 0.
- Matched-regime violations: 0.
- Outer-year data files read for this protocol: none.
- New PIT-FCG test suite: 28 passed.
- Core model/control code coverage before runner: 90%.
- Ruff: passed.
- Bandit: passed.

## Interpretation for the paper

PIT-FCG-LSTM should not be presented as the proposed winning architecture. Its
defensible use is a registered negative development result showing that a
mechanistically plausible, leakage-safe modality falsification gate can detect
useful aligned-news advantage in one period yet fail to transfer across adjacent
development years. This finding is consistent with the paper's reliability
audit framing, but it does not improve the final five-model benchmark and should
be placed in an exploratory architecture subsection or appendix if space
permits.

## Artifact map

- Novelty audit: `test/pit_fcg_lstm_novelty_audit_v1.md`
- Protocol: `test/pit_fcg_lstm_protocol_v1.md`
- Freeze: `test/pit_fcg_lstm_freeze_v1.json`
- Model: `models/pit_fcg_lstm.py`
- Control matching: `models/pit_fcg_controls.py`
- Development data contract: `models/pit_fcg_development.py`
- Runner: `models/pit_fcg_runner.py`
- Tests: `tests/test_pit_fcg_*.py`
- Machine-readable results:
  `outputs/pit_fcg_lstm_inner_development_v1/`
