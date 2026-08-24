# PIT-TLDN pre-result development protocol v1

## Status and purpose

This protocol registers the first implementation of the Point-in-Time
Trend-Level Debate Network (PIT-TLDN) before any PIT-TLDN result is produced.
It is a post-benchmark architecture-development experiment.  It does not alter
the frozen five-model benchmark and it does not claim an untouched outer test.

The target is next-trading-day direction:

`y_t = 1[Close_(t+1) > Close_t]`.

The primary development metric is balanced accuracy at a fixed probability
threshold of 0.5. Direction accuracy, MCC, binary cross-entropy and Brier score
are secondary metrics.

## Architecture

PIT-TLDN contains two separately supervised workers and a constrained leader.

1. **CNN Trend Worker**: a W=20 causal multi-scale CNN with kernel sizes 2, 3
   and 5. It estimates the probability of positive next-day direction.
2. **LSTM Price Worker**: a W=5 LSTM with direction and standardized next-return
   heads. The next-return head supplies the price-change anchoring objective.
3. **PIT worker-specific SHAP masks**: the CNN mask ranks features against the
   direction output; the LSTM mask ranks features against the next-return
   output. Each mask is estimated only from the training prefix belonging to
   its temporal cross-fit split. The top 30 features are retained per worker.
4. **Debate Leader**: the leader receives the two worker logits, absolute
   disagreement, the two confidence scores, and four soft regime context
   variables. It returns a convex worker weight plus a bounded logit correction
   (absolute cap 0.5). The leader is fitted only to temporal out-of-fold worker
   claims; it never receives in-sample claims from the final worker fits.

The four regime context variables are `prob_bull`, `prob_sideway`, `prob_bear`
and `routing_entropy`. No hard regime routing is used by PIT-TLDN.

## Development cohort and chronology

- Source cohort: the 2019--2021 portion of fold 1 training data.
- Inner validation years: 2020 and 2021.
- Seeds: 42, 123, 456, 789 and 2025.
- Within each inner training period, three expanding temporal cross-fit splits
  begin after 55% of sequences and use a 20-row purge gap.
- Scalers are fitted on each inner training period only. The existing fold
  preparation contract ensures validation rows do not affect scaling.
- SHAP selection seed: 31415, fixed independently of training seed.
- SHAP background cap: 48; explanation cap: 64; nsamples: 100.

## Frozen optimization settings

- Adam learning rate: 0.001.
- Worker epochs: 15.
- Leader epochs: 30.
- Batch size: 32.
- Shuffle: false.
- Early stopping: false.
- Hyperparameter optimization: none.
- CNN filters per scale: 8; worker dense units: 8.
- LSTM units: 16; return loss weight: 0.25.
- Leader hidden units: 4; correction-logit cap: 0.5.

## Frozen ablations

1. `cnn_trend_shap`
2. `lstm_price_shap`
3. `simple_average_shap`
4. `leader_no_disagreement_shap`
5. `pit_tldn_all_features`
6. `pit_tldn`

The all-feature variant reuses the preliminary full-feature workers that are
required to calculate SHAP. This prevents an unequal training-budget artifact.
The no-disagreement variant uses the same leader and inputs but sets the
absolute-disagreement channel to zero in both cross-fit and validation claims.

## Promotion gate

PIT-TLDN is promoted to an outer extension only if all conditions hold:

1. integrity and leakage audits pass;
2. mean inner balanced accuracy is strictly higher than
   `simple_average_shap`, `leader_no_disagreement_shap`, and
   `pit_tldn_all_features`;
3. its balanced-accuracy delta over `simple_average_shap` is non-negative in
   both inner validation years;
4. mean absolute worker disagreement is at least 0.02, so the leader is not
   arbitrating identical claims;
5. mean leader weight is strictly between 0.05 and 0.95 in both years, so the
   leader does not collapse to a constant single-worker selector.

Failure means the candidate is retained as a documented negative architecture
experiment and is not inserted into the frozen five-model headline table.

## Evidence boundary

The registered run writes only to
`outputs/pit_tldn_inner_development_v1` on drive D. It makes no API calls and has
zero incremental API cost. Outer years 2022--2025 are excluded from this
development protocol.
