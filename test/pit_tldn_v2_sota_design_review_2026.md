# Post-mortem and current-method design review for an Ours model

Date: 2026-08-06

## Locked research scope

The research scope is SET50 only. SET100 and foreign markets are excluded from
model development and validation. The target remains the next-trading-day
direction of the SET50 index. Historical stocks that were point-in-time SET50
members may be used as internal explanatory inputs; this does not expand the
market scope because their weighted aggregation defines the SET50 index itself.

## Decision on PIT-TLDN v1

PIT-TLDN v1 is rejected as a headline Ours architecture. Its registered inner
gate passed, but the complete model did not beat its strongest component: mean
inner balanced accuracy was 51.998% for PIT-TLDN versus 52.557% for the LSTM
Price Worker. The CNN Trend Worker was approximately chance-level (50.052%) and
poorly calibrated. Consequently, the central semantic premise--that the CNN
supplied reliable trend evidence for arbitration--was not established.

This is a scientific rejection, not a software or leakage failure. All ten
cells passed the integrity audit.

## What strong recent methods actually add

There is no single universal stock-forecasting SOTA because datasets, targets,
periods and trading assumptions differ. Recent strong methods repeatedly use
five ideas:

1. Cross-sectional information: MASTER alternates intra-stock and inter-stock
   aggregation and uses market information for dynamic feature selection.
2. Lightweight multi-scale mixing: StockMixer performs indicator, temporal and
   stock mixing and explicitly models stock-to-market and market-to-stock
   influence; TimeMixer separates and mixes fine/coarse temporal structure.
3. Distribution-shift adaptation: TRA routes samples among latent trading
   patterns; DoubleAdapt adapts both data and model parameters through
   incremental meta-learning.
4. Non-stationarity-aware representation: TimeBridge treats short-term
   non-stationarity differently from long-term cointegrated dependencies;
   TimeStacker models time-varying frequency structure through multilevel
   windows.
5. Conflict-aware multimodal fusion: MSGCA uses price indicators as the primary
   modality and gated cross-attention to suppress sparse or semantically
   conflicting text/graph evidence.

Adding CNN, LSTM, attention or a generic gate is therefore not enough for a
current novelty claim. Each of those mechanisms and their ordinary
combinations already have close precedents.

## Recommended high-value architecture

Working name: **PIT SET50 Constituent-Reconciled Index Network
(PIT-SET50-CRIN)**.

The proposed contribution is not another feature encoder. It changes the
information set and imposes the economic identity connecting constituent
returns to index return.

1. A shared lightweight multi-scale encoder predicts a return distribution for
   every point-in-time SET50 constituent from returns, volume, technical and
   optional stock-linked news features.
2. A permutation-invariant dynamic set/graph layer learns contemporaneous and
   lead-lag constituent relations, conditioned on soft market regime.
3. A bottom-up head constructs the implied SET50 return distribution using the
   constituent membership and official/proxy index weights known at time t.
4. A separate top-down head predicts the index-return distribution from index,
   macro, regime and market-level news features.
5. A differentiable reconciliation layer forces the top-down and bottom-up
   forecasts toward aggregation coherence rather than asking an unconstrained
   leader to choose between them.
6. Breadth and concentration auxiliary heads predict whether the expected move
   is broad-based or driven by a small number of high-weight stocks.
7. The reported direction probability is P(reconciled next-day index return >
   0), with Brier/calibration loss in addition to directional loss.

### Why this is stronger than PIT-TLDN

- The two branches have structurally different information, not merely
  different neural layers applied to the same 122 index features.
- The combination is governed by an auditable financial constraint.
- The model can explain whether a predicted index move comes from broad market
  participation, sector concentration, or a few dominant weights.
- It creates testable falsification checks: shuffled constituents, uniform
  weights, stale membership, no-reconciliation and no-cross-sectional-relation
  controls.

### Novelty boundary

Graph-based constituent/index forecasting, stock mixing and forecast
reconciliation already exist separately. A defensible novelty claim would have
to be the exact point-in-time combination of (a) dynamic constituent relations,
(b) membership/weight-aware probabilistic index reconciliation, (c)
breadth/concentration supervision, and (d) regime/news reliability for
next-day index direction. The current search found close work in each adjacent
area but did not establish that this exact combination is absent. A systematic
literature review is still required before using the word "first".

## Data feasibility

The current SET50 modeling files contain index-level histories and engineered
features but do not provide the complete panel of individual historical SET50
members, point-in-time memberships and index weights needed by PIT-SET50-CRIN.
The existing `set100_data` files are outside the newly locked scope and will not
be used. PIT-SET50-CRIN requires:

- daily OHLCV for every historical SET50 constituent;
- point-in-time entry/exit membership dates;
- point-in-time official index weights, or a fully disclosed free-float market
  capitalization proxy;
- sector mapping valid at each date;
- optional ticker-linked news coverage.

Using the current index-only files cannot test a constituent-reconciled model.

## Lower-cost fallback using current data

If constituent data cannot be obtained, the technically sensible candidate is
a small distribution-shift-aware multi-scale LSTM/mixer:

- train in return/difference space rather than raw price levels;
- use fixed W=1/3/5/10/20 branches with shared weights;
- add train-only selective normalization;
- use soft regime adapters rather than hard routing;
- use a return-distribution head and derive direction probability from it;
- fuse news with a missingness/conflict gate;
- allow strictly prequential parameter updates only after each label resolves.

This fallback is more likely to be stable than PIT-TLDN but has only moderate
novelty because TimeMixer, TimeBridge, DoubleAdapt and gated multimodal fusion
cover neighboring ideas. It should be positioned as a reliability-oriented
adaptation, not claimed as a fundamentally new SOTA architecture.

## Required promotion rule for any v2

Before accessing a comparable outer extension, freeze the following gate:

1. full model must beat every individual component and simple fusion in mean
   development BAcc;
2. BAcc improvement over the strongest component must be non-negative in every
   development year;
3. Brier score must not be worse than the strongest component;
4. the result must remain positive across five seeds;
5. all point-in-time, membership, weight, news and cross-fit audits must pass;
6. the full model must beat ablations for reconciliation, dynamic relations,
   regime context and news reliability.

## Primary sources reviewed

- MASTER, AAAI 2024: https://doi.org/10.1609/aaai.v38i1.27767
- StockMixer, AAAI 2024: https://doi.org/10.1609/aaai.v38i8.28681
- TRA, KDD 2021: https://www.microsoft.com/en-us/research/publication/learning-multiple-stock-trading-patterns-with-temporal-routing-adaptor-and-optimal-transport/
- DoubleAdapt, KDD 2023: https://doi.org/10.1145/3580305.3599315
- TimeMixer, ICLR 2024: https://openreview.net/pdf?id=7oLshfEIC2
- iTransformer, ICLR 2024: https://openreview.net/pdf?id=JePfAI8fah
- TimeBridge, ICML 2025: https://openreview.net/forum?id=pyKO0ZZ5lz
- TimeStacker, ICML 2025: https://openreview.net/forum?id=5RYSqSKz9b
- MSGCA, 2025: https://doi.org/10.1007/s40747-025-02023-3
- Probabilistic hierarchical reconciliation, UAI 2024:
  https://proceedings.mlr.press/v244/zambon24a.html
- Stock-index forecast reconciliation, Quantitative Finance 2025:
  https://doi.org/10.1080/14697688.2024.2412687
- Inter/intra graph index prediction, EAAI 2026:
  https://doi.org/10.1016/j.engappai.2025.113273
