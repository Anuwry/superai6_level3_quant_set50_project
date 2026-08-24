# PIT Trend-Level Debate Network: novelty review v1

Date: 2026-08-06 (Asia/Bangkok)

## Verdict

The broad recipe "CNN + LSTM + SHAP + gating/leader" is not novel by itself. A defensible contribution requires an exact methodological distinction: semantically heterogeneous trend and price experts, branch-specific point-in-time SHAP evidence, explicit disagreement/confidence variables, out-of-fold leader training, and soft regime-aware arbitration for next-day direction forecasting.

Targeted searches did not identify an exact financial forecasting architecture containing all of those elements together. This is evidence of a plausible research gap, not proof that no related work exists.

## Established components

1. CNNs have already been used for automatic feature extraction and next-day stock-index direction prediction. CNNpred applied CNNs to diverse financial variables and five US indices.
2. Hybrid CNN-LSTM and feature-selection architectures already exist in stock forecasting.
3. Mixture-of-Experts and learned routing already exist in stock prediction. MIGA uses routed experts and expert interaction; more recent adaptive recurrent MoE work uses learnable softmax gating across financial regimes.
4. SHAP-guided feature reselection already exists as a general methodology (REFRESH).
5. SHAP information has also been supplied to an MoE gate in non-financial time-series forecasting, so SHAP-guided routing cannot be claimed as a new general mechanism.
6. SHAP analysis has been combined with hybrid CNN/recurrent financial forecasting architectures.

## Potentially defensible novelty

- A CNN Trend Worker trained to produce a multi-scale directional claim and calibrated confidence rather than merely a latent feature vector.
- An LSTM Price Worker that preserves a short-horizon price/delta anchor and produces a separate directional claim.
- A disagreement-aware leader that receives the two claims, their uncertainty, disagreement magnitude, regime probabilities, and volatility state.
- Separate SHAP feature masks for trend and price workers, derived only from the corresponding training partition and frozen before outer-test inference.
- Out-of-fold expert claims used to train the leader, preventing stacking leakage.
- Soft point-in-time regime arbitration and a price-anchor residual intended to reduce the discontinuities observed under hard regime routing.
- A reliability-audit evaluation that reports when and why the experts agree, disagree, or fail, instead of claiming only higher accuracy.

## Claim boundary

Do not claim that CNNs discovering market trends is new. The local result currently supports only that the CNN output visually follows medium-term index levels; its final-arm balanced accuracy is 51.49%, the lowest among the five models, and a diagnostic cross-correlation indicates an approximately two-trading-day level lag. A dedicated trend target and trend metrics are required before stating that the CNN predicts trends well.

Recommended claim:

> We propose a point-in-time, disagreement-aware dual-expert architecture that preserves the semantic separation between multi-scale trend extraction and short-horizon price anchoring, and arbitrates their calibrated directional claims using fold-frozen, regime-specific SHAP evidence.

## Required ablations

1. CNN Trend Worker only.
2. LSTM Price Worker only.
3. Simple average of the two claims.
4. Generic learned gate without disagreement variables.
5. Disagreement-aware leader without SHAP evidence.
6. SHAP-guided fusion without out-of-fold leader training.
7. Full PIT Trend-Level Debate Network.

The full model should be promoted only if it improves the primary directional metric under the frozen outer folds and seeds, while calibration, fold stability, and economic proxies do not materially deteriorate.

## Primary sources reviewed

- CNNpred: https://doi.org/10.1016/j.eswa.2019.03.029
- MIGA: https://arxiv.org/abs/2410.02241
- GateTS: https://arxiv.org/abs/2508.17515
- REFRESH: https://doi.org/10.1145/3600211.3604706
- Adaptive recurrent expert gating for financial time series: https://doi.org/10.1016/j.procs.2026.06.366
- SHAP-MoE for day-ahead wind forecasting: https://doi.org/10.3390/en19010124
- Explainable CNN-BiLSTM-TCN with SHAP: https://doi.org/10.1016/j.rineng.2026.109865
- FTS-Text-MoE: https://arxiv.org/abs/2507.20535

## Search methodology

The review used targeted searches spanning CNN/LSTM hybrids, stock-direction CNNs, financial Mixture-of-Experts, SHAP-based feature selection, SHAP-guided routing, disagreement-aware forecasting, and dual-stream trend/level architectures. Primary paper and publisher pages were prioritized. Exact absence cannot be established through web search alone; a final manuscript should repeat the search in Scopus, Web of Science, IEEE Xplore, and Google Scholar using the finalized architecture terminology.
