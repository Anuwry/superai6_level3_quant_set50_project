# Next proposed model candidate: research decision v1

Date: 2026-08-04

Status: **superseded on 2026-08-04 after target clarification; not frozen and not executed**

Revision note: the user clarified that the required output is only next-day Up/Down. The PIT-REFM proposal below is retained as an audit trail but will not be the next implementation. The replacement screen will use direct binary LSTM-family classifiers (LSTM, BiLSTM, and compact xLSTM) before considering one minimal regime-aware modification.

## Motivation from the PIT-DERN failure

PIT-DERN produced low close-level RMSE but failed on next-day direction. Standard and dual retrieval reduced RMSE further while degrading balanced accuracy. The next model must therefore treat Up/Down classification as the primary learning problem rather than obtain direction by converting a price-level regression or similarity-weighted delta.

The four-fold SET50 result also shows temporal domain shift: performance varied materially by test year and market condition. A useful new method should address both objective alignment and regime-dependent distribution shift.

## Candidate review

| Candidate | Strength | Main risk for this project | Decision |
|---|---|---|---|
| ModernTCN direct classifier | Strong modern convolutional time-series backbone | Most benefits require longer receptive fields; current effective samples are small and the strongest locked windows are short | Benchmark candidate only |
| TimeMIL/wavelet token classifier | Can localize sparse temporal patterns | Transformer/MIL complexity is high relative to 723-1,451 training sequences and its novelty would mostly be adoption | Reject as primary Ours |
| Plain direct-classification LSTM | Aligns training with direction and is cheap | Necessary baseline but insufficient architectural novelty | Required ablation |
| Fixed FIC-LSTM | Fisher-information constraint targets classification under train/test domain shift and is efficient | Direct adoption of FIC-TSC is not novel by itself | Required ablation |
| Regime-soft mixture LSTM | Uses existing causal Bull/Sideway/Bear probabilities without hard expert fragmentation | Regime gating and financial MoE are already known | Required component, not the central novelty |
| **PIT-REFM-LSTM** | Couples causal regime uncertainty to Fisher-constrained optimization and a regime-class directional margin objective | New mechanism is plausible but unproven; must be validated without reusing outer tests for selection | **Selected proposal** |

## Selected proposal

Working name: **PIT-REFM-LSTM — Point-in-Time Regime-Entropy Fisher-Margin LSTM**

The proposed model has one shared causal LSTM encoder. Its hidden representation passes through three small residual adapters associated with Bull, Sideway, and Bear. The frozen point-in-time regime probabilities softly mix the adapters; no hard routing and no separately trained regime experts are used.

The primary output is a next-day Up probability. The model does not forecast price and convert the price to direction. A small auxiliary signed-return head may be retained only if a pre-registered ablation shows that it regularizes the classifier; it cannot determine the final sign.

### Central proposed mechanism

FIC-TSC uses a Fisher-information constraint to guide time-series classifiers toward flatter minima under distribution shift. PIT-REFM-LSTM makes the Fisher budget depend on causal regime entropy:

`epsilon_batch = epsilon_stable * exp(-alpha * mean_normalized_regime_entropy)`

When Bull/Sideway/Bear probabilities are diffuse, regime uncertainty is high and the Fisher budget becomes tighter. When the regime is clear, the budget relaxes. This links an observable, point-in-time market uncertainty measure to optimization flatness instead of applying one global regularization strength to all market states.

The directional loss uses a class-sensitive margin computed within regime groups. The purpose is to avoid an aggregate objective that performs well mainly in the dominant regime or direction class. No threshold search on outer-test labels is allowed; the registered decision threshold remains 0.5 unless calibrated solely inside the development period.

## Registered component ladder for later ablation

1. Direct LSTM classifier with ordinary binary cross-entropy.
2. Direct LSTM plus fixed Fisher-information constraint.
3. Direct LSTM plus soft regime adapters, without Fisher adaptation.
4. Soft regime adapters plus a fixed Fisher constraint.
5. Soft regime adapters plus regime-entropy-adaptive Fisher constraint.
6. Full PIT-REFM-LSTM with the regime-class margin objective.

The ablation must report balanced accuracy, direction accuracy, MCC, calibration/Brier score, per-regime recall, runtime, and parameter count. The full model is credible only if it beats the direct classifier, fixed-FIC variant, and regime-adapter variant; otherwise the proposed interaction mechanism is unsupported.

## Data and evaluation boundary

The 2022-2025 SET50 outer results have already been accessed repeatedly. They must not be used to select PIT-REFM-LSTM hyperparameters and then be described as untouched confirmatory evidence.

Architecture development and all hyperparameter choices must use point-in-time inner walk-forward splits ending no later than 2021. Candidate selection should first use the numeric arm because it provides a longer pre-2022 history. Once the mechanism is frozen, the same configuration is applied without arm-specific tuning to all four existing integrated arms.

The 2022-2025 evaluation can be reported as a post-freeze exploratory architecture extension. A stronger confirmation requires an untouched target, preferably the pending SET100 same-exchange robustness dataset under a pre-registered freeze, or a future 2026 period after it becomes complete.

## Proposed promotion gates

- Primary final-arm mean BAcc improvement of at least 1 percentage point over frozen LSTM-CNN-Attention.
- Positive BAcc delta in at least three of four temporal folds.
- Full model must beat direct LSTM, fixed FIC-LSTM, and soft-regime-only ablations.
- Per-regime BAcc must not fall by more than 1 percentage point in any regime relative to the direct classifier.
- Parameter count must remain within 15% of the frozen final comparator.
- All predictions must be finite and full-coverage.
- No outer-test threshold tuning, Optuna search, seed removal, or selective fold reporting.

## Preliminary novelty position

Fisher-constrained time-series classification, regime-aware mixture models, directional financial losses, and class-sensitive margins each exist separately. The potentially novel contribution is their point-in-time coupling: causal market-regime entropy dynamically controls the Fisher-information budget while causal regime probabilities mix lightweight LSTM adapters and the primary loss targets regime-class directional margins.

This is a preliminary novelty position, not a claim of being the first method. A formal systematic literature search and exact-method comparison are required before using novelty language in the manuscript.

## Sources reviewed

- Chen et al. (2025), FIC-TSC: Learning Time Series Classification with Fisher Information Constraint, ICML/PMLR.
- Cortes et al. (2025), Balancing the Scales: A Theoretical and Algorithmic Framework for Learning from Imbalanced Data, ICML/PMLR.
- Luo and Wang (2024), ModernTCN, ICLR.
- Chen et al. (2024), TimeMIL, ICML/PMLR.
- Cui et al. (2019), Class-Balanced Loss Based on Effective Number of Samples, CVPR.
- Khosla et al. (2020), Supervised Contrastive Learning, NeurIPS.
