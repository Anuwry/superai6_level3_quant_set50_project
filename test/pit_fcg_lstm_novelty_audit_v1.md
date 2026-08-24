# PIT-FCG-LSTM novelty and collision audit v1

Date: 2026-08-04  
Search status: completed for open primary literature; no subscription-wide
Scopus/Web of Science export was available  
Implementation decision: **proceed as a post-freeze exploratory prototype with
a narrow mechanism claim**

## Bottom line

The exact name **PIT-FCG-LSTM** and the exact proposed financial implementation
were not located in the literature searched. This is not evidence that nobody
has ever built it. Several core ingredients already exist:

- permutation/shuffle loss has been used to measure and regulate modality
  utilization;
- shuffled modality pairs have been used as synthetic mismatch augmentation;
- learned gates have been used to suppress uncertain or misaligned modalities;
- anchor-plus-delta gated multimodal models already exist; and
- regime-conditioned sentiment/price gates already exist in financial
  forecasting.

Consequently, the defensible contribution is **not** "the first shuffled
multimodal gate". The residual mechanism-level claim to test is:

> A point-in-time next-day financial classifier in which a bounded news
> correction to a numerical LSTM anchor is admitted by a sample-level gate
> trained against strictly past, regime-and-coverage-matched placebo news, with
> the aligned-versus-placebo predictive advantage used as gate supervision.

This combination was not located in the searched primary literature. Its
estimated architecture novelty is **moderate (about 3.2/5)**, lower than the
pre-audit 4.0/5 estimate. Novelty depends on the matched past-only control and
sample-level gate-calibration objective, not on the acronym.

## Closest collisions

| Work | Existing mechanism | Collision with PIT-FCG-LSTM | Remaining distinction |
|---|---|---|---|
| Singh et al. (Sensors, 2024), [Regulating Modality Utilization within Multimodal Fusion Networks](https://doi.org/10.3390/s24186054) | Shuffles one modality, compares original and shuffled loss, and adds a modality-utilization loss during training | **High conceptual collision**: loss after breaking modality-label association already regulates fusion | Uses a global utilization target, not a sample-level past-only matched control, bounded financial residual, or inference gate trained to predict aligned advantage |
| Xu et al. (Interspeech, 2025), [Mitigating Audiovisual Mismatch in Visual-Guide Audio Captioning](https://doi.org/10.21437/Interspeech.2025-1593) | Batch-wise stochastic modality shuffling creates mismatched pairs; an attention-entropy gate suppresses misleading visual cues | **High component collision**: shuffling and gating coexist in one framework | Shuffle is mismatch augmentation; gate is driven by attention entropy rather than aligned-versus-matched-placebo predictive advantage |
| He et al. (CVPR Workshops, 2026), [Delta-Gated Incremental Multi-Forward-Pass Modeling](https://openaccess.thecvf.com/content/CVPR2026W/CV4Edu/html/He_Delta-Gated_Incremental_Multi-Forward-Pass_Modeling_for_Robust_Multimodal_Classroom_Video_Understanding_CVPRW_2026_paper.html) | Treats text as the primary modality and adds audio/video increments through delta gates isolated by multiple masked forward passes | **High architecture collision**: primary anchor plus gated incremental modalities already exists | Does not construct temporal matched placebo controls or calibrate a financial news gate from their predictive advantage |
| Ma et al. (ICML, 2023), [Calibrating Multimodal Learning](https://proceedings.mlr.press/v202/ma23i.html) | Regularizes predictive confidence so removing a modality cannot increase confidence | Medium: reliability calibration through modality intervention is established | Calibrates confidence under removal, not admission of a bounded residual using a matched temporal negative control |
| Zou et al. (ACL Findings, 2026), [DEAR](https://aclanthology.org/2026.findings-acl.1517/) | Estimates reconstruction reliability and uses reliability-driven dual-stream gating under missing modalities | Medium: reliability-gated fusion is established | Reliability is reconstruction/distribution based, not aligned-versus-placebo predictive falsification |
| Haroon (arXiv, 2026), [RAML](https://arxiv.org/abs/2607.23370) | A regime-aware sigmoid gate weights social sentiment relative to technical price features for direction prediction | **High domain collision**: adaptive price/sentiment gating for financial direction exists | Does not use negative controls, past-only matched placebos, or an anchor-preserving falsification objective |
| Kummerfeld et al. (JMLR, 2024), [DANCE](https://jmlr.org/papers/v25/22-1062.html) | Searches for and validates negative controls for causal inference | Low architecture collision; establishes negative-control terminology and cautions | PIT-FCG is predictive reliability testing, not causal identification or effect estimation |
| Lin and Hu (TACL, 2023), [MissModal](https://aclanthology.org/2023.tacl-1.94/) | Enforces representation consistency between complete and missing-modality inputs | Medium robustness collision | Missing-modality consistency is different from time-alignment falsification with matched placebo news |

## Searches performed

Mechanism queries included combinations of:

- `negative control`, `placebo`, `permutation control`, `modality shuffling`,
  `shuffled modality`, `falsification gate`, and `negative-control loss`;
- `matched placebo news`, `date-shuffled news`, `time-shifted news`, `stock
  prediction`, `financial LSTM`, and `multimodal gate`; and
- primary-source searches across ISCA Archive, ACL Anthology, PMLR, CVF Open
  Access, JMLR, arXiv, and publisher pages.

The screen also checked related work on missing/noisy modalities, confidence
calibration, regime-aware price/sentiment fusion, and modality utilization.

## Frozen wording constraints

Permitted manuscript wording:

- "We propose PIT-FCG-LSTM ..."
- "To the best of our search, prior work has not combined ..."
- "The architecture adapts permutation-based modality auditing into a
  point-in-time matched-control gate for financial direction prediction."

Prohibited wording without a complete subscription-index search:

- "the first ever negative-control multimodal network";
- "no previous work uses modality shuffling and gating";
- "the placebo branch proves that news causally moves the market"; or
- novelty claims based only on the acronym or application to SET50.

## Go/no-go conditions before outer evaluation

Implementation may begin, but the architecture is not promoted merely because
it runs. Before any 2022--2025 exploratory outer evaluation, the pre-2022 inner
development evidence must show all of the following:

1. The full matched-control model exceeds a direct numerical LSTM, ordinary
   numerical-news concatenation, and a bounded-residual model without
   falsification calibration on mean inner-fold BAcc.
2. Correctly aligned validation news produces higher BAcc than matched placebo
   news under the same fitted model.
3. The median gate is not trivially always closed or always open.
4. The parameter increase over the matched direct LSTM is no more than 15%.
5. Every placebo source is inside the training partition, strictly earlier than
   its anchor, and separated by the registered temporal gap.

Failure is retained as a registered negative architecture result. It must not
trigger tuning on the already observed 2022--2025 outcomes.

## Research limitation

The audit is strong enough to prevent an obvious open-literature collision, but
it cannot certify worldwide priority. A final manuscript should still run an
author-accessible Scopus/Web of Science title/abstract query and report the
closest-work table rather than a universal first-of-kind claim.
