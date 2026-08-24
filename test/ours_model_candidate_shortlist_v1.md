# Ours-model architecture shortlist v1

Date: 2026-08-04  
Status: PIT-FCG-LSTM audited and executed on frozen pre-2022 development data  
Primary task: point-in-time next-day SET index direction, `P(Close[t+1] > Close[t])`

## Decision summary

The initially strongest candidate was **PIT-FCG-LSTM**, a falsification-calibrated model in
which news may alter a numerical LSTM anchor only when correctly dated news
outperforms distribution-matched placebo news inside training data. Its central
claim is not ordinary multimodal gating; it is that a modality must pass an
internal negative-control test before its residual is admitted to the decision.

The best numeric-only alternative is **TCU-LSTM**, which admits recurrent state
updates only when lightweight chronological-environment heads agree. The best
low-complexity alternative is **RUT-LSTM**, which separates regime transition
evidence (reset old memory) from regime uncertainty (write new memory
conservatively).

The subsequent collision audit reduced PIT-FCG-LSTM's novelty estimate to about
3.2/5 because permutation-based modality regulation, shuffled mismatch gating,
and anchor-plus-delta gating have close precedents. Its 2019--2021 inner run was
technically valid but failed the registered promotion rule (mean BAcc 50.428%
versus 50.551% for Direct LSTM; aligned news failed to beat matched placebo in
both inner folds). It was therefore not evaluated on 2022--2025. See
`test/pit_fcg_lstm_novelty_audit_v1.md` and
`test/pit_fcg_lstm_execution_log_v1.md`.

These novelty scores were preliminary research judgements, not proof of worldwide
first publication. A broader Scopus/Web of Science search should be completed
before manuscript wording uses a first-of-kind claim.

## Project constraints used in screening

- Direct binary direction must be the primary objective.
- All inputs must be available by the end of trading day `t`.
- Integrated fits can have as few as 723 training sequences.
- Inputs contain 122 numerical or 130 numerical-news variables.
- Candidate windows are short (1, 3, 5, and 10); the frozen integrated comparison
  currently uses `W=5`.
- A useful architecture should add few parameters and have clean component
  ablations.
- The model should target measured failures: weak signal, temporal shift, news
  that is not uniformly useful, regime uncertainty, and the observed mismatch
  between low regression error and directional skill.

## Ranked candidates

Scores use 1 (weak) to 5 (strong). Complexity is reversed: 1 is easiest and 5
is hardest to implement and validate.

| Rank | Candidate | Novelty | Project fit | Empirical upside | Complexity | Decision |
|---:|---|---:|---:|---:|---:|---|
| 1 | **PIT-FCG-LSTM** | **3.2** | **4.5** | **2.0 observed** | 3.0 | Executed; failed inner promotion |
| 2 | **TCU-LSTM** | 3.7 | 4.0 | 3.8 | 4.0 | Strong numeric-only alternative |
| 3 | **RUT-LSTM** | 3.2 | 4.3 | 3.4 | 2.5 | Best lightweight fallback |
| 4 | RBR-LSTM | 2.5 | 4.7 | 3.8 | 2.0 | Strong baseline, novelty too low alone |
| 5 | ODM-LSTM | 2.2 | 3.5 | 3.0 | 2.5 | Mechanism ablation only |
| 6 | VMD-SM-LSTM | 1.8 | 2.8 | 2.5 | 3.5 | Reject as Ours |

No score is a promise of higher accuracy. "Empirical upside" estimates how
directly the mechanism addresses failures already observed in this project.

## 1. PIT-FCG-LSTM — Point-in-Time Falsification-Calibrated Gated LSTM

### Mechanism in brief

1. A numerical LSTM produces a stable anchor logit `z_num`.
2. A small news branch proposes a bounded correction `delta_news`.
3. During training only, each real news vector is paired with distribution-matched
   placebo news drawn from an earlier training date in the same year/regime and
   with similar news coverage. This breaks day-specific alignment without using
   future information.
4. A falsification gate opens only when aligned news improves direction loss over
   the placebo by a registered margin.
5. The final prediction is
   `P(Up) = sigmoid(z_num + g_falsification * tanh(delta_news))`.

At inference, the model uses correctly aligned news only; placebo generation is
not required. This is a prediction reliability mechanism, not an estimator of a
causal news effect.

### Why it may be novel

Negative controls are established in causal and reliability analysis, and
reliability-gated multimodal fusion is established. The search did not locate a
financial recurrent classifier that uses a distribution-matched, time-shifted
negative control to calibrate whether a news residual is permitted to change a
point-in-time next-day direction logit. That interaction is the proposed
mechanism-level contribution.

### Why it fits this project

- It protects the longer-history numerical signal rather than allowing noisy news
  concatenation to overwrite it.
- It reuses the existing Track B relevance, coverage, and sentiment variables.
- The news branch is small and the residual is bounded, which suits limited data.
- The placebo arm is a built-in falsification test and fits the reliability-audit
  paper narrative.

### Main risks

- Placebo construction must be frozen and training-only; otherwise selection or
  temporal leakage is possible.
- A weak news signal may keep the gate near zero. That is a scientifically valid
  negative result, but it may not raise accuracy.
- The paper must call the control a predictive falsification control, not proof of
  causal news impact.

### Required ablation ladder

1. Direct numerical LSTM.
2. Numerical-news concatenation LSTM.
3. Numerical anchor plus unbounded news residual.
4. Bounded residual without falsification loss.
5. Falsification gate with randomly shuffled controls.
6. Full model with same-year/regime/coverage-matched past-only controls.

## 2. TCU-LSTM — Temporal Consensus Update LSTM

### Mechanism in brief

Split development data into chronological environments. Lightweight environment
heads propose a candidate cell update. The shared LSTM writes only the component
whose sign and magnitude are sufficiently consistent across past environments;
disputed updates are shrunk toward zero. The final direction head remains a
single direct sigmoid classifier.

One implementation uses a lower-confidence-bound update:
`u_consensus = sign(mean(u_e)) * relu(abs(mean(u_e)) - lambda * std(u_e))`.

### Novelty position

Domain generalization, gradient/moment alignment, invariant risk minimization,
and gated domain units already exist. The residual gap is applying explicit
cross-environment consensus to the innovation written into an LSTM cell for
walk-forward financial direction. This is moderately strong novelty, but not an
entirely new domain-generalization principle.

### Suitability and risk

It directly targets fold instability and can work without news. Its main risk is
fragmenting already limited development data among environment heads. Heads
should therefore be low-rank and used only to estimate update agreement, not to
become full expert models.

### Required ablations

Direct LSTM; environment-loss regularization without cell gating; mean-update
gate; sign-only consensus; full mean-minus-variance consensus.

## 3. RUT-LSTM — Regime-Uncertainty Transition LSTM

### Mechanism in brief

Use the existing causal Bull/Sideway/Bear posterior in two different ways:

- `transition_t = 1 - dot(p_t, p_[t-1])` accelerates forgetting when the regime
  distribution changes;
- normalized entropy `H(p_t)` suppresses new memory writes when regime identity
  is uncertain.

The model therefore does not equate "regime changed" with "current regime is
known." Transition controls reset; uncertainty controls trust in the new update.

### Novelty position

Regime-aware models, regime-conditioned fusion, and regime-specific state
transitions already exist. Separating transition and uncertainty and injecting
them into different LSTM gates is a narrower potentially new mechanism. Its
novelty is lower than PIT-FCG-LSTM but its implementation is cleaner.

### Suitability and risk

It adds very few parameters and avoids expert fragmentation. The risk is that
five-day windows may be too short for adaptive memory half-life to matter; it
should therefore be tested at fixed `W=5` first and not justified by long-memory
claims.

### Required ablations

Direct LSTM; regime probabilities as ordinary inputs; transition-only reset;
entropy-only write gate; full transition-reset plus uncertainty-write cell.

## 4. RBR-LSTM — Reliability-Bounded Residual LSTM

Numerical LSTM supplies the anchor and a small news branch supplies a bounded
correction weighted by news coverage, relevance, and regime confidence. This is
probably the most practical model, but reliability-driven gating, regime-aware
multimodal fusion, and anchor-residual fusion all have close 2026 precedents. It
is a strong baseline for PIT-FCG-LSTM, not a sufficiently novel Ours model alone.

## 5. ODM-LSTM — Orthogonal Direction-Magnitude LSTM

Split the LSTM representation into direction and magnitude subspaces; the
direction head sees only the direction subspace, while an auxiliary return-size
head is prevented from sending gradients into it. This addresses the project's
regression/direction mismatch, but financial direction-plus-magnitude multi-task
LSTMs and orthogonal multi-task representations already exist. Keep it as an
ablation or secondary model.

## 6. VMD-SM-LSTM — VMD Stability-Masked LSTM

Use a learned stability/noise score to admit or suppress individual VMD modes
before the LSTM. Decomposition-LSTM and mode-attention models are crowded, while
the project's VMD gains were already small. This has the lowest novelty and is
not recommended.

## Executed choice and next decision

**PIT-FCG-LSTM was frozen and prototyped first.** It provided the clearest link among
the paper's main components:

- Track A supplies the numerical anchor;
- Track B supplies news plus relevance/coverage;
- regime probabilities define safe placebo matching and optional gate context;
- the negative-control arm turns the architecture into a reliability audit rather
  than another unconstrained accuracy model.

PIT-FCG-LSTM failed its inner-development mechanism checks and was not tuned or
run on 2022--2025. Under the original decision rule, RUT-LSTM is the low-risk
alternative. TCU-LSTM should be used only if enough pre-2022 chronological
environments have adequate class counts; neither fallback has been authorized,
frozen, or executed by this update.

## Promotion gates to register before execution

- Full model must beat Direct LSTM, concatenation LSTM, and bounded-residual
  baseline on inner walk-forward mean BAcc.
- Correctly aligned news must outperform matched-placebo news; otherwise the
  falsification gate mechanism is unsupported.
- At least 3/4 outer folds must improve over the frozen final comparator for any
  promotion claim.
- Mean final-arm BAcc improvement should be at least 1 percentage point.
- Parameter count no more than 15% above the frozen comparator.
- Five seeds, threshold 0.5, complete coverage, no outer-label hyperparameter
  search or selective fold/seed removal.
- Because 2022-2025 have already been inspected, resulting evidence remains a
  post-freeze exploratory architecture extension unless a new untouched target
  is available.

## Closest literature used for collision screening

1. Zou et al. (2026), DEAR: reliability-driven dual-stream multimodal gating. https://aclanthology.org/2026.findings-acl.1517/
2. Haroon (2026), regime-aware sentiment/technical fusion for price direction. https://arxiv.org/abs/2607.23370
3. Lin and Hu (2023), MissModal: complete/missing-modality representation consistency. https://aclanthology.org/2023.tacl-1.94/
4. Zeng et al. (2022), semantic consistency under missing modalities. https://aclanthology.org/2022.emnlp-main.189/
5. Kummerfeld et al. (2024), data-driven negative control estimation. https://jmlr.org/papers/v25/22-1062.html
6. Chen et al. (2025), closed-form gradient/Hessian moment alignment for domain generalization. https://proceedings.mlr.press/v286/chen25f.html
7. Kamath et al. (2021), limitations and sampling fragility of practical invariant risk minimization. https://proceedings.mlr.press/v130/kamath21a.html
8. Chen et al. (2025), FIC-TSC for time-series classification under domain shift. https://proceedings.mlr.press/v267/chen25cq.html
9. Yoo et al. (2021), financial direction and magnitude multi-task LSTM-Forest. https://doi.org/10.1016/j.asoc.2021.108106
10. Liu et al. (2016), Coupled-LSTMs. https://aclanthology.org/D16-1176/

## Search limitations

The screen used mechanism-level queries across ACL Anthology, PMLR, OpenReview,
CVF, arXiv, general web search, and bibliographic indexing available on
2026-08-04. It did not have a complete export from subscription Scopus or Web of
Science. The safe manuscript wording is therefore "we propose" plus an explicit
closest-work comparison, not "the first ever."
