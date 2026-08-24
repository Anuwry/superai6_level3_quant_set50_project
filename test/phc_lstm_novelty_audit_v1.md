# PHC-LSTM mechanism-level novelty and suitability audit v1

Date: 2026-08-04  
Decision: **REJECT as the paper's primary proposed architecture**  
Execution status: not frozen, not implemented, not evaluated  

## Executive decision

PHC-LSTM is directly aligned with next-day Up/Down classification, but it is not
novel enough to support a strong architecture claim. A search did not find the
complete PHC equation under one existing model name; however, every mechanism
that carries the scientific idea already has a close precedent. The closest
financial collision, B4, is especially material: it creates `[UP]` and `[DOWN]`
representations, models bullish and bearish competition, and defines the target
as whether the next close is above or below the current close.

The defensible description would therefore be "a new combination of established
components." That is insufficient for the intended Ours contribution and creates
a high reviewer risk of architecture-by-renaming. No compute should be spent on
PHC-LSTM under the current design.

## Audited claim

The proposed model would send the same causal feature sequence through two
shared-weight forward LSTM states conditioned by opposite hypothesis tokens.
Each state would receive a signed difference from the other state at every time
step, and a final difference head would produce one Up logit.

The audit decomposed this into five claims rather than searching only for the
invented model name:

1. shared or tied recurrent weights;
2. paired label or hypothesis conditioning;
3. mutually coupled recurrent states;
4. bull/bear or Up/Down competition for market direction;
5. antisymmetric difference scoring.

## Search protocol

The search covered primary papers and proceedings available through 2026-08-04.
Sources included ACL Anthology, PMLR/ICML, CVF, OpenReview, arXiv for recent
preprints, and an OpenAlex mechanism-keyword collision pass. Search expressions
included variants of:

- `shared weight LSTM`, `Siamese recurrent network`;
- `coupled LSTM`, `interdependent LSTM states`, `recurrent state interaction`;
- `paired hypothesis recurrent network`, `hypothesis-conditioned LSTM`;
- `bull bear competition market prediction`, `UP DOWN token stock direction`;
- `antisymmetric neural classifier`, `pairwise difference neural ranker`;
- `dual evidence accumulator recurrent network`, `positive negative evidence LSTM`.

An absence from these searches is not proof that no unpublished or differently
named method exists. Consequently, this audit does not support a "first ever"
claim even for the exact assembly.

## Closest prior work and collision analysis

| PHC mechanism | Closest precedent | Collision | Residual difference |
|---|---|---|---|
| Shared LSTM weights | Barrow and Peskov (2017), shared-weight LSTM; earlier Siamese recurrent work | High | PHC applies the same idea to two label hypotheses rather than two sentences |
| Two recurrent states influencing one another | Liu et al. (2016), loosely and tightly Coupled-LSTMs | High | PHC uses the signed state difference and the same observed sequence |
| Explicit Up/Down representations and competition | B4 (2025 preprint) | Very high | B4 uses a sentiment-aware encoder and contrastive losses; PHC would use recurrent counter-evidence |
| Antisymmetric difference head | DirectRanker (2019) and the wider pairwise-scoring literature | High | PHC applies difference scoring to binary time-series classification |
| "Antisymmetric RNN" terminology | Moe et al. (2020) | Name collision, different mechanism | LARNN uses antisymmetric dynamics for stability, not label-hypothesis exchange |

### Decisive financial collision

B4 extracts market, `[UP]`, and `[DOWN]` representations, interprets the latter
two as bullish and bearish features, contrasts the competing forces, and uses
the exact next-day rule `Close[t+1] > Close[t]` versus `Close[t+1] < Close[t]`.
This removes the strongest potential PHC claim: PHC cannot claim to be the first
model to represent opposing Up/Down market hypotheses or let them compete for
next-day direction.

### Coupled-state collision

Coupled-LSTMs explicitly make two LSTMs interdependent at different positions.
Their loosely coupled form sends the paired hidden states forward, while the
tightly coupled form combines preceding hidden and memory states. PHC's signed
state difference is a narrower update rule, not a new class of interacting
recurrent architecture.

### Antisymmetry is not a sufficient contribution

For binary classification, an ordinary sigmoid already defines complementary
probabilities: `P(Down)=1-P(Up)`. A difference head guarantees that swapping the
two internal branches reverses the logit, but that constraint does not create a
new output semantics. Moreover, Up and Down are not necessarily exchangeable
physical states: positive market drift, asymmetric reactions, and market rules
can make an exact label symmetry an unjustified inductive bias. The constraint
would need an independent theorem or a demonstrated generalization advantage;
PHC currently has neither.

## Novelty score

Scale: 0 = standard; 5 = clearly distinct mechanism with a defensible research
claim.

| Claim | Score / 5 | Audit finding |
|---|---:|---|
| Shared/tied recurrent weights | 0.5 | Established Siamese/shared-weight design |
| Paired Up/Down hypothesis tokens | 1.0 | Hypothesis conditioning is established and B4 has explicit `[UP]`/`[DOWN]` tokens |
| Recurrent counter-evidence exchange | 1.5 | Signed difference is a small variation of coupled recurrent interaction |
| Antisymmetric difference readout | 1.0 | Established pairwise construction and partly vacuous for a binary sigmoid |
| Exact complete assembly | 2.0 | Exact equation not located, but novelty is combinational rather than conceptual |
| **Overall architecture novelty** | **1.5 / 5** | Too incremental for the intended Ours claim |

Reviewer risk if presented as a new architecture: **high**.

## Suitability for this experiment

The project has 122 numerical features or 130 integrated numerical-news
features, short candidate windows of 1, 3, 5, and 10 days, and as few as 723
training sequences in an integrated fit. The strongest frozen comparison is
already only modestly above chance, so variance control and objective alignment
matter more than extra internal branches.

| Criterion | Score / 5 | Reason |
|---|---:|---|
| Direct alignment with Up/Down target | 4.5 | It optimizes the correct binary estimand |
| Fit to sample size | 2.0 | Two state trajectories add optimization variance despite tied parameters |
| Fit to short windows | 1.5 | At W=1 recurrent counter-evidence has no temporal depth; W=3-5 remains very short |
| Identifiability/stability | 2.0 | The cell can ignore hypothesis tokens and collapse the branches, or overuse them as class-prior shortcuts |
| Expected directional uplift | 2.0 | No mechanism directly addresses the observed temporal domain shift or weak signal |
| Ablation clarity | 4.0 | Weight tying, state exchange, and readout can be removed cleanly |
| Pipeline coherence | 3.0 | It accepts all current features but does not exploit the audit framework's measured reliability variables |
| **Overall suitability** | **2.5 / 5** | Implementable, but not a good risk-adjusted experiment |

## Failure modes that would remain even if accuracy improved

1. **Branch collapse:** shared weights can learn to ignore the opposite tokens,
   making the two states nearly identical and the difference logit nearly zero.
2. **Token shortcut:** the learned tokens can dominate the weak market inputs,
   making the model encode class priors rather than day-specific evidence.
3. **Noise amplification:** recurrent subtraction can amplify unstable hidden
   differences in a low-signal daily direction task.
4. **Vacuous symmetry claim:** complementary binary probabilities already exist
   without two streams.
5. **Post-hoc evidence class:** the 2022-2025 outer results have already been
   inspected for several architectures, so a positive PHC result would remain an
   exploratory extension rather than untouched confirmation.

## Decision rule and outcome

The architecture would be approved for implementation only if it satisfied both:

- mechanism novelty at least 3/5 after closest-work comparison; and
- suitability at least 3/5 under the project's sample size and windows.

PHC-LSTM scored 1.5/5 and 2.5/5, respectively. It fails both gates and is
therefore rejected before implementation.

## What should replace it

Do not replace PHC with another renamed dual-stream model. A future candidate
must begin from a failure measured in this project and introduce one auditable,
low-parameter cell mechanism. The most coherent unresolved target is whether a
single recurrent state should reduce or defer memory updates when causal market
regime uncertainty is high and numerical/news evidence conflicts. This is only a
research direction, not an approved architecture: multimodal recurrent fusion,
conflictive multi-view learning, evidential fusion, and regime gating already
exist, so its exact mechanism requires a separate collision audit before any
claim or code.

If no candidate clears the same novelty gate, the scientifically stronger choice
is to keep the reliability-audit framework as the contribution and report the
five frozen models, rather than weaken the manuscript with a nominal Ours model.

## Primary sources

1. Liu et al. (2016), *Modelling Interaction of Sentence Pair with Coupled-LSTMs*, EMNLP. https://aclanthology.org/D16-1176/
2. Barrow and Peskov (2017), *End-to-End Shared Weight LSTM Model for Semantic Textual Similarity*, SemEval. https://aclanthology.org/S17-2026/
3. Neculoiu et al. (2016), *Learning Text Similarity with Siamese Recurrent Networks*, ACL Workshop. https://aclanthology.org/W16-1617/
4. Luo et al. (2025), *From Bias to Behavior: Learning Bull-Bear Market Dynamics with Contrastive Modeling* (preprint). https://arxiv.org/abs/2507.14182
5. Koppel et al. (2019), *Pairwise Learning to Rank by Neural Networks Revisited*. https://arxiv.org/abs/1909.02768
6. Moe et al. (2020), *Linear Antisymmetric Recurrent Neural Networks*, PMLR. https://proceedings.mlr.press/v120/moe20a.html
7. Chen et al. (2025), *FIC-TSC: Learning Time Series Classification with Fisher Information Constraint*, ICML. https://proceedings.mlr.press/v267/chen25cq.html
8. Liang et al. (2018), *Multimodal Language Analysis with Recurrent Multistage Fusion*, EMNLP. https://aclanthology.org/D18-1014/
9. Pandey and Yu (2023), *Learn to Accumulate Evidence from All Training Samples*, ICML. https://proceedings.mlr.press/v202/pandey23a.html
