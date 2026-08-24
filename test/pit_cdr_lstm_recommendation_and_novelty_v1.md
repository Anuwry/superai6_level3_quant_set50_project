# PIT-CDR-LSTM recommendation and novelty review v1

Date: 2026-08-07  
Status: **SUPERSEDED; DIRECT 2024--2025 RUN COMPLETED; CANDIDATE REJECTED**

Post-review note: at the user's direction, the inner promotion sequence proposed
in this recommendation was not used for comparison. The architecture was frozen
and evaluated directly on the same 2024--2025 final-arm cohort as the five
benchmarks. It failed the registered mechanism gate and has been closed. See
`test/pit_cdr_lstm_direct_execution_log_v1.md`.

Proposed name: **Point-in-Time Counter-Directional Relational LSTM
(PIT-CDR-LSTM)**

## 1. Decision

PIT-CDR-LSTM is the recommended next “Ours” candidate. It keeps one strong LSTM
encoder at inference and changes how its directional representation is learned.
During training, a shared-weight twin tower compares historical windows linked
by two training-only relation types:

1. **state-matched counter-direction pairs:** similar causal market state but
   opposite next-day direction; and
2. **cross-state same-direction pairs:** different causal market states but the
   same next-day direction.

The first relation forces the encoder to identify subtle evidence that separates
Up from Down after broad trend, volatility, regime, and news-availability
conditions are matched. The second discourages the representation from treating
regime identity as the direction label. At inference, only a single LSTM tower
and its direction head are used. There is no test-time retrieval, voting,
multi-worker gate, or learned Leader over weak branches.

## 2. Why this candidate follows from the project evidence

The attempted architecture extensions identified a repeated failure mode:

| Candidate | Main failure |
|---|---|
| PIT-CMM-LSTM | −0.435 BAcc pp versus the LSTM final arm; only 2/4 positive folds |
| PIT-DERN | −3.483 pp versus its frozen comparator; retrieval improved level error but harmed direction |
| PIT-FCG-LSTM | −0.123 pp versus direct LSTM; matched news beat placebo in only one of two inner years |
| PIT-TLDN | lost to its LSTM worker by 0.559 pp and collapsed to 50% BAcc in 2020 |
| PIT-SET50-CRIN | 51.870% mean BAcc, below all five frozen models on 2024–2025 |
| TCRC-LSTM | −0.69 pp versus its LSTM anchor; reversal gate was temporally unstable |

The common lesson is not that an additional branch is automatically helpful.
CNN, retrieval, constituent, and news branches were sometimes individually
weak, and learned gates allocated non-trivial weight to them. PIT-CDR-LSTM does
not ask a weak branch to correct the LSTM. It creates additional directional
constraints for the LSTM representation during training and adds no branch at
deployment.

## 3. Proposed architecture

Let \(x_i\) be a length-5 point-in-time window and \(y_i\in\{0,1\}\) the next-
session direction. One shared LSTM encoder produces \(z_i=f_\theta(x_i)\), and a
small direction head produces logit \(s_i\).

For every training anchor, candidate partners are restricted to rows whose
label-observation dates precede the fold boundary. Matching variables are all
known at date \(t\):

- causal Bull/Sideway/Bear label and continuous regime score;
- trailing 20- and 60-day volatility;
- multi-horizon past trend score;
- standardized current level/drawdown state; and
- for the later Leader ablation only, news availability/count and Debate-Leader
  confidence.

Partner selection is deterministic after train-only scaling:

- hard negative: nearest eligible row with the same regime and opposite label;
- transport positive: nearest eligible row with the same label and a different
  regime; and
- minimum temporal separation: 20 trading sessions, preventing near-duplicate
  overlapping windows from becoming trivial pairs.

The proposed loss is

\[
\mathcal{L}=\mathcal{L}_{\mathrm{balanced\ BCE}}
+\lambda_r\mathcal{L}_{\mathrm{relation}}
+\lambda_c\mathcal{L}_{\mathrm{calibration}},
\]

where the relation loss combines a margin term for state-matched opposite-label
pairs and a supervised contrastive term for same-label cross-state pairs.
Calibration is a small Brier component. The exact margin, temperature, and loss
weights must be selected only on pre-promotion years and then frozen.

## 4. Debate-Leader integration

The core architecture must first be evaluated with the same numerical input as
the LSTM benchmark. If it passes its numerical promotion gate, a second frozen
arm appends dated Debate-Leader daily features without adding a fusion gate:

- Leader sentiment mean;
- Leader confidence/entropy;
- Bull-versus-Bear disagreement;
- article count; and
- news-availability flag.

This gives a clean two-factor ablation:

| Arm | Relational training | Debate-Leader features |
|---|---|---|
| LSTM anchor | No | No |
| LSTM + Leader | No | Yes |
| PIT-CDR-LSTM | Yes | No |
| PIT-CDR-LSTM + Leader | Yes | Yes |

The design separates an architecture gain from a news-source gain and tests
their interaction. Local-NLP values must not be relabelled as Leader outputs.

## 5. Required ablations and falsification controls

1. direct LSTM anchor with identical window and parameter budget;
2. LSTM with ordinary random supervised-contrastive pairs;
3. state-matched counter-direction pairs only;
4. cross-state same-direction pairs only;
5. full two-relation PIT-CDR-LSTM;
6. full model with regime labels permuted inside the training period;
7. numerical-only versus Debate-Leader feature arms; and
8. inference runtime and parameter count against the LSTM anchor.

The random-pair and permuted-regime controls are essential. If they match or
beat the proposed pairing rule, the claimed state-controlled relational
mechanism is not supported even if the full model's raw BAcc looks favourable.

## 6. Evaluation sequence and no-go gate

1. Use 2018–2019 only to select the small loss-weight/margin grid.
2. Freeze architecture and hyperparameters.
3. Run the 2020 and 2021 promotion screen with seeds 42, 123, 456, 789, 2025.
4. Do not open a new candidate-specific 2022–2025 run unless all promotion
   conditions pass.
5. If promoted, compare the frozen candidate with all five models on identical
   dates and then run the same-exchange SET100 robustness arm.

Recommended promotion conditions:

- full model exceeds the direct LSTM in both 2020 and 2021;
- mean BAcc improvement is at least +0.75 percentage point;
- full model exceeds random-pair contrastive training;
- full model exceeds both single-relation ablations in mean BAcc;
- predicted-Up share remains between 20% and 80% in each validation year;
- Brier score is not worse than the anchor by more than 0.01;
- inference parameters increase by no more than 10%; and
- all point-in-time, finite-output, cardinality, and hash audits pass.

Because 2022–2025 outcomes have already been observed elsewhere in this
project, a later candidate run on those years is a frozen retrospective
extension, not an untouched confirmatory test. A prospective period or a truly
independent market would still be required for the strongest generalization
claim.

## 7. Novelty assessment

### Closest primary work

- Wu, Gattami, and Flierl (2020) already proposed conditional-mutual-
  information contrastive learning for financial direction and generated
  same/opposite-class pairs. This creates a direct collision with any broad
  claim that pairwise or contrastive financial direction learning is new:
  https://arxiv.org/abs/2002.07638
- TS2Vec performs hierarchical contrastive representation learning across
  temporal contexts, so generic time-series contrastive encoding is established:
  https://doi.org/10.1609/aaai.v36i8.20881
- TimesURL shows that time-series hard-negative construction and augmentation
  choices are themselves established research questions:
  https://doi.org/10.1609/aaai.v38i12.29299
- Supervised contrastive learning for limited-label high-frequency time series
  is also established:
  https://doi.org/10.1609/aaai.v37i6.25863

### Defensible distinction

The literature scan did not locate the exact complete mechanism of deterministic
point-in-time state matching, opposite-direction hard negatives, cross-regime
same-direction transport positives, a shared LSTM direction head, and single-
tower inference for next-day index direction. Absence from this search is not
proof of worldwide first use. A subscription-wide search is still required
before submission.

Safe novelty estimate: **3.5/5 (moderate to moderately high)**.

Permitted claim if ablations pass:

> We propose a point-in-time relational training architecture that constructs
> market-state-matched counter-direction pairs and cross-state same-direction
> pairs to learn a direction-discriminative LSTM representation without adding
> test-time experts or retrieval.

Blocked claims:

- first contrastive model for financial time series;
- first pairwise LSTM for market direction;
- causal effect identification from observational matching;
- state of the art before symmetric external benchmarks; or
- guaranteed accuracy improvement.

## 8. Expected result and stopping rule

No architecture can honestly guarantee a gain on next-day SET50 direction. A
realistic development expectation is a small BAcc change, approximately 0 to
+1.5 percentage points. The candidate is worth attempting because its mechanism
targets the observed failure—directional confounding by broad state—while
remaining compact. If it misses the frozen promotion gate, close it and retain
the five-model reliability-audit paper rather than tuning on 2022–2025.
