---
title: "A Point-in-Time Reliability Audit of Multimodal and Regime-Aware Deep Learning for Next-Day SET Index Direction Forecasting"
author:
  - "[Author name(s) to be inserted]"
date: "Draft version 1 — 11 August 2026"
bibliography: references.bib
link-citations: true
---

> **Draft-control note — remove before submission.** This manuscript reports
> the completed, frozen five-architecture audit. A proposed “Ours” architecture
> is intentionally absent. It should be added only after its design is frozen
> and an untouched evaluation is completed; the results below must not be
> changed or retrospectively relabelled to accommodate it. The author-reported
> downstream Bull/Bear/Leader extension is now included in the design, but its
> quantitative fields remain marked pending until the corresponding result
> artifact is imported and reconciled.

# Abstract

Market-direction studies often emphasize the best-performing configuration,
although apparent gains may depend on temporal leakage, repeated model
selection, market regime, class imbalance, or additional inference budget. We
conduct a point-in-time reliability audit of numerical denoising, predicted
financial-news sentiment, role-structured large-language-model (LLM)
inference, regime-aware feature selection, and neural complexity. The study
uses daily SET50 price-index data from 2012–2025, expanding pre-2022 model
selection, four outer test years (2022–2025; 962 sessions), five fixed seeds,
and five architectures: LSTM, CNN, LSTM–CNN, LSTM–Attention, and
LSTM–CNN–Attention. Labels are purged by observation date, transformations are
fitted on training data only, and balanced accuracy is the primary endpoint.
Causal rolling variational mode decomposition changed balanced accuracy by
−0.60 to +0.35 percentage points across architectures, with no conclusive
effect. Predicted news produced no multiplicity-adjusted gain over market-only
and falsification controls. In a locked 2023 sentiment benchmark, a
Bull/Bear/Leader LLM system exceeded equal-call and near-cost self-consistency
controls by 5.93 and 6.00 points, respectively, but remained below a local
character TF–IDF classifier. Regime-specific SHAP selection improved CNN by
1.46 points but produced mixed or negative effects elsewhere, without
Holm-adjusted significance. A partial-2026 stress test revealed one-sided
prediction collapse for several objectives, and frozen SET100 transfer reduced
mean balanced accuracy for all five models. These results show that plausible
enhancements can improve individual configurations without yielding reliable,
generalizable superiority across architectures, periods, and robustness
checks.

**Keywords:** point-in-time evaluation; stock-market direction; SET50;
balanced accuracy; multimodal learning; financial sentiment; variational mode
decomposition; market regime; SHAP; reliability audit

# 1. Introduction

Predicting whether a market index will rise or fall on the next trading day is
an unusually compact statement of a difficult forecasting problem. The target
is binary, but the data-generating process is non-stationary, the usable signal
is weak relative to market noise, and small implementation choices can alter
the apparent outcome [@fama1970efficient; @olorunnimbe2023survey]. Deep
recurrent networks, convolutional networks, and attention mechanisms offer
flexible representations of sequential financial data
[@hochreiter1997lstm; @vaswani2017attention; @qin2017darnn]. Representative
financial applications have reported gains from LSTM and CNN architectures
[@fischer2018lstmfinance; @hoseinzade2019cnnpred]. That flexibility also
increases the number of choices available to a researcher: feature families,
lookback windows, decomposition settings, text encoders, fusion mechanisms,
random seeds, thresholds, regimes, and post-hoc explanations. A result selected
from many such choices can look stronger than the underlying signal.

The financial forecasting literature contains increasingly elaborate hybrid
systems. Reviews document extensive use of LSTM, CNN, and hybrid deep-learning
models for price and direction prediction, while also identifying continuing
gaps in backtesting, reproducibility, and practical evaluation
[@sezer2020review; @olorunnimbe2023survey]. Earlier multimodal systems combined
text and market signals through attention, graph structure, expert-opinion
aggregation, and cross-modal interaction [@wang2020opinionsignals;
@sawhney2020stockmovement; @luo2023cmin]. More recent work models interactions
among news items or uses sentiment-attention mechanisms for index prediction
[@wang2024finin; @liu2024newsattention]. Signal-decomposition methods offer a
second route: separate a price process into frequency components and expose a
denoised representation to the predictor. Yet a component can improve
squared-price error without improving the sign of the next movement, and a
text model can classify sentiment well without adding conditional information
to an already strong market-data feature set. These distinctions are often
blurred when performance is summarized by a single best accuracy.

This study treats those distinctions as the object of analysis. Rather than
asking which configuration wins a broad model tournament, we ask whether five
common enhancement classes remain useful after applying the same temporal and
inferential pressure:

1. **Point-in-time data reliability:** Are features, labels, scalers, and
   sequence boundaries consistent with information that existed at the
   prediction time?
2. **Numerical and denoising reliability:** Does causal rolling VMD add a
   repeatable directional benefit beyond a full technical-analysis feature
   set?
3. **Multimodal and LLM reliability:** Does out-of-sample predicted news add
   forecasting information, and does a role-structured LLM system retain an
   intrinsic sentiment advantage after compute-matched controls?
4. **Regime-aware explainability reliability:** Do causal Bull/Sideway/Bear
   routing and SHAP-selected feature subsets generalize across architectures,
   or do they merely produce attractive model-specific explanations?
5. **Forward and transfer reliability:** Do conclusions survive a later
   source-contingent period and transfer from SET50 to the broader SET100 index
   without retuning?

The five registered neural architectures form a common horizontal panel for
these questions. They are not presented as five separate contributions, and
the study does not claim a new state-of-the-art model. Its contribution is a
single audit framework that connects point-in-time reconstruction,
architecture-paired ablations, negative-control falsification, compute-matched
LLM evaluation, capacity-aware regime analysis, local-explanation diagnostics,
serial-dependence sensitivity, and frozen same-exchange transfer. The resulting
evidence is intentionally mixed. Some enhancements help selected models or
intrinsic endpoints, but no primary forecasting balanced-accuracy contrast
survives the registered multiplicity controls. Reporting that outcome is
central to the study, not a failure to locate a winning specification.

# 2. Related work and reliability gap

## 2.1 Deep sequence models for financial direction

LSTM networks were designed to retain information over long sequences while
controlling vanishing gradients [@hochreiter1997lstm]. Their use in financial
prediction is motivated by delayed dependencies, while CNNs provide local
filters that can extract short-range temporal patterns. Attention adds a
learned mechanism for weighting representations across a sequence
[@vaswani2017attention; @qin2017darnn]. Large empirical studies have reported
advantages for LSTM-based financial prediction [@fischer2018lstmfinance], and
CNN-based systems have combined diverse market variables for next-day index
direction [@hoseinzade2019cnnpred]. However, the existence of a flexible
mapping does not imply a stable signal. Market dynamics change across years,
and a high-capacity hybrid can magnify optimization variance or collapse to a
majority-side decision while retaining superficially acceptable raw accuracy.
Recent review evidence likewise treats architecture choice as only one part of
a credible, backtested, reproducible financial-learning pipeline
[@olorunnimbe2023survey].

The present benchmark therefore keeps training budgets deliberately compact
and constant, repeats each outer cell across five seeds, and treats the year—not
the seed—as the temporal inferential unit. This design distinguishes stochastic
fit variability from independent evidence over time.

## 2.2 Denoising and multimodal financial information

Variational mode decomposition separates a signal into band-limited intrinsic
mode functions through a constrained variational formulation
[@dragomiretskiy2014vmd]. In financial applications, decomposition can reduce
high-frequency variation or provide additional frequency-domain features, and
recent stock-forecasting systems have paired VMD with LSTM, meta-learning, and
attention-based networks [@liu2022vml; @liu2024vmdattention]. The key causal
requirement is that decomposition at date (t) must use only observations
available through (t). A full-series decomposition, even when followed by a
temporal train/test split, can transmit future information into past
components; windowed decomposition has been proposed specifically to address
this problem [@liu2022vml]. We therefore use a rolling past-only implementation
and test VMD as a feature addition rather than assuming it is beneficial
denoising.

Financial text offers a different information channel. Prior work has used
social media, analyst or expert opinions, and news to enrich stock-movement
models [@wang2020opinionsignals; @sawhney2020stockmovement;
@luo2023cmin]. Recent systems further model interactions among news items or
sentiment-attention mechanisms for index prediction
[@wang2024finin; @liu2024newsattention]. An intrinsic sentiment score and an
incremental forecasting signal are nevertheless different estimands. Text may
be labelled accurately but arrive too late, be redundant with price adjustment,
be diluted during daily aggregation, or exhibit source and length shift. A
credible multimodal test therefore needs date-aware prediction of text labels,
equal forecasting cohorts, and controls that distinguish chronology from
generic extra dimensions.

## 2.3 LLM role systems and compute matching

Multi-agent debate and judge-style LLM systems can improve reasoning by
eliciting divergent intermediate positions before a final decision
[@du2024debate; @liang2024debate]. Self-consistency provides a strong
alternative explanation: repeated sampling itself can improve a result by
aggregating several reasoning paths [@wang2023selfconsistency]. Comparing a
three-call role system only with one call therefore confounds role structure
with inference budget. Our intrinsic benchmark uses both an equal-call
three-pass control and a four-pass near-cost control. Even then, the comparison
is system-level: all roles use the same underlying proprietary model, so the
result cannot identify abstract “debate reasoning” as the sole causal
mechanism. A recent survey of financial LLM agents similarly identifies prompt
sensitivity, real-time adaptation, and institutional deployment constraints as
open evaluation problems [@dong2025financeagents].

## 2.4 Market regimes and explanation reliability

Bull and bear phases have long been analysed as persistent market states
[@pagan2003bullbear]. Trend-following evidence motivates using returns across
several horizons [@moskowitz2012tsm], while the Average Directional Index (ADX)
provides a conventional measure of trend strength [@wilder1978concepts]. A
regime router used for next-day forecasting must be fitted without future
labels and should produce semantically valid states in both training and test
periods.

SHAP unifies feature attribution through Shapley values
[@lundberg2017shap]. LIME fits local surrogate explanations around individual
predictions [@ribeiro2016lime]. Neither method converts model association into
causal market evidence, and an explanation is useful only if it is sufficiently
faithful to the fitted model. Sanity checks are consequently part of the
explanation audit rather than an optional visualization exercise
[@adebayo2018sanity]. Finance-specific review evidence also emphasizes that
explanation methods must be assessed with domain, transparency, and adoption
constraints in view [@yeo2025finxai]. We use SHAP for a train-only selector and
LIME only as a post-selection diagnostic; LIME never determines a feature
subset.

## 2.5 Evaluation reliability

Time-series evaluation must respect order and dependence
[@bergmeir2018cv]. Financial machine-learning backtests additionally require
explicit information boundaries, pre-specified choices, and separation of
development from evaluation [@arnott2019backtest; @olorunnimbe2023survey]. We
use expanding annual folds and a label-observation purge instead of random
cross-validation. Balanced accuracy—the mean of class-wise recalls—is the
primary directional metric because raw accuracy can reward a one-sided
classifier when Up and Down frequencies differ [@brodersen2010balanced].
Families of related comparisons are adjusted by the Holm procedure
[@holm1979procedure]. Economic summaries are explicitly exploratory; the
Deflated Sharpe Ratio is retained to reveal selection and short-sample risk
rather than to certify deployment [@bailey2014dsr].

# 3. Materials and methods

## 3.1 Study design and evidence hierarchy

Figure 1 summarizes a unified five-pillar reliability audit. The numerical
market stream is available for the full historical period. The text stream
enters only on dates for which a defensible out-of-sample news estimate exists.
Causal regime labels and train-only explanation rankings act on the numerical
feature pool. Later-period and cross-index modules stress the frozen
configuration.

![**Figure 1. Five-pillar reliability-audit pipeline.** Market-data governance
and point-in-time reconstruction support all later modules. Causal rolling VMD
tests numerical denoising; out-of-sample Local NLP and a compute-matched
Bull/Bear/Leader system test multimodal claims; causal daily Bull/Sideway/Bear
routing and train-only SHAP feature sets test regime-dependent selection; and
frozen partial-2026 and SET100 evaluations test forward and same-exchange
robustness. Solid arrows denote data or forecasting flow, whereas dashed arrows
denote evaluation-only links. Two dated news routes enter the common SET50
forecasting contract; the intrinsic 2023 LLM comparison remains separate from
forecasting.](figures/figure1_reliability_audit_pipeline.png){#fig:audit-pipeline
width=100%}

The evidence tiers were fixed in the reporting specification. The corrected
numerical ablation is the primary/secondary architecture-paired experiment.
The multimodal negative controls are frozen retrospective falsification tests:
their protocols were fixed before those control outputs were opened, but the
2022–2025 market period had appeared in earlier analyses. The integrated
numeric–news–regime factorial and the regime-explainability analysis are
reported as post-hoc robustness evidence because their questions were refined
after inspecting shortcomings in earlier components. The partial-2026 module
was frozen before 2026 access but required a documented source deviation, so
it is called source-contingent forward evidence. SET100 was frozen before
model fitting and is a same-exchange transfer audit, not an independent-market
replication. These labels prevent retrospective work from being presented as
an untouched confirmatory study.

## 3.2 Market data, provenance, and temporal convention

Daily, weekly, and monthly SET50 and SET100 price-index observations were
obtained by manual browser download from publicly accessible Investing.com
historical-data pages [@investingSet50; @investingSet100]. The retained
provenance manifest records source URLs, file sizes, SHA-256 digests, row
counts, date ranges, acquisition evidence, and method. Public access is not
represented as an open-data licence; provider terms continue to apply and raw
rows are not redistributed.

Date-only observations were interpreted as Stock Exchange of Thailand sessions
in the Asia/Bangkok time zone. Features were treated as available at 17:00
local time, after the normal trading session. Weekly and monthly bars were
shifted by one completed period before forward filling to the daily calendar.
Thus a daily row never used a still-incomplete weekly or monthly bar. The study
uses provider-published price-index levels, not total-return indices, and
applies no researcher-side split, dividend, constituent, or corporate-action
adjustment.

Across the six raw index-frequency files, 40 of 42 integrity checks passed.
Two OHLC-containment warnings occurred on 11 December 2025 in both daily index
files. They were retained and disclosed rather than silently repaired; no
critical check failed. The aligned SET100 cohort contained 3,360 of 3,381
required SET50 reference dates (99.3789%). Twenty-one leading dates were
excluded because a completed monthly SET100 period was not yet available;
backfilling was prohibited.

After the 60-day technical-indicator warm-up, the effective SET50 feature
period begins on 3 May 2012. Outer evaluation ends on 18 December 2025. The
four outer SET50 test years contain 241, 243, 244, and 234 rows, respectively.

## 3.3 News data and out-of-sample sentiment construction

The labelled text source is Bilingual StockTBSA, published by the VISTEC-depa
AI Research Institute of Thailand [@uthayopas2025stocktbsa]. The local source
snapshot spans 3 January 2018 to 28 December 2023 and contains 10,295 articles,
15,949 article–ticker labels, and 12,706 valid positive, neutral, or negative
pairs. Labels marked exclude, not-stock, or ambiguous were not relabelled as
neutral. The locked 2023 intrinsic sentiment cohort contains 1,333 pairs from
738 unique articles, with class counts of 92 negative, 585 neutral, and 656
positive.

For 2024–2025, 69,824 deduplicated headlines were obtained from the official
SET news service [@setNews]. A point-in-time membership filter produced 4,619
article–symbol pairs associated with contemporaneous SET50 constituents. Six
non-overlapping membership versions cover the period, including the
March–April 2025 GULF/GULFI/INTUCH/VGI transition. These headlines are
unlabelled; their sentiment predictions are evaluated only through downstream
forecasting. An additional 68,514-row 2015–2017 corpus lacks comparable ticker
and sentiment labels and was excluded from the main benchmark. No defensible,
free, consistently dated 2012–2014 news source was identified. The paper
therefore does not claim homogeneous news coverage from 2012 to 2025.

The operational sentiment model is a balanced logistic regression over
character TF–IDF features, with the target ticker prepended to the text. It
requires no Thai word segmentation. Annual expanding predictions were made as
follows: train 2018 and predict 2019; train through 2019 and predict 2020; and
continue analogously through a 2018–2022 fit predicting 2023. For 2024–2025,
the classifier was fitted once on labelled 2018–2023 data and frozen. It was
not retrained on pseudo-labels, and no Optuna tuning was used.

The labelled source lacks reliable intraday publication times. To preserve a
conservative information boundary, a news item dated on calendar day (d)
was assigned to the first trading session strictly after (d); weekend items
were assigned to the next session. Eight daily variables were constructed:
sentiment mean and standard deviation; positive, negative, and neutral ratios;
article count; ticker-mention count; and a news-availability indicator. The
forecasting cohort begins in 2019, the first year with out-of-sample sentiment
predictions.

## 3.4 Point-in-time prediction contract

For feature date (t), all market variables use information available through
the close at (t). The price target for the regression-based historical
modules is

\[
y_t=C_{t+1},
\]

and the directional target is

\[
d_t=\mathbb{1}(C_{t+1}>C_t).
\]

Each row carries both `Date`, the feature timestamp, and `Label_Date`, the next
observed session when (C_{t+1}) becomes known. At every training/evaluation
boundary, a supervised training row is retained only when

\[
\text{Label\_Date}_{t}<\min(\text{Date in evaluation split}).
\]

This rule removes the apparently harmless last training row whose feature date
precedes the split but whose target is observed on the first evaluation date.
The purged row remains as context-only history for construction of the first
evaluation sequence: it is excluded from model fitting and scaler estimation,
transformed with the training-fitted scaler, and never contributes a target.
Reconstruction of all saved context transforms produced a maximum absolute
error of (2.22\times10^{-16}).

Actual zero-return observations remain available to price regression but are
excluded from binary direction metrics. A predicted exact zero movement is an
abstention. Direction accuracy, balanced accuracy, and Matthews correlation
coefficient (MCC) are calculated on non-tied, non-abstaining cases, and
coverage is reported. Historical regression models had full direction
coverage. The later direct classifier predicts an Up probability.

**[Figure 2 near here]**

**Figure 2. Point-in-time expanding evaluation.** Four pretest folds select
the sequence window using validation years 2018–2021. Frozen windows are then
used in four outer folds testing 2022–2025. Training rows are purged by label
observation date, scalers use supervised training rows only, and a final
pre-evaluation feature row is admitted only as sequence context.

## 3.5 Temporal splits and frozen windows

Candidate sequence lengths were 1, 3, 5, 10, and 20 trading days. Window
selection used expanding training periods 2012–2017, 2012–2018, 2012–2019,
and 2012–2020, with validation years 2018, 2019, 2020, and 2021. Balanced
accuracy was the primary selection metric, followed by direction accuracy,
RMSE, and the shorter window as tie-breakers. Selection was symmetric across
the Full-TA and Full-TA+VMD conditions so that neither side of the numerical
ablation received a privileged lookback.

The locked windows were 5 days for LSTM, 20 for CNN, 20 for LSTM–CNN, 10 for
LSTM–Attention, and 20 for LSTM–CNN–Attention. The outer folds trained through
2021, 2022, 2023, and 2024 and tested 2022, 2023, 2024, and 2025. Each outer
cell used seeds 42, 123, 456, 789, and 2025.

**Table 1. Frozen protocol and benchmark cohort.**

| Model | Window | Outer years | Folds | Seeds/fold | Primary metric | Numerical features | News features |
|---|---:|---|---:|---:|---|---:|---:|
| LSTM | 5 | 2022–2025 | 4 | 5 | Balanced accuracy | 122 | 8 |
| CNN | 20 | 2022–2025 | 4 | 5 | Balanced accuracy | 122 | 8 |
| LSTM–CNN | 20 | 2022–2025 | 4 | 5 | Balanced accuracy | 122 | 8 |
| LSTM–Attention | 10 | 2022–2025 | 4 | 5 | Balanced accuracy | 122 | 8 |
| LSTM–CNN–Attention | 20 | 2022–2025 | 4 | 5 | Balanced accuracy | 122 | 8 |

*Note:* All dates use Asia/Bangkok and a 17:00 information cutoff. News
variables enter only the relevant multimodal arms.

## 3.6 Numerical features and causal VMD

The Full-TA control contains 116 variables generated from daily and previous
completed weekly/monthly OHLCV information. The pool includes price and volume
lags; 1-, 3-, 5-, 10-, 20-, and 60-day returns; simple and weighted moving
averages; volatility, momentum, and rate-of-change measures; cross-timeframe
ratios; candlestick features; volume ratios; direction lags; and Stochastic,
RSI, MACD, Williams %R, CCI, ADX, and directional-indicator variables.

Causal rolling VMD adds six variables. At each date (t), the algorithm
decomposes only the 60 closes from (t-59) through (t). It uses five modes,
penalty (alpha=1000), dual-ascent step (	au=0), a DC mode, tolerance
(10^{-7}), and at most 500 iterations. The mode with the highest final center
frequency is treated as noise. Four retained intrinsic mode functions, the
denoised close, and the removed-mode energy ratio form the six additional
features, giving 122 numerical inputs. VMD is therefore evaluated as a causal
auxiliary representation, not as an assumed improvement.

## 3.7 Neural architectures and training

The historical numerical, news, and regime modules predict the next close and
derive direction from the sign of predicted next close minus current close.
Architectures were deliberately compact:

- **LSTM:** LSTM(16) → Dense(8, ReLU) → linear output.
- **CNN:** causal Conv1D(32, kernel 3) → global average pooling → Dense(8,
  ReLU) → linear output.
- **LSTM–CNN:** LSTM(16, sequence output) → causal Conv1D(32, kernel 3) →
  global average pooling → Dense(8, ReLU) → linear output.
- **LSTM–Attention:** LSTM(16, sequence output) → causal two-head attention
  (key dimension 8) → global average pooling → Dense(8, ReLU) → linear output.
- **LSTM–CNN–Attention:** LSTM → causal convolution → causal two-head
  attention → global average pooling → Dense(8, ReLU) → linear output.

All fits used Adam, mean squared error, 20 epochs, batch size 32, and
`shuffle=False`. Random states were reset before every cell, deterministic
TensorFlow operations were enabled, and the Keras session was cleared between
fits. Min–max scaling was fitted separately on the supervised training portion
of each fold.

## 3.8 Multimodal falsification design

Observed predicted-news features were compared with a date-matched market-only
arm. Four additional controls tested alternative explanations:

1. **News-only:** tests whether aggregated sentiment can forecast without the
   numerical market state.
2. **Shuffled news:** permutes the daily news block while preserving its
   empirical marginal distribution, testing whether observed chronology adds
   more than generic additional dimensions.
3. **Lagged news:** shifts the news features by five rows, weakening their
   contemporaneous alignment.
4. **Random feature:** adds matched noise variables, testing generic capacity
   expansion.

All arms retained the frozen architecture, window, cohort, seed, and training
budget. The registered family contained 100 model–control cells and 400 fits.
The principal reported contrasts are Observed-News minus Market-Only and
Observed-News minus Shuffled-News. News-only, lagged-news, and random-feature
controls remain in the complete evidence bundle.

## 3.9 Local NLP, Bull/Bear/Leader debate, and downstream news routes

The local relevance and sentiment pipelines were evaluated annually on
out-of-sample pairs. The LLM benchmark was separate. A class-stratified set of
60 pairs from 2022 was used once to validate a fixed prompt; prompts were then
frozen before opening the 2023 cohort. The proprietary model was
`gpt-5.6-terra`, called through structured outputs at low reasoning effort with
response storage disabled.

The role system contained a Bull worker, a Bear worker, and a Leader that
returned positive/neutral/negative probabilities and a scalar sentiment score.
All roles used the same model under different prompts. Controls were: one
single pass; three identical-prompt passes aggregated by self-consistency,
matching the three role-system calls; and four passes, whose token-accounted
cost fell within the frozen ±15% near-cost band around the Leader system.
Uncertainty was clustered by article ID because an article could generate
several ticker-labelled pairs. The primary intrinsic endpoint was accuracy,
with Macro-F1 as supporting evidence. The two registered Leader comparisons
were Holm-adjusted.

The original audited downstream run used daily features from the expanding
local classifier. A subsequent extension uses the final Leader output as the
sentiment source. For each eligible article, the Leader receives the Bull and
Bear analyses and returns class probabilities and a scalar score. Only outputs
available by the market-close cutoff may be aggregated to date \(t\); those
daily aggregates may predict \(t+1\), never the same-day close used to form the
target. The Leader arm must be compared with Market-Only and Local-NLP arms on
the identical overlapping dates, architecture, frozen window, seed set,
training budget, and target rows. Its final artifact must record the article
cutoff/timezone, daily aggregation fields, coverage, missing-news handling,
API model/version, prompt hash, runtime, token cost, and paired uncertainty.

At this draft stage, the author reports that Leader-derived features improve
SET50 performance in the additional test. The quantitative cohort, baseline,
effect size, confidence interval, and multiplicity-adjusted result are left as
explicit verification fields rather than inferred from the Local-NLP run.

## 3.10 Causal market regimes and SHAP selection

The daily regime router uses only past/current numerical features. For horizon
(h\in\{1,3,5,10,20,60\}), define a risk-adjusted return

\[
z_{t,h}=\frac{R_{t,h}}{\sigma_{t,v(h)}\sqrt{h}},
\]

where (v(h)=20) for (h\leq20) and (v(60)=60). The horizon weights are
(0.05,0.10,0.15,0.20,0.25,0.25), respectively. Trend evidence is
(T_t=\sum_h w_h z_{t,h}), trend strength is (D_t=ADX_{14,t}/100), and the
regime score is a span-3 exponentially weighted mean of (Q_t=T_tD_t). For
each fold, the symmetric boundary

\[
\theta=\operatorname{quantile}_{0.35}(|S_t|)
\]

is fitted on training data only. Bull is (S_t>\theta), Bear is
(S_t<-\theta), and Sideway is the remaining interval. The 0.35 quantile was
selected on a pre-outer development period from candidates 0.30, 0.35, and
0.40. Distance-based soft memberships with temperature 0.35 are descriptive;
hard routing is primary and the memberships are not calibrated posterior
probabilities.

All four training folds passed semantic gates: Bull had positive and Bear
negative 20-day returns, Sideway was closest to zero and had the smallest
absolute score, and every training regime exceeded the minimum share. Outer
years contained 268 Bull (27.86%), 351 Sideway (36.49%), and 343 Bear (35.65%)
rows in total.

SHAP rankings used `GradientExplainer`, 100 training-only background
sequences, at most 128 evenly spaced training ranking sequences, 200 samples,
deterministic cell seeds, and float32 tensors. Candidate feature counts were
10, 20, 30, 40, 60, 80, 100, and 122. A one-standard-error rule plus balanced-
accuracy, model-error, RMSE, and temporal-Jaccard guardrails selected the
smallest stable subset. The frozen counts were 122 globally, 30 for Bull, 122
for Sideway, and 80 for Bear. A Spearman selector used the same counts as a
size-matched non-SHAP comparator.

Seven outer arms separated routing, capacity, and selector effects:
Global-All, a three-replica Global3-All capacity control, Global-SHAP,
Global-Spearman, Regime-All, Regime-SHAP, and Regime-Spearman. Because the
global selector retained all 122 features, Global-SHAP and Global-Spearman are
identity controls. The primary selector contrast is Regime-SHAP minus
Regime-All.

LIME was applied after selection to 1,800 instance–repeat explanations. It did
not influence training or feature selection. A local surrogate with
(R^2<0.70) was labelled low fidelity. Agreement with SHAP was summarized only
for reliable LIME repeats, while every failed repeat remained in the
denominator.

## 3.11 Partial-2026 objective-alignment stress test

Before accessing 2026 observations, the protocol froze the five architectures,
windows, features, seeds, objectives, thresholds, and economic rules. The
registered Yahoo Finance source failed its overlap/availability gate. A dated
deviation then accepted the Investing.com historical interface; accordingly,
the result is source-contingent rather than a pristine source-frozen holdout.
The final extension contains 138 rows from 5 January to 30 July 2026.

Two objectives used the same backbones. **Direct** predicts Up with a sigmoid
and binary cross-entropy. **Multitask** adds a standardized next-day log-return
head to the shared backbone, with direction BCE weight 1.0 and return MSE
weight 0.25; return scaling is fitted on training data only. Threshold
candidates 0.50, 0.55, 0.60, and 0.65 were evaluated on 2019–2021 validation
data under registered coverage and economic gates. All ten model–objective
cells failed the gate and reverted to 0.50. The 2026 positive-class share was
58.70%.

An exploratory next-session open-to-close proxy evaluated long/flat and
long/short positions at 5, 10, and 20 basis points per active round trip. The
complete grid, not only the best cell, is retained in the Supplement. These
short-sample results are not evidence of a profitable or deployable strategy.

## 3.12 Frozen SET100 same-exchange transfer

SET100 transfer used the same 122 numerical features, windows, seeds, epochs,
and outer years as the matched SET50 global numerical comparator. Features,
VMD, and scaling were reconstructed within each SET100 fold. There was no
SET100 retuning, Optuna search, early stopping, or model exclusion. Five models
× four years × five seeds produced 100/100 planned fits.

SET100 news was excluded because a historically point-in-time SET100
constituent/news universe was unavailable. Adding SET50-oriented news would
create an asymmetric test. The matched SET50 values in the transfer table are
therefore the numerical Full-TA+VMD global comparator, not the later integrated
regime–news arm. Because SET50 is nested within SET100 and both share the same
exchange, macro regimes, period, and many constituents, the analysis is a
same-exchange breadth transfer, not external-market validation.
The official index profiles define the respective top-50 and top-100 liquid
large-capitalization universes and their semiannual review convention
[@set50Profile; @set100Profile].

## 3.13 Metrics and statistical inference

Balanced accuracy is the primary forecasting endpoint. Direction accuracy,
MCC, coverage, RMSE, and MAE are secondary where applicable; AUC and Brier
score are included for probabilistic 2026 objectives. Seed predictions or
metrics are averaged within each model–year cell before temporal inference.
Seeds measure optimization variability and are not treated as independent
market samples.

For the four historical outer years, paired model contrasts use exact two-sided
sign-flip tests over the four fold-level effects. The minimum attainable
two-sided p-value is therefore 0.125. Descriptive t intervals communicate
effect uncertainty but do not override the exact test. Holm adjustment controls
the registered familywise error rate. A 10-day circular moving-block bootstrap
over paired daily predictions provides a serial-dependence sensitivity
[@politis1992circular]; it cannot promote a squared-error finding into a
directional claim. Intrinsic LLM intervals use 5,000 article-cluster bootstrap
replicates, with article-cluster sign-flip sensitivity and Holm adjustment
across the two accuracy contrasts.

# 4. Results

## 4.1 Numerical denoising was architecture dependent

Table 2 reports the corrected point-in-time numerical ablation. VMD increased
mean balanced accuracy only for LSTM–CNN (+0.35 percentage points) and reduced
it for the other four architectures (−0.30 to −0.60 points). Every interval
crossed zero and the smallest exact p-value was 0.375. VMD therefore cannot be
described as a generally beneficial denoiser for next-day direction.

**Table 2. Full technical analysis versus causal rolling VMD, 2022–2025.**

| Model (window) | Full TA BAcc | +VMD BAcc | Δ BAcc (pp) | 95% CI (pp) | Exact p | Runtime Full/+VMD (s) |
|---|---:|---:|---:|---:|---:|---:|
| LSTM (5) | 53.49% | 53.16% | −0.335 | [−1.737, 1.068] | 0.500 | 7.84 / 7.71 |
| CNN (20) | 52.14% | 51.84% | −0.299 | [−1.384, 0.786] | 0.625 | 6.55 / 6.57 |
| LSTM–CNN (20) | 52.08% | 52.43% | +0.355 | [−1.612, 2.321] | 0.625 | 14.26 / 14.55 |
| LSTM–Attention (10) | 52.51% | 52.01% | −0.495 | [−3.478, 2.487] | 0.750 | 14.21 / 15.55 |
| LSTM–CNN–Attention (20) | 51.58% | 50.98% | −0.600 | [−1.980, 0.780] | 0.375 | 20.38 / 19.64 |

*Note:* BAcc is balanced accuracy. Values are means over four outer years
after within-fold seed aggregation. Runtime is mean model build, fit, and
evaluation inference time per seed–fold cell; VMD feature generation is not
included. The complete selection plus outer numerical protocol recorded
4,158.40 seconds of model fit/inference time.

The error and direction endpoints also diverged. In the corrected numerical
run, VMD reduced RMSE for LSTM and CNN but reduced their balanced accuracy.
Conversely, the small LSTM–CNN balanced-accuracy gain did not imply a universal
price-error gain. This is exactly the distinction that a direction-focused
audit must preserve.

## 4.2 Predicted news did not add reliable directional information

The observed-news effect relative to the market-only arm was negative for all
five architectures in the serial-dependence sensitivity, ranging from −0.009
points for CNN to −1.895 points for LSTM–Attention (Table 3A). None survived
Holm adjustment. Observed news also failed to beat the shuffled-news control
after adjustment. Although the unadjusted moving-block intervals for LSTM and
LSTM–CNN–Attention versus shuffled news excluded zero in the negative
direction, both Holm-adjusted p-values were 0.098. The news-only, five-row
lagged-news, and random-feature controls likewise produced no adjusted
balanced-accuracy superiority. Across the registered exact fold-level families,
zero of 30 primary balanced-accuracy contrasts survived Holm correction.

**Table 3A. Multimodal falsification using a 10-day circular moving-block bootstrap.**

| Model | Observed news − market only (pp) | 95% CI | Holm p | Observed − shuffled (pp) | 95% CI | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | −1.582 | [−5.311, 2.108] | 1.000 | −3.239 | [−5.952, −0.473] | 0.098 |
| CNN | −0.009 | [−2.114, 1.989] | 1.000 | −1.715 | [−3.679, 0.049] | 0.171 |
| LSTM–CNN | −0.172 | [−1.784, 1.392] | 1.000 | +1.020 | [−0.876, 3.076] | 0.616 |
| LSTM–Attention | −1.895 | [−4.752, 0.835] | 0.701 | +0.128 | [−1.371, 1.560] | 0.853 |
| LSTM–CNN–Attention | −1.224 | [−2.796, 0.220] | 0.481 | −1.785 | [−3.423, −0.270] | 0.098 |

*Note:* Each contrast uses 960 paired eligible daily directional rows. The
moving-block analysis is a serial-dependence sensitivity; registered
fold-level exact tests govern the primary historical inference.

The negative-control result should not be interpreted as proof that news has
no financial information. It shows that this dated, aggregated,
out-of-sample sentiment representation did not provide a robust conditional
increment beyond the numerical pipeline. The 2024–2025 official headlines
were also much shorter than the labelled headline proxies (approximately 65
versus 260 characters on average), and sentiment confidence fell from 0.756 in
2023 to about 0.495, documenting a material source shift.

## 4.3 The LLM role system improved intrinsic sentiment and was extended to SET50 forecasting

The local sentiment classifier achieved out-of-sample accuracies of 81.24%,
81.55%, 83.43%, 79.85%, and 82.82% for 2019–2023, respectively; 2023 Macro-F1
was 78.44%. The 2023 relevance classifier achieved 88.85% accuracy and 85.91%
Macro-F1.

On the locked 2023 intrinsic cohort, a one-call LLM scored 69.84% accuracy,
the three-pass self-consistency control 70.67%, the four-pass control 70.59%,
and the Bull/Bear/Leader system 76.59%. Table 3B shows that the Leader retained
an advantage of 5.93–6.00 points under equal-call and near-cost controls. Both
article-cluster intervals excluded zero and both comparisons remained
significant after Holm adjustment. However, the local TF–IDF classifier
remained 6.23 points more accurate than the Leader. Role structure is therefore
useful relative to repeated calls of the same LLM in this bounded intrinsic
task. This intrinsic comparison alone does not prove downstream value; that
question is evaluated separately by the added Leader-derived SET50 arm.

**Table 3B. Separate intrinsic LLM benchmark on the locked 2023 cohort.**

| Comparison | Pairs/articles | Control accuracy | Leader accuracy | Δ accuracy (pp) | Article-cluster 95% CI | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| Leader − self-consistency 3 | 1,333 / 738 | 70.67% | 76.59% | +5.926 | [3.491, 8.487] | 0.000040 |
| Leader − self-consistency 4 | 1,333 / 738 | 70.59% | 76.59% | +6.002 | [3.613, 8.477] | 0.000040 |

*Note:* Table 3B is the intrinsic sentiment benchmark. The downstream use of
Leader outputs belongs to the distinct common-cohort forecasting comparison in
Table 3C and must not be inferred from the intrinsic accuracy values.

Token-accounted costs were USD 4.64 for one pass, USD 13.93 for three-pass
self-consistency, USD 18.58 for four passes, and USD 19.98 for the role system.
The four-pass/Leader cost ratio was 0.930, within the frozen near-cost band.
The project-accounted intrinsic-LLM ledger was approximately USD 38.56. These
are reconstructed experimental costs, not a billing invoice.

**Table 3C. Downstream SET50 contribution of Debate-Leader news features.**

| Model and frozen window | Common test cohort | Market-only BAcc | Local-NLP BAcc | Debate-Leader BAcc | Leader − market (pp) | 95% paired CI | Holm p | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [from added test] | [dates/rows] | [pending] | [pending] | [pending] | [pending] | [pending] | [pending] | Artifact reconciliation pending |

The table will be populated only from the additional test artifact. A positive
point estimate will be described as an improvement; a reliable improvement
will be claimed only if its paired interval, temporal consistency, and the
registered multiplicity analysis support that stronger wording.

The later integrated 2×2 robustness analysis crossed global versus
regime-SHAP numerical inputs with absence versus presence of predicted news on
the 2019–2025 common cohort. The highest mean balanced accuracy was 54.07% for
LSTM–CNN with Regime-SHAP numerical inputs *without* news. Adding news within
the regime-selected arm improved mean balanced accuracy for two of five models
(LSTM +0.13 and LSTM–Attention +1.03 points) and reduced it for the other
three. No balanced-accuracy contrast survived Holm adjustment. The complete
factorial is retained as post-hoc integrated evidence in Table S5 rather than
promoted as a success claim.

## 4.4 Regime-SHAP selection helped one architecture but did not generalize

The causal router produced all three regimes in every outer year. The pooled
distribution was 27.86% Bull, 36.49% Sideway, and 35.65% Bear. Training-fitted
thresholds varied only from 0.0632 to 0.0660, and all semantic quality gates
passed. This resolves the construct-validity problem of an earlier router that
produced implausibly sparse Sideway observations.

Regime-specific SHAP selection then produced mixed effects (Table 4). CNN
improved from 53.07% to 54.53% balanced accuracy (+1.46 points), whereas
LSTM–Attention declined by 1.03 points. LSTM–CNN was nearly unchanged. No exact
contrast was significant, and no result survived Holm adjustment. The exact
four-fold test could not yield a p-value below 0.125, so effect sizes and
cross-year uncertainty are more informative than a binary significance label.

**Table 4. Primary regime-specific SHAP reduction contrast.**

| Model | Regime-All BAcc | Regime-SHAP BAcc | Δ BAcc (pp) | 95% CI (pp) | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 53.07% | 54.53% | +1.458 | [−0.093, 3.009] | 0.125 | 0.625 |
| LSTM | 51.27% | 51.17% | −0.101 | [−1.751, 1.549] | 1.000 | 1.000 |
| LSTM–Attention | 50.65% | 49.61% | −1.035 | [−2.191, 0.121] | 0.125 | 0.625 |
| LSTM–CNN | 53.18% | 53.23% | +0.053 | [−2.446, 2.553] | 1.000 | 1.000 |
| LSTM–CNN–Attention | 53.75% | 53.01% | −0.740 | [−3.102, 1.623] | 0.500 | 1.000 |

*Note:* The contrast is Regime-SHAP minus Regime-All. Seeds are averaged
within outer years before inference. SHAP values are fitted-model
attributions, not causal effects of market variables.

LIME did not provide an independent validation of SHAP. Of 1,800 registered
instance–repeat explanations, 1,293 (71.83%) had local surrogate
(R^2<0.70). Low-fidelity repeats were retained and disclosed. Because the
diagnostic failed for most repetitions, complete LIME agreement and stability
tables are placed in the Supplement and no feature-level causal narrative is
built from them.

## 4.5 Partial-2026 performance exposed majority-side collapse

The 138-row partial-2026 stress test provided weak and objective-dependent
evidence (Table 5A). Direct LSTM had the highest balanced accuracy at 54.42%
and MCC 0.088, but its direction accuracy was only 52.90%. LSTM multitask had
the highest raw direction accuracy, 59.42%, but balanced accuracy fell to
51.40%. Several CNN and LSTM–CNN–Attention cells obtained 58.70% raw accuracy,
exactly matching the Up-class share, while balanced accuracy was 50% and MCC
was zero. These cells predicted only the majority side. The apparent accuracy
was therefore not directional discrimination.

**Table 5A. Source-contingent partial-2026 forward robustness.**

| Model | Objective | Accuracy | BAcc | MCC | AUC | Brier |
|---|---|---:|---:|---:|---:|---:|
| LSTM | Direct | 52.90% | 54.42% | 0.088 | 0.533 | 0.2497 |
| LSTM | Multitask | 59.42% | 51.40% | 0.074 | 0.502 | 0.2492 |
| CNN | Direct | 58.70% | 50.00% | 0.000 | 0.496 | 0.2492 |
| CNN | Multitask | 58.70% | 50.00% | 0.000 | 0.494 | 0.2490 |
| LSTM–CNN | Direct | 49.28% | 52.11% | 0.044 | 0.485 | 0.2500 |
| LSTM–CNN | Multitask | 58.70% | 50.00% | 0.000 | 0.517 | 0.2488 |
| LSTM–Attention | Direct | 55.80% | 49.61% | −0.011 | 0.507 | 0.2491 |
| LSTM–Attention | Multitask | 57.97% | 51.20% | 0.038 | 0.512 | 0.2493 |
| LSTM–CNN–Attention | Direct | 58.70% | 50.00% | 0.000 | 0.484 | 0.2489 |
| LSTM–CNN–Attention | Multitask | 58.70% | 50.00% | 0.000 | 0.515 | 0.2488 |

*Note:* All objectives used the validation-gate fallback threshold 0.50. The
positive-class share was 58.70%. This is a source-contingent partial year, not
a full-year or live deployment result.

The complete 10-basis-point economic grid is exploratory. Its best observed
cell—direct LSTM–CNN, long/flat—had 34.31% coverage, 8.55% net cumulative
return, annualized Sharpe 1.75, maximum drawdown −3.54%, and break-even cost
27.91 basis points over 137 executable rows. Its Deflated-Sharpe probability
was only 0.441, below 0.5. This short, selected result is therefore reported as
a diagnostic sensitivity, not evidence of profitability or deployment
readiness.

## 4.6 Frozen SET100 transfer was uniformly weaker

Mean balanced accuracy declined on SET100 for all five architectures by 0.95
to 2.17 percentage points (Table 5B). CNN had the highest SET100 balanced
accuracy at 51.89%, followed by LSTM at 51.63%; LSTM–CNN–Attention was close
to chance at 50.03%. No paired cross-index difference survived Holm
adjustment. Although CNN was lower on SET100 in all four years and its
descriptive interval excluded zero, the exact p-value was 0.125 and the Holm
p-value was 0.625. It is not a statistically established degradation.

**Table 5B. Frozen SET100 same-exchange numerical transfer.**

| Model | Matched SET50 BAcc | SET100 BAcc | SET100 − SET50 (pp) | 95% CI (pp) | Holm p |
|---|---:|---:|---:|---:|---:|
| LSTM | 53.12% | 51.63% | −1.497 | [−6.139, 3.145] | 1.000 |
| CNN | 53.08% | 51.89% | −1.194 | [−2.080, −0.308] | 0.625 |
| LSTM–CNN | 52.68% | 50.51% | −2.166 | [−6.560, 2.229] | 1.000 |
| LSTM–Attention | 52.37% | 51.42% | −0.955 | [−2.987, 1.077] | 1.000 |
| LSTM–CNN–Attention | 51.10% | 50.03% | −1.070 | [−6.852, 4.712] | 1.000 |

*Note:* Matched SET50 is the frozen 122-feature global numerical comparator,
not the final post-hoc integrated regime–news arm. SET100 is a broader,
overlapping index on the same exchange and is not an external-market
replication.

**[Figure 3 near here]**

**Figure 3. Architecture-wise effects and uncertainty across audit pillars.**
Forest-style panels show paired balanced-accuracy changes for VMD,
Observed-News versus Market-Only, Regime-SHAP versus Regime-All, and SET100
minus matched SET50. The vertical zero line makes the absence of a consistent
cross-architecture direction visible; LLM intrinsic accuracy is shown in a
separate panel because it has a different endpoint and cohort.

# 5. Discussion

## 5.1 Reliability, rather than peak accuracy, is the principal result

The audit gives a consistent answer across otherwise different interventions:
local improvements occurred, but they were rarely stable across architectures,
years, endpoints, or controls. Causal VMD helped LSTM–CNN balanced accuracy by
only 0.35 points while reducing it elsewhere. Regime-SHAP helped CNN by 1.46
points but harmed LSTM–Attention by 1.03 points. The original Local-NLP news
arm produced no Holm-adjusted forecasting gain. The strongest intrinsic LLM
result survived a compute-matched comparison, and the Debate Leader was then
carried into an additional downstream SET50 test that the author reports as
positive. Its evidential strength will be classified after the exact artifact
is reconciled. Finally, partial-2026 and SET100 tests did not reveal a hidden
robust winner.

These findings identify distinct reliability boundaries. VMD fails at
cross-architecture transport; the original Local-NLP representation fails at
incremental forecasting value; the LLM role system succeeds intrinsically but
remains below the practical/local comparator; the added Leader downstream arm
tests whether richer deliberative sentiment closes that forecasting bridge;
SHAP reduction fails at universal predictive improvement; LIME fails local
fidelity; and the frozen numeric pipeline weakens under same-exchange breadth
transfer. A reliability audit is valuable precisely because it retains these
boundaries.

## 5.2 Why added neural complexity did not guarantee improvement

Next-day index direction offers a weak signal and a limited number of
independent market years. Adding convolution or attention changes the
inductive bias but does not create new information. A CNN can smooth local
patterns and appear to follow price trends, while an LSTM can minimize
next-price error. Their serial composition may nonetheless be worse if the
convolution removes small sign-relevant deviations, if the regression loss
rewards price proximity rather than directional calibration, or if the larger
network increases fit variance. Attention can also concentrate on unstable
timesteps or redundant technical transforms. These mechanisms are consistent
with the observed endpoint divergence and the weak performance of the most
complex architecture in several panels.

The partial-2026 results make the danger concrete. Raw accuracy near 58.7%
looked higher than many historical results, but balanced accuracy and MCC
revealed that several cells simply predicted the majority Up class. Any paper
that reported accuracy alone could mistakenly frame collapse as improvement.

## 5.3 Denoising and news were useful hypotheses, not universal treatments

Rolling VMD obeyed the information boundary and sometimes improved price
error, so the decomposition was not technically unsuccessful. Its failure was
more specific: better reconstruction or lower RMSE did not reliably improve
the next-day sign. The highest-frequency component may contain both noise and
short-horizon directional information, and the relevant frequency band may
vary across regimes and architectures. A fixed five-mode, one-mode-removal
rule should therefore be interpreted as a controlled treatment, not an optimal
universal decomposition.

The text results show a similar separation. The local classifier achieved
approximately 80–83% annual sentiment accuracy, demonstrating that the label
task was learnable. Yet daily aggregation, conservative next-session mapping,
limited news availability, and source shift attenuated the signal available to
the forecasting models. Shuffled-news controls further showed that adding a
high-dimensional block can change predictions without preserving useful
chronology. The appropriate conclusion is not that financial news is
irrelevant, but that sentiment accuracy alone is insufficient evidence for
incremental index-direction value. Recent multimodal work likewise reports
that news interactions, delayed diffusion, and information not captured by a
single sentiment score can matter for market prediction [@wang2024finin].

## 5.4 What the LLM comparison establishes

The role system's 5.93–6.00 point advantage over compute-matched
self-consistency is one of the clearest positive results. The cluster intervals
exclude zero, the comparison respects the number/cost of calls, and the same
locked cohort is used throughout. However, all three roles share one model,
the prompts and ordered information differ, and no component-isolation study
was performed. The intrinsic evidence establishes that the complete
Bull/Bear/Leader configuration outperformed repeated identical-prompt
inference on the locked sentiment task. It is not proof that independent
models debated or that debate is the unique causal mechanism. Forecasting
value is instead assessed by the separate Leader-derived feature arm, for
which the common-cohort result and paired uncertainty must be reported in
Table 3C.

For a cost-sensitive operational system, the local classifier remains the
low-cost comparator because it was more accurate on the intrinsic 2023 labels
and inexpensive to rerun. The Debate Leader is justified as a downstream layer
only to the extent that its added SET50 gain survives identical-cohort controls
and is commensurate with its API cost and latency. This system-level framing is
consistent with current financial-agent research, which treats cost, latency,
prompt sensitivity, and real-world constraints as part of the evaluation
target [@dong2025financeagents].

## 5.5 Regimes and explanations are model-conditional

The regime router itself passed semantic and temporal checks. Sideway was not
an afterthought: it comprised 36.49% of outer observations, and its mean
20-day return stayed nearest zero. This provides a defensible context variable
for conditional analysis. The predictive consequence, however, depended on
the backbone. CNN benefited most from the regime-specific SHAP subsets, whereas
LSTM and LSTM–Attention did not. The LSTM–CNN Spearman comparator even exceeded
its SHAP counterpart descriptively, reinforcing that a sophisticated
attribution method is not automatically a superior feature selector.

SHAP rankings describe how a fitted network distributes attribution under a
specified background; they do not establish that a feature causes a market
move. The poor LIME fidelity further warns against building a strong economic
narrative from local explanation agreement. Retaining the failed LIME rows in
the Supplement converts an apparent weakness into a reproducibility result:
the explanations themselves were subjected to an audit and did not pass it
consistently. This qualification follows current financial-XAI reviews that
distinguish post-hoc explanation from transparency and trustworthy adoption
[@yeo2025finxai].

## 5.6 Implications for SET-focused forecasting research

The SET50 focus is substantively appropriate for a study of Thailand's large-
capitalization market, and SET100 provides a useful same-market breadth test.
The latter does not solve external validity because the indices share exchange
microstructure, macroeconomic conditions, dates, and constituents. Its value
is narrower: a frozen feature/model protocol did not become stronger merely by
moving to a broader index on the same exchange. Future work seeking independent
external replication should add a separately governed ASEAN or other emerging-
market index, preferably with a reduced but fully frozen protocol.

The low absolute balanced accuracies are also economically plausible. Daily
index moves incorporate many unobserved shocks, and market efficiency limits
stable predictive structure [@fama1970efficient]. The correct response is not
to tune until a desired accuracy appears. More informative extensions are
longer prospective testing, uncertainty-aware abstention, cross-market
validation, and explicitly matched objectives and execution rules; uncertainty
quantification remains an active issue in deep financial time-series models
[@blasco2024uncertainty].

# 6. Limitations

First, the primary temporal inference uses four outer years. Exact sign-flip
tests consequently have a minimum two-sided p-value of 0.125, and confidence
intervals are wide. Daily moving-block analyses use more observations but do
not create additional independent economic regimes. Conclusions emphasize
effect stability rather than proof of equivalence.

Second, the study is centred on one exchange. SET100 broadens index membership
but is strongly related to SET50 and must not be called independent external
validation. News coverage is also heterogeneous: comparable labelled text
begins in 2018, official 2024–2025 headlines are unlabelled and shorter, and
no defensible 2012–2014 text source was included.

Third, daily news dates in the labelled source lack reliable publication
times. Shifting every item to the next trading session is conservative and
reduces leakage risk, but it may also discard same-day information that would
be available in a timestamped real-time system.

Fourth, the five neural architectures and compact training budget are a
controlled benchmark panel, not an exhaustive comparison with every current
forecasting model. No Optuna search was performed in the frozen audit. This
reduces search-induced optimism but does not prove that a different,
pre-specified architecture could not perform better.

Fifth, the regime rule is a causal semantic classifier developed after an
earlier hidden-state router failed construct-validity checks. The corrected
regime/SHAP analysis is therefore post-hoc robustness evidence. Soft regime
memberships are distance scores rather than calibrated state probabilities.

Sixth, SHAP and LIME describe the fitted model, not the causal market process.
LIME had low fidelity in most registered repeats. Feature-level explanation
claims should therefore be treated as conditional diagnostics.

Seventh, the partial-2026 forward period is short and source-contingent. A
registered data source failed and a dated alternative was used after a
protocol deviation. The economic proxy covers only 137 executable rows, uses
simplified open-to-close positions and fixed costs, and omits slippage,
liquidity, tax, and capacity. It is exploratory and cannot support a live-
trading claim.

Finally, provider-hosted market rows cannot be redistributed. The public
package supplies code, schemas, hashes, manifests, split contracts, and
aggregate results, enabling validation and fresh reconstruction but not a
one-click exact rerun without independently reacquiring the data.

# 7. Reproducibility and data availability

The repository separates persisted-artifact verification from a fresh
reconstruction. Frozen protocols record feature counts, windows, seeds,
training budgets, temporal boundaries, label-date rules, environment versions,
source and code hashes, and output manifests. Aggregate tables are generated
from authoritative CSV/JSON artifacts and paired with SHA-256 source/output
hashes. Deterministic operations, explicit seeds, finite-output gates, expected
cell counts, date-pair checks, and restricted-path/secret scans are automated.

Market-index observations were obtained from the identified publicly
accessible Investing.com historical-data pages
[@investingSet50; @investingSet100]. Access and reuse remain subject to provider
terms; public accessibility is not represented as an open-data licence. Raw and
reconstructive market rows are excluded from the public package. Researchers
may obtain the series independently and validate them against the published
schemas, date ranges, row counts, and hashes. Raw news text, private LLM
responses, response identifiers, checkpoints, and credentials are likewise
excluded. The public package includes model and evaluation code, protocols,
aggregate evidence, data contracts, non-reconstructive integrity summaries,
and exact temporal split specifications.

# 8. Conclusion

This study evaluated a full multimodal and regime-aware forecasting pipeline
without reducing the question to its best-looking model. Under a corrected
point-in-time contract, causal VMD, predicted news, regime-specific SHAP
selection, direct direction training, and additional neural complexity each
produced selected positive results, but none yielded a forecasting balanced-
accuracy advantage that was reliable across architectures and survived the
registered multiplicity controls. The LLM Bull/Bear/Leader system survived
compute-matched intrinsic sentiment controls, remained below the local
classifier on the intrinsic labels, and was subsequently used as the news
source in an additional downstream SET50 evaluation. The author reports a
positive forecasting effect; the final claim will use the exact common-cohort
estimate and uncertainty after artifact reconciliation. Partial-2026 and
SET100 tests further exposed class-collapse and transfer fragility.

The main contribution is therefore methodological and empirical: an auditable
framework for distinguishing an attractive component result from a reliable
forecasting improvement. For next-day SET direction, the evidence supports
modest, architecture-dependent predictive structure rather than a universal
high-accuracy model. Future claims should be strengthened through a frozen new
architecture, longer prospective observation, and independent-market
replication—not through additional selection on the already inspected
2022–2026 outcomes.

# Acknowledgements

[Insert only the authors' actual acknowledgements.]

# Submission fields to complete for the selected journal

- Author contributions: [to be completed from actual roles].
- Funding: [to be completed; do not state “none” unless verified].
- Competing interests: [to be completed according to the target journal].
- Corresponding author and ORCID identifiers: [to be completed].
- Code-repository DOI or archival URL: [to be added after public-package
  release].

# References
