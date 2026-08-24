**A Point-in-Time of Multimodal and Regime-Aware Deep Learning for  
Next-Day SET Index Direction Forecasting  **

Arsanchai Sukkuea<sup>1,2</sup>, Tanawat Rungwallapa<sup>3</sup>

<sup>1</sup> School of Engineering and Technology, Walailak University,
222 Thaiburi, Thasala District, Nakhon Si Thammarat 80160, Thailand.

<sup>2</sup> Research Center for Intelligent Technology and Integration,
School of Engineering and Technology, Walailak University, Nakhon Si
Thammarat 80160, Thailand.

<sup>3</sup> Faculty of Engineering, Kasetsart University, Bangkok,
10900 Thailand.

\* Correspondence: tanawat.run@ku.th

**Abstract**

Market-direction studies often emphasize the best-performing
configuration but whether reported gains persist under strict
point-in-time evaluation and robustness checks remain unclear.  
We conduct a point-in-time reliability audit of numerical denoising,
predicted financial-news sentiment, role-structured large-language-model
inference, regime-aware feature selection and neural complexity. The
study uses daily SET50 price index data from 2012-2025, expanding pre
2022 model selection, four outer test years (2022-2025; 962 sessions),
five fixed seeds and five architectures: LSTM, CNN, LSTM-CNN,
LSTM-Attention and LSTM-CNN-Attention. Labels are purged by observation
date, transformations are fitted on training data only and balanced
accuracy is the primary endpoint. Casual rolling variational mode
decomposition changed balanced accuracy by -0.60 to +0.35 percentage
points across architectures, with no conclusive effect. Predicted news
produced no multiplicity-adjusted gain over market-only and
falsification controls. In a locked 2023 sentiment benchmark, a
Bull/Bear/Leader large-language-model system exceeded equal-call and
near-cost self-consistency controls by 5.93 and 6.00 points,
respectively. Regime-specific SHAP selection improved CNN by 1.46 points
but produced mixed or negative effects elsewhere without Holm adjusted
significance. Partial-2026 testing exposed one-sided prediction collapse
while frozen SET100 transfer was uniformly weaker, indicating that the
observed gains were configuration-specific rather than generalizable.

**Keywords:** point-in-time evaluation; stock-market direction; SET50;  
balanced accuracy; multimodal learning; financial sentiment; variational
mode  
decomposition; market regime; SHAP; reliability audit

**1. Introduction**

Predicting whether a market index will rise or fall on the next trading
day is an unusually compact statement of a difficult forecasting
problem. The target is binary, but the data-generating process is
non-stationary, the usable signal is weak relative to market noise, and
small implementation choices can alter the apparent outcome (Fama, 1970;
Olorunnimbe & Viktor, 2023) Deep recurrent networks, convolutional
networks, and attention mechanisms offer flexible representations of
sequential financial data (Hochreiter & Schmidhuber, 1997; Qin et al.,
2017; Vaswani et al., 2017) Representative financial applications have
reported gains from LSTM and CNN architectures (Fischer & Krauss, 2018;
Hoseinzade & Haratizadeh, 2019) That flexibility also increases the
number of choices available to a researcher: feature families, lookback
windows, decomposition settings, text encoders, fusion mechanisms,
random seeds, thresholds, regimes, and post-hoc explanations. A result
selected from many such choices can look stronger than the underlying
signal.

The financial forecasting literature contains increasingly elaborate
hybrid systems. Reviews document extensive use of LSTM, CNN, and hybrid
deep-learning models for price and direction prediction, while also
identifying continuing gaps in backtesting, reproducibility, and
practical evaluation (Olorunnimbe & Viktor, 2023; Sezer et al., 2020)
Earlier multimodal systems combined text and market signals through
attention, graph structure, expert-opinion aggregation, and cross-modal
interaction (Luo et al., 2023; Sawhney et al., 2020; H. Wang et al.,
2020) More recent work models interactions among news items or uses
sentiment-attention mechanisms for index prediction (W.-J. Liu et al.,
2024; M. Wang et al., 2024) Signal-decomposition methods offer a second
route: separate a price process into frequency components and expose a
denoised representation to the predictor. Yet a component can improve
squared-price error without improving the sign of the next movement, and
a text model can classify sentiment well without adding conditional
information to an already strong market-data feature set. These
distinctions are often blurred when performance is summarized by a
single best accuracy.

This study treats these distinctions as the object of analysis. Rather
than selecting the best-performing configuration, we examine whether
five audit dimensions remain informative  
under a common point-in-time and inferential protocol

1.  **Point-in-time data reliability:** Are features, labels, scalers,
    and sequence boundaries consistent with the information available at
    the prediction time?

2.  **Numerical and denoising reliability:** Does causal rolling VMD
    provide a repeatable directional benefit beyond a comprehensive
    technical-analysis feature set?

3.  **Multimodal and LLM reliability:** Does out-of-sample predicted
    news add incremental forecasting information, and does a
    role-structured LLM system retain an intrinsic sentiment advantage
    over compute-matched controls?

4.  **Regime-aware explainability reliability:** Do causal
    Bull/Sideway/Bear routing and SHAP-selected feature subsets
    generalize across architectures, or are their benefits
    architecture-specific?

5.  **Forward and transfer reliability:** Do the conclusions persist in
    a later source-contingent period and under frozen transfer from
    SET50 to the broader SET100 index?

The five registered neural architectures provide a common benchmark
panel across these audit dimensions. They are not presented as separate
contributions, and the study does not claim a new state-of-the-art
forecasting model. Instead, the contribution is an integrated
reliability-audit framework combining point-in-time reconstruction,
architecture-paired ablations, negative-control falsification,
compute-matched LLM evaluation, capacity-aware regime analysis,
local-explanation diagnostics, serial-dependence sensitivity, and frozen
same-exchange transfer. The results are heterogeneous: selected
enhancements improve particular architectures or intrinsic endpoints,
but none of the completed primary forecasting contrasts survive the
registered multiplicity controls. This lack of generalizable superiority
is a substantive finding of the audit.

**2. Related works**

**2.1 Deep Sequence models for financial direction**

LSTM networks were designed to retain information over long sequences
while controlling vanishing gradients (Hochreiter & Schmidhuber, 1997)
Their use in financial prediction is motivated by delayed dependencies,
while CNNs provide local filters that can extract short-range temporal
patterns. Attention adds a learned mechanism for weighting
representations across a sequence (Qin et al., 2017; Vaswani et al.,
2017) Large empirical studies have reported advantages for LSTM-based
financial prediction (Fischer & Krauss, 2018) and CNN-based systems have
combined diverse market variables for next-day index direction
(Hoseinzade & Haratizadeh, 2019) However, the existence of a flexible
mapping does not imply a stable signal. Market dynamics change across
years, and a high-capacity hybrid can magnify optimization variance or
collapse to a majority-side decision while retaining superficially
acceptable raw accuracy. Recent review evidence likewise treats
architecture choice as only one part of a credible, backtested,
reproducible financial-learning pipeline (Olorunnimbe & Viktor, 2023)

The present benchmark therefore keeps training budgets deliberately
compact and constant, repeats each outer cell across five seeds, and
treats the year not the seed as the temporal inferential unit. This
design distinguishes stochastic fit variability from independent
evidence over time. **  
2.2 Denoising and multimodal financial information**

Variational mode decomposition separates a signal into band-limited
intrinsic mode functions through a constrained variational formulation
(Dragomiretskiy & Zosso, 2014) In financial applications, decomposition
can reduce high-frequency variation or provide additional
frequency-domain features, and recent stock-forecasting systems have
paired VMD with LSTM, meta-learning, and attention-based networks (T.
Liu et al., 2022; Y. Liu et al., 2024) The key causal requirement is
that decomposition at date (t) must use only observations available
through (t). A full-series decomposition, even when followed by a
temporal train/test split, can transmit future information into past
components; windowed decomposition has been proposed specifically to
address this problem (T. Liu et al., 2022) We therefore use a rolling
past-only implementation and test VMD as a feature addition rather than
assuming it is beneficial denoising.

Financial text offers a different information channel. Prior work has
used social media, analyst or expert opinions, and news to enrich
stock-movement models (Luo et al., 2023; Sawhney et al., 2020; H. Wang
et al., 2020) Recent systems further model interactions among news items
or sentiment-attention mechanisms for index prediction (W.-J. Liu et
al., 2024; M. Wang et al., 2024) An intrinsic sentiment score and an
incremental forecasting signal are nevertheless different estimands.
Text may be labelled accurately but arrive too late, be redundant with
price adjustment, be diluted during daily aggregation, or exhibit source
and length shift. A credible multimodal test therefore needs date-aware
prediction of text labels, equal forecasting cohorts, and controls that
distinguish chronology from generic extra dimensions.

**2.3** **LLM role systems and compute matching**

Multi-agent debate and judge-style LLM systems can improve reasoning by
eliciting divergent intermediate positions before a final decision (Du
et al., 2024; Liang et al., 2024) Self-consistency provides a strong
alternative explanation: repeated sampling itself can improve a result
by aggregating several reasoning paths (X. Wang et al., 2023) Comparing
a three-call role system only with one call therefore confounds role
structure with inference budget. Our intrinsic benchmark uses both an
equal-call three-pass control and a four-pass near-cost control. Even
then, the comparison is system-level: all roles use the same underlying
proprietary model, so the result cannot identify abstract “debate
reasoning” as the sole causal mechanism. A recent survey of financial
LLM agents similarly identifies prompt sensitivity, real-time
adaptation, and institutional deployment constraints as open evaluation
problems (Dong et al., 2025)

**2.4 Market regimes and explanation reliability**

Bull and bear phases have long been analyses as persistent market states
(Pagan & Sossounov, 2003) Trend-following evidence motivates using
returns across several horizons (Moskowitz et al., 2012) while the
Average Directional Index (ADX) provides a conventional measure of trend
strength (Wilder, 1978) A regime router used for next-day forecasting
must be fitted without future labels and should produce semantically
valid states in both training and test periods.

SHAP unifies feature attribution through Shapley values (Lundberg & Lee,
2017) LIME fits local surrogate explanations around individual
predictions (Ribeiro et al., 2016) Neither method converts model
association into causal market evidence, and an explanation is useful
only if it is sufficiently faithful to the fitted model. Sanity checks
are consequently part of the explanation audit rather than an optional
visualization exercise (Adebayo et al., 2018) Finance-specific review
evidence also emphasizes that explanation methods must be assessed with
domain, transparency, and adoption constraints in view (Yeo et al.,
2025) We use SHAP for a train-only selector and LIME only as a
post-selection diagnostic; LIME never determines a feature subset.

**2.5 Evaluation reliability**

Time-series evaluation must respect order and dependence (Bergmeir et
al., 2018) Financial machine-learning back tests additionally require
explicit information boundaries, pre-specified choices, and separation
of development from evaluation (Arnott et al., 2019; Olorunnimbe &
Viktor, 2023) We use expanding annual folds and a label-observation
purge instead of random cross-validation. Balanced accuracy the mean of
class-wise recalls is the primary directional metric because raw
accuracy can reward a one-sided classifier when Up and Down frequencies
differ (Brodersen et al., 2010) Families of related comparisons are
adjusted by the Holm procedure (Holm, 1979) Economic summaries are
explicitly exploratory; the Deflated Sharpe Ratio is retained to reveal
selection and short-sample risk rather than to certify deployment
(Bailey & López de Prado, 2014)

**3 Materials and Methods**

**3.1 Study design and evidence hierarchy**

Figure 1 summarizes a unified five-pillar reliability audit. The
numerical market stream is available for the full historical period. The
text stream enters only on dates for which a defensible out-of-sample
news estimate exists. Causal regime labels and train-only explanation
rankings act on the numerical feature pool. Later period and cross-index
modules stress the frozen configuration

**<span class="mark">\[ Figure here later \]</span>**

**Figure 1, Five-pillar reliability-audit pipeline.** Market-data
governance and point-in-time reconstruction support all later modules.
Causal VMD tests numerical denoising; out-of-sample Local NLP and a
compute-matched Bull/Bear/Leader system test multimodal claims; daily
Bull/Sideway/Bear routing and SHAP test regime-dependent selection;
partial-2026 and SET100 evaluate temporal and same-exchange robustness.
The diagram must show two dated news routes into SET50 forecasting: the
original expanding Local-NLP route and the subsequent Debate-Leader
route. The intrinsic 2023 comparison is shown as a separate evaluation
box connected to the same Leader system.

The evidence tiers were fixed in the reporting specification; The
corrected numerical ablation is the primary/secondary
architecture-paired experiment. The multimodal negative controls are
frozen retrospective falsification tests: their protocols were fixed
before those control outputs were opened but the 2022–2025 market period
had appeared in earlier analyses. The integrated numeric–news–regime
factorial and the regime-explainability analysis are reported as
post-hoc robustness evidence because their questions were refined after
inspecting shortcomings in earlier components. The partial-2026 module
was frozen before 2026 access but required a documented source
deviation, so it is called source-contingent forward evidence. SET100
was frozen before model fitting and is a same-exchange transfer audit,
not an independent-market replication. These labels prevent
retrospective work from being presented as an untouched confirmatory
study.

**3.2 Market data, provenance, and temporal convention**

Daily, weekly, and monthly SET50 and SET100 price-index observations
were obtained by manual browser download from publicly accessible
Investing.com historical-data pages (Investing.com, n.d.-a, n.d.-b) The
retained provenance manifest records source URLs, file sizes, SHA-256
digests, row counts, date ranges, acquisition evidence, and method.
Public access is not represented as an open-data license; provider terms
continue to apply and raw rows are not redistributed.

Date-only observations were interpreted as Stock Exchange of Thailand
sessions in the Asia/Bangkok time zone. Features were treated as
available at 17:00 local time, after the normal trading session. Weekly
and monthly bars were shifted by one completed period before filing
forward to the daily calendar. Thus, a daily row never used a
still-incomplete weekly or monthly bar. The study uses
provider-published price-index levels, not total-return indices, and
applies no researcher-side split, dividend, constituent, or
corporate-action adjustment**.**

Across the six raw index-frequency files, 40 of 42 integrity checks
passed. Two OHLC-containment warnings occurred on 11 December 2025 in
both daily index files. They were retained and disclosed rather than
silently repaired; no critical check failed. The aligned SET100 cohort
contained 3,360 of 3,381 required SET50 reference dates (99.3789%).
Twenty-one leading dates were excluded because a completed monthly
SET100 period was not yet available; backfilling was prohibited.

After the 60-day technical-indicator warm-up, the effective SET50
feature period begins on 3 May 2012. Outer evaluation ends on 18
December 2025. The four outer SET50 test years contain 241, 243, 244,
and 234 rows, respectively.

<span class="mark">Plot Graph data figure</span>

**3.3 News data and out-of-sample sentiment construction**

The labelled text source is Bilingual StockTBSA, published by the
VISTEC-depa AI Research Institute of Thailand (Uthayopas et al., 2025)
The local source snapshot spans 3 January 2018 to 28 December 2023 and
contains 10,295 articles, 15,949 article–ticker labels, and 12,706 valid
positive, neutral, or negative pairs. Labels marked exclude, not stock,
or ambiguous were not relabeled as neutral. The locked 2023 intrinsic
sentiment cohort contains 1,333 pairs from 738 unique articles, with
class counts of 92 negative, 585 neutral, and 656 positives.

For 2024–2025, 69,824 deduplicated headlines were obtained from the
official SET news service (The Stock Exchange of Thailand, n.d.) A
point-in-time membership filter produced 4,619 article–symbol pairs
associated with contemporaneous SET50 constituents. Six non-overlapping
membership versions cover the period, including the March–April 2025
GULF/GULFI/INTUCH/VGI transition. These headlines are unlabeled; their
sentiment predictions are evaluated only through downstream forecasting.
An additional 68,514-row 2015–2017 corpus lacks comparable ticker and
sentiment labels and was excluded from the main benchmark. No
defensible, free, consistently dated 2012–2014 news source was
identified. The paper therefore does not claim homogeneous news coverage
from 2012 to 2025

The operational sentiment model is a balanced logistic regression over
character TF–IDF features, with the target ticker prepended to the text.
It requires no Thai word segmentation. Annual expanding predictions were
made as follows: train 2018 and predict 2019; train through 2019 and
predict 2020; and continue analogously through a 2018–2022 fit
predicting 2023. For 2024–2025, the classifier was fitted once on
labelled 2018–2023 data and frozen. It was not retrained on
pseudo-label, and no Optuna tuning was used.

The labelled source lacks reliable intraday publication times. To
preserve a conservative information boundary, a news item dated on
calendar day (d) was assigned to the first trading session strictly
after (d); weekend items were assigned to the next session. Eight daily
variables were constructed: sentiment mean and standard deviation;
positive, negative, and neutral ratios; article count; ticker-mention
count; and a news-availability indicator. The forecasting cohort begins
in 2019, the first year with out-of-sample sentiment predictions.

<span class="mark">Table for news sentiments</span>

**3.4 Point-in-time prediction contract**

**<span class="mark">\[ Equation Later \]</span>**

**<span class="mark">\[ Figure here later \]</span>**

**3.5 Temporal splits and frozen windows**

Candidate sequence lengths were 1, 3, 5, 10, and 20 trading days. Window
selection used expanding training periods 2012–2017, 2012–2018,
2012–2019, and 2012–2020, with validation years 2018, 2019, 2020, and
2021. Balanced accuracy was the primary selection metric, followed by
direction accuracy, RMSE, and the shorter window as tiebreakers.
Selection was symmetric across the Full-TA and Full-TA+VMD conditions so
that neither side of the numerical ablation received a privileged
lookback**.**

The locked windows were 5 days for LSTM, 20 for CNN, 20 for LSTM–CNN, 10
for LSTM–Attention, and 20 for LSTM–CNN–Attention. The outer folds
trained through 2021, 2022, 2023, and 2024 and tested 2022, 2023, 2024,
and 2025. Each outer cell used seeds 42, 123, 456, 789, and 2025**.**

**<span class="mark">\[ Table 1 Frozen protocol and benchmark\]</span>**

All dates use Asia/Bangkok and a 17:00 information cutoff. News
variables enter only the relevant multimodal arms.

**3.6 Numerical features and casual VMD**

The Full-TA control contains 116 variables generated from daily and
previous completed weekly/monthly OHLCV information. The pool includes
price and volume lags; 1-, 3-, 5-, 10-, 20-, and 60-day returns; simple
and weighted moving averages; volatility, momentum, and rate-of-change
measures; cross-timeframe ratios; candlestick features; volume ratios;
direction lags; and Stochastic, RSI, MACD, Williams %R, CCI, ADX, and
directional-indicator variables**.**

Causal rolling VMD adds six variables. At each date (t), the algorithm
decomposes only the 60 closes from <span class="mark">(t-59)</span>
through (t). It uses five modes, penalty
<span class="mark">(alpha=1000)</span>, dual-ascent step
<span class="mark">(au=0)</span>, a DC mode, tolerance
<span class="mark">(10^{-7}),</span> and at most 500 iterations. The
mode with the highest final center frequency is treated as noise. Four
retained intrinsic mode functions, the denoised close, and the
removed-mode energy ratio form the six additional features, giving 122
numerical inputs. VMD is therefore evaluated as a causal auxiliary
representation, not as an assumed improvement**.**

**<span class="mark">Equation for Technical Indicators</span>**

**3.7 Neural architecture and training**

The historical numerical, news, and regime modules predict the next
close and derive direction from the sign of predicted next close minus
current close. Architectures were deliberately compact**:**

**<span class="mark">\[ Equation Later \]</span>**

All fits used Adam, mean squared error, 20 epochs, batch size 32, and
`shuffle=False`. Random states were reset before every cell,
deterministic TensorFlow operations were enabled, and the Keras session
was cleared between fits. Min–max scaling was fitted separately on the
supervised training portion of each fold**.**

**3.8 Multimodal falsification design**

Observed predicted news features were compared with a date-matched
market-only arm. Four additional controls tested alternative
explanations**:**

1.  **News-only:** tests on whether aggregated sentiment can forecast
    without the numerical market state

2.  **Shuffled news:** permutes the daily news block while preserving
    its empirical marginal distribution, testing whether observed
    chronology adds more than generic additional dimensions

3.  **Lagged news**: shifts the news features by five rows, weakening
    their contemporaneous alignment

4.  **Random feature:** adds matched noise variables, testing generic
    capacity expansion

All arms retained the frozen architecture, window, cohort, seed, and
training budget. The registered family contained 100 model–control cells
and 400 fits. The principal reported contrasts are Observed-News minus
Market-Only and Observed-News minus Shuffled-News. News-only,
lagged-news, and random-feature controls remain in the complete evidence
bundle**.**

**3.9 Local NLP, Bull/Bear/Leader Debate and downstream news routes**

**<span class="mark">\[ Table Here Later\]</span>**

The local relevance and sentiment pipelines were evaluated annually on
out-of-sample pairs. The LLM benchmark was separate. A class-stratified
set of 60 pairs from 2022 was used once to validate a fixed prompt;
prompts were then frozen before opening the 2023 cohort. The proprietary
model was `gpt-5.6-terra`, called through structured outputs at low
reasoning effort with response storage disabled**.**

The role system contained a Bull worker, a Bear worker, and a Leader
that returned positive/neutral/negative probabilities and a scalar
sentiment score. All roles used the same model under different prompts.
Controls were: one single pass; three identical-prompt passes aggregated
by self-consistency, matching the three role-system calls; and four
passes, whose token-accounted cost fell within the frozen ±15% near-cost
band around the Leader system. Uncertainty was clustered by article ID
because an article could generate several ticker-labelled pairs. The
primary intrinsic endpoint was accuracy, with Macro-F1 as supporting
evidence. The two registered Leader comparisons were Holm-adjusted**.**

The original audited downstream run used daily features from the
expanding local classifier. A subsequent extension uses the final Leader
output as the sentiment source. For each eligible article, the Leader
receives the Bull and Bear analyses and returns class probabilities and
a scalar score. Only outputs available by the market-close cutoff may be
aggregated to date (t); those daily aggregates may predict (t+1), never
the same-day close used to form the target. The Leader arm must be
compared with Market-Only and Local-NLP arms on the identical
overlapping dates, architecture, frozen window, seed set, training
budget, and target rows. Its final artifact must record the article
cutoff/timezone, daily aggregation fields, coverage, missing-news
handling, API model/version, prompt hash, runtime, token cost, and
paired uncertainty**.**

At this draft stage, the author reports that Leader-derived features
improve SET50 performance in the additional test. The quantitative
cohort, baseline, effect size, confidence interval, and
multiplicity-adjusted result are left as explicit verification fields
rather than inferred from the Local-NLP run.

**3.10 Casual market regimes and SHAP selection**

**<span class="mark">\[ Equation Later \]</span>**

SHAP rankings used `Gradient Explainer`, 100 training-only background
sequences, at most 128 evenly spaced training ranking sequences, 200
samples, deterministic cell seeds, and float32 tensors. Candidate
feature counts were 10, 20, 30, 40, 60, 80, 100, and 122. A
one-standard-error rule plus balanced- accuracy, model-error, RMSE, and
temporal-Jaccard guardrails selected the smallest stable subset. The
frozen counts were 122 globally, 30 for Bull, 122 for Sideway, and 80
for Bear. A Spearman selector used the same counts as a size-matched
non-SHAP comparator.

Seven outer arms separated routing, capacity, and selector effects:
Global-All, a three-replica Global3-All capacity control, Global-SHAP,
Global-Spearman, Regime-All, Regime-SHAP, and Regime-Spearman. Because
the global selector retained all 122 features, Global-SHAP and
Global-Spearman are identity controls. The primary selector contrast is
Regime-SHAP minus Regime-All. LIME was applied after selection to 1,800
instance–repeat explanations. It did not influence training or feature
selection. A local surrogate with <span class="mark">(R^2\<0.70)</span>
was labelled low fidelity. Agreement with SHAP was summarized only for
reliable LIME repeats, while every failed repeat remained in the
denominator**.**

**<span class="mark">3.11 Partial 2026 objectives-alignment stress
test</span>**

<span class="mark">Before accessing 2026 observations, the protocol
froze the five architectures, windows, features, seeds, objectives,
thresholds, and economic rules. The registered Yahoo Finance source
failed its overlap/availability gate. A dated deviation then accepted
the Investing.com historical interface; accordingly, the result is
source-contingent rather than a pristine source-frozen holdout. The
final extension contains 138 rows from 5 January to 30 July 2026.</span>

<span class="mark">Two objectives used the same backbones. **Direct**
predicts Up with a sigmoid and binary cross-entropy. **Multitask** adds
a standardized next-day log-return head to the shared backbone, with
direction BCE weight 1.0 and return MSE weight 0.25; return scaling is
fitted on training data only. Threshold candidates 0.50, 0.55, 0.60, and
0.65 were evaluated in 2019–2021 validation data under registered
coverage and economic gates. All ten model–objective cells failed the
gate and reverted to 0.50. The 2026 positive-class share was
58.70%.</span>

<span class="mark">An exploratory next-session open-to-close proxy
evaluated long/flat and long/short positions at 5, 10, and 20 basis
points per active round trip. The complete grid, not only the best cell,
is retained in the Supplement. These short-sample results are not
evidence of a profitable or deployable strategy.</span>
<span class="mark">Results</span>

**<span class="mark">3.12 Frozen SET100 same-exchange transfer</span>**

<span class="mark">SET100 transfer used the same 122 numerical features,
windows, seeds, epochs, and outer years as the matched SET50 global
numerical comparator. Features, VMD, and scaling were reconstructed
within each SET100 fold. There was no SET100 retuning, Optuna search,
early stopping, or model exclusion. Five models × four years × five
seeds produced 100/100 planned fits.</span>

<span class="mark">SET100 news was excluded because a historically
point-in-time SET100 constituent/news universe was unavailable. Adding
SET50-oriented news would create an asymmetric test. The matched SET50
values in the transfer table are therefore the numerical Full-TA+VMD
global comparator, not the later integrated regime–news arm. Because
SET50 is nested within SET100 and both share the same exchange, macro
regimes, period, and many constituents, the analysis is a same-exchange
breadth transfer, not external-market validation. The official index
profiles define the respective top-50 and top-100 liquid
large-capitalization universes and their semiannual review convention
(The Stock Exchange of Thailand, n.d.)</span>
<span class="mark">Results</span>

**3.13 Metrics and statistics inference**

Balanced accuracy is the primary forecasting endpoint. Direction
accuracy, MCC, coverage, RMSE, and MAE are secondary where applicable;
AUC and Brier score are included for probabilistic 2026 objectives. Seed
predictions or metrics are averaged within each model–year cell before
temporal inference. Seeds measure optimization variability and are not
treated as independent market samples**.**

For the four historical outer years, paired model contrasts use exact
two-sided sign-flip tests over the four-fold-level effects. The minimum
attainable two-sided p-value is therefore 0.125. Descriptive t intervals
communicate effect uncertainty but do not override the exact test. Holm
adjustment controls the registered familywise error rate. A 10-day
circular moving-block bootstrap over paired daily predictions provides a
serial-dependence sensitivity (Politis & Romano, 1992)**;** it cannot
promote a squared-error finding into a directional claim. Intrinsic LLM
intervals use 5,000 article-cluster bootstrap replicates, with
article-cluster sign-flip sensitivity and Holm adjustment across the two
accuracy contrasts**.**

**<span class="mark">Method 11 -\> 8</span>**

**4 Results** **and Discussion**

**4.1 Numerical denoising was architecture dependent**

Table 2 reports the corrected point-in-time numerical ablation. VMD
increased mean balanced accuracy only for LSTM–CNN (+0.35 percentage
points) and reduced it for the other four architectures (−0.30 to −0.60
points). Every interval crossed zero and the smallest exact p-value was
0.375. VMD therefore cannot be described as a generally beneficial
denoiser for next-day direction**.**

**<span class="mark">\[ Table Here Later (full rolling 2022-2025)
\]</span>**

*Note:* BAcc is balanced accuracy. Values are means over four outer
years after within-fold seed aggregation. Runtime is mean model build,
fit, and evaluation inference time per seed–fold cell; VMD feature
generation is not included. The complete selection plus outer numerical
protocol recorded 4,158.40 seconds of model fit/inference time

The error and direction endpoints also diverged. In the corrected
numerical run, VMD reduced RMSE for LSTM and CNN but reduced their
balanced accuracy. Conversely, the small LSTM–CNN balanced-accuracy gain
did not imply a universal price-error gain. This is exactly the
distinction that a direction-focused audit must preserve**.**

**4.2 Predicted news did not add reliable directional information**

The observed-news effect relative to the market-only arm was negative
for all five architectures in the serial-dependence sensitivity, ranging
from −0.009 points for CNN to −1.895 points for LSTM–Attention (Table
3A). None survived Holm adjustment. Observed news also failed to beat
the shuffled-news control after adjustment. Although the unadjusted
moving-block intervals for LSTM and LSTM–CNN–Attention versus shuffled
news excluded zero in the negative direction, both Holm-adjusted
p-values were 0.098. The news-only, five-row lagged-news, and
random-feature controls likewise produced no adjusted balanced-accuracy
superiority. Across the registered exact fold-level families, zero of 30
primary balanced-accuracy contrasts survived Holm correction**.**

**<span class="mark">\[ Table Here Later (10 day circular moving-block )
\]</span>**

*Note:* Each contrast uses 960 paired eligible daily directional rows.
The moving-block analysis is a serial-dependence sensitivity; registered
fold-level exact tests govern the primary historical inference.

The negative-control result should not be interpreted as proof that news
has no financial information. It shows that this dated, aggregated,
out-of-sample sentiment representation did not provide a robust
conditional increment beyond the numerical pipeline. The 2024–2025
official headlines were also much shorter than the labelled headline
proxies (approximately 65 versus 260 characters on average), and
sentiment confidence fell from 0.756 in 2023 to about 0.495, documenting
a material source shift**.**

**4.3 The LLM role system improved intrinsic sentiment and was extended
to SET50 forecasting**

The local sentiment classifier achieved out-of-sample accuracies of
81.24%, 81.55%, 83.43%, 79.85%, and 82.82% for 2019–2023, respectively;
2023 Macro-F1 was 78.44%. The 2023 relevance classifier achieved 88.85%
accuracy and 85.91% Macro-F1**.**

On the locked 2023 intrinsic cohort, a one-call LLM scored 69.84%
accuracy, the three-pass self-consistency control 70.67%, the four-pass
control 70.59%, and the Bull/Bear/Leader system 76.59%. Table 3B shows
that the Leader retained an advantage of 5.93–6.00 points under
equal-call and near-cost controls. Both article-cluster intervals
excluded zero and both comparisons remained significant after Holm
adjustment.

**<span class="mark">\[ Table Here Later (LLM Benchmark on locked 2023 )
\]</span>**

*Note:* Table 3B is the intrinsic sentiment benchmark. The downstream
use of Leader outputs belongs to the distinct common-cohort forecasting
comparison in Table 3C and must not be inferred from the intrinsic
accuracy values.

The later integrated 2×2 robustness analysis crossed global versus
regime-SHAP numerical inputs with absence versus presence of predicted
news on the 2019–2025 common cohort. The highest mean balanced accuracy
was 54.07% for LSTM–CNN with Regime-SHAP numerical inputs *without*
news. Adding news within the regime-selected arm improved mean balanced
accuracy for two of five models (LSTM +0.13 and LSTM–Attention +1.03
points) and reduced it for the other three. No balanced-accuracy
contrast survived Holm adjustment. The complete factorial is retained as
post-hoc integrated evidence in Table S5 rather than promoted as a
success claim**.**

**4.4 Regime-SHAP selection helped one architecture but did not
generalize**

**<span class="mark">\[ Table Here Later (Primary regime-specific SHAP
reduction contrast) \]</span>**

*Note:* The contrast is Regime-SHAP minus Regime-All. Seeds are averaged
within outer years before inference. SHAP values are fitted-model
attributions, not causal effects of market variables.

LIME did not provide an independent validation of SHAP. Of 1,800
registered instance–repeat explanations, 1,293 (71.83%) had local
surrogate <span class="mark">(R^2\<0.70).</span> Low-fidelity repeats
were retained and disclosed. Because the diagnostic failed for most
repetitions, complete LIME agreement and stability tables are placed in
the Supplement and no feature-level causal narrative is built from
them**.**

**4.5 Partial-2026 performance exposed majority-side collapse**

The 138-row partial-2026 stress test provided weak and
objective-dependent evidence (Table 5A). Direct LSTM had the highest
balanced accuracy at 54.42% and MCC 0.088, but its direction accuracy
was only 52.90%. LSTM multitask had the highest raw direction accuracy,
59.42%, but balanced accuracy fell to 51.40%. Several CNN and
LSTM–CNN–Attention cells obtained 58.70% raw accuracy, exactly matching
the Up-class share, while balanced accuracy was 50% and MCC was zero.
These cells predicted only the majority side. The apparent accuracy was
therefore not directional discrimination**.**

**<span class="mark">\[ Table Here Later (partial-2026 forward
robustness) \]</span>**

*Note:* All objectives used the validation-gate fallback threshold 0.50.
The positive-class share was 58.70%. This is a source-contingent partial
year, not a full-year or live deployment result.

The complete 10-basis-point economic grid is exploratory. Its best
observed cell direct LSTM–CNN, long/flat—had 34.31% coverage, 8.55% net
cumulative return, annualized Sharpe 1.75, maximum drawdown −3.54%, and
break-even cost 27.91 basis points over 137 executable rows. Its
Deflated-Sharpe probability was only 0.441, below 0.5. This short,
selected result is therefore reported as a diagnostic sensitivity, not
evidence of profitability or deployment readiness

**4.6 Frozen SET100 transfer was uniformly weaker**

Mean balanced accuracy declined on SET100 for all five architectures by
0.95 to 2.17 percentage points (Table 5B). CNN had the highest SET100
balanced accuracy at 51.89%, followed by LSTM at 51.63%;
LSTM–CNN–Attention was close to chance at 50.03%. No paired cross-index
difference survived Holm adjustment. Although CNN was lower on SET100 in
all four years and its descriptive interval excluded zero, the exact
p-value was 0.125 and the Holm p-value was 0.625. It is not a
statistically established degradation**.**

**<span class="mark">\[ Table Here Later (SET100 same-exchange numerical
transfer) \]</span>**

*Note:* Matched SET50 is the frozen 122-feature global numerical
comparator, not the final post-hoc integrated regime–news arm. SET100 is
a broader, overlapping index on the same exchange and is not an
external-market replication**.**

**<span class="mark">\[ Figure Here Later \]</span>**

**Figure 3. Architecture-wise effects and uncertainty across audit
pillars.** Forest-style panels show paired balanced-accuracy changes for
VMD, Observed-News versus Market-Only, Regime-SHAP versus Regime-All,
and SET100 minus matched SET50. The vertical zero line makes the absence
of a consistent cross-architecture direction visible; LLM intrinsic
accuracy is shown in a separate panel because it has a different
endpoint and cohort**.**

**5 Conclusion** **( Limitations 1 Paragraph , Future Works 1 Paragraph
)**

**Acknowledgements**

**<span class="mark">Reproducibility and data availability</span>**

**References**
