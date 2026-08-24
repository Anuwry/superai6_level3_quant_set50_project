**A Point-in-Time of Multimodal and Regime-Aware Deep Learning for
Next-Day SET Index Direction Forecasting**

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
point-in-time evaluation and robustness checks remain unclear. We
conduct a point-in-time reliability audit of numerical denoising,
predicted financial-news sentiment, role-structured large-language-model
inference, regime-aware feature selection and neural complexity. The
study uses daily SET50 price index data from 2012-2025, expanding
pre-2022 model selection, four outer test years (2022-2025; 962
sessions), five fixed seeds and five architectures: LSTM, CNN, LSTM-CNN,
LSTM-Attention and LSTM-CNN-Attention. Labels are purged by observation
date, transformations are fitted on training data only and balanced
accuracy is the primary endpoint. Causal rolling variational mode
decomposition changed balanced accuracy by -0.60 to +0.35 percentage
points across architectures, with no conclusive effect. Predicted news
produced no multiplicity-adjusted gain over market-only and
falsification controls. In a locked 2023 sentiment benchmark, a
Bull/Bear/Leader large-language-model system exceeded equal-call and
near-cost self-consistency controls by 5.93 and 6.00 points,
respectively. Regime-specific SHAP selection improved CNN by 1.46 points
but produced mixed or negative effects elsewhere without Holm-adjusted
significance. Partial-2026 testing exposed one-sided prediction collapse
while frozen SET100 transfer was uniformly weaker, indicating that the
observed gains were configuration-specific rather than generalizable.

**Keywords:** point-in-time evaluation; stock-market direction; SET50;
balanced accuracy; multimodal learning; financial sentiment; variational
mode decomposition; market regime; SHAP; reliability audit

**1. Introduction**

Predicting whether a market index will rise or fall on the next trading
day is an unusually compact statement of a difficult forecasting
problem. The target is binary, but the data-generating process is
non-stationary, the usable signal is weak relative to market noise, and
small implementation choices can alter the apparent outcome **(Fama,
1970; Olorunnimbe & Viktor, 2023)** Deep recurrent networks,
convolutional networks, and attention mechanisms offer flexible
representations of sequential financial data **(Hochreiter &
Schmidhuber, 1997; Qin et al., 2017; Vaswani et al., 2017)**
Representative financial applications have reported gains from LSTM and
CNN architectures **(Fischer & Krauss, 2018; Hoseinzade & Haratizadeh,
2019)** That flexibility also increases the number of choices available
to a researcher: feature families, lookback windows, decomposition
settings, text encoders, fusion mechanisms, random seeds, thresholds,
regimes, and post-hoc explanations. A result selected from many such
choices can look stronger than the underlying signal.

The financial forecasting literature contains increasingly elaborate
hybrid systems. Reviews document extensive use of LSTM, CNN, and hybrid
deep-learning models for price and direction prediction, while also
identifying continuing gaps in backtesting, reproducibility, and
practical evaluation **(Olorunnimbe & Viktor, 2023; Sezer et al.,
2020)** Earlier multimodal systems combined text and market signals
through attention, graph structure, expert-opinion aggregation, and
cross-modal interaction **(Luo et al., 2023; Sawhney et al., 2020; H.
Wang et al., 2020)** More recent work models interactions among news
items or uses sentiment-attention mechanisms for index prediction
**(W.-J. Liu et al., 2024; M. Wang et al., 2024)** Signal-decomposition
methods offer a second route: separate a price process into frequency
components and expose a denoised representation to the predictor. Yet a
component can improve squared-price error without improving the sign of
the next movement, and a text model can classify sentiment well without
adding conditional information to an already strong market-data feature
set. These distinctions are often blurred when performance is summarized
by a single best accuracy.

This study treats these distinctions as the object of analysis. Rather
than selecting the best-performing configuration, we examine whether
five audit dimensions remain informative under a common point-in-time
and inferential protocol

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
while controlling vanishing gradients **(Hochreiter & Schmidhuber,
1997)** Their use in financial prediction is motivated by delayed
dependencies, while CNNs provide local filters that can extract
short-range temporal patterns. Attention adds a learned mechanism for
weighting representations across a sequence **(Qin et al., 2017; Vaswani
et al., 2017)** Large empirical studies have reported advantages for
LSTM-based financial prediction **(Fischer & Krauss, 2018)** and
CNN-based systems have combined diverse market variables for next-day
index direction **(Hoseinzade & Haratizadeh, 2019)** However, the
existence of a flexible mapping does not imply a stable signal. Market
dynamics change across years, and a high-capacity hybrid can magnify
optimization variance or collapse to a majority-side decision while
retaining superficially acceptable raw accuracy. Recent review evidence
likewise treats architecture choice as only one part of a credible,
backtested, reproducible financial-learning pipeline **(Olorunnimbe &
Viktor, 2023)**.

The present benchmark therefore keeps training budgets deliberately
compact and constant, repeats each outer cell across five seeds, and
treats the year not the seed as the temporal inferential unit. This
design distinguishes stochastic fit variability from independent
evidence over time.

**2.2 Denoising and multimodal financial information**

Variational mode decomposition separates a signal into band-limited
intrinsic mode functions through a constrained variational formulation
**(Dragomiretskiy & Zosso, 2014)** In financial applications,
decomposition can reduce high-frequency variation or provide additional
frequency-domain features, and recent stock-forecasting systems have
paired VMD with LSTM, meta-learning, and attention-based networks **(T.
Liu et al., 2022; Y. Liu et al., 2024)** The key causal requirement is
that decomposition at date (t) must use only observations available
through (t). A full-series decomposition, even when followed by a
temporal train/test split, can transmit future information into past
components; windowed decomposition has been proposed specifically to
address this problem **(T. Liu et al., 2022)** We therefore use a
rolling past-only implementation and test VMD as a feature addition
rather than assuming it is beneficial denoising.

Financial text offers a different information channel. Prior work has
used social media, analyst or expert opinions, and news to enrich
stock-movement models **(Luo et al., 2023; Sawhney et al., 2020; H. Wang
et al., 2020)** Recent systems further model interactions among news
items or sentiment-attention mechanisms for index prediction **(W.-J.
Liu et al., 2024; M. Wang et al., 2024)** An intrinsic sentiment score
and an incremental forecasting signal are nevertheless different
estimands. Text may be labelled accurately but arrive too late, be
redundant with price adjustment, be diluted during daily aggregation, or
exhibit source and length shift. A credible multimodal test therefore
needs date-aware prediction of text labels, equal forecasting cohorts,
and controls that distinguish chronology from generic extra dimensions.

**2.3 LLM role systems and compute matching**

Multi-agent debate and judge-style LLM systems can improve reasoning by
eliciting divergent intermediate positions before a final decision **(Du
et al., 2024; Liang et al., 2024)** Self-consistency provides a strong
alternative explanation: repeated sampling itself can improve a result
by aggregating several reasoning paths **(X. Wang et al., 2023)**
Comparing a three-call role system only with one call therefore
confounds role structure with inference budget. Our intrinsic benchmark
uses both an equal-call three-pass control and a four-pass near-cost
control. Even then, the comparison is system-level: all roles use the
same underlying proprietary model, so the result cannot identify
abstract “debate reasoning” as the sole causal mechanism. A recent
survey of financial LLM agents similarly identifies prompt sensitivity,
real-time adaptation, and institutional deployment constraints as open
evaluation problems **(Dong et al., 2025)**.

**2.4 Market regimes and explanation reliability**

Bull and bear phases have long been analyses as persistent market states
**(Pagan & Sossounov, 2003)** Trend-following evidence motivates using
returns across several horizons **(Moskowitz et al., 2012)** while the
Average Directional Index (ADX) provides a conventional measure of trend
strength **(Wilder, 1978)** A regime router used for next-day
forecasting must be fitted without future labels and should produce
semantically valid states in both training and test periods.

SHAP unifies feature attribution through Shapley values **(Lundberg &
Lee, 2017)** LIME fits local surrogate explanations around individual
predictions **(Ribeiro et al., 2016)** Neither method converts model
association into causal market evidence, and an explanation is useful
only if it is sufficiently faithful to the fitted model. Sanity checks
are consequently part of the explanation audit rather than an optional
visualization exercise **(Adebayo et al., 2018)** Finance-specific
review evidence also emphasizes that explanation methods must be
assessed with domain, transparency, and adoption constraints in view
**(Yeo et al., 2025)** We use SHAP for a train-only selector and LIME
only as a post-selection diagnostic; LIME never determines a feature
subset.

**2.5 Evaluation reliability**

Time-series evaluation must respect order and dependence **(Bergmeir et
al., 2018)** Financial machine-learning back tests additionally require
explicit information boundaries, pre-specified choices, and separation
of development from evaluation **(Arnott et al., 2019; Olorunnimbe &
Viktor, 2023)** We use expanding annual folds and a label-observation
purge instead of random cross-validation. Balanced accuracy, the mean of
class-wise recalls, is the primary directional metric because raw
accuracy can reward a one-sided classifier when Up and Down frequencies
differ **(Brodersen et al., 2010)** Families of related comparisons are
adjusted by the Holm procedure **(Holm, 1979)** Economic summaries are
explicitly exploratory; the Deflated Sharpe Ratio is retained to reveal
selection and short-sample risk rather than to certify deployment
**(Bailey & López de Prado, 2014)**.

**3. Materials and Methods**

**3.1 Study design, data governance, and evidence hierarchy**

Figure 1 summarizes a five-pillar reliability audit in which
point-in-time market-data governance supports numerical denoising,
multimodal news, regime-aware explanation, and later robustness tests.
The numerical stream spans the full historical period; news enters only
on dates with defensible out-of-sample sentiment estimates; and causal
regime labels and train-only explanation rankings act only on the
numerical feature pool. The corrected numerical ablation is the primary
architecture-paired experiment. Multimodal negative controls are frozen
retrospective falsification tests, the integrated regime-news and
explanation analyses are post-hoc robustness evidence, the partial-2026
module is source-contingent forward evidence, and SET100 is a frozen
same-exchange transfer audit rather than independent-market replication.

<span class="mark">\[Insert Figure 1 near here\]</span>

**Figure 1.** Five-pillar reliability-audit pipeline. Market-data
governance and point-in-time reconstruction support the causal VMD,
out-of-sample Local-NLP and Bull/Bear/Leader, causal Bull/Sideway/Bear
routing with train-only SHAP, partial-2026, and frozen SET100 modules.
Solid arrows denote data or forecasting flow; evaluation-only links are
shown separately.

Daily, weekly, and monthly SET50 and SET100 price-index observations
were manually downloaded from the publicly accessible Investing.com
historical-data pages **(Investing.com, n.d.-a, n.d.-b)** The provenance
manifest records source URLs, acquisition evidence, file sizes, SHA-256
digests, row counts, date ranges, and processing method. Public
accessibility is not treated as an open-data license, provider terms
continue to apply, and raw rows are not redistributed. Date-only
observations were interpreted as Stock Exchange of Thailand sessions in
Asia/Bangkok, with features available at 17:00 after the normal close.
Weekly and monthly bars were shifted by one completed period before
forward filling. The study uses provider-published price-index levels
and applies no researcher-side split, dividend, constituent, or
corporate-action adjustment.

Forty of 42 raw-file integrity checks passed. Two OHLC-containment
warnings on 11 December 2025, present in both daily index files, were
retained and disclosed; no critical check failed. The aligned SET100
cohort contained 3,360 of 3,381 required SET50 reference dates
(99.3789%); 21 leading dates were excluded rather than backfilled. After
the 60-day indicator warm-up, SET50 features begin on 3 May 2012. The
four outer SET50 tests end on 18 December 2025 and contain 241, 243,
244, and 234 rows for 2022-2025.

**3.2 News data and out-of-sample sentiment construction**

The labelled Bilingual StockTBSA snapshot spans 3 January 2018 to 28
December 2023 and contains 10,295 articles, 15,949 article-ticker
labels, and 12,706 valid positive, neutral, or negative pairs
**(Uthayopas et al., 2025)** Exclude, not-stock, and ambiguous labels
were not relabeled as neutral. The locked 2023 intrinsic cohort contains
1,333 pairs from 738 articles: 92 negative, 585 neutral, and 656
positives. For 2024-2025, 69,824 deduplicated official SET headlines
yielded 4,619 point-in-time article-symbol pairs under six
non-overlapping SET50 membership versions, including the March-April
2025 transition. These headlines are unlabeled and are evaluated only
through downstream forecasting. The 2015-2017 corpus lacked comparable
ticker and sentiment labels, and no defensible free 2012-2014 source was
identified; homogeneous news coverage over 2012-2025 is therefore not
claimed.

The operational sentiment model is balanced logistic regression over
character TF-IDF features with the target ticker prepended. Annual
expanding predictions trained on 2018 to predict 2019 and then expanded
one year at a time through a 2018-2022 fit predicting 2023. For
2024-2025, the model was fitted once on labelled 2018-2023 data and
frozen; pseudo-label retraining and Optuna tuning were not used. Because
reliable intraday timestamps were unavailable in the labelled source,
each item dated d was assigned conservatively to the first trading
session strictly after d. Eight daily variables were constructed:
sentiment mean and standard deviation; positive, negative, and neutral
ratios; article count; ticker-mention count; and news availability. The
forecasting cohort begins in 2019, the first year with out-of-sample
sentiment predictions.

**3.3 Point-in-time contract, temporal splits, and frozen windows**

For feature date t, market variables use only information available
through the close at t. The regression target is y_t = C\_(t+1), and
direction is d_t = I(C\_(t+1) \> C_t). Each row records both its feature
Date and the next-session Label_Date on which C\_(t+1) becomes
observable. At every boundary, a supervised training row is retained
only when Label_Date_t is earlier than the first Date in the evaluation
split. The final pre-evaluation row may supply sequence context, but it
is excluded from fitting and scaler estimation, transformed with the
training-fitted scaler, and never contributes a target. Zero-return
observations remain available to price regression but are excluded from
binary metrics; a predicted exact-zero move is treated as an abstention
and coverage is reported.

Candidate sequence lengths were 1, 3, 5, 10, and 20 trading days.
Expanding validation folds trained on 2012-2017, 2012-2018, 2012-2019,
and 2012-2020 and validated on 2018-2021. Selection prioritized balanced
accuracy, followed by direction accuracy, RMSE, and the shorter-window
tie-breaker, and was symmetric across Full-TA and Full-TA+VMD. Frozen
windows were 5 days for LSTM, 20 for CNN, 20 for LSTM-CNN, 10 for
LSTM-Attention, and 20 for LSTM-CNN-Attention. Outer folds trained
through 2021, 2022, 2023, and 2024 and tested 2022, 2023, 2024, and 2025
using seeds 42, 123, 456, 789, and 2025. All dates use Asia/Bangkok and
the 17:00 information cutoff.

**3.4 Numerical features and causal VMD**

The Full-TA control contains 116 variables constructed from daily and
previously completed weekly/monthly OHLCV information: price and volume
lags; 1-, 3-, 5-, 10-, 20-, and 60-day returns; moving averages;
volatility, momentum, rate-of-change, candlestick, cross-timeframe, and
volume-ratio features; direction lags; and Stochastic, RSI, MACD,
Williams %R, CCI, ADX, and directional indicators.

Causal rolling VMD adds six variables. At each t, only the 60 closes
from t-59 through t are decomposed using five modes, alpha = 1000,
dual-ascent step tau = 0, a DC mode, tolerance 10^-7, and at most 500
iterations. The mode with the highest final centre frequency is treated
as noise. Four retained intrinsic mode functions, the denoised close,
and removed-mode energy ratio produce 122 numerical inputs. VMD is
tested as a causal auxiliary representation rather than assumed to
improve direction.

**3.5 Neural architectures and training**

Historical numerical, news, and regime modules predict the next close
and derive direction from predicted next close minus current close. The
compact architectures were: LSTM, LSTM(16) followed by Dense(8, ReLU);
CNN, causal Conv1D(32, kernel 3) followed by global-average pooling and
Dense(8, ReLU); LSTM-CNN, sequence-output LSTM(16) followed by the
causal convolution and pooling stack; LSTM-Attention, sequence-output
LSTM(16) followed by causal two-head attention (key dimension 8),
pooling, and Dense(8, ReLU); and LSTM-CNN-Attention, LSTM followed by
causal convolution, causal two-head attention, pooling, and Dense(8,
ReLU). Every architecture uses a linear output.

All fits used Adam, mean squared error, 20 epochs, batch size 32, and
shuffle = false. Random states were reset for every cell, deterministic
TensorFlow operations were enabled, the Keras session was cleared
between fits, and min-max scaling was fitted only on the supervised
training portion of each fold.

**3.6 Multimodal falsification and Bull/Bear/Leader evaluation**

Observed predicted news features were compared with a date-matched
Market-Only arm and four falsification controls: News-Only,
date-shuffled news, news lagged by five rows, and matched random
features. All arms retained the frozen architecture, window, cohort,
seeds, and training budget. The registered family comprised 100
model-control cells and 400 fits. Primary contrasts were Observed-News
minus Market-Only and Observed-News minus Shuffled-News; the remaining
controls are retained in the complete evidence bundle.

Local relevance and sentiment pipelines were evaluated annually on
out-of-sample pairs, while the LLM benchmark used a separate locked
design. A class-stratified set of 60 pairs from 2022 was used once to
validate prompts before the 2023 cohort was opened. The Bull worker,
Bear worker, and Leader used the same proprietary model under distinct
prompts; the Leader returned positive, neutral, and negative
probabilities and a scalar sentiment score. Controls were one pass,
three identical-prompt self-consistency passes matching the role-system
call count, and four passes within the frozen +/-15% cost band. Accuracy
was the primary intrinsic endpoint, Macro-F1 was supporting evidence,
uncertainty was clustered by article, and the two Leader contrasts were
Holm-adjusted.

The audited forecasting route used to expand Local-NLP daily features. A
separately specified Leader-derived route aggregates only outputs
available by the market-close cutoff on date t and may predict t+1. Any
downstream comparison must use identical overlapping dates,
architecture, frozen window, seed set, training budget, and targets for
Market-Only, Local-NLP, and Leader arms, and must report coverage,
missing-news handling, model/version, prompt hash, runtime, token cost,
and paired uncertainty. Intrinsic sentiment accuracy is not interpreted
as proof of forecasting value.

**3.7 Causal market regimes and SHAP selection**

SHAP rankings used Gradient Explainer with 100 training-only background
sequences, at most 128 evenly spaced training ranking sequences, 200
samples, deterministic cell seeds, and float32 tensors. Candidate
feature counts were 10, 20, 30, 40, 60, 80, 100, and 122. A
one-standard-error rule with balanced-accuracy, model-error, RMSE, and
temporal-Jaccard guardrails selected the smallest stable subset. Frozen
counts were 122 globally, 30 for Bull, 122 for Sideway, and 80 for Bear;
a Spearman selector used the same counts as a size-matched comparator.

Seven outer arms separated routing, capacity, and selector effects:
Global-All, Global3-All, Global-SHAP, Global-Spearman, Regime-All,
Regime-SHAP, and Regime-Spearman. Because the global selector retained
all 122 features, Global-SHAP and Global-Spearman are identity controls;
the primary selector contrast is Regime-SHAP minus Regime-All. LIME was
an explanation-fidelity diagnostic only: 1,800 instance-repeat
explanations used an R^2 \< 0.70 threshold for low fidelity, failed
repeats remained in the denominator, and LIME did not affect training or
selection.

**3.8 Metrics and statistical inference**

Balanced accuracy is the primary forecasting endpoint. Direction
accuracy, MCC, coverage, RMSE, and MAE are secondary where applicable;
AUC and Brier score are reported for probabilistic 2026 objectives.
Seeds are averaged within each model-year cell before temporal inference
and are not treated as independent market samples.

Historical paired contrasts use exact two-sided sign-flip tests over
four outer-year effects; the minimum attainable two-sided p-value is
0.125. Descriptive t intervals communicate uncertainty but do not
override the exact test, and Holm adjustment controls the registered
familywise error rate. A 10-day circular moving-block bootstrap provides
a serial-dependence sensitivity **(Politis & Romano, 1992)**. Intrinsic
LLM intervals use 5,000 article-cluster bootstrap replicates,
article-cluster sign-flip sensitivity, and Holm adjustment across the
two accuracy contrasts. Economic summaries are exploratory and are not
used to certify deployment.

**4. Results and Discussion**

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

Before opening the 2026 observations, the five architectures, windows,
features, seeds, objectives, thresholds, and economic rules were frozen.
The registered Yahoo Finance source failed its overlap and availability
gate, after which a dated deviation accepted the Investing.com
historical interface; the 138-row sample from 5 January to 30 July 2026
is therefore source-contingent. Direct models predicted Up with sigmoid
binary cross-entropy. Multitask models added a standardized next-day
log-return head with direction BCE weight 1.0 and return MSE weight
0.25, using train-only return scaling. Threshold candidates 0.50, 0.55,
0.60, and 0.65 were evaluated on 2019-2021 validation data; all ten
model-objective cells failed the registered gate and reverted to 0.50.

The same frozen extension evaluated an exploratory next-session
open-to-close proxy for long/flat and long/short rules at 5, 10, and 20
basis points per active round trip. The full grid is retained in the
Supplement. These short-sample economic outputs are descriptive
diagnostics, not evidence of profitability or deployment readiness.

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

SET100 transfer retained the same 122 numerical features, frozen
windows, seeds, epochs, and 2022-2025 outer years as the matched SET50
global numerical comparator. Features, VMD, and scaling were
reconstructed within each SET100 fold; no SET100 retuning, Optuna
search, early stopping, or model exclusion was allowed. All 100 planned
fits (five models x four years x five seeds) completed.

SET100 news was excluded because a historically point-in-time
constituent and news universe was unavailable; adding SET50-oriented
news would have made the comparison asymmetric. The matched SET50 arm is
therefore the Full-TA+VMD global numerical comparator rather than the
later integrated regime-news arm. Because SET50 is nested within SET100
and both share the exchange, macro regimes, period, and many
constituents, this is a same-exchange breadth transfer rather than
external-market validation **(The Stock Exchange of Thailand, n.d.)**.

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

**5. Conclusion**

Under a common point-in-time protocol, the five frozen architectures
showed modest but architecture-dependent next-day SET50 directional
structure rather than a universally superior configuration. Causal
rolling VMD changed balanced accuracy by only -0.60 to +0.35 percentage
points, predicted Local-NLP news did not survive the registered
falsification and multiplicity controls, and regime-specific SHAP
improved CNN while producing mixed or negative effects elsewhere. The
Bull/Bear/Leader system delivered a clear intrinsic sentiment advantage
over compute-matched self-consistency, but that endpoint is distinct
from downstream forecasting value. Partial-2026 testing exposed
majority-side collapse in several objectives, and frozen SET100 transfer
weakened all five models. Collectively, the findings support the paper's
central contribution: a reproducible audit framework that distinguishes
selected component gains from improvements that remain reliable across
time, architectures, controls, and transfer settings.

**Limitations.** Temporal inference is based on four outer test years,
so the exact sign-flip design has limited power and cannot establish
small but consistent effects below its attainable p-value resolution.
The 2026 sample is partial-year and source-contingent, SET100 is an
overlapping same-exchange index rather than an independent market, and
news coverage varies in source, length, labelling, and timestamp
precision across periods. The LLM roles share one proprietary base
model, and intrinsic sentiment gains do not isolate debate as the unique
mechanism or guarantee forecasting value. SHAP describes fitted-model
attribution rather than causal market effects, while the low-fidelity
LIME diagnostic limits feature-level interpretation. These constraints
bound the claims to reliability under the audited SET setting rather
than universal market predictability or deployment readiness.

**Future work.** The next study should preregister the complete protocol
before opening a full untouched forward year, evaluate at least one
independent ASEAN or comparable emerging-market index, and construct a
consistently licensed point-in-time news archive with verifiable
publication timestamps and constituent membership. The Leader-derived
forecasting arm should be frozen and assessed on an identical common
cohort with article coverage, latency, cost, failures, and paired
uncertainty reported alongside predictive metrics. Model development
should then focus on a single prespecified architecture and calibration
strategy, followed by prospective shadow evaluation, rather than further
selection on the already inspected 2022-2026 outcomes.

**Acknowledgements**

The authors thank their respective institutions for providing the
research environment in which this study was conducted.

**Reproducibility and data availability**

The reproducibility package contains the model and evaluation code,
fixed seeds, architecture and window specifications, temporal split
definitions, point-in-time data contracts, prompt and protocol hashes,
aggregate outputs, runtime and cost ledgers, and non-reconstructive
integrity summaries. Raw SET50 and SET100 market rows are not
redistributed; they can be obtained from the publicly accessible
Investing.com historical-data pages subject to the provider's terms.
Official 2024-2025 headlines remain subject to the Stock Exchange of
Thailand website terms, and Bilingual StockTBSA is available from its
public repository under the licence stated in its dataset card.
Processed artifacts are shared only when they do not permit
reconstruction of restricted raw records. The archival repository URL or
DOI will be added after deposit.

**References**
