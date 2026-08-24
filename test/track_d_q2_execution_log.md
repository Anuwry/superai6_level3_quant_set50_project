# Track D: Q2 Upgrade Execution Log

Status: **COMPLETED**  
Protocol: `track-d-direction-forward-v1`  
Execution date: 2026-07-31  
Market: SET50  
Forecast target: next-observed-day direction (`Close[t+1] > Close[t]`)

## 1. Scope completed

Track D implements the four prespecified Q2-oriented upgrades without
overwriting the completed Track C controls:

1. direct direction classification and multi-task direction/return learning
   for the same five registered neural architectures;
2. a protocol freeze followed by a 2026 forward evaluation;
3. transaction-cost-aware long/flat and long/short backtests, including
   validation-only confidence thresholds and selective-prediction coverage;
4. SHAP parameter/label randomization and top-feature deletion faithfulness
   tests.

The registered model windows remained fixed: LSTM=5, CNN=20, LSTM-CNN=20,
LSTM-Attention=10, and LSTM-CNN-Attention=20. The final seeds were 42, 123,
456, 789, and 2025. No Optuna search or 2026-informed model selection was
performed.

## 2. Freeze and data chronology

- The model, objective, threshold, cost, seed, and XAI protocol was frozen at
  `2026-07-31T15:17:42+00:00`, before any 2026 market-data access.
- The registered Yahoo Finance symbol `^SET50.BK` failed closed: it returned
  only one non-overlapping row with zero volume.
- A dated source deviation was frozen before the first full alternative-source
  request. Investing.com's SET50 instrument 41049 was then used.
- The first alternative request exposed a pre-existing 2025-12-11 OHLC
  containment anomaly also present in the local source. Parser amendment V2
  was frozen before the accepted repeated snapshot. The raw anomaly was kept
  without clipping, repair, or imputation.
- The accepted source snapshot had 200 rows, 55 overlapping dates, maximum
  and median overlapping close difference of 0.00 points, and 100% positive
  volume coverage. It appended 145 rows after the local 2025-12-22 endpoint.
- The forward data comprised 3,307 training rows, 138 test rows from
  2026-01-05 through 2026-07-30, and 122 causal Full-TA plus rolling-VMD
  features.

Consequently, this is a **source-contingency partial-2026 forward evaluation**,
not a pristine registered-source confirmatory holdout. The model protocol was
genuinely frozen before 2026 access, but the source and parser deviations must
remain disclosed in the paper.

## 3. Validation-only confidence selection

Threshold selection used only 2019, 2020, and 2021 validation folds, seed 42,
a long/short next-open-to-close strategy, and a 10-bps round-trip cost. None of
the ten model-objective pairs satisfied the registered gate of at least 20%
coverage and positive net return in at least two of the three years. All ten
therefore used the prespecified fallback threshold of 0.50.

This result is important: threshold 0.50 was not selected because it was best
on 2026. It was the frozen fallback after every confidence threshold failed
the validation gate.

## 4. Forward predictive results

The table reports metrics from probabilities averaged over the five frozen
seeds. `DA` is direction accuracy and `BAcc` is balanced accuracy.

| Model | Objective | DA | BAcc | MCC | AUC | Brier | ECE-10 |
|---|---|---:|---:|---:|---:|---:|---:|
| LSTM | Direct | 0.5290 | **0.5442** | **0.0882** | **0.5332** | 0.2497 | 0.0872 |
| LSTM | Multitask | **0.5942** | 0.5140 | 0.0736 | 0.5018 | 0.2492 | 0.0896 |
| CNN | Direct | 0.5870 | 0.5000 | 0.0000 | 0.4960 | 0.2492 | 0.0824 |
| CNN | Multitask | 0.5870 | 0.5000 | 0.0000 | 0.4945 | 0.2490 | 0.0811 |
| LSTM-CNN | Direct | 0.4928 | 0.5211 | 0.0439 | 0.4854 | 0.2500 | 0.0876 |
| LSTM-CNN | Multitask | 0.5870 | 0.5000 | 0.0000 | 0.5168 | 0.2488 | 0.0796 |
| LSTM-Attention | Direct | 0.5580 | 0.4961 | -0.0109 | 0.5068 | 0.2491 | 0.0825 |
| LSTM-Attention | Multitask | 0.5797 | 0.5120 | 0.0380 | 0.5116 | 0.2493 | 0.0830 |
| LSTM-CNN-Attention | Direct | 0.5870 | 0.5000 | 0.0000 | 0.4843 | 0.2489 | 0.0804 |
| LSTM-CNN-Attention | Multitask | 0.5870 | 0.5000 | 0.0000 | 0.5151 | 0.2488 | 0.0799 |

The positive-class share was 0.5870. Therefore, a DA of 0.5870 together with
BAcc=0.50 and MCC=0 denotes a one-sided/degenerate decision rule, not useful
direction discrimination. The highest DA (LSTM multitask, 0.5942) likewise
reduced to BAcc=0.5140 and AUC=0.5018. The strongest discrimination was the
LSTM direct ensemble, but its BAcc=0.5442, MCC=0.0882, and AUC=0.5332 still
represent a weak signal.

Multi-task learning did not improve BAcc consistently. Its BAcc changes
relative to direct learning were CNN 0.0000, LSTM -0.0302, LSTM-CNN -0.0211,
LSTM-Attention +0.0159, and LSTM-CNN-Attention 0.0000. Both objectives must
therefore remain in the paper; the results do not support promoting multitask
learning as a general improvement.

## 5. Selective prediction and economics

The seed-averaged probabilities were tightly concentrated near 0.50. Every
registered threshold above 0.50 (0.55, 0.60, and 0.65) produced zero selected
rows for all ten model-objective pairs. Selective prediction therefore did not
yield a usable confidence/coverage trade-off in this forward period.

The primary executable proxy used information through close t, entered at
open t+1, exited at close t+1, and charged 10 bps round trip on every active
day. All 5/10/20-bps and long/flat/long/short cells are retained in the raw
economic table. Selected 10-bps results are summarized below.

| Model | Objective | Strategy | Coverage | Net return | Sharpe | Max DD |
|---|---|---|---:|---:|---:|---:|
| LSTM | Direct | Long/flat | 0.4234 | 0.0798 | 1.5619 | -0.0352 |
| LSTM | Direct | Long/short | 1.0000 | -0.0054 | -0.0205 | -0.1038 |
| LSTM | Multitask | Long/flat | 0.9708 | -0.0169 | -0.2146 | -0.1179 |
| LSTM | Multitask | Long/short | 1.0000 | -0.0399 | -0.5462 | -0.1385 |
| CNN | Direct | Long/flat | 1.0000 | -0.0024 | 0.0244 | -0.1093 |
| CNN | Direct | Long/short | 1.0000 | -0.0024 | 0.0244 | -0.1093 |
| CNN | Multitask | Long/flat | 1.0000 | -0.0024 | 0.0244 | -0.1093 |
| CNN | Multitask | Long/short | 1.0000 | -0.0024 | 0.0244 | -0.1093 |
| LSTM-CNN | Direct | Long/flat | **0.3431** | **0.0855** | **1.7515** | -0.0354 |
| LSTM-CNN | Direct | Long/short | 1.0000 | -0.0172 | -0.1985 | -0.1106 |
| LSTM-CNN | Multitask | Long/flat | 1.0000 | -0.0024 | 0.0244 | -0.1093 |
| LSTM-CNN | Multitask | Long/short | 1.0000 | -0.0024 | 0.0244 | -0.1093 |
| LSTM-Attention | Direct | Long/flat | 0.8540 | 0.0031 | 0.1067 | -0.0996 |
| LSTM-Attention | Direct | Long/short | 1.0000 | -0.0318 | -0.4210 | -0.1204 |
| LSTM-Attention | Multitask | Long/flat | 0.8905 | 0.0245 | 0.4377 | -0.0881 |
| LSTM-Attention | Multitask | Long/short | 1.0000 | 0.0204 | 0.3638 | -0.0833 |
| LSTM-CNN-Attention | Direct | Long/flat | 1.0000 | -0.0024 | 0.0244 | -0.1093 |
| LSTM-CNN-Attention | Direct | Long/short | 1.0000 | -0.0024 | 0.0244 | -0.1093 |
| LSTM-CNN-Attention | Multitask | Long/flat | 1.0000 | -0.0024 | 0.0244 | -0.1093 |
| LSTM-CNN-Attention | Multitask | Long/short | 1.0000 | -0.0024 | 0.0244 | -0.1093 |

The best observed cell was LSTM-CNN direct long/flat: 8.55% net cumulative
return, Sharpe 1.75, 34.31% coverage, and -3.54% maximum drawdown. Its
break-even round-trip cost was 27.91 bps. However, its Deflated-Sharpe
probability was only 0.4411, the period is only 138 observations, and all
strategy/cost cells were inspected. It is an encouraging exploratory result,
not reliable evidence of deployable profitability.

## 6. SHAP sanity and deletion results

The registered XAI audit used every direct seed-42 model, 30 evenly spaced
2026 endpoints, 100 train-only background sequences, GradientExplainer with
200 samples, trained/random-initialized/permuted-label conditions, top-10
feature deletion, and 100 size-matched random deletions per endpoint.

| Model | Seed-42 probability SD | Unique probabilities | Random-init rank corr. | Permuted-label rank corr. | Mean top deletion | Mean random deletion | Median faithfulness percentile |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | 0.012707 | 138 | 0.4098 | 0.4339 | 0.006647 | 0.001638 | 0.99 |
| CNN | 0.000000 | 1 | 0.1598 | 0.1176 | 0.000000 | 0.000000 | 1.00 |
| LSTM-CNN | 0.005621 | 138 | 0.4545 | 0.0000 | 0.002817 | 0.000748 | 1.00 |
| LSTM-Attention | approximately 0 | 2 | 0.0000 | 0.0000 | approximately 0 | approximately 0 | 0.99 |
| LSTM-CNN-Attention | 0.000045 | 2 | 0.2358 | 0.4082 | 0.000000 | 0.0000003 | 1.00 |

Low rank correlation is desirable in a randomization sanity check, while a
large top-feature deletion effect relative to random deletion supports local
faithfulness. Those statistics cannot be interpreted alone when model output
is constant. The CNN, LSTM-Attention, and LSTM-CNN-Attention seed-42 models
were constant or near-constant, making their high deletion percentile
uninformative. Their SHAP rankings must not be presented as validated
explanations.

Only LSTM and LSTM-CNN showed materially varying seed-42 predictions and
non-zero deletion effects. Their median faithfulness percentiles were 0.99 and
1.00, respectively, although their random-initialization rank correlations
remained moderate (approximately 0.41 and 0.45). The most prominent LSTM
features were daily/weekly close-position variables and direction lag 1. The
most prominent LSTM-CNN features were monthly close position, weekly open,
and VMD-denoised close. These rankings are exploratory because the underlying
forward discrimination was weak.

The zero permuted-label correlation for LSTM-CNN arose from a constant
randomized attribution vector and is not, by itself, proof of a perfect sanity
test. All rows were retained; no XAI outcome fed back into model selection.

## 7. Runtime and computational record

Mean fit time per seed was 8.23-23.36 seconds, increasing from LSTM/CNN to the
attention hybrid. Across the 50 forward cells, summed fit time was 723.50
seconds and summed inference time was 21.79 seconds. The complete forward
runner wall time was 794.14 seconds. The XAI audit required 117.95 seconds.
The earlier 30 validation cells contained 313.13 seconds of recorded
fit-plus-inference time.

| Model | Direct fit s/seed | Multitask fit s/seed | Direct parameters | Multitask parameters |
|---|---:|---:|---:|---:|
| LSTM | 8.23 | 9.14 | 9,041 | 9,050 |
| CNN | 8.60 | 9.96 | 12,017 | 12,026 |
| LSTM-CNN | 16.48 | 17.73 | 10,737 | 10,746 |
| LSTM-Attention | 13.99 | 15.08 | 10,129 | 10,138 |
| LSTM-CNN-Attention | 22.13 | 23.36 | 12,865 | 12,874 |

Execution used Windows 11, Python 3.12, TensorFlow 2.21.0, NumPy 2.4.3,
pandas 3.0.1, SciPy 1.17.1, and scikit-learn 1.8.0. Native-Windows TensorFlow
2.21 ran on CPU. Deterministic operations were enabled and oneDNN numerical
reordering was disabled. Runtime/cache paths were kept on drive D.

## 8. Verification record

- Track D test suite: **47 passed**.
- Ruff static checks for every Track D implementation/test file: **passed**.
- XAI smoke test: finite outputs for all five architectures before forward
  access.
- Final integrity audit: **all checks passed**.
- Verified cardinalities included 6,900 seed-level forward predictions, 1,380
  seed-averaged predictions, 240 economic summaries, 32,880 economic daily
  rows, 54,900 XAI attributions, 300 randomization comparisons, 150 top-feature
  deletion rows, and 15,000 random deletion rows.
- The audit confirmed frozen model/objective/seed registries, no duplicate
  keys, finite outputs, 2026-only forward dates, data artifact hashes, and
  freeze time preceding accepted forward-source access.

Passing integrity means the experiment is internally complete and
reproducible. It does not convert weak/degenerate predictive results into a
positive finding.

## 9. Paper claim recommendation

Track D should be included as a rigorous robustness/negative-result section:

- objective alignment and multi-task learning did not consistently strengthen
  next-day directional discrimination;
- confidence-based abstention failed because probabilities did not reach the
  frozen thresholds above 0.50;
- some long/flat cells were economically encouraging, but the short holdout,
  multiple inspected strategies, and low Deflated-Sharpe probability prevent
  a profitability claim;
- SHAP deletion was meaningful only for models with non-degenerate outputs,
  demonstrating why explanation sanity checks are necessary before publishing
  feature rankings.

Do not claim that 59.42% DA is strong performance, that the LSTM-CNN strategy
is ready for real trading, or that SHAP validated all five models. A defensible
claim is that the locked forward protocol exposed model collapse and separated
raw accuracy, balanced discrimination, economic utility, and explanation
faithfulness.

## 10. Artifacts

Core outputs are under `outputs/track_d_q2/`. Paper-ready derived tables are:

- `paper_predictive_summary.csv`
- `paper_economic_primary_10bps.csv`
- `paper_runtime_summary.csv`
- `paper_xai_summary.csv`
- `paper_xai_top10_features.csv`

These summary tables were derived after the experiment and do not alter any
frozen decision. Full raw predictions, economic daily paths, all registered
cost/threshold sensitivities, XAI rows, source snapshots, access ledgers,
freeze manifests, and the integrity audit remain available for reviewer
inspection.
