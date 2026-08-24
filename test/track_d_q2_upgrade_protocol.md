# Track D Q2 Upgrade Protocol

Status: **FROZEN BEFORE ANY 2026 DATA ACCESS (2026-07-31)**  
Protocol version: `track-d-direction-forward-v1`  
Forward year: 2026

This protocol adds four prespecified extensions to the completed Track C:
direct direction learning, multi-task direction/return learning, a genuinely
forward 2026 holdout, transaction-cost-aware selective backtesting, and SHAP
sanity/faithfulness checks. Track C artifacts are immutable controls and are
not overwritten.

## 1. Iteration compact

```text
Goal: test whether objective alignment produces a reproducible next-day
      direction signal and whether that signal survives implementation lag,
      trading costs, and explanation sanity tests.
Decision: Up probability controls long/flat or long/short next-session action.
Primary predictive metric: forward balanced accuracy at threshold 0.50.
Guardrails: direction accuracy, MCC, AUC, Brier score, calibration, coverage,
            seed dispersion, runtime, and finite-output checks.
Unacceptable mistakes: any 2026 access before freeze; test-fitted scaling;
                       threshold selection on 2026; same-close execution;
                       silent XAI fidelity filtering.
Labels: Up iff next observed close > current close; exact ties are excluded
        from classification metrics and supervised sequence endpoints.
Feature snapshot: 122 Full-TA plus causal rolling-VMD variables.
Fallback: Track C remains the registered fallback result if Track D fails a
          data, prediction, economic, or XAI integrity gate.
```

## 2. Registered models and objectives

Exactly the existing five architectures are used. No new hidden layer,
Optuna search, or 2026-informed hyperparameter is allowed.

| Model | Window |
|---|---:|
| LSTM | 5 |
| CNN | 20 |
| LSTM-CNN | 20 |
| LSTM-Attention | 10 |
| LSTM-CNN-Attention | 20 |

Two objectives are evaluated:

1. `direct`: sigmoid Up probability with binary cross-entropy;
2. `multitask`: a shared registered backbone with sigmoid Up probability and
   standardized next-day log-return output. Direction BCE has weight 1.0 and
   return MSE has weight 0.25. Return mean and standard deviation are fit on
   each training fold only.

Epochs, batch sizes, optimizer, shuffle=False, deterministic operations, and
the five final seeds remain inherited from Track A. Objective selection is
not permitted after observing 2026: both objectives must be reported.
Native-Windows CPU execution fixes `TF_DETERMINISTIC_OPS=1` and
`TF_ENABLE_ONEDNN_OPTS=0`; the values are captured in run metadata.

## 3. Validation-only threshold selection

Threshold selection uses the three existing point-in-time folds with
validation years 2019, 2020, and 2021. Only seed 42 is used for this decision.
The registered symmetric confidence thresholds are 0.50, 0.55, 0.60, and
0.65.

For each model-objective pair, choose the threshold with the highest pooled
mean validation net return under:

- long/short validation strategy;
- 10 bps round-trip cost per active daily position;
- next-session open-to-close implementation return;
- at least 20% non-zero-position coverage;
- positive total net return in at least two of three validation years.

Ties prefer higher coverage and then the smaller threshold. If no threshold
passes, use 0.50 and record the gate failure. The selected threshold cannot be
changed from a 2026 result.

## 4. Untouched 2026 forward holdout

No 2026 SET50 price, query result, or row count may be inspected before the
protocol and implementation hash manifest is written.

Registered source:

```text
Yahoo Finance chart API
symbol = ^SET50.BK
interval = 1d
requested overlap start = 2025-10-01
end = retrieval time after freeze
```

The source must overlap the existing local close series on at least 20 dates,
with maximum absolute close difference no greater than 0.50 SET50 points.
OHLCV rows must be unique, ordered, complete, and finite. Non-positive or
mostly unavailable volume causes a fail-closed data gate. There is no silent
source fallback. If the registered source fails, an official user-supplied
SET-compatible file requires a dated protocol deviation before use.

The daily series is extended only with dates later than the local source.
Weekly and monthly values are aggregated from completed daily periods and
lagged by one period before daily alignment. Rolling TA and VMD variables use
past/current observations only. The final feature row lacking an observed
next close is excluded. Train labels require `Label_Date < first 2026 feature
Date`.

The 2026 holdout is executed once after freeze. No model, threshold, cost,
seed, or XAI setting may be modified from the result.

## 5. Economic evaluation

Prediction date t uses information through close t. The primary executable
proxy enters at open t+1 and exits at close t+1. The idealized close-to-close
return is retained only as a labelled sensitivity.

Strategies:

- long/flat: +1 when p >= q, otherwise 0;
- long/short: +1 when p >= q, -1 when p < 1-q, otherwise 0.

The validation-selected threshold is the primary economic result. To retain
a complete risk-coverage curve even if the gate falls back to 0.50, all four
already registered thresholds are also reported on 2026 as labelled
sensitivity rows; none may be promoted based on the forward result. At
q=0.50, exact p=0.50 follows the registered Up tie-break.

Costs are 5, 10, and 20 bps as round-trip costs per absolute daily position.
Because the executable proxy enters at open and exits at close, every active
day is a separate trade and pays its round-trip cost; consecutive equal
signals are not treated as a free carried position. Report coverage,
round-trip units, signal-position changes, gross/net cumulative return,
annualized mean/volatility/Sharpe, maximum drawdown, win rate, trade count,
and break-even round-trip cost. Report all model-objective-strategy-cost
cells, not only the winner. The number of tried strategies is explicit;
Deflated Sharpe probabilities are a selection-bias sensitivity, not a
promotion guarantee.

## 6. SHAP sanity and feature-deletion audit

The XAI audit is fixed to all five `direct` seed-42 forward models, never the
best 2026 model only. Thirty 2026 endpoints are selected at evenly spaced
chronological indexes without using correctness. Background comprises 100
evenly spaced train-only sequences. GradientExplainer uses `nsamples=200`.

For each instance, signed lag contributions are summed per feature. Save and
compare:

1. trained model attributions;
2. same architecture with randomly initialized parameters;
3. same architecture trained on a deterministic permutation of training
   direction labels;
4. top-10 feature-trajectory deletion effect;
5. 100 size-matched random feature-trajectory deletion effects.

Randomization sensitivity is described by absolute-rank Spearman
correlation. Feature deletion replaces the complete lag trajectory with the
corresponding train-background mean trajectory. Faithfulness is the
percentile of the top-feature output change among random deletions. All rows
are retained; no sanity result can feed back into model selection.

## 7. Required outputs and claims

Required root artifacts:

```text
outputs/track_d_q2/
  freeze_manifest.json
  validation_predictions.csv
  validation_metrics.csv
  threshold_selection.csv
  selected_thresholds.csv
  forward_predictions_by_seed.csv
  forward_predictions_seed_averaged.csv
  forward_metrics.csv
  selective_prediction_metrics.csv
  economic_daily.csv
  economic_summary.csv
  xai_attributions.csv
  xai_randomization_summary.csv
  xai_deletion_summary.csv
  runtime_summary.csv
  integrity_audit.json
  run_metadata.json
```

Track D is confirmatory only with respect to the timestamped 2026 forward
holdout after freeze. Comparisons motivated by already viewed 2022-2025 Track
C results remain explicitly exploratory. Neither predictive accuracy nor XAI
agreement is causal evidence or proof of profitability.
