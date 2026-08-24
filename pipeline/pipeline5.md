┌──────────────────────────────┐
│     Raw Stock Data           │
└──────────────┬───────────────┘
               │
               ├─ Daily OHLCV
               ├─ Weekly OHLCV
               └─ Monthly OHLCV
               │
               ▼
┌──────────────────────────────┐
│ Data Cleaning & Alignment    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Multi-Timeframe Generator    │
└──────────────┬───────────────┘
               │
               ├─ Daily Features
               ├─ Weekly Features
               └─ Monthly Features
               │
               ▼
┌──────────────────────────────┐
│ Multi-Scale Feature Pool     │
└──────────────┬───────────────┘
               │
               ├─ Lag Features
               ├─ Rolling Stats
               ├─ Indicators
               └─ Cross-Timeframe Features
               │
               ▼
┌──────────────────────────────┐
│ Stage 1 Baseline Training    │
└──────────────┬───────────────┘
               │
               ├─ Ridge Regression ( Basic Regression Model )
               ├─ XGBoost, LightGBM ( Gradient Boosting Model)
               ├─ LSTM ( Deep Learning Model )
               ├─ Chronos ( Foundation Model )
               └─ AutoGluon ( AutoML Model ( State-of-the-Art ) )
               │
               ▼
┌──────────────────────────────┐
│ Initial SHAP Analysis        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Progressive SHAP Refinement  │
└──────────────┬───────────────┘
               │
               ├─ Remove Bottom 20%
               ├─ Retrain
               ├─ Recompute SHAP
               └─ Iterate
               │
               ▼
┌──────────────────────────────┐
│ Stability Selection          │
└──────────────┬───────────────┘
               │
               ├─ 10 Seeds
               ├─ Frequency Ranking
               └─ Stable Feature Set
               │
               ▼
┌──────────────────────────────┐
│ Adaptive Timeframe Fusion    │
└──────────────┬───────────────┘
               │
               ├─ Daily Weight
               ├─ Weekly Weight
               └─ Monthly Weight
               │
               ▼
┌──────────────────────────────┐
│ Final Forecasting Models     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Walk Forward Evaluation      │
└──────────────┬───────────────┘
               │
               ├─ RMSE
               ├─ MAE
               ├─ MAPE
               ├─ R²
               └─ Direction Accuracy
               │
               ▼
┌──────────────────────────────┐
│ Financial Backtesting        │
└──────────────┬───────────────┘
               │
               ├─ Long Only
               ├─ Long/Short
               ├─ Buy & Hold
               └─ Equal Weight
               │
               ▼
┌──────────────────────────────┐
│ Financial Metrics            │
└──────────────┬───────────────┘
               │
               ├─ Sharpe Ratio
               ├─ Sortino Ratio
               ├─ Max Drawdown
               ├─ CAGR
               ├─ Annual Return
               └─ Calmar Ratio
               │
               ▼
┌──────────────────────────────┐
│ Ablation Study               │
└──────────────┬───────────────┘
               │
               ├─ Raw Features
               ├─ + Multi-Timeframe
               ├─ + Progressive SHAP
               ├─ + Stability Selection
               └─ Full Framework
               │
               ▼
┌──────────────────────────────┐
│ Statistical Significance     │
└──────────────┬───────────────┘
               │
               ├─ Wilcoxon Test
               ├─ Paired t-test
               └─ Effect Size
               │
               ▼
┌──────────────────────────────┐
│ Explainability & Insights    │
└──────────────────────────────┘

Ablation:

Method	#Features	RMSE	Sharpe
Raw	150	12.5	0.72
+ Multi-Timeframe	150	11.8	0.84
+ Progressive SHAP	60	10.9	1.01
+ Stability	40	10.6	1.08
Full Framework	30	10.2	1.15

จุดขายหลัก

150 Features
↓
30 Features

Performance ดีขึ้น

Sharpe ดีขึ้น

คุณได้:
Accuracy
Simplicity
Explainability
Practical Utility
