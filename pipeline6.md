┌──────────────────────────────┐
│       SET50 Index Data       │
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
               ├─ Missing Values
               ├─ Trading Calendar Alignment
               ├─ Target Construction
               └─ Leakage Check
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
               ├─ Raw Features
               ├─ Lag Features
               ├─ Rolling Statistics
               ├─ Technical Indicators
               └─ Cross-Timeframe Features
               │
               ▼
┌──────────────────────────────┐
│ Walk-Forward Validation Loop │
└──────────────┬───────────────┘
               │
               │
               ├──────────────────────────────┐
               │                              │
               ▼                              │
      Train: 2012-2021                        │
      Test : 2022                             │
                                              │
      Train: 2012-2022                        │
      Test : 2023                             │
                                              │
      Train: 2012-2023                        │
      Test : 2024                             │
                                              │
      Train: 2012-2024                        │
      Test : 2025                             │
                                              │
               ▼                              │
┌──────────────────────────────┐              │
│ Hyperparameter Optimization  │              │
└──────────────┬───────────────┘              │
               │                              │
               ├─ Ridge + Optuna              │
               ├─ XGBoost + Optuna            │
               ├─ LightGBM + Optuna           │
               ├─ LSTM + Optuna               │
               └─ Best Config                 │
               ▼                              │
┌──────────────────────────────┐              │
│ Initial Model Training       │              │
└──────────────┬───────────────┘              │
               │                              │
               ├─ Ridge                       │
               ├─ XGBoost                     │
               ├─ LightGBM                    │
               ├─ LSTM                        │
               ├─ Chronos                     │
               └─ AutoGluon                   │
               ▼                              │
┌──────────────────────────────┐              │
│ Fold-Specific SHAP Analysis  │              │
└──────────────┬───────────────┘              │
               │                              │
               ├─ Global SHAP                 │
               ├─ Feature Ranking             │
               └─ Timeframe Ranking           │
               ▼                              │
┌──────────────────────────────┐              │
│ Progressive SHAP Refinement  │              │
└──────────────┬───────────────┘              │
               │                              │
               ├─ Remove Bottom 20%           │
               ├─ Retrain                     │
               ├─ Recompute SHAP              │
               ├─ Remove Bottom 20%           │
               └─ Repeat Until Stop           │
               ▼                              │
┌──────────────────────────────┐              │
│ Stability Selection          │              │
└──────────────┬───────────────┘              │
               │                              │
               ├─ Multiple Seeds              │
               ├─ Feature Frequency           │
               └─ Stable Feature Set          │
               ▼                              │
┌──────────────────────────────┐              │
│ Adaptive Timeframe Fusion    │              │
└──────────────┬───────────────┘              │
               │                              │
               ├─ Daily Weight                │
               ├─ Weekly Weight               │
               └─ Monthly Weight              │ 
               ▼                              │
┌──────────────────────────────┐              │
│ Final Retraining             │              │
└──────────────┬───────────────┘              │
               │                              │
               └─ Predict Test Fold           │
               ▼                              │
        Collect Results ◄─────────────────────┘

               ▼
┌──────────────────────────────┐
│ Forecasting Evaluation       │
└──────────────┬───────────────┘
               │
               ├─ RMSE
               ├─ MAE
               ├─ MAPE
               ├─ R²
               └─ Direction Accuracy
               ▼
┌──────────────────────────────┐
│ Financial Backtesting        │
└──────────────┬───────────────┘
               │
               ├─ Buy & Hold
               ├─ Long Only
               ├─ Long/Short
               ├─ Transaction Cost
               └─ Slippage
               ▼
┌──────────────────────────────┐
│ Financial Metrics            │
└──────────────┬───────────────┘
               │
               ├─ Sharpe Ratio
               ├─ Sortino Ratio
               ├─ CAGR
               ├─ Annual Return
               ├─ Max Drawdown
               └─ Calmar Ratio
               ▼
┌──────────────────────────────┐
│ Ablation Study               │
└──────────────┬───────────────┘
               │
               ├─ Raw Features
               ├─ + Multi-Timeframe
               ├─ + SHAP
               ├─ + Progressive SHAP
               ├─ + Stability Selection
               ├─ + Adaptive Fusion
               └─ Full Framework
               ▼
┌──────────────────────────────┐
│ Statistical Significance     │
└──────────────┬───────────────┘
               │
               ├─ Paired t-test
               ├─ Wilcoxon Test
               └─ Effect Size
               ▼
┌──────────────────────────────┐
│ Explainability & Findings    │
└──────────────────────────────┘

Contribution ที่จะเขียนใน Paper
C1 — Adaptive Multi-Timeframe Fusion
Daily
Weekly
Monthly

ไม่ได้ถูกกำหนดน้ำหนักตายตัว
แต่ถูกเรียนรู้จากข้อมูล

C2 — Progressive SHAP Refinement
150 Features
↓
120
↓
90
↓
60
↓
30

เลือก feature แบบ iterative แทน SHAP รอบเดียว

C3 — Stability-Aware Feature Selection
เลือกเฉพาะ feature
ที่ถูกเลือกซ้ำอย่างสม่ำเสมอ
ลด overfitting

C4 — Explainable Forecasting
วิเคราะห์ว่า

Daily Signal
Weekly Signal
Monthly Signal

มีผลต่อการพยากรณ์มากน้อยเพียงใด

C5 — Cross-Paradigm Benchmark
Linear
Tree-Based
Deep Learning
Foundation Model
AutoML

ในสภาพแวดล้อมเดียวกัน