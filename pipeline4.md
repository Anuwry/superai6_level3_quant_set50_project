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
               ├─ Missing Value
               ├─ Split Adjustment
               ├─ Alignment
               └─ Target Creation
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
│ Multi-Scale Lag Constructor  │
└──────────────┬───────────────┘
               │
               ├─ Lag 1~5
               ├─ Lag 10
               ├─ Lag 20
               ├─ Lag 60
               └─ Lag 120
               │
               ▼
┌──────────────────────────────┐
│ Temporal Feature Pool        │
└──────────────┬───────────────┘
               │
               ├─ Raw Features
               ├─ Technical Indicators
               ├─ Rolling Statistics
               └─ Multi-Timeframe Features
               │
               ▼
┌──────────────────────────────┐
│ Stage 1 SHAP Analysis        │
└──────────────┬───────────────┘
               │
               ├─ Train Initial Model
               ├─ Compute SHAP
               └─ Rank Features
               │
               ▼
┌──────────────────────────────┐
│ Progressive Refinement       │
└──────────────┬───────────────┘
               │
               ├─ Remove Bottom 20%
               ├─ Retrain
               ├─ Recompute SHAP
               ├─ Remove Bottom 20%
               ├─ Retrain
               └─ Repeat Until Stop
               │
               ▼
┌──────────────────────────────┐
│ Stability Selection          │
└──────────────┬───────────────┘
               │
               ├─ Run 10 Seeds
               ├─ Track Feature Frequency
               └─ Keep Stable Features
               │
               ▼
┌──────────────────────────────┐
│ Adaptive Timeframe Scoring   │
└──────────────┬───────────────┘
               │
               ├─ Daily Contribution
               ├─ Weekly Contribution
               └─ Monthly Contribution
               │
               ▼
┌──────────────────────────────┐
│ Timeframe Weight Generator   │
└──────────────┬───────────────┘
               │
               ├─ Daily Weight
               ├─ Weekly Weight
               └─ Monthly Weight
               │
               ▼
┌──────────────────────────────┐
│ Adaptive Feature Fusion      │
└──────────────┬───────────────┘
               │
               ├─ Weighted Daily
               ├─ Weighted Weekly
               └─ Weighted Monthly
               │
               ▼
┌──────────────────────────────┐
│ Forecasting Engine           │
└──────────────┬───────────────┘
               │
               ├─ XGBoost
               ├─ LightGBM
               ├─ Chronos
               └─ AutoGluon
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
│ Explainability Layer         │
└──────────────┬───────────────┘
               │
               ├─ Global SHAP
               ├─ Local SHAP
               ├─ Daily Importance
               ├─ Weekly Importance
               └─ Monthly Importance
               │
               ▼
┌──────────────────────────────┐
│ Knowledge Discovery          │
└──────────────────────────────┘

Contribution ที่ reviewer มองเห็น:

Contribution 1
Adaptive Multi-Timeframe Representation

Daily
Weekly
Monthly

ไม่ใช่แค่เอามาต่อกัน
แต่เรียนรู้ความสำคัญของแต่ละ timeframe

Contribution 2
Progressive SHAP Refinement

150 Features
↓
120
↓
90
↓
60
↓
30

แทนการทำ SHAP รอบเดียว

Contribution 3
Feature Stability Selection

Feature ไหน
ถูกเลือกซ้ำบ่อย

ลดความเสี่ยง overfitting

Contribution 4
Explainable Forecasting
ตอบได้ว่า

Daily Signal
Weekly Signal
Monthly Signal

ใครสำคัญ

Contribution 5
Cross-Paradigm Benchmark

Traditional ML
vs
AutoML
vs
Foundation Models