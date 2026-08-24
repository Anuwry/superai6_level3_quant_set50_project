┌──────────────────────────┐
│     Raw Data Sources     │
└────────────┬─────────────┘
             │
             ├─ Daily OHLCV
             ├─ Weekly OHLCV
             └─ Monthly OHLCV
             │
             ▼
┌──────────────────────────┐
│      Data Cleaning       │
└────────────┬─────────────┘
             │
             ├─ Missing Values
             ├─ Date Alignment
             └─ Target Construction
             │
             ▼
┌──────────────────────────┐
│   Feature Engineering    │
└────────────┬─────────────┘
             │
             ├─ Raw Features
             ├─ Lag Features
             ├─ Rolling Statistics
             └─ Technical Indicators
             │
             ▼
┌──────────────────────────┐
│  Multi-Timeframe Fusion  │
└────────────┬─────────────┘
             │
             ├─ Daily Features
             ├─ Weekly Features
             └─ Monthly Features
             │
             ▼
┌──────────────────────────┐
│  Feature Set Creation    │
└────────────┬─────────────┘
             │
             ├─ Set A: Daily
             ├─ Set B: Daily+Weekly
             └─ Set C: Daily+Weekly+Monthly
             │
             ▼
┌──────────────────────────┐
│ Walk-Forward Validation  │
└────────────┬─────────────┘
             │
             ├─ 2012-2021 → 2022
             ├─ 2012-2022 → 2023
             ├─ 2012-2023 → 2024
             └─ 2012-2024 → 2025
             │
             ▼
┌──────────────────────────┐
│      Model Training      │
└────────────┬─────────────┘
             │
             ├─ Persistence Baseline
             ├─ Linear Regression
             ├─ Random Forest
             ├─ XGBoost
             ├─ LightGBM
             ├─ AutoGluon Tabular
             ├─ AutoGluon TimeSeries
             └─ Chronos
             │
             ▼
┌──────────────────────────┐
│      Evaluation          │
└────────────┬─────────────┘
             │
             ├─ RMSE
             ├─ MAE
             ├─ MAPE
             ├─ R²
             └─ Direction Accuracy
             │
             ▼
┌──────────────────────────┐
│     Best Performing      │
│         Models           │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│      SHAP Analysis       │
└────────────┬─────────────┘
             │
             ├─ Global SHAP
             ├─ Feature Ranking
             ├─ Daily Importance
             ├─ Weekly Importance
             └─ Monthly Importance
             │
             ▼
┌──────────────────────────┐
│ SHAP-Based Selection     │
└────────────┬─────────────┘
             │
             ├─ Select Top-K
             ├─ Retrain Models
             └─ Re-Evaluate
             │
             ▼
┌──────────────────────────┐
│     Ablation Study       │
└────────────┬─────────────┘
             │
             ├─ Raw Only
             ├─ + Lag
             ├─ + Indicators
             ├─ + Weekly
             ├─ + Monthly
             └─ + SHAP Selection
             │
             ▼
┌──────────────────────────┐
│ Statistical Significance │
└────────────┬─────────────┘
             │
             ├─ Before vs After SHAP
             ├─ Daily vs Multi-Timeframe
             └─ Model-to-Model
             │
             ▼
┌──────────────────────────┐
│     Research Findings    │
└──────────────────────────┘

Contribution 1
เปรียบเทียบ

Traditional ML
AutoML
Foundation Models

บนตลาดหุ้นไทย

Contribution 2
ศึกษาผลของ

Daily
Weekly
Monthly

ต่อประสิทธิภาพการพยากรณ์

Contribution 3
เสนอ
SHAP-Guided Feature Selection
เพื่อคัดเลือก Feature ที่สำคัญที่สุด

Contribution 4
วิเคราะห์ Explainability ว่า

Daily Signal
Weekly Signal
Monthly Signal

กลุ่มใดมีผลต่อการตัดสินใจของโมเดลมากที่สุด

ถ้าผมเป็น PI (Principal Investigator) ของโปรเจกต์นี้ ผมจะตั้งชื่อ Theme หลักของงานเป็น
Multi-Timeframe Stock Forecasting with SHAP-Guided Feature Selection: A Comparative Study of Traditional Machine Learning, AutoML, and Foundation Models
เพราะชื่อแบบนี้สะท้อนสิ่งที่เป็น "ของคุณ" จริง ๆ คือ