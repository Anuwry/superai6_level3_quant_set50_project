=======================================================================
                   SET50 DATA ACQUISITION PIPELINE
                     (Timeframe: ล่าสุด 5-6 ปี)
=======================================================================
                               |
       +-----------------------+-----------------------+
       |                                               |
[1. Price Data (OHLCV)]                  [2. Fundamental Data (Qtrly)]
(ข้อมูลราคารายวันของ SET50)                   (งบการเงินและอัตราส่วนต่างๆ)
       |                                               |
       v                                               v
=======================================================================
                      PHASE 1: FEATURE EXTRACTION
=======================================================================
       |                                               |
[Chronos2 (Foundation Model)]            [Data Preprocessing]
- Zero-Shot Forecasting                  - Handling Missing Values
- ดึงสัญญาณแนวโน้มและโมเมนตัม                - Data Scaling (Standard/MinMax)
       |                                               |
       v                                               v
[Time-Series Features]                   [Cross-Sectional Features]
- Predicted Future Trajectory            - Value (e.g., P/E, P/BV)
- Volatility / Trend Signals             - Profitability (e.g., ROE)
       |                                               |
       +-----------------------+-----------------------+
                               |
=======================================================================
                      PHASE 2: FEATURE MERGING
=======================================================================
                               v
                     [Data Alignment (Join)]
       (นำ Time-Series Signals มาประกบกับงบการเงินรายไตรมาส)
                               |
=======================================================================
                      PHASE 3: PREDICTIVE MODELING
=======================================================================
                               v
                 [AutoGluon TabularPredictor]
                 +--------------------------+
                 | - XGBoost                |
                 | - LightGBM               |
                 | - CatBoost               |
                 | - Weighted Ensemble      |
                 +--------------------------+
                               |
=======================================================================
                      PHASE 4: OUTPUT & EVALUATION
=======================================================================
                               v
               [Stock Classification / Scoring]
          (ประเมินศักยภาพการลงทุนของหุ้น SET50 แต่ละตัว)
                               |
                               v
                 [Backtesting Engine (Walk-Forward)]
                (ประเมิน MAE/RMSE และ Financial Metrics)