# RETIRED PIPELINE DRAFT

This file does not match the executed experiment. Use
`pipeline/pipeline8.md` and `test/pre_shap_experiment_manifest.md`.
BiLSTM, legacy LLM names, placeholder folds, historical backtesting, and live
execution below must not be cited as completed work.

=======================================================================
           FULL SYSTEM ARCHITECTURE: SET50 ALGORITHMIC TRADING
=======================================================================

[PHASE 1: DATA INGESTION & FEATURE ENGINEERING]
   │
   ├─ 1.1 Market Data Collection
   │   └─ SET50 Index Data (01/2012 -> 12/2025)[cite: 2]
   │   └─ Timeframes: Daily, Weekly, Monthly[cite: 2]
   │
   ├─ 1.2 Feature Engineering (8 Categories)[cite: 2]
   │   ├─ 1. Metadata & Target (DATE, TARGET_NEXT_CLOSE)[cite: 2]
   │   ├─ 2. Raw Multi-Timeframe (Open, High, Low, Close, Vol, Change)[cite: 2]
   │   ├─ 3. Lag Features (Close, Pct, Vol, Direction lags)[cite: 2]
   │   ├─ 4. Rolling/Lookback (Return, SMA, Momentum, Volatility, ROC)[cite: 2]
   │   ├─ 5. Cross-Timeframe (Ratio D/W, Ratio D/M)[cite: 2]
   │   ├─ 6. Candlestick/Price Action (Spread, Body, Shadows)[cite: 2]
   │   ├─ 7. Technical Analysis (WMA, RSI, MACD, Stoch, CCI)[cite: 2]
   │   └─ 8. Directional Movement (ADX, PLUSDI, MINUSDI)[cite: 2]
   │
   └─ 1.3 News Data Collection
       └─ Scrap the news (Text Data)[cite: 2]

                             │
                             ▼

[PHASE 2: DUAL-TRACK PROCESSING (NOISE REDUCTION & SENTIMENT)]
   │
   ├─ TRACK A: NUMERIC DATA (Regime-System)
   │   └─ VMD (Variational Mode Decomposition)[cite: 2]
   │      * Chosen for high stability and cleaner signal separation[cite: 2]
   │      * (STL and CEEMDAN rejected due to lack of clear seasonality and high randomness)[cite: 2]
   │
   └─ TRACK B: TEXT DATA (Multi-Agents System)[cite: 2]
       ├─ Worker 1: Scrap the news / Filter[cite: 2]
       ├─ Worker 2: Debate Model (Worker A & Worker B)[cite: 2]
       └─ Leader: LLMs Decision Maker (LLAMA-3 8B, MISTRAL, SEALLMS)[cite: 2]
          * Outputs Sentiment Score

                             │
                             ▼

[PHASE 3: REGIME-SPECIFIC DIMENSIONALITY REDUCTION]
   │
   ├─ Combine VMD Features + Sentiment Scores
   ├─ Market Regime Labeling[cite: 2]
   │   ├─ Bull Market[cite: 2]
   │   ├─ Bear Market[cite: 2]
   │   └─ Sideway Market[cite: 2]
   │
   └─ SHAP Feature Selection[cite: 2]
       └─ Select the most impactful features independently for each regime[cite: 2]

                             │
                             ▼

[PHASE 4: WALK-FORWARD VALIDATION & MODELING]
   │
   ├─ 4.1 Data Splitting (Walk-Forwards Folds)[cite: 2]
   │   └─ Target Folds: 2015-2016, 2019-2020, 2020-2021, 2024-2025, etc.[cite: 2]
   │
   └─ 4.2 LSTM-Based Evolution Models[cite: 2]
       ├─ Vanilla LSTM[cite: 2]
       ├─ BiLSTM[cite: 2]
       └─ Attention-LSTM[cite: 2]

                             │
                             ▼

[PHASE 5: EVALUATION & REAL-WORLD TESTING]
   │
   ├─ 5.1 Historical Backtesting (over Walk-Forward Folds)
   │
   └─ 5.2 Live Execution
       └─ Test with the real market (Around 1 weeks)[cite: 2]

=======================================================================
                             END OF PIPELINE
=======================================================================
