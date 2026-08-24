# Executed SET50 Multimodal Reliability Pipeline

Status: Track A--D and the post-hoc integrated news + regime-SHAP extension
are complete. The integrated extension passed its 100-cell/800-fit integrity
audit on 2026-08-01 (Asia/Bangkok).

```text
SET50 market data                         Historical financial news
2012-05-03 to 2025-12-18                 heterogeneous sources
        |                                          |
        v                                          v
Label_Date point-in-time purge             frozen expanding Local NLP
boundary feature row = context only         relevance + sentiment
        |                                          |
        v                                          v
116 causal TA features                  8 out-of-sample daily news features
        |                                          |
        v                                          |
+6 causal rolling VMD                              |
        |                                          |
        v                                          |
122 numerical features                             |
        |                                          |
        +----------------------+-------------------+
                               |
                               v
Frozen five-model contract
LSTM W5 / CNN W20 / LSTM-CNN W20
LSTM-Attention W10 / LSTM-CNN-Attention W20
five seeds; 2022, 2023, 2024, partial-2025 folds
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
Global model                       causal daily regime router
                                  Bull / Sideway / Bear
                                  + frozen numerical SHAP sets
                 |                           |
                 +-------------+-------------+
                               |
                               v
Executed 2 x 2 post-hoc extension
Global-Numeric / Global-Numeric-News
Regime-SHAP-Numeric / Regime-SHAP-Numeric-News
                               |
                               v
Seed-averaged fold evaluation
BAcc primary; DA/MCC/RMSE/MAE secondary
exact sign-flip + Holm across five models
circular moving-block bootstrap sensitivity
                               |
                               v
Mixed/negative result
news improved regime-pipeline BAcc in 2/5 models
no BAcc contrast survived Holm correction
```

The LLM single-pass and Bull/Bear/Leader debate experiment is an intrinsic
text benchmark. It is not the source of the downstream news features above.
The integrated extension makes no new API calls and costs USD 0 incrementally.

The partial-2026 Track D forward test remains numerical because no matching
frozen 2026 news feature file exists. Therefore this is an executed integrated
2019--2025 forecasting extension, not a fully integrated live/deployed trading
system and not a pristine confirmatory holdout.

Controlling evidence:

- `test/integrated_multimodal_protocol_v1.md`
- `test/integrated_multimodal_freeze_v1.json`
- `test/integrated_multimodal_execution_log_v1.md`
- `outputs/integrated_multimodal_posthoc_v1/integrity_audit.json`
- `outputs/integrated_multimodal_posthoc_v1/paper_integrated_table.csv`
