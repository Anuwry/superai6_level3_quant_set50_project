# Six-model comparison: final integrated arm

Arm: Regime-SHAP-Numeric-News. Metrics are calculated after averaging predictions over five seeds within each of four temporal folds.

| Model | BAcc (%) | Direction accuracy (%) | MCC | RMSE | MAE | Evidence status |
|---|---:|---:|---:|---:|---:|---|
| LSTM | 52.007 | 51.983 | 0.0443 | 15.972 | 12.731 | Frozen existing result |
| CNN | 51.495 | 51.756 | 0.0308 | 22.741 | 17.729 | Frozen existing result |
| LSTM-CNN | 52.810 | 52.111 | 0.0657 | 29.794 | 23.445 | Frozen existing result |
| LSTM-Attention | 52.620 | 52.398 | 0.0618 | 20.405 | 15.897 | Frozen existing result |
| LSTM-CNN-Attention | **53.642** | **53.428** | **0.0893** | 31.768 | 25.342 | Frozen existing result |
| PIT-DERN | 50.159 | 49.497 | 0.0037 | **13.688** | **11.013** | Post-freeze exploratory extension |

Primary direction conclusion: PIT-DERN failed its predeclared promotion criteria and does not replace LSTM-CNN-Attention. Secondary descriptive finding: PIT-DERN reduced close-level regression error, illustrating that level accuracy and directional accuracy are different objectives.
