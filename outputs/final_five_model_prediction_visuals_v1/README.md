# Final five-model prediction visualizations

Generated: 2026-08-06 (Asia/Bangkok)

## Scope

- Five frozen benchmark models: LSTM, CNN, LSTM-CNN, LSTM-Attention, and LSTM-CNN-Attention.
- Final integrated arm: `Regime-SHAP-Numeric-News`.
- Strict out-of-sample walk-forward test predictions only: 2022-01-04 to 2025-12-18.
- Four temporal folds and five seeds; the plotted prediction at each date is the mean across the five seeds.
- 962 prediction dates per model.
- This visualization step did not refit, tune, or select any model.

## Files

- `actual_vs_predicted_oos_2022_2025.*`: complete OOS next-day actual and predicted SET50 level for all five models.
- `actual_vs_predicted_oos_2025_zoom.*`: readable 2025 detail; annotations use metrics calculated for the visible 2025 fold.
- `observed_vs_predicted_scatter_oos_2022_2025.*`: observed-versus-predicted scatter plots with a 45-degree identity line.
- `direction_alignment_oos_2022_2025.*`: daily realized direction and correct/incorrect prediction strip for every model.
- `final_five_model_results_table.*`: ranked final table; balanced accuracy is the primary ranking metric.
- `final_arm_prediction_series.csv`: plotted model-level OOS prediction series.
- `final_five_model_metrics.csv`: exact table values.
- `verification_report.json`: coverage, source, and metric-recomputation checks.

PNG files are provided for review and slides; matching PDF files are vector outputs intended for manuscript production.

## Final results

| Rank | Model | Window | Balanced accuracy, mean ± SD (%) | Direction accuracy, mean ± SD (%) | MCC | RMSE | MAE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | LSTM-CNN-Attention | 20 | 53.64 ± 2.42 | 53.43 ± 2.96 | 0.0893 | 31.77 | 25.34 |
| 2 | LSTM-CNN | 20 | 52.81 ± 2.29 | 52.11 ± 2.58 | 0.0657 | 29.79 | 23.44 |
| 3 | LSTM-Attention | 10 | 52.62 ± 1.46 | 52.40 ± 0.80 | 0.0618 | 20.41 | 15.90 |
| 4 | LSTM | 5 | 52.01 ± 1.64 | 51.98 ± 0.43 | 0.0443 | 15.97 | 12.73 |
| 5 | CNN | 20 | 51.49 ± 1.83 | 51.76 ± 1.75 | 0.0308 | 22.74 | 17.73 |

## Interpretation

LSTM is closest to the realized next-day SET50 level by RMSE and MAE. LSTM-CNN-Attention is best for the primary direction task by balanced accuracy, direction accuracy, and MCC. These are not contradictory: the next-day index level is highly persistent, so a forecast can be close to the realized level while still missing the sign of a small daily move. The direction-alignment figure should therefore accompany the level plots when reporting direction-forecasting performance.

## Provenance and verification

- Predictions: `outputs/integrated_multimodal_posthoc_v1/predictions_seed_averaged.csv`
- Frozen metrics: `outputs/integrated_multimodal_posthoc_v1/arm_summary.csv`
- Frozen sequence windows: `outputs/track_a_final_point_in_time_v2/locked_windows.csv`
- The actual series is identical across all five model panels.
- Recomputed fold-mean balanced accuracy and direction accuracy match the frozen summary to numerical precision.
