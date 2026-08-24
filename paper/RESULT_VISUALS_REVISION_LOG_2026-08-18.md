# Results visuals and discussion revision log

Date: 18 August 2026

## Scope

This revision updates the two requested prediction plots, adds a combined
SHAP/LIME diagnostic, and integrates all three displays into the current
manuscript without refitting, retuning, or selecting a new model.

## Plot revisions

1. `observed_vs_predicted_scatter_oos_2022_2025.pdf`
   - removed the overall title and subtitle;
   - removed the explanatory "How to read" panel;
   - retained model labels, axes, the identity line, and model-level metrics;
   - arranged the five registered architectures in a balanced 3-over-2 layout.
2. `actual_vs_predicted_oos_2025_zoom.pdf`
   - removed the overall title and subtitle;
   - removed the bottom interpretation note;
   - retained the shared legend, panel labels, actual/predicted series, and
     model-level metrics.
3. `figure10_shap_lime_result_audit.pdf`
   - Panel A reports Regime-SHAP minus Regime-All BAcc effects with 95%
     intervals for all five architectures;
   - Panel B reports the low-fidelity proportion of the LIME diagnostic for
     every model-regime cell;
   - the figure is descriptive/diagnostic and does not mark any unadjusted
     result as significant.

## Manuscript integration

Output manuscript:

`paper/newest_original_manuscript_results_visuals_discussed.docx`

Section 4.7 was added before the standalone Discussion and contains:

- Figure 8: observed versus predicted SET50 levels, 2022–2025;
- Figure 9: 2025 actual-versus-predicted trajectories;
- Figure 10: Regime-SHAP effects and LIME fidelity diagnostics;
- model-by-model interpretation that distinguishes index-level tracking from
  next-day directional discrimination;
- cross-reference discussion with prior financial forecasting and XAI work.

Section 5 additionally explains why high observed-predicted level correlation
does not imply strong next-day sign prediction and why unstable local
explanations should not be treated as confirmatory evidence.

## Quantitative evidence used in the text

Across the 2022–2025 common OOS cohort (962 sessions per architecture):

| Model | BAcc (%) | RMSE | MAE | Observed-predicted correlation |
|---|---:|---:|---:|---:|
| LSTM | 52.01 | 15.97 | 12.73 | 0.983 |
| CNN | 51.49 | 22.74 | not highlighted | within the audited 0.927–0.983 range |
| LSTM-CNN | 52.81 | 29.79 | not highlighted | within the audited 0.927–0.983 range |
| LSTM-Attention | 52.62 | 20.41 | not highlighted | within the audited 0.927–0.983 range |
| LSTM-CNN-Attention | 53.64 | 31.77 | 25.34 | 0.927 |

The manuscript does not infer superior directional skill from lower RMSE or
visual trend proximity. In 2025, all models remained strongly Up-skewed
(74.79–86.32% predicted Up versus 48.72% actual Up), with Down recall of only
16.67–28.33%. LSTM-CNN had the highest 2025 BAcc (55.15%), while LSTM had the
lowest 2025 RMSE (21.57), illustrating the difference between level error and
directional discrimination.

Regime-SHAP BAcc effects were −0.10, +1.46, +0.05, −1.03, and −0.74 percentage
points for LSTM, CNN, LSTM-CNN, LSTM-Attention, and LSTM-CNN-Attention,
respectively. All intervals crossed zero and no effect survived Holm
adjustment. The LIME audit classified 1,293 of 1,800 repeats (71.83%) as low
fidelity; model-regime cells ranged from 65.8% to 77.5%.

## Quality assurance

- persisted predictions were used; no experiment was rerun or selected after
  inspecting these plots;
- all five models contain 962 common OOS rows and the same actual series;
- plotted metrics were reconciled against the authoritative prediction files;
- the manuscript contains 10 inline figures, with descriptive alternative text
  for all figures;
- the document rendered to 36 pages through Microsoft Word and all pages were
  visually checked for overlap, clipping, caption separation, and readability;
- accessibility audit result: 0 high-, 0 medium-, and 0 low-severity findings.

## Final artifacts

- manuscript: `paper/newest_original_manuscript_results_visuals_discussed.docx`
- clean scatter: `outputs/final_five_model_prediction_visuals_v1/observed_vs_predicted_scatter_oos_2022_2025.pdf`
- clean 2025 zoom: `outputs/final_five_model_prediction_visuals_v1/actual_vs_predicted_oos_2025_zoom.pdf`
- SHAP/LIME visual: `paper/assets/figure10_shap_lime_result_audit.pdf`
- diagnostic metrics: `outputs/final_five_model_prediction_visuals_v1/model_behavior_diagnostics.csv`
- QA PDF: `paper/qa/render_results_visuals_discussed_word/final_qa.pdf`

## SHA-256 record

| Artifact | SHA-256 |
|---|---|
| Final manuscript DOCX | `DD186206E50007B8FC740F32E6C60DCE519EBF59206A6B57FC079F4C1ABA5C0F` |
| Clean scatter PDF | `B8C0C987AED8B8348E1C0D5CB98E0704CA149A37E77347495F86EEB3019A7F3A` |
| Clean 2025 zoom PDF | `A44CCAA90F35C20DD920A18BBB257ED61946293D60D807EEF5487C6AF18B2FC4` |
| SHAP/LIME PDF | `4C6110C9261B9D3DFF3EC673CB238341D75AA758FB58C61BCEC3AC671A864546` |
| Behaviour diagnostics CSV | `1DAEEBBD291B9E20022E4219F8444E53F7F19F296D685622A2E3186D5496B2FE` |
| Prediction-visual generator | `FDC1830147502BA4431E535B2CF746B89F70118954CCCA974003FEFFAB3EC40D` |
| SHAP/LIME generator | `CCE83C32613960F40F9262EE2BD57C13BE71E487BEB2BF1D06D2DE5BB6CE6DCB` |
| Manuscript integration script | `98C91D92AEBD00116CC8572A6F9DCFE8F3DC019A09B36B09A8F1C14815CE9EBE` |
