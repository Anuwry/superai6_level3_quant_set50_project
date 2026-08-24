# Q2 Evidence Consolidation Log

Status: **COMPLETE - NON-MANUSCRIPT AUDIT ARTIFACT**

This task consolidates existing corrected Track A-D evidence and does not start manuscript writing.
No model was retrained and no result was overwritten.

## Scope decision

- Shadow deployment: omitted. A five-day pilot would add little inferential evidence beyond the existing 138-day partial-2026 forward evaluation.
- The five registered architectures and their frozen windows remain unchanged.
- Balanced Accuracy is the primary directional improvement metric; DA and RMSE remain separate secondary endpoints.

## Does the same three models improve throughout?

No. There is no stable same three models winner set across the pipeline. The set changes with the intervention, comparator, and metric:

| Comparison | Improved models | Count |
|---|---|---:|
| Track A +VMD, BAcc | lstm_cnn | 1/5 |
| Track B +News, BAcc | lstm, lstm_attention | 2/5 |
| Track C Regime-SHAP vs Global-All, BAcc (descriptive) | cnn, lstm_cnn, lstm_cnn_attention | 3/5 |
| Track C Regime-SHAP vs Regime-All, BAcc (isolated) | cnn, lstm_cnn | 2/5 |
| Integrated Regime-SHAP +News vs Regime-SHAP Numeric, BAcc | lstm, lstm_attention | 2/5 |
| Integrated final vs Global-Numeric, BAcc | lstm, lstm_cnn_attention | 2/5 |
| Track D Multitask vs Direct, BAcc | lstm_attention | 1/5 |
| Track D Multitask vs Direct, DA | lstm, lstm_cnn, lstm_attention | 3/5 |

A positive DA change in Track D is not accepted as improvement when BAcc remains flat or worsens, because several outputs collapse toward the majority Up class.

## Evidence classification

- Track A: corrected confirmatory paired ablation with four temporal units and low power.
- Track B fusion: corrected paired ablation; intrinsic text/LLM evidence remains separate from downstream forecasting.
- Track C: post-hoc robustness evidence. Capacity-confounded end-to-end comparisons are descriptive; registered within-router and capacity-matched contrasts carry Holm-adjusted inference.
- Integrated multimodal v1: post-hoc 2 x 2 extension using a common 2019-start cohort; no BAcc contrast passed Holm correction. Downstream news comes from frozen Local NLP at USD 0 incremental API cost.
- Track D: frozen model protocol on a source-contingency partial-2026 forward set; not a pristine registered-source confirmatory holdout.

## Claim rule frozen for the next stage

Every improvement statement must name: model, exact comparator, metric, effect size, evidence class, and multiplicity status. Counts such as '3 of 5 improved' cannot stand alone.

## Generated artifacts

- `outputs/q2_evidence_package/master_evidence_matrix.csv`
- `outputs/q2_evidence_package/q2_claim_status.csv`
- `outputs/q2_evidence_package/source_manifest.json`
- `outputs/integrated_multimodal_posthoc_v1/paper_integrated_table.csv`
- `outputs/integrated_multimodal_posthoc_v1/fold_inference_holm.csv`
- `test/integrated_multimodal_execution_log_v1.md`
