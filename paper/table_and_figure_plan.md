# Table and figure production plan

This plan maps every proposed manuscript display to an authoritative artifact.
The mapping is journal-neutral; numbering may be compressed after a target
journal and page limit are selected.

## Main-text tables

| Display | Purpose | Authoritative source | Production status |
|---|---|---|---|
| Table 1 | Frozen models, windows, seeds, outer folds, and common SET50 cohort | `outputs/manuscript_tables_v1/table_1_protocol_cohort.csv` | Ready |
| Table 2 | Full-TA versus causal rolling-VMD BAcc ablation | `outputs/manuscript_tables_v1/table_2_numerical_ablation.csv` | Ready |
| Table 3A | Observed-news falsification against market-only and shuffled-news controls | `outputs/manuscript_tables_v1/table_3a_multimodal_falsification.csv`; `outputs/integrated_multimodal_posthoc_v1/daily_block_bootstrap_holm.csv` | Ready |
| Table 3B | Intrinsic Bull/Bear/Leader versus compute-matched self-consistency | `outputs/manuscript_tables_v1/table_3b_llm_intrinsic_separate.csv` | Ready |
| Table 4 | Regime-specific SHAP reduction by architecture and regime | `outputs/manuscript_tables_v1/table_4_regime_shap.csv` | Ready; main text uses primary rows, detailed rows move to Supplement |
| Table 5A | Source-contingent partial-2026 forward stress test | `outputs/manuscript_tables_v1/table_5a_forward_robustness.csv` | Ready |
| Table 5B | Frozen SET100 same-exchange transfer | `outputs/manuscript_tables_v1/table_5b_set100_transfer.csv` | Ready |

Table values must be copied or generated from the listed files, not manually
recomputed from rounded prose. The current main-text tables use percentages for
readability, while source CSVs retain full precision.

## Main-text figures

### Figure 1 — Five-pillar reliability-audit pipeline

Format: horizontal or two-row vector flow diagram.

Status: generated and visually inspected on 11 August 2026.

Assets:

- editable vector: `figures/figure1_reliability_audit_pipeline.svg`;
- high-resolution raster: `figures/figure1_reliability_audit_pipeline.png`
  (4000 × 3000 pixels).

Required elements:

1. point-in-time market-data governance and target definition as the shared
   foundation;
2. numerical branch: Full TA → causal rolling VMD → five frozen models;
3. text branch: relevance filter → Bull worker and Bear worker → Leader →
   dated daily news aggregation;
4. two downstream text routes: expanding Local NLP and Debate Leader, both
   joined to the same SET50 forecasting contract;
5. causal Bull/Sideway/Bear router → regime-specific train-only SHAP feature
   sets;
6. partial-2026 and SET100 robustness checks; and
7. distinct evaluation boxes for intrinsic sentiment, historical forecasting,
   explanation fidelity, forward robustness, and same-exchange transfer.

Visual rule: do not draw future news, future regime labels, full-sample VMD, or
test-set SHAP rankings as upstream inputs. The intrinsic 2023 LLM evaluation is
connected to the Leader system but is not itself a forecasting arrow.

### Figure 2 — Point-in-time expanding evaluation

Format: timeline with training, purge, test, and label-observation markers.

Show:

- pre-2022 expanding window selection;
- outer tests 2022, 2023, 2024, and 2025;
- five repeated seeds inside each fixed model/fold cell;
- train-only fitting for scaling, VMD reconstruction, NLP, regime/SHAP ranks,
  and model weights;
- label at trading day \(t+1\) becoming observable only at \(t+1\); and
- conservative news mapping from an eligible headline at date \(t\) to the
  next trading session.

Source contract: `test/primary_estimand_and_confirmatory_protocol_v1.md`,
`test/integrated_multimodal_protocol_v1.md`, and the fold fields in Table 1.

### Figure 3 — Architecture-wise treatment effects

Format: a three-panel effect-size plot with a shared zero line.

Panels retained in the auditable effect-size display:

- A: causal VMD minus Full TA;
- B: Local-NLP observed news minus Market-Only; and
- C: regime-SHAP reduced minus Full features.

Each panel should display percentage-point BAcc differences and 95% paired
intervals for the five architectures. A downstream Debate-Leader panel is not
included because no authoritative common-cohort artifact is available. Never
substitute the intrinsic LLM accuracy difference into a downstream forecasting
panel.

Primary source files are Tables 2, 3A, and 4. Use full-precision source
values. Mark Holm-significant contrasts only after adjustment; an interval that
excludes zero in a sensitivity analysis is not enough by itself.

### Figures 7–10 — Explanation, prediction behaviour and cross-pillar summary

Status: generated, integrated beside their corresponding Results evidence, and
visually checked on 18 August 2026.

| Display | Content | Authoritative source |
|---|---|---|
| Figure 7 | Regime-SHAP BAcc effects and LIME low-fidelity diagnostic in Section 4.4 | `paper/assets/figure10_shap_lime_result_audit.pdf`; `outputs/manuscript_tables_v1/table_4_regime_shap.csv`; `outputs/manuscript_tables_v1/supplement_lime_diagnostic.csv` |
| Figure 8 | Observed versus predicted SET50 levels, 2022–2025 | `outputs/final_five_model_prediction_visuals_v1/observed_vs_predicted_scatter_oos_2022_2025.pdf`; persisted five-seed OOS predictions |
| Figure 9 | Actual and predicted SET50 levels during 2025 | `outputs/final_five_model_prediction_visuals_v1/actual_vs_predicted_oos_2025_zoom.pdf`; persisted five-seed OOS predictions |
| Figure 10 | Architecture-wise point-estimate summary after Section 4.6 | `paper/assets/figure7_heatmap_summary.pdf`; Tables 2, 3A, 4 and 5B |

Figures 8 and 9 deliberately omit the former overall title, subtitle, and
how-to-read panel so that the manuscript caption carries the explanation.
Figure 7 treats LIME as a fidelity stress test, not as confirmatory feature
importance evidence. The former standalone Section 4.7 was removed: Figure 7
now follows the SHAP/LIME results in Section 4.4, Figures 8–9 follow the
post-hoc integrated-arm result, and Figure 10 remains the final cross-pillar
summary before Discussion.

## Supplementary displays

| Display | Content | Source |
|---|---|---|
| Table S1 | Data provenance, access, timezone, close convention, adjustment, row counts, and hashes | `test/market_data_governance_v1.md`; governance manifests |
| Table S2 | Full feature dictionary and availability timing | numerical feature contract and news daily-feature schema |
| Table S3 | Complete VMD results by year/seed and runtime | Track A/VMD outputs and runtime summaries |
| Table S4 | All multimodal control arms and all endpoints | integrated multimodal outputs |
| Table S5 | Local relevance/sentiment annual metrics and coverage | Track B evaluation outputs |
| Table S6 | Single pass, SC3, SC4, and Leader class metrics, cost, and latency | Track B compute-matched outputs |
| Table S8 | Regime prevalence, transition counts, frozen thresholds, and selected top-k | Track C regime outputs |
| Table S9 | Full regime-SHAP results and feature stability | `table_4_regime_shap.csv` and Track C outputs |
| Table S10 | LIME fidelity diagnostic | `outputs/manuscript_tables_v1/supplement_lime_diagnostic.csv` |
| Table S11 | Partial-2026 confusion matrices and objective-alignment details | Track D outputs |
| Table S12 | SET100 per-year/per-seed transfer and runtime | SET100 robustness outputs |
| Table S13 | Exploratory economic proxy and DSR | `outputs/manuscript_tables_v1/supplement_economic_exploratory.csv` |
| Figure S1 | Actual versus predicted index levels for the frozen five-model benchmark | `outputs/final_five_model_prediction_visuals_v1/actual_vs_predicted_oos_2022_2025.pdf` |
| Figure S2 | Direction alignment across the 2022–2025 OOS period | `outputs/final_five_model_prediction_visuals_v1/direction_alignment_oos_2022_2025.pdf` |
| Figure S5 | Regime timeline and transition matrix | Track C causal daily regime artifact |

## Quality-control checklist

- Use vector PDF/SVG for diagrams and forest plots; use at least 300 dpi for
  raster fallbacks.
- Use a color-blind-safe palette and retain distinct line styles in grayscale.
- Report BAcc as percentages or proportions consistently within each display.
- State whether intervals are exact fold sign-flip or moving-block-bootstrap
  sensitivities in every caption.
- Keep partial-2026 labelled by actual cutoff date and row count.
- Label SET100 as same-exchange breadth transfer, not independent-market
  validation.
- Present LIME only as a fidelity diagnostic; keep the economic proxy in the
  Supplement.
- Do not add a significance star to an unadjusted p-value.
- Record the generating command, script version, source hashes, and output hash
  for every final figure.
