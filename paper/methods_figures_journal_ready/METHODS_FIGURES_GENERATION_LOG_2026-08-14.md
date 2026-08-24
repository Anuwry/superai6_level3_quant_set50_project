# Methods figures generation and QA log

Date: 2026-08-14 (Asia/Bangkok)

## Scope

Six restrained, journal-style figures were generated for Methods Sections 3.1–3.8. The visual language uses a white background, dark-gray text and borders, one muted-blue accent, and limited red only where a warning or negative class must be distinguished. The figures were constructed as scientific vector diagrams/data plots rather than generative artwork so that labels, counts, and pipeline relationships remain exact and editable.

## Deliverables

1. `figure1_reliability_audit_pipeline_simple` — overall point-in-time reliability-audit pipeline (Section 3.1).
2. `figure2_news_data_and_oos_sentiment` — news coverage, class composition, forward filtering, and next-session mapping (Section 3.2).
3. `figure3_point_in_time_expanding_evaluation` — point-in-time label contract, expanding selection folds, and frozen outer evaluation (Section 3.3).
4. `figure4_numerical_vmd_and_architectures` — technical feature families, causal rolling VMD, and the five registered neural architectures (Sections 3.4–3.5).
5. `figure5_multimodal_falsification_and_leader` — multimodal controls, locked Bull/Bear/Leader benchmark, and downstream sentiment routes (Section 3.6).
6. `figure6_regime_shap_and_inference` — causal regime routing, SHAP selection controls, and hierarchical inference (Sections 3.7–3.8).

Each figure is supplied as:

- SVG for lossless editing and preferred Word insertion when supported.
- PDF for journal production and archival use.
- 400-dpi PNG as the compatibility fallback.

## Data contracts used in Figure 2

- Labeled StockTBSA period: 2018–2023.
- Valid positive/neutral/negative article–ticker pairs: 12,706.
- Valid pairs by year: 2018 = 2,840; 2019 = 2,740; 2020 = 2,314; 2021 = 2,154; 2022 = 1,325; 2023 = 1,333.
- Locked 2023 class counts: negative = 92; neutral = 585; positive = 656.
- 2024 forward source: 32,966 raw records; 2,099 point-in-time SET50 records; 1,223 selected records; 225 mapped sessions.
- 2025 forward source: 36,858 raw records; 2,520 point-in-time SET50 records; 1,569 selected records; 224 mapped sessions.

## Verification completed

- All six SVG files parse successfully as XML.
- All six PDF files contain exactly one page.
- All six PNG files are exported at 400 dpi.
- Minimum PNG dimensions exceed 2,200 pixels in width.
- Data counts in Figure 2 were programmatically checked against the stated contracts.
- Every figure was visually inspected for clipping, label overlap, arrow direction, hierarchy, and grayscale readability.
- Figure 4 was checked to ensure the architecture chains include their correct dense layer before the output layer.
- Figure 5 was checked after final layout adjustment to prevent overlap in the downstream-route panel.

## Reproducibility

Generation script: `paper/tools/generate_methods_figures.py`

SHA-256 hashes of the final PNG files:

- Figure 1: `8879AD61C99B591CA1C41281118AACB885E02F9266A68A2468519FF72C872C42`
- Figure 2: `FC863B771DFD2C17553B832876D726D07C11C177AA9DD10A6D7A0328811F6CA3`
- Figure 3: `849AC98AD10C4209359A645417DAED58C95D2CB363BB8DBBFA7CE8A0B2B03F87`
- Figure 4: `D3DF6CD1D243B5709725246BEC1A5C439E2C7723F9B20D752F587EEDCBE17DF0`
- Figure 5: `FE8B92395A8797BBD82D7C3A25537F27473EFF15BC94C7DDAFB630C072646625`
- Figure 6: `15DC8F8336A8E4DCE5AA64AD6932610A1D67C70C1B501B46071FAE39B2896F0E`

## Manuscript note

Recommended captions and exact placement are recorded in `METHODS_FIGURES_CAPTIONS_AND_PLACEMENT.md`. If all six figures remain in the main text, the existing Results figures must be renumbered after insertion.
