# Manuscript revision log: Table 3C removal and Figure 7 redesign

Date: 2026-08-14

## Input and output

- Input: `paper/newest_original_manuscript_spacing_corrected.docx`
- Output: `paper/newest_original_manuscript_revised.docx`
- The input file was preserved unchanged.

## Evidence-alignment edits

1. Removed the unavailable placeholder Table 3C, including its caption, three pending rows and all statements that presented a Leader-derived downstream SET50 comparison as completed evidence.
2. Renamed Section 4.3 to restrict its claim to the locked intrinsic sentiment benchmark.
3. Revised the Table 3B note to state explicitly that intrinsic sentiment accuracy is not evidence of downstream forecasting value.
4. Revised Section 3.6 so that the completed downstream route uses expanding out-of-sample Local-NLP sentiment and the registered falsification controls, while the Bull/Bear/Leader system remains a separate locked intrinsic benchmark.
5. Retained the completed post-hoc Local-NLP x Regime-SHAP 2x2 analysis and moved it to Section 4.4 after Table 4. This analysis is not a substitute for the removed Leader-derived downstream comparison.
6. Retained the Leader-derived forecasting experiment only as future work.

## Figure edits

- Replaced Figure 5 with a two-panel diagram that visually separates the completed Local-NLP downstream forecasting audit from the locked Bull/Bear/Leader intrinsic sentiment audit. No arrow links the Leader output to a forecasting arm.
- Replaced the forest-style Figure 7 with an annotated heatmap of architecture-wise balanced-accuracy point estimates and a separate horizontal-bar panel for the Leader intrinsic sentiment endpoint.
- Figure 7 directs readers to Tables 2, 3A, 4 and 5B for confidence intervals and Holm-adjusted p-values; it does not imply that cell colors establish statistical significance.

## Verification

- Word-rendered length: 27 pages.
- Tables: 7.
- Inline figures: 7.
- Word equations: 14.
- Word fields: 36, preserving the existing Zotero citation fields.
- Search checks confirmed that `Table 3C`, `Authoritative common-cohort artifact pending`, `Leader-derived route`, `extended to SET50 forecasting` and `Forest-style panels` no longer appear.
- All 27 pages were exported through Microsoft Word and visually inspected. The revised Table 3B-to-Section 4.4 transition, Figure 5 and Figure 7 render without clipping or overlap.

## Figure 5 layout correction

- Rebuilt Figure 5 after the first two-panel version proved visually crowded at manuscript scale.
- Replaced intersecting/adjacent terminal boxes with a strict top-to-bottom hierarchy and separated the paired forecast result from the falsification arms.
- Consolidated the LLM controls into one intrinsic-comparison endpoint to eliminate crossing connectors and to avoid implying a downstream Leader forecasting arm.
- Used a compact source canvas so labels remain readable after scaling to 15.0 cm in Word.
- Final corrected manuscript: `paper/newest_original_manuscript_revised_figure5_fixed.docx`.
- Re-rendered and visually checked all 27 pages; Figure 5 is unclipped, non-overlapping and remains adjacent to its caption on page 13.
