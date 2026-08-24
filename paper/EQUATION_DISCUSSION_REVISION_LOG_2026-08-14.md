# Equation and Results–Discussion Revision Log

Date: 2026-08-14  
Source manuscript: `paper/newest_original_manuscript_revised_figure5_fixed.docx`  
Revised manuscript: `paper/newest_original_manuscript_equations_discussion_revised.docx`

## Scope completed

- Renumbered every displayed equation as one continuous manuscript-wide sequence, `(1)` through `(14)`.
- Removed compound equation labels such as `(3a)`, `(3b)`, `(3c)`, `(5a)` and `(8a)` and updated all in-text equation references.
- Added a source citation in the sentence that introduces or interprets each equation or equation group.
- Preserved citations as live Zotero `ADDIN ZOTERO_ITEM CSL_CITATION` fields rather than replacing them with plain text.
- Expanded Section 4 into integrated **Results and Discussion**, using the sequence: study result, comparison/contrast with prior evidence, plausible mechanism, and bounded inference.
- Added comparative discussion for causal VMD, predicted financial news, role-structured LLM inference, regime-aware SHAP selection, the partial-2026 stress test, and frozen SET100 transfer.

## Equation sequence and supporting sources

| Equation | Content | Source(s) cited in surrounding text |
|---:|---|---|
| (1) | Daily sentiment score, coverage and article-level aggregation | Uthayopas et al. (2025) |
| (2) | Next-session regression target and directional labels | Fischer and Krauss (2018); Hoseinzade and Haratizadeh (2019) |
| (3)–(5) | Trailing-window VMD, removed high-frequency mode, denoised close and energy ratio | Dragomiretskiy and Zosso (2014); T. Liu et al. (2022) |
| (6) | Mean squared error training objective | T. Liu et al. (2022); Y. Liu et al. (2024) |
| (7)–(9) | Causal regime score, horizon set and train-only deadband | Pagan and Sossounov (2003); Moskowitz et al. (2012); Wilder (1978) |
| (10) | Regime-specific mean absolute SHAP importance and top-k selection | Lundberg and Lee (2017) |
| (11) | Balanced accuracy | Brodersen et al. (2010) |
| (12)–(14) | Paired model-year effect, sign-flip statistics and exact probability | Holm (1979); Politis and Romano (1992) |

## Results–Discussion additions

1. **VMD:** contrasted continuous-price denoising evidence with the manuscript's next-day sign endpoint and explained why smoothing can improve magnitude error while removing small sign-bearing movements.
2. **Predicted news:** contrasted attention-rich news interaction studies with daily scalar sentiment aggregation and interpreted shuffled-news results as a temporal-information falsification test.
3. **LLM debate:** separated role diversity from extra inference calls, compared the Leader against compute-matched controls and the local classifier, and distinguished intrinsic sentiment accuracy from downstream forecasting value.
4. **Regime-SHAP:** distinguished sustained market-phase definitions from causal day-level routing and explained architecture-dependent gains through compression, redundancy and smaller regime-conditioned samples.
5. **Partial 2026:** interpreted one-sided prediction collapse using balanced accuracy and treated the economic proxy as an exploratory selection-risk diagnostic.
6. **SET100 transfer:** framed the test as frozen same-exchange transportability rather than independent-market replication and discussed frozen representation mismatch and limited transfer power.

## Integrity and visual QA

- Pages: 31
- Tables: 7
- Figures: 7
- Displayed equations: 14
- Zotero citation fields: 52, all with non-empty displayed results
- Zotero bibliography fields: 1
- Residual compound equation labels or placeholder citation tags: none
- Rendered all 31 pages to PDF/PNG and inspected every contact sheet plus the equation-heavy pages at full resolution.
- Verified that equation numbers are right-aligned, citations are visible, tables and figures remain inside the page area, discussion paragraphs do not overlap, and the bibliography renders without clipping.

## Results and Discussion separation

Revised output: `paper/newest_original_manuscript_results_discussion_separated.docx`

- Renamed Section 4 from **Results and Discussion** to **Results**.
- Retained concise literature comparison and mechanism-level interpretation beside the relevant result in Sections 4.1-4.6.
- Added a separate Section 5 **Discussion** containing four cross-cutting synthesis paragraphs: architecture-representation fit, the compression-capacity trade-off, intrinsic LLM performance versus downstream forecasting value, and inferential/generalizability implications.
- Renumbered **Conclusion** from Section 5 to Section 6; its limitations and future-work paragraphs remain within the conclusion as requested.
- Added four live Zotero citation fields to the new Discussion; the final document contains 56 non-empty Zotero citation fields and one Zotero bibliography field.
- Structural checks found no residual `4. Results and Discussion` heading and no citation placeholders.
- Final render: 32 pages, 7 tables, 7 figures and 14 equations. All pages were visually inspected; no clipping, overlap, broken tables, misplaced figures or bibliography overflow was observed.

## LLM benchmark scope correction

Revised output: `paper/newest_original_manuscript_llm_benchmark_scope_corrected.docx`

- Clarified that the locked intrinsic benchmark compares the Bull/Bear/Leader debate system with equal-call and near-cost LLM controls.
- Removed the Local-NLP-versus-Leader accuracy comparison and its untested lexical/label-policy explanation from Section 4.3.
- Removed the Local-NLP superiority statement from Section 5 while retaining the distinction between intrinsic sentiment evaluation and unavailable Leader-derived downstream forecasting evidence.
- Preserved Local-NLP references in the separate out-of-sample news-construction and downstream forecasting components, where they remain methodologically relevant.
- The document retains 56 non-empty Zotero citation fields, one Zotero bibliography field, 14 equations, 7 tables and 7 figures.
- Rendered and inspected all 32 pages; the revised paragraphs on pages 20 and 26 are clean and no layout regression was observed.
