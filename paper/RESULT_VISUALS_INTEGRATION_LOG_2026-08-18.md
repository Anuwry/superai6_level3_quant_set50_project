# Results-visual integration log

Date: 18 August 2026

## Purpose

The prediction and explanation visuals were moved from a standalone late
Results subsection to the exact result blocks that define their evidence. The
change is editorial and presentational only. No model was refitted, retuned or
reselected, and no numerical result changed.

## Final Results order

1. Section 4.4 reports Table 4 and the LIME fidelity result.
2. Figure 7 immediately displays the Regime-SHAP effects and LIME low-fidelity
   rates, followed by the regime/XAI interpretation.
3. The post-hoc integrated 2x2 paragraph defines the
   Regime-SHAP-Numeric-News arm.
4. Figure 8 displays the 2022-2025 observed-versus-predicted levels for that
   arm.
5. Figure 9 displays its 2025 trajectories, followed immediately by the 2025
   metrics and model-by-model interpretation.
6. Sections 4.5 and 4.6 retain the partial-2026 and frozen SET100 results.
7. Figure 10 remains after Section 4.6 as the cross-pillar point-estimate
   summary immediately before Discussion.

The former heading `4.7 Prediction-level diagnostics separated level tracking
from directional discrimination` was removed. The wording `final integrated
arm` was replaced with `post-hoc integrated arm` to match the registered
evidence hierarchy.

## Figure renumbering

| Final number | Display | Previous number |
|---|---|---|
| Figure 7 | Regime-SHAP and LIME explainability audit | Figure 10 |
| Figure 8 | Observed versus predicted levels, 2022-2025 | Figure 8 |
| Figure 9 | Actual versus predicted trajectories, 2025 | Figure 9 |
| Figure 10 | Cross-pillar architecture-wise summary | Figure 7 |

All accompanying textual references, captions and image alternative text were
updated to the final numbering.

## Pagination and quality assurance

- Final manuscript length: 35 pages.
- Document structure: 206 paragraphs, 7 tables and 10 inline figures.
- Figure locations in the rendered manuscript: Figure 7 page 21, Figure 8 page
  22, Figure 9 page 23 and Figure 10 page 28.
- Moving the 2025 results paragraph after Figure 9 removed the large blank area
  produced by the earlier forced page break.
- Microsoft Word rendering was inspected for all 35 pages; no clipping,
  overlap, separated caption, broken table or unreadable figure was found.
- Accessibility audit: 0 high-, 0 medium- and 0 low-severity findings.
- `Table 3C`, `pending artifact`, the former Section 4.7 and `final integrated
  arm` have zero occurrences in the final manuscript.

## Final artifact

`paper/newest_original_manuscript_results_visuals_integrated.docx`

SHA-256:

`4C67803B89C6929939D56905A21349DD7E62A9E61AA56D204B67F6B2222C8658`

Builder script:

`paper/tools/add_result_visuals_and_discussion.py`

Builder SHA-256:

`F6611DB4D88BE5E78A41F86FA9D82A2B6A229E426FD60711AC6B4066E1F4E21E`
