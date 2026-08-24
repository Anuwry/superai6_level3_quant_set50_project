# Journal formatting log — 20 August 2026

## Authoritative output

- File: `SET_direction_manuscript_journal_formatted_v3.docx`
- Title: *Evaluating Multimodal and Regime-Aware Deep Learning for Next-Day SET Index Direction Forecasting*
- Correspondence: `Correspondence: Arsanchai.su@wu.ac.th`
- Size: 3,362,610 bytes
- SHA-256: `84231B0AAD6CB15F446793B22CA07739993ECB5F52658910E0463F63D8064059`

## Formatting work completed

1. Replaced the manuscript title and correspondence line with the author-specified text.
2. Converted all 30 numbered and back-matter headings from manually bolded body paragraphs to genuine Word `Heading 1` and `Heading 2` styles.
3. Shortened explanatory subsection labels to concise academic headings. For example, `3.3 Point-in-time contract, temporal splits and frozen windows` became `3.3 Point-in-Time Evaluation`.
4. Separated `Limitations` and `Future Work` from their body text and converted both labels to genuine subsection headings.
5. Set all 17 Figure/Table captions and all 6 table notes to 10 pt Times New Roman. Caption labels remain bold, while their descriptions use regular weight; `Note:` labels use italics.
6. Typeset inline mathematical notation using italic variables, Greek symbols, subscripts and superscripts where appropriate. Examples include `N_t`, `n_t`, `C_(t+1)`, `u_(k,t)`, `omega_(k,t)`, `sigma_(t,v(h))`, `ADX_(14,t)`, `lambda_f`, `S_r(k)`, `R^2`, `alpha`, `tau`, and `10^-7`.
7. Preserved the existing displayed equations and their sequential numbering (1–14).
8. Preserved all embedded tables, figures, Word fields and Zotero citation metadata.
9. Standardized all 14 displayed equations to 12 pt Cambria Math, set equation numbers to 11 pt Times New Roman, and applied consistent 5 pt spacing above and below each equation. The longer inferential equations (12)–(14) were rendered and confirmed to fit without collision or overflow.
10. Rebuilt Figure 5 at print resolution with larger horizontal and vertical gaps between the data, feature, model and evaluation stages. The two audit panels remain semantically separate, all arrows terminate cleanly outside text, and the original manuscript image aspect ratio was retained so that pagination and surrounding text were unchanged.

## Structural verification

- Body paragraphs: 208
- Tables: 7
- Embedded figures/drawings: 10
- Heading 1 paragraphs: 9
- Heading 2 paragraphs: 21
- Figure/Table captions: 17
- Table notes: 6
- Live Zotero CSL citation fields: 67
- Zotero bibliography fields: 1
- Valid parsed Zotero citation JSON records: 67/67
- Field-character balance: 68 begin, 68 separate, 68 end
- Embedded media retained: 10/10
- Raw authoring notation such as `_` and `^` remaining before References: none
- Placeholder/error scan (`TODO`, `TBD`, `Error!`): none

## Visual verification

The DOCX was exported through Microsoft Word and rendered to page images. All 35 pages were visually inspected. The final layout showed no clipped content, overlapping objects, broken equations, orphaned headings, lost captions, missing tables, or missing figures. Figure 5 was additionally inspected at full-page resolution on page 13; its boxes, arrows, panel divider and endpoint notes are separated and legible. Figure and table descriptions render at 10 pt and the mathematical inline notation displays with the intended subscript/superscript typography.

## Reproduction scripts

- `paper/tools/journal_format_manuscript.py`
- `paper/tools/audit_journal_format.py`
- `paper/tools/render_journal_formatted.ps1`
- `paper/tools/generate_figure5_spacious.py`
- `paper/tools/replace_figure5_v3.py`
