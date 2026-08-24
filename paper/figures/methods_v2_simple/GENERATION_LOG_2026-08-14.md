# Methods figures v2 — generation log

Date: 2026-08-14 (Asia/Bangkok)

The original Methods figures were retained in `methods_v1`. A second set was
created after visual review to match a conventional academic flowchart:

- Times-style serif typography.
- Thin grey borders and straight arrows.
- Pale cream for inputs, pale blue for processing/output, pale green for
  models/selection, and pale grey for controls.
- No embedded figure titles, panel banners, gradients, icons, shadows, or
  decorative callouts.
- Short process labels; detailed assumptions and counts moved to captions.

Six figures were generated in SVG, one-page PDF, and 400-dpi PNG formats. All
SVG files were parsed successfully, all PDFs were verified as one page, and all
PNGs were visually inspected for clipping, overlaps, disconnected arrows, and
directional ambiguity.

Reproduction script: `paper/tools/generate_methods_figures_simple_v2.py`
