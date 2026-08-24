# Figure 1 generation log — 11 August 2026

## Deliverable

Figure 1 was generated as a two-row vector flow diagram for the five-pillar
point-in-time reliability audit. The editable source is
`figures/figure1_reliability_audit_pipeline.svg`; the Word-ready raster is
`figures/figure1_reliability_audit_pipeline.png` (4000 × 3000 pixels).

## Content contract

The figure was derived from Section 3.1 and `table_and_figure_plan.md`. It shows:

1. the common market-data governance and point-in-time foundation;
2. the Full-TA versus causal-rolling-VMD numerical ablation;
3. separate expanding Local-NLP and Bull/Bear/Leader news routes;
4. a causal daily regime router and train-only SHAP feature sets;
5. the common panel of five frozen neural architectures;
6. the common SET50 next-day direction forecasting contract; and
7. historical, intrinsic-sentiment, explanation-fidelity, partial-2026, and
   frozen SET100 evaluation endpoints.

Solid arrows encode data or forecasting flow. Dashed teal arrows encode
evaluation-only links. In particular, the intrinsic 2023 LLM evaluation is
connected to the Leader system but is not drawn as an upstream forecasting
input. No future news, future regime labels, full-sample VMD, or test-set SHAP
rankings are depicted as available features.

## Quality checks

- SVG parsed successfully as XML.
- The SVG was rasterized at 2× scale in headless Chrome.
- The final PNG was visually inspected at original detail.
- No clipped labels, overlapping boxes, or ambiguous branch crossings remained
  after the final routing correction.
- SHA-256 (SVG):
  `C2A428E907802314935395C180A483DEFC31B5A44203570FF8D2189D34563532`.
- SHA-256 (PNG):
  `E0C67DD82AE5BBE586E9CFFD61748D9A08C5CEC1F4955C97153A960F2936110B`.
