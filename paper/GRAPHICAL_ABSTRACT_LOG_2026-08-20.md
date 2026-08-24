# Graphical abstract log - 20 August 2026

## Authoritative real-evidence v4

- PNG: `output/graphical_abstract_v4/graphical_abstract_set_reliability_real_images_v4.png`
- PDF: `output/pdf/graphical_abstract_set_reliability_real_images_v4.pdf`
- Manuscript asset copies: `paper/assets/graphical_abstract_set_reliability_real_images_v4.png` and `.pdf`
- Source layout: `paper/tools/graphical_abstract_real_images_v4.html`
- PNG dimensions: 3,600 x 1,800 pixels (2:1 aspect ratio)
- PDF dimensions: 864 x 432 points (12 x 6 inches), one page

Version 4 replaces the illustrative mini-icons in v3 with unmodified project figures and experimental plots. The left panel contains the actual 2025 out-of-sample observed-versus-predicted model plot and the separated downstream-news/intrinsic-LLM audit. The center panel contains the point-in-time expanding-fold protocol, causal VMD five-architecture experiment and causal regime-XAI pipeline. The right panel contains the cross-track effect heatmap, SHAP/LIME result diagnostic and final five-model held-out table. Only layout, scaling, captions, borders and arrows are added; chart values, labels and plotted traces are not redrawn or altered.

The v4 PNG was visually inspected at full 3,600 x 1,800 resolution after shortening every caption to a single line. No caption, panel or arrow is clipped or overlapping. The PDF was generated from the same fixed-size HTML canvas and verified as a one-page PDF with a 864 x 432 point MediaBox. This image-rich version is recommended when the journal permits small embedded result panels; v3 remains preferable when the journal requests a simpler, more schematic graphical abstract.

## Authoritative image-rich soft-minimal v3

- PNG: `output/graphical_abstract_v3/graphical_abstract_set_reliability_visual_v3.png`
- Vector PDF: `output/pdf/graphical_abstract_set_reliability_visual_v3.pdf`
- PNG dimensions: 3,600 x 1,800 pixels (2:1 aspect ratio)
- PDF dimensions: 864 x 432 points (12 x 6 inches), one page
- PNG SHA-256: `B58CF045D5B811941994DB0F4F1AA98056251CCC7295D4D9DC1673A445D9306A`
- PDF SHA-256: `88F2341B8E1D54E02A9887BDF59DADBE7F6242FE0CD934AD7BECFF36C4DF2975`

Version 3 removes the overall title, the large `INPUT/METHOD/OUTPUT` headings and the take-home band. The reading order remains left to right through two large arrows, while the space is reassigned to scientific illustrations: the observed SET50 series, news sentiment symbols, a neural-network sketch, expanding-fold blocks, causal VMD waveforms, market/news fusion, Bull/Bear/Leader roles, regime routing, SHAP bars, a LIME fidelity ring, partial-2026 diagnostics and SET50-to-SET100 transfer. The palette remains a restrained soft blue-grey with one neutral news tint. Versions 1 and 2 remain archived intermediate assets.

## Content represented

The graphical abstract summarizes the complete five-dimension reliability audit:

1. point-in-time inputs and expanding out-of-sample evaluation;
2. Full-TA versus causal rolling VMD;
3. predicted-news forecasting and the separate intrinsic Bull-LLM/Bear-LLM/Leader benchmark;
4. causal Bull/Sideway/Bear market regimes, train-only SHAP and the LIME fidelity diagnostic; and
5. the partial-2026 forward stress test and frozen SET100 same-exchange transfer.

The left-side sparkline is drawn from the project's dated SET50 daily close file rather than from a decorative or externally sourced stock image. The five registered architectures, common next-session Up/Down target and held-out 2022-2025 cohort are shown explicitly.

## Evidence boundaries

- The Bull-LLM/Bear-LLM/Leader system is written within the multimodal row, separately from the Bull/Sideway/Bear market-regime row.
- The Leader gain is labelled as an intrinsic sentiment result rather than downstream forecasting evidence.
- VMD, predicted-news, Regime-SHAP, LIME, partial-2026 and SET100 statements match the manuscript's reported evidence and do not imply universal superiority.
- The locked-evaluation result states that no primary balanced-accuracy contrast passed Holm adjustment without adding a separate headline or take-home-message panel.

## Visual verification

The v3 PNG was inspected at full resolution. The PDF was rendered back to a PNG and inspected independently. The final files contain no clipped labels, overlapping panels, broken glyphs or unreadable arrows. All mini-plots and diagrams remain separated at journal scale, and the three visual groups are connected without section-title clutter. The regenerated PDF contains one page and extractable text, while the diagram and chart remain vector elements where supported by Matplotlib.

## Reproduction

- Authoritative v3 generator: `paper/tools/generate_graphical_abstract_visual_v3.py`
- Archived v2 generator: `paper/tools/generate_graphical_abstract_minimal.py`
- Archived v1 generator: `paper/tools/generate_graphical_abstract.py`
