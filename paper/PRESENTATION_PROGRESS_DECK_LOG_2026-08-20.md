# SET50 reliability framework progress deck log - 20 August 2026

## Deliverable

- PowerPoint: `outputs/presentation_progress_v1/SET50_reliability_framework_progress_presentation.pptx`
- Slide count: 10
- Format: 16:9 widescreen
- Presenter support: Thai speaker notes are embedded on all 10 slides

## Narrative

The deck is designed for a ten-minute Super AI engineering progress presentation. It frames the project as a point-in-time reliability framework rather than a model tournament. The sequence covers the forecasting problem, the five-dimension framework, point-in-time data controls, causal VMD, predicted news and the Bull/Bear/Leader sentiment audit, regime-aware SHAP and LIME diagnostics, the final five-model result panel, observed model behavior and the engineering conclusion.

## Evidence used

All scientific visuals come from existing project artifacts. The deck uses the out-of-sample scatter and trajectory plots, the overall reliability pipeline, the expanding-window point-in-time design, the VMD architecture figure, the separated forecasting/LLM audit, the SHAP/LIME result figure and the cross-track heatmap. Reported values match the current manuscript artifacts, including the 53.64% leading mean balanced accuracy, the VMD effect range of -0.60 to +0.35 percentage points, the Leader gains of +5.93 and +6.00 points, the CNN regime-SHAP gain of +1.46 points and the 71.83% LIME low-fidelity rate.

## Visual and structural QA

Every slide was rendered at 1,920 x 1,080 pixels and inspected at full size. The cover was revised to remove title/subtitle overlap, and the multimodal slide title was shortened to eliminate clipping. The final montage shows no overlapping panels, clipped titles or unreadable footer text. The PPTX archive contains 10 slide XML files, 10 notes-slide XML files and passes ZIP integrity validation. A separate boundary check confirmed that all full-slide image objects exactly match the 16:9 slide canvas with no out-of-bounds shapes.

The standard artifact-tool renderer and its automated `slides_test.py` route could not run because the bundled internal `@oai/artifact-tool` runtime is missing from this environment. The deck was therefore validated through the original high-resolution slide renders, PowerPoint package integrity checks and explicit shape-boundary inspection.
