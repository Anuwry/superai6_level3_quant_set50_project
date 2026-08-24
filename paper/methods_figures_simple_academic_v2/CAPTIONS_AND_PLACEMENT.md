# Simple Methods figures: captions and placement

These figures use a conventional academic-flowchart style. Detailed parameter
values remain in the Methods text and caption rather than inside the diagram.

## Figure 1 — Section 3.1

**Figure 1. Overall point-in-time reliability-audit pipeline.** Historical
SET50 data and dated financial news are processed under a common point-in-time
contract. Numerical denoising, regime-aware SHAP feature selection, and
out-of-sample sentiment feed the same five registered neural architectures.
The resulting next-day forecasts are evaluated with expanding outer tests,
followed by frozen SET100 transfer and a source-contingent partial-2026 stress
test.

Place after the first paragraph of Section 3.1.

## Figure 2 — Section 3.2

**Figure 2. Point-in-time financial-news and sentiment construction.** Dated
StockTBSA labels from 2018–2023 and forward news from 2024–2025 are normalized,
filtered for SET50 relevance, and processed using only previously observed
labels. Predicted sentiment is mapped from publication time to the next
tradable session before daily aggregation. The locked audit includes 12,706
valid labelled article–ticker pairs, 1,223 selected 2024 records across 225
sessions, and 1,569 selected 2025 records across 224 sessions.

Place after the second paragraph of Section 3.2.

## Figure 3 — Section 3.3

**Figure 3. Point-in-time target contract and expanding-window evaluation.**
Features known by the close of day t predict the direction of day t+1, whose
label is not observed until after t+1. Training data expand through time, while
2022, 2023, 2024, and 2025 are retained as separate held-out outer years. Model
selection is completed before the outer evaluation, and all transforms are
fitted on training data only.

Place after the paragraph defining the outer folds and seeds in Section 3.3.

## Figure 4 — Sections 3.4–3.5

**Figure 4. Numerical inputs, causal VMD, and registered architectures.** The
116-feature technical-analysis representation is compared with a 122-feature
representation produced by adding six variables from causal rolling VMD fitted
to a trailing 60-day window. Both representations are evaluated through the
same LSTM, CNN, LSTM–CNN, LSTM–Attention, and LSTM–CNN–Attention panel using a
paired TA versus TA-plus-VMD comparison.

Place after the final paragraph of Section 3.5.

## Figure 5 — Section 3.6

**Figure 5. Multimodal forecasting and role-structured sentiment audit.** The
five registered models produce paired forecasts with and without point-in-time
sentiment. Financial news is filtered for SET50 relevance before Bull and Bear
workers provide arguments to a Leader that returns the dated sentiment score.
Forecasting falsification arms and compute-matched sentiment controls are kept
outside the main prediction path but included in the audit.

Place after the paragraph describing forecasting and LLM controls in Section
3.6.

## Figure 6 — Sections 3.7–3.8

**Figure 6. Causal regime-aware SHAP selection and inference.** Past returns
and volatility define daily Bull, Sideway, and Bear states. The reference model,
SHAP values, and regime-specific feature subsets are fitted within each
training fold, frozen, and applied to the held-out year. Paired effects are
aggregated across seeds and years before Holm adjustment.

Place after the final paragraph of Section 3.8.

## File format

- Prefer SVG in Microsoft Word when supported.
- Use vector PDF for LaTeX or journal production.
- Use the 400-dpi PNG only when the workflow rejects vector files.
- Keep the figure title and full explanation in the manuscript caption, not
  inside the image.
