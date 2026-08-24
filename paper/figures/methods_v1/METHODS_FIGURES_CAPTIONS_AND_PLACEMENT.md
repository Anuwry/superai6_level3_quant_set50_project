# Methods figures: captions and placement

The captions below are written for the current Methods structure (Sections
3.1–3.8). If all six figures remain in the main text, later figures already in
the Results section must be renumbered sequentially.

## Figure 1 — place in Section 3.1

**Figure 1. Point-in-time reliability-audit framework.** A common market-data
governance and target-definition layer supports three architecture-paired audit
branches: numerical denoising, multimodal news and LLM inference, and
regime-aware explainability. All branches are evaluated through the same panel
of five frozen neural architectures and the same SET50 next-day direction
contract. Historical 2022–2025 evaluation, source-contingent partial-2026
testing, and frozen SET100 same-exchange transfer provide distinct robustness
checks.

Recommended insertion point: after the first paragraph of Section 3.1.

## Figure 2 — place in Section 3.2

**Figure 2. Temporal coverage, class composition, and point-in-time construction
of the financial-news data.** (A) SET50 market data span 2012–2025, whereas the
labelled Bilingual StockTBSA source spans 2018–2023, expanding out-of-sample
sentiment begins in 2019, and the model frozen on labelled data through 2023 is
applied to official SET headlines in 2024–2025. (B) Valid positive, neutral,
and negative article–ticker pairs in the labelled source by year; the locked
2023 cohort contains 92 negative, 585 neutral, and 656 positive pairs. (C)
Point-in-time SET50 membership and relevance filtration of official 2024–2025
headlines. These forward headlines are unlabeled and are not treated as an
intrinsic sentiment benchmark. (D) Conservative next-session assignment and
daily feature aggregation used for next-day forecasting.

Recommended insertion point: after the second paragraph of Section 3.2.

## Figure 3 — place in Section 3.3

**Figure 3. Point-in-time target contract and expanding evaluation design.**
(A) A sequence ending at close t uses only information available through t to
predict the next close and direction; its label becomes observable at t+1. At
each boundary, a training row is admissible only when its label-observation
date precedes the first evaluation date. (B) Four expanding pre-2022 validation
folds select the sequence window before four frozen outer tests in 2022–2025.
Scaling, feature selection, model weights, regime thresholds, and SHAP rankings
are fitted on training data only. Each fixed model–fold cell is repeated with
five registered seeds.

Recommended insertion point: after the paragraph defining the outer folds and
five seeds in Section 3.3.

## Figure 4 — place after Section 3.5

**Figure 4. Numerical feature construction, causal rolling VMD, and the fixed
neural architecture panel.** (A) The Full-TA control comprises 116 variables
from daily and completed weekly/monthly market information. (B) At every date
t, VMD is fitted only to the trailing 60 closes. Four retained intrinsic mode
functions, the denoised close, and the removed-mode energy ratio add six causal
variables, yielding 122 numerical inputs. (C) Layer sequences and frozen
windows for LSTM, CNN, LSTM–CNN, LSTM–Attention, and
LSTM–CNN–Attention. All architectures use the same optimizer, loss, epoch,
batch, shuffle, and determinism settings.

Recommended insertion point: after the final paragraph of Section 3.5. This
single display supports Sections 3.4 and 3.5.

## Figure 5 — place in Section 3.6

**Figure 5. Multimodal falsification and role-structured LLM evaluation.** (A)
Observed predicted-news features are compared with Market-Only, News-Only,
date-shuffled, five-row-lagged, and matched-random-feature arms under identical
dates, windows, seeds, architectures, and training budgets. (B) In the separate
locked 2023 intrinsic benchmark, Bull and Bear workers provide distinct
arguments to a Leader and are compared with single-pass and compute-matched
self-consistency controls. (C) Expanding Local-NLP features and dated Leader
outputs are retained as distinct downstream routes and require an
identical-cohort paired SET50 forecasting comparison.

Recommended insertion point: after the paragraph describing the LLM controls
in Section 3.6.

## Figure 6 — place after Section 3.8

**Figure 6. Causal regime routing, train-only SHAP selection, and temporal
inference.** (A) Past-only multi-horizon returns, volatility, and ADX form a
risk-adjusted trend score. A training-only symmetric deadband maps each close-t
observation to Bull, Sideway, or Bear. (B) Regime-specific SHAP rankings and
candidate feature counts are selected within admissible training data using a
one-standard-error rule and stability guardrails; size-matched Spearman and
capacity controls remain in the outer comparison. (C) Seeds are averaged within
model–year cells before four annual paired effects enter exact sign-flip and
Holm-adjusted inference. Moving-block and article-cluster bootstraps are used as
serial-dependence and intrinsic-LLM uncertainty analyses, respectively.

Recommended insertion point: after the final paragraph of Section 3.8. This
single display supports Sections 3.7 and 3.8.

## File-format guidance

- Use the `.svg` file in Word when the journal workflow accepts SVG; it remains
  editable and scales without loss.
- Use the `.pdf` file for LaTeX or production workflows that prefer vector PDF.
- Use the `.png` file when Word or a journal submission portal rejects vector
  graphics. PNG files were rendered at 400 dpi.
- Do not paste screenshots of these figures into the manuscript.
