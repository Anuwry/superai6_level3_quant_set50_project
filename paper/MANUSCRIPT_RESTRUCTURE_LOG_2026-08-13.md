# Manuscript restructuring log — 13 August 2026

## Source and output

- Source supplied by the author: `D:\Downloads\Manuscript_lastest.docx`
- Revised manuscript: `Manuscript_latest_methods8_conclusion.docx`
- The source file was preserved; all edits were applied to a copy through
  Microsoft Word automation.

## Methods consolidation

The original draft contained 11 Method subsections after moving the two
robustness protocols out of Methods. The section was reduced to eight
subsections while retaining the point-in-time contract, frozen settings,
control arms, model specifications, and inferential rules:

1. `3.1 Study design, data governance, and evidence hierarchy` combines the
   former study-design and market-data/provenance sections.
2. `3.2 News data and out-of-sample sentiment construction` retains the news
   sources, counts, expanding prediction schedule, date mapping, and eight
   daily variables.
3. `3.3 Point-in-time contract, temporal splits, and frozen windows` combines
   the former prediction-contract and split/window sections and restores the
   target, label-date purge, context-only row, windows, folds, and seeds.
4. `3.4 Numerical features and causal VMD` retains the Full-TA feature scope
   and all causal rolling-VMD settings.
5. `3.5 Neural architectures and training` retains all five architecture
   definitions and the common optimizer, loss, epoch, batch, determinism, and
   scaling settings.
6. `3.6 Multimodal falsification and Bull/Bear/Leader evaluation` combines the
   news negative-control design with the compute-matched LLM and downstream
   common-cohort contract.
7. `3.7 Causal market regimes and SHAP selection` retains the SHAP sample
   design, candidate subset sizes, guardrails, seven arms, and LIME fidelity
   diagnostic.
8. `3.8 Metrics and statistical inference` retains the primary and secondary
   endpoints, year-level inference, exact sign-flip tests, Holm control,
   moving-block sensitivity, and clustered LLM uncertainty.

Method length decreased from approximately 2,742 to 1,865 tokenized words, a
32.0% reduction. Obsolete subsection numbers `3.9`–`3.13` and the draft marker
`Method 11 -> 8` no longer occur.

## Material moved to Results

- The former `3.11 Partial 2026 objective-alignment stress test` was condensed
  and placed at the start of Results `4.5`, before the reported 2026 outcomes.
- The former `3.12 Frozen SET100 same-exchange transfer` was condensed and
  placed at the start of Results `4.6`, before the cross-index results.
- Both subsections retain their frozen-design qualifications so the Results do
  not appear to report an unspecified experiment.

## Conclusion and end matter

Section 5 now contains:

1. one synthesis paragraph stating the audit's principal finding;
2. one paragraph labelled `Limitations.`; and
3. one paragraph labelled `Future work.`

These are followed by unnumbered `Acknowledgements` and `Reproducibility and
data availability` sections. The availability statement distinguishes
reproducibility artifacts from raw market/news rows and retains provider-term
and licence constraints.

## Quality assurance

- Required Method headings found: 8/8 (`3.1`–`3.8`).
- Old Method headings found: 0.
- Required Results sections `4.5` and `4.6` contain their corresponding design
  summaries before outcomes.
- Conclusion, Limitations, Future work, Acknowledgements, Reproducibility and
  data availability, and References each occur in the intended order.
- Pandoc successfully extracted the revised DOCX for structural review.
- The DOCX ZIP/OOXML package was readable after Word saved the changes.
- SHA-256:
  `1190A484BEF636FD95E5F1DE228D16F22B1829D0FE217A55F65A0888335FE13A`.

Visual page rendering could not be completed in the current environment:
LibreOffice and the packaged renderer dependencies are unavailable, and Word's
PDF export did not complete. The document therefore passed structural and
content QA but not page-image visual QA. Existing figure/table insertion
markers elsewhere in the author's working draft were intentionally retained.
