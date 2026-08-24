# PIT-SET50-CRIN 2024-2025 Execution Log v1

## Final decision

PIT-SET50-CRIN v1 **does not pass as the headline Ours model**. On the frozen
common cohort from test years 2024 and 2025, its mean annual balanced accuracy
was 51.870%, ranking sixth of six. The best model was LSTM-Attention at
53.755%. The gap was -1.885 percentage points.

No result from 2020 or 2021 is included in the comparison. Those years were
used only to fit and validate the constituent worker. The reported accuracy
cohort contains test dates from 2024 and 2025 only.

## Requested comparison scope

| Item | Frozen value |
|---|---|
| Evaluation years | 2024 and 2025 only |
| Common test observations | 243 in 2024; 234 in 2025; 477 total |
| Base models | CNN; LSTM; LSTM-CNN; LSTM-Attention; LSTM-CNN-Attention |
| Proposed model | PIT-SET50-CRIN |
| Seeds | 42, 123, 456, 789, 2025 |
| Seed aggregation | Average probabilities first, then apply threshold 0.5 |
| Primary metric | Mean of annual balanced accuracy for 2024 and 2025 |

The constituent worker used observations through 2020 for training and 2021
for early-stopping validation. Its predictions from 2022 onward were therefore
out of sample. The reconciliation leader used prior frozen predictions in a
walk-forward design: 2022/2023 supported the 2024 forecast and 2022-2024
supported the 2025 forecast. These earlier dates are training history, not
reported test results.

## Data and point-in-time controls

- Official SET membership evidence contains 300 rows, six effective versions,
  50 names per version, and 59 distinct symbols across 2024-2025.
- Membership changes were applied only on their official effective dates,
  including the 2025 GULF/GULFI/INTUCH/VGI transition.
- Provisional price histories were downloaded at zero API cost from the Yahoo
  Finance chart endpoint for internal feasibility testing. Fifty-six of 59
  symbols downloaded. BANPU, GULFI, and INTUCH had no usable file from this
  endpoint; GULF and TIDLOR also had provider-history discontinuities associated
  with corporate restructuring.
- No constituent close price was forward-filled. Missing names were masked.
- Mean usable test membership was 47.0/50 in 2024 and approximately 48.4/50 in
  2025. Every included test day exceeded the registered 35/50 coverage gate.
- Raw constituent rows are non-redistributable and remain a provisional input.
  This extension is not paper-ready until rerun with institution-authorized
  constituent data.

## Architecture executed

Each possible constituent passed through the same eight-unit LSTM and
eight-unit projection. A point-in-time membership/data mask removed unavailable
names before learned attention pooled member embeddings. The bottom worker
jointly predicted next-day SET50 direction and next-day constituent breadth.

The reconciliation leader combined:

1. the five frozen top-down forecast returns;
2. the bottom-up constituent logit;
3. usable-member coverage; and
4. disagreement between the top-down forecasts.

It learned a soft top-versus-bottom gate and a bounded correction. Required
controls were equal majority vote, top-only learned stacking, and bottom-only
prediction.

## Six-model results: test years only

### Per-year balanced accuracy

| Model | 2024 BAcc (%) | 2025 BAcc (%) | Mean (%) |
|---|---:|---:|---:|
| CNN | 53.999 | 51.075 | 52.537 |
| LSTM | 53.083 | 53.202 | 53.142 |
| LSTM-CNN | 49.878 | **55.154** | 52.516 |
| LSTM-Attention | 53.694 | 53.816 | **53.755** |
| LSTM-CNN-Attention | 53.236 | 53.070 | 53.153 |
| PIT-SET50-CRIN | 50.977 | 52.763 | 51.870 |

### Mean 2024-2025 metrics on the 477-day common cohort

| Rank | Model | BAcc (%) | Direction accuracy (%) | MCC | Predicted-up share (%) |
|---:|---|---:|---:|---:|---:|
| 1 | LSTM-Attention | 53.755 | 52.833 | 0.0938 | 79.709 |
| 2 | LSTM-CNN-Attention | 53.153 | 51.994 | 0.0931 | 86.784 |
| 3 | LSTM | 53.142 | 52.414 | 0.0712 | 73.401 |
| 4 | CNN | 52.537 | 52.168 | 0.0524 | 63.739 |
| 5 | LSTM-CNN | 52.516 | 51.417 | 0.0689 | 84.892 |
| 6 | PIT-SET50-CRIN | 51.870 | 51.171 | 0.0424 | 72.570 |

The proposed model was 1.885 percentage points below LSTM-Attention and 0.646
points below the lowest-scoring frozen base model on the two-year mean.

## Required ablation

| Variant | Mean BAcc (%) | Mean direction accuracy (%) | Mean MCC |
|---|---:|---:|---:|
| Majority vote of frozen five | **53.066** | 52.010 | 0.0845 |
| Top-only learned stack | 52.127 | 50.720 | 0.0967 |
| Full PIT-SET50-CRIN | 51.870 | 51.171 | 0.0424 |
| Bottom-only constituent worker | 50.749 | 49.929 | 0.0126 |

The full model improved over bottom-only by 1.121 points but lost to top-only
stacking by 0.257 points and to majority vote by 1.196 points. Therefore the
data do not support a claim that constituent reconciliation adds predictive
value in v1.

## Failure localization

1. **Weak bottom-up signal.** Bottom-only BAcc was 49.634% in 2024 and 51.864%
   in 2025. In 2024 it predicted upward movement on 90.1% of test days although
   the actual up-day share was 48.1%.
2. **Gate over-relied on the weak branch.** Mean top-down gate weight was 45.3%
   in 2024 and 44.5% in 2025, leaving roughly 55% weight on the bottom branch.
3. **Initialization instability.** Full-model seed BAcc in 2024 ranged from
   49.91% to 55.59%. Seed averaging reduced this instability but could not
   recover the loss against the frozen models.
4. **Constituent provenance/continuity is provisional.** Missing and reset
   provider histories reduce the credibility and effective coverage of the
   cross-sectional branch, even though all days passed the coverage gate.
5. **Two-year inference is descriptive.** With only two reported temporal
   folds, no confirmatory significance claim is made.

## Runtime and compute

| Item | Result |
|---|---:|
| Completed year-seed cells | 10/10 |
| Total recorded runtime | 3,983.46 s (66.39 min) |
| Mean runtime per cell | 398.35 s |
| Bottom-worker fit total | 3,726.13 s (62.10 min) |
| Reconciliation-leader fit total | 41.61 s |
| Top-only-control fit total | 51.07 s |
| Bottom-worker trainable parameters | 586 |
| Leader trainable parameters | 63 |

The run used native Windows TensorFlow on CPU. TensorFlow emitted compatibility
warnings concerning dataset attributes, but all ten cells completed, outputs
were finite, and the warnings did not change the registered execution grid.
NaN attention summaries occurred only for symbols with zero active test days;
these symbols were already masked from prediction.

## Verification

- Ten of ten registered cells completed.
- All 1,908 seed-averaged date-variant rows contained exactly five seeds.
- All probabilities were finite.
- All six models were rescored on the same 243 dates in 2024 and 234 dates in
  2025.
- Freeze hashes were checked inside every cell.
- Source, predictions, attention, leader diagnostics, runtime, and hashes were
  retained.

## Paper decision

Do not present PIT-SET50-CRIN v1 as a successful accuracy-improving architecture
and do not replace the frozen five-model headline with it. It can be retained
as a documented negative feasibility experiment or reliability-audit appendix.
Because the 2024-2025 outcomes have now been observed, redesigning and tuning
another version on these same years would be exploratory. A later confirmatory
claim requires a newly frozen nested protocol or a genuinely untouched forward
period.

## Output index

- `outputs/pit_set50_crin_2024_2025_v1/six_model_metrics_by_year_2024_2025.csv`
- `outputs/pit_set50_crin_2024_2025_v1/six_model_comparison_2024_2025.csv`
- `outputs/pit_set50_crin_2024_2025_v1/crin_ablation_2024_2025.csv`
- `outputs/pit_set50_crin_2024_2025_v1/variant_metrics_by_year.csv`
- `outputs/pit_set50_crin_2024_2025_v1/predictions_seed_averaged.csv`
- `outputs/pit_set50_crin_2024_2025_v1/runtime_summary.csv`
- `outputs/pit_set50_crin_2024_2025_v1/verification_report.json`
- `test/pit_set50_crin_protocol_v1.md`
- `test/pit_set50_crin_freeze_v1.json`

