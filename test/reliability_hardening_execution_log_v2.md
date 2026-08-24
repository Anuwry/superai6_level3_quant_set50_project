# Reliability hardening execution log v2

Completion date: 2026-08-04 (Asia/Bangkok)  
Controlling framing: **leakage-controlled reliability audit**  
New experiment evidence class: **pre-frozen retrospective falsification and robustness**

## 1. Completion map

| Item | Completed action | Auditable evidence |
|---|---|---|
| 1 | Reconciled market-data provenance, access basis, non-redistribution boundary, Asia/Bangkok session convention, 17:00 information cutoff, and price-index adjustment convention | `test/market_data_governance_v1.md`; `outputs/market_data_governance_v1/` |
| 2 | Removed `.env` from repository tracking, expanded secret/restricted-path gates, recorded two exact Python environments, and moved pytest temporary files to drive D | `.gitignore`; `requirements-paper-py312.txt`; `requirements-integrated-py311.txt`; `pytest.ini`; `REPRODUCIBILITY.md` |
| 3 | Froze the present-paper estimand hierarchy and an honest future prospective protocol before the new control results were inspected | `test/primary_estimand_and_confirmatory_protocol_v1.md`; `test/reliability_extension_freeze_v1.json` |
| 4 | Added News-Only, shuffled-news, five-row lagged-news, and random-feature falsification arms to all five architectures | `outputs/multimodal_falsification_v1/` |
| 5 | Retained four temporal folds and five seeds, averaged seeds before inference, applied exact sign-flip/Holm and a 10-day 10,000-replicate moving-block sensitivity, and described 16 quarterly origins without treating them as independent confirmation | `fold_inference_holm.csv`; `daily_block_bootstrap_holm.csv`; `quarterly_origin_summary.csv` |
| 6 | Re-audited regime-SHAP and grouped LIME without feature or result reselection | `outputs/track_c/outer_v2/`; `outputs/track_c/dual_xai_lime_v1/` |
| 7 | Re-audited the fixed 10-bps economic proxy and retained it as exploratory Supplement evidence | `outputs/track_d_q2/paper_economic_primary_10bps.csv` |
| 8 | Locked the reliability-audit claim hierarchy and generated compact main/Supplement tables from authoritative aggregates | `test/manuscript_reporting_lock_v1.md`; `outputs/manuscript_tables_v1/` |
| 9 | Added reproducibility, lint, typed-core, test, security, and environment gates without rewriting source files frozen before observed results | `REPRODUCIBILITY.md`; `pyrightconfig.json`; `tests/` |
| 10 | Implemented a fail-closed public replication package v3 and an independent hash/secret audit; raw data, row-level predictions, private checkpoints, and credentials remain excluded | `models/public_replication_package.py`; `scripts/audit_public_replication_package.py` |

## 2. New falsification experiment

The frozen experiment covered five architectures, four outer test years
(2022--2025), five fitting seeds, and four newly fitted arms. It therefore
produced 100 complete model-fold-seed cells and 400 new fits. The two persisted
reference arms, Market-Only and Observed-News, were reused from the prior
integrated run and were not refitted or selected after the controls were seen.

The aggregate integrity audit passed:

- expected/completed cells: 100/100;
- new fit rows: 400;
- seed-averaged fold metric rows: 120;
- paired fold rows: 120;
- fold-inference rows: 150;
- moving-block-bootstrap rows: 60;
- quarterly-origin rows: 480; and
- all reference and control predictions finite and date-aligned.

Six files in the new freeze and 24 files inherited from the integrated freeze
were re-hashed successfully. Incremental API cost was USD 0.

### Process-deviation record

Two isolated workers briefly selected the same deterministic
`lstm_attention/fold_2/seed_2025` cell after the main worker completed
LSTM-CNN. The redundant main child was stopped. Before aggregation, that exact
cell was rerun once with `--force` in a single process and passed its post-write
audit. A subsequent full-grid validation read every expected artifact and
reported 100 skipped-as-complete cells and zero incomplete cells. The clean
rerun is retained as the authoritative cell; it did not change any registered
setting or estimand.

## 3. Primary forecasting and falsification results

Balanced Accuracy (BAcc) is the primary endpoint. Values below are percentage-
point differences from the paired 2022--2025 outer folds. Exact fold-level
inference remains authoritative; the moving-block result is a serial-
dependence sensitivity.

### Observed-News minus Market-Only

| Model | BAcc delta (pp) | Four-fold 95% descriptive CI | Exact p | Holm p | Block-bootstrap Holm p |
|---|---:|---:|---:|---:|---:|
| LSTM | -1.582 | [-4.400, 1.236] | 0.250 | 1.000 | 1.0000 |
| CNN | -0.009 | [-3.749, 3.731] | 1.000 | 1.000 | 1.0000 |
| LSTM-CNN | -0.172 | [-1.830, 1.485] | 0.750 | 1.000 | 1.0000 |
| LSTM-Attention | -1.895 | [-6.833, 3.044] | 0.375 | 1.000 | 0.7008 |
| LSTM-CNN-Attention | -1.224 | [-3.879, 1.431] | 0.250 | 1.000 | 0.4810 |

### Observed-News minus shuffled-news control

| Model | BAcc delta (pp) | Four-fold 95% descriptive CI | Exact p | Holm p | Block-bootstrap Holm p |
|---|---:|---:|---:|---:|---:|
| LSTM | -3.239 | [-7.028, 0.551] | 0.125 | 0.625 | 0.0980 |
| CNN | -1.715 | [-4.549, 1.119] | 0.125 | 0.625 | 0.1710 |
| LSTM-CNN | +1.020 | [-1.496, 3.535] | 0.375 | 0.750 | 0.6164 |
| LSTM-Attention | +0.128 | [-2.781, 3.036] | 1.000 | 1.000 | 0.8528 |
| LSTM-CNN-Attention | -1.785 | [-5.179, 1.610] | 0.250 | 0.750 | 0.0980 |

Across all six registered BAcc contrast families, zero of 30 exact fold rows
and zero of 30 moving-block rows survived Holm adjustment at 0.05. Two
unadjusted block-bootstrap intervals for the primary shuffled control were
negative, but both adjusted p-values were 0.098 and cannot be reported as
significant. The correct finding is that observed news did not establish a
reliable incremental forecasting benefit; shuffled, lagged, random-feature,
and news-only behaviour indicates architecture/capacity/timing sensitivity.
It is not evidence that randomised news is a usable forecasting method.

The 16 quarterly origins per model were descriptive only. Direction and sign
varied across architectures and origins, reinforcing temporal instability
rather than supplying additional independent sample size.

## 4. Runtime and cost

| Model | New fits | Fit + inference compute minutes |
|---|---:|---:|
| CNN | 80 | 6.67 |
| LSTM | 80 | 6.90 |
| LSTM-Attention | 80 | 10.62 |
| LSTM-CNN | 80 | 12.27 |
| LSTM-CNN-Attention | 80 | 16.36 |
| **Total** | **400** | **52.82** |

The sum of per-cell wall times was 1.168 hours; parallel execution reduced the
observed elapsed interval. The registered fit plus inference ledger totals
0.880 compute-hours. No paid API call, Optuna study, or accuracy-directed
search was used.

## 5. XAI and economic re-audit

- Regime-SHAP: 25 exact four-fold BAcc comparisons, zero Holm-significant;
  minimum adjusted p=0.625 and maximum absolute point effect=2.760 pp.
- Grouped LIME: 1,293/1,800 instance-repeat rows (71.83%) fell below the
  registered local-fidelity threshold of 0.70. All failed rows remain in the
  audit. LIME is a Supplement-only diagnostic and does not validate SHAP.
- Economic proxy: 20 fixed 10-bps rows; maximum deflated-Sharpe probability
  0.441096. The analysis does not establish deployability or profitability and
  remains Supplement-only exploratory evidence.
- SET100: all five mean BAcc transfers were below their paired SET50 values and
  none was Holm-significant. This is negative same-exchange breadth evidence,
  not external-market replication.

## 6. Manuscript and prospective boundary

The generated manuscript bundle contains separate main panels for the
numerical ablation, multimodal falsification, intrinsic compute-matched LLM
benchmark, regime-SHAP, partial-2026 robustness, and SET100 transfer. LIME and
economics are explicitly labelled Supplement diagnostic/exploratory.
Intrinsic LLM debate is not represented as the downstream news-feature source;
the frozen local-NLP features are the downstream source.

No untouched historical confirmatory set remains because 2012--2025 and the
available partial-2026 outcomes were inspected before the prospective freeze.
The first genuinely untouched prospective session is after 2026-07-30. The
locked LSTM-CNN window-20 comparison requires 252 newly labelled sessions; 126
sessions may be reported only as an interim reliability check. This future
outcome cannot be manufactured or marked complete in the present execution.

## 7. Interpretation for journal positioning

The work is defensible as a broad, transparent reliability-audit framework,
not as a state-of-the-art accuracy paper. Completing these controls removes
several avoidable reviewer risks (provenance ambiguity, uncontrolled feature-
count effects, missing null controls, hidden diagnostic failures, and an
unverifiable release). It does not create a forecasting gain. A carefully
written submission is reasonably positioned for a relevant Q2 journal, while
Q1 remains a stretch because there is no independent external-market
replication and no completed prospective 252-session confirmation. Quartile
and acceptance are journal/year dependent and cannot be guaranteed.

## 8. Final verification record

- Python compilation: passed for `models`, `scripts`, and `tests`.
- Typed maintained core: pyright 0 errors, 0 warnings.
- Maintained-source lint: passed. The frozen falsification runner retains two
  style-only findings (import grouping and an explicit UTF-8 argument); it was
  not rewritten after result access because its SHA-256 is part of the freeze.
- Full test suite: 426 passed, 22 non-failing warnings.
- Targeted maintained-core coverage: 91% overall (manuscript artifacts 97%,
  falsification pure functions 87%, public package 91%).
- Security scan: Bandit passed after resolving the Git executable and retaining
  fixed argv with `shell=False`; 294 public-eligible source files had zero
  secret-pattern findings before the final snapshot rebuild.
- Dependency consistency: `pip check` reported no broken requirements.
- Diff whitespace gate: passed; only normal Windows LF-to-CRLF notices were
  emitted.

The security review influenced the release by keeping credentials outside the
repository snapshot, rejecting restricted paths and symlinks, scanning every
eligible text artifact, using exact hashes, and independently auditing the
finished package. No authentication, network endpoint, or live trading action
was introduced.
