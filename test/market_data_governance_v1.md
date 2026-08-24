# Market-data governance and SET100 preparation log v1

> Access clarification (2026-08-04): the controlling position is publicly
> accessible provider data, provider terms apply, and raw rows are not
> redistributed. The dated amendment in
> `test/public_market_data_access_amendment_2026-08-04.md` preserves the change
> history; this governance log no longer imposes a separate private-evidence
> gate.

Execution date: 2026-08-03 (Asia/Bangkok)  
Protocol: `market-data-governance-v1`  
Implementation: `models/market_data_governance.py`  
Runner: `scripts/run_market_data_governance.py`

## 1. Outcome and scope

This stage closes the market-data provenance, integrity, timezone, adjustment,
redistribution, and point-in-time preparation controls for SET50 and SET100.
SET100 is data-ready for a same-exchange robustness benchmark, but no SET100
model result is claimed in this log.

| Gate | Result | Interpretation |
|---|---|---|
| Internal research use | PASS | The provider historical pages are publicly accessible and explicitly offer a download option for analysis and reporting. |
| SET50 integrity | PASS | No critical integrity check failed. |
| SET100 integrity | PASS | No critical integrity check failed. |
| SET100 common-date coverage | PASS | 3,360/3,381 SET50 reference dates matched (99.3789%), above the pre-set 99% gate. |
| Timezone convention | PASS | Date-only observations are treated as SET session dates in Asia/Bangkok. |
| Adjustment convention | PASS with retained warnings | Provider-published price-index observations are used without researcher-side adjustment. Two source OHLC warnings are retained. |
| Raw redistribution | EXCLUDED | Provider terms apply; raw and reconstructive row-level derivatives are not released. |
| Public release | CLEAN PACKAGE ONLY | The verified public package excludes provider rows; the private working repository is not itself a release artifact. |
| SET100 benchmark | NOT STARTED; DATA READY | This run prepares the data and folds only. |

SET100 is a same-exchange robustness check. It must not be described as an
independent external-market replication because SET50 and SET100 share the Thai
market, macroeconomic period, and overlapping constituents.

## 2. Data inventory and provenance

| Index | Frequency | Rows | First label | Last label | SHA-256 prefix |
|---|---:|---:|---:|---:|---|
| SET50 | Daily | 3,403 | 2012-01-04 | 2025-12-22 | `b77dc6792165` |
| SET50 | Weekly | 730 | 2012-01-01 | 2025-12-21 | `b57aad9187f7` |
| SET50 | Monthly | 168 | 2012-01-01 | 2025-12-01 | `4478d0d20d07` |
| SET100 | Daily | 3,549 | 2012-01-04 | 2026-08-03 | `a712a8182213` |
| SET100 | Weekly | 762 | 2012-01-01 | 2026-08-02 | `3f03f76ce9f9` |
| SET100 | Monthly | 175 | 2012-02-01 | 2026-08-01 | `2da2124af1c2` |

The raw files were supplied by the user as manual browser downloads from the
Investing.com SET50 and SET100 historical-data pages. The Windows
`Zone.Identifier` alternate data streams preserve host evidence for
`https://th.investing.com/` (SET50) and `https://www.investing.com/` (SET100).
The exact historical-page URLs, byte sizes, full SHA-256 digests, row counts,
date ranges, and source evidence are recorded in
`outputs/market_data_governance_v1/market_data_manifest.csv` and JSON.

Acquisition times in the manifest are explicitly labelled as local-file
creation-time evidence, not as vendor-issued receipts. This distinction avoids
overstating the strength of the timestamp evidence.

Source pages:

- SET50 historical data: https://th.investing.com/indices/set-50-historical-data
- SET100 historical data: https://www.investing.com/indices/set-100-historical-data
- SET100 profile and calculation method: https://www.set.or.th/en/market/index/set100/profile
- SET trading hours: https://www.set.or.th/en/market/information/trading-procedure/trading-hours

## 3. Access and availability convention

The access register records that the provider's SET50 and SET100 historical-
data pages are publicly accessible and offer a download option. The project
does not represent public accessibility as public-domain or open-licence
status. Provider terms apply, and raw observations are not redistributed.

The public replication package may contain code, schemas, hashes, manifests,
and non-reconstructive summary statistics. It must not contain raw SET50/SET100
rows, aligned OHLCV rows, point-in-time fold CSVs, or any artifact from which
provider row-level values can be reconstructed.

Suggested Data Availability wording:

> Market-index observations were obtained from publicly accessible provider
> historical-data pages that offer a download option. Access and reuse remain
> subject to the provider's terms; public accessibility is not represented as
> an open-data licence. The public replication package provides preprocessing
> code, data contracts, checksums, integrity summaries, and point-in-time split
> specifications, but not row-level provider observations.

## 4. Timezone and temporal convention

Raw CSVs contain dates without intraday timestamps. Each daily label is treated
as the corresponding SET trading-session date in `Asia/Bangkok`. The registered
decision cutoff is 17:00 Bangkok time, after the normal market close. SET states
that the random closing-price auction occurs between 16:35 and 16:40 and that
the market closes at 17:00; the 17:00 cutoff is therefore conservative.

Weekly labels are provider period labels on Sunday, and monthly labels are
provider period labels on the first calendar day. They are not interpreted as
trade timestamps. To prevent leakage, each weekly and monthly series is shifted
by one completed provider period before forward filling onto daily session
dates. The target is the close on the next observed trading session, and
supervised training rows are retained only when `Label_Date` is earlier than the
first evaluation date.

Suggested Methods wording:

> Date-only observations were interpreted as SET session dates in the
> Asia/Bangkok timezone. Features were considered available at 17:00 local time.
> Provider-labelled weekly and monthly bars were lagged by one completed period
> before forward filling to the daily calendar. The prediction target was the
> close of the next observed SET session. At each temporal split, a supervised
> row was eligible for fitting only when its label-observation date preceded the
> first evaluation date.

## 5. Adjustment convention

SET50 and SET100 observations are treated as provider-published price-index
levels, not total-return indices. The official SET index methodology determines
the index construction and treatment of index operations and corporate actions.
The research pipeline does not apply a second split, dividend, constituent, or
corporate-action adjustment to the downloaded index levels. Results therefore
must not be described as dividend-adjusted or total-return predictions.

Suggested Methods wording:

> We used provider-published SET50 and SET100 price-index levels rather than
> total-return indices. Official index operations are embedded in the published
> series; no additional researcher-side corporate-action, dividend, or
> constituent adjustment was applied.

## 6. Integrity audit

Forty-two automated checks were run: 40 passed and two produced retained
warnings. No critical check failed. Checks cover unique dates, finite OHLCV,
non-negative volume, positive index levels, OHLC containment, agreement between
reported percentage change and close-to-close change, and the registered
frequency-label convention.

Reported percentage change agreed with close-to-close calculations on all six
raw series at the 0.011 percentage-point tolerance. The largest observed
absolute discrepancy was 0.005003504 percentage points.

Two OHLC containment warnings occurred on 2025-12-11:

| Dataset | Date | Warning | Action |
|---|---|---|---|
| SET50 daily | 2025-12-11 | Open exceeds reported High | Retain restricted raw pending source verification |
| SET100 daily | 2025-12-11 | Open exceeds reported High | Retain restricted raw pending source verification |

Both source rows have an opening value slightly above the reported high. The
pipeline does not silently clip, impute, or repair either observation. Full
OHLC values are stored only in the ignored `private/anomaly_ledger.csv`; the
public summary is non-reconstructive. The private correction ledger remains
empty; any later correction must record the old and new values, reason,
verification source, and verification time.

## 7. SET100 canonical preparation

Generated restricted artifacts:

| Artifact | Rows | Date range |
|---|---:|---|
| Daily prepared target rows | 3,548 | 2012-01-04 to 2026-07-31 |
| Weekly prepared target rows | 761 | 2012-01-01 to 2026-07-26 |
| Monthly prepared target rows | 174 | 2012-02-01 to 2026-07-01 |
| Daily calendar | 3,549 | 2012-01-04 to 2026-08-03 |
| Causally aligned SET100 | 3,507 | 2012-03-01 to 2026-07-31 |
| Common SET50-date cohort | 3,360 | 2012-03-01 to 2025-12-18 |

The aligned cohort omits 21 leading SET50 dates from 2012-02-01 onward because
SET100 monthly data begin in February 2012 and a completed monthly period is
required before causal use. Backfilling these dates would introduce future
information, so the pipeline uses the observed date intersection. Coverage is
99.3789%, which passes the pre-set 99% minimum.

## 8. Point-in-time folds

Selection folds are reserved for pre-2022 selection; outer folds cover the same
2022-2025 years used by the registered SET50 evaluation.

| Role | Test year | Training rows after purge | Test rows | Boundary labels purged |
|---|---:|---:|---:|---:|
| Selection | 2018 | 1,424 | 245 | 1 |
| Selection | 2019 | 1,669 | 244 | 1 |
| Selection | 2020 | 1,913 | 243 | 1 |
| Selection | 2021 | 2,156 | 241 | 1 |
| Outer | 2022 | 2,397 | 241 | 1 |
| Outer | 2023 | 2,638 | 243 | 1 |
| Outer | 2024 | 2,881 | 244 | 1 |
| Outer | 2025 | 3,125 | 234 | 1 |

Every fold records the original and retained training counts, first test date,
maximum retained label date, removed boundary dates, filenames, and SHA-256
digests in its `point_in_time_contract.json` and top-level `run_metadata.json`.

## 9. Repository distribution audit

SET100 raw, prepared, and fold paths are now ignored by Git. The current Git
index nevertheless contains seven pre-existing restricted SET50 files: three
raw frequency files and four prepared/aligned files. `.gitignore` does not
remove files that were already committed.

This does not block internal research in the present private workspace, but it
does block treating the current repository as a public replication package.
Git-history inspection found all seven paths in history: the six raw/frequency-
prepared files first appear in commit `4b148507a171` (2026-06-22), and the
aligned file first appears in commit `267258023432` (2026-06-24). Before any
public release, create a clean publication artifact or remove the seven files
from the public index after confirming the intended repository and release
method. History rewriting was deliberately not attempted because it is
destructive and requires an explicit release decision.

## 10. Reproduction and verification

Run from the repository root:

```powershell
$env:PYTHONPATH=(Get-Location).Path
D:\conda_envs\my_env\python.exe scripts\run_market_data_governance.py
```

Verification completed on 2026-08-03:

- focused governance tests: 14 passed;
- governance plus point-in-time regression tests: 20 passed;
- module coverage: 92.73%, above the required 80% gate;
- Ruff static checks: passed;
- real-data governance run: completed without an exception;
- SET50 data gate: PASS;
- SET100 data gate: PASS.

Machine-readable evidence is stored under
`outputs/market_data_governance_v1/`. Row-level SET100 prepared data and folds
remain under `set100_data/` and are excluded from version control.
