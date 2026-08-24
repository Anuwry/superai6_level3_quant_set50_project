# Paper-ready market-data statements

## Data source and provenance

Daily, weekly, and monthly SET50 and SET100 price-index observations were
obtained from publicly accessible Investing.com historical-data pages, which
explicitly provide an option to download the data for analysis and reporting.
The primary files were supplied as manual browser downloads. The retained
provenance manifest records the source URLs, file sizes, SHA-256 digests, row
counts, date ranges, local-file creation-time evidence, and acquisition method.
Local creation times are used only as acquisition evidence and are not
represented as vendor-issued receipt timestamps. A source-contingent partial-
2026 SET50 forward extension was retrieved separately through the provider's
public historical-data interface and is identified as such in the Track D
access ledger; it is not represented as a manual download.

## Temporal convention

Date-only observations were interpreted as Stock Exchange of Thailand session
dates in the Asia/Bangkok timezone. Features were considered available at 17:00
local time, after the normal market close. Provider-labelled weekly and monthly
bars were shifted by one completed period before forward filling to the daily
calendar. The target was the close on the next observed SET session. At each
temporal split, supervised observations were eligible for training only when
their label-observation date preceded the first evaluation date.

## Adjustment convention

The analysis used provider-published SET50 and SET100 price-index levels rather
than total-return indices. Official index operations are embedded in the
published series; the research pipeline applied no additional split, dividend,
constituent, or corporate-action adjustment.

## Integrity and SET100 preparation

Across six raw index-frequency series, 40 of 42 integrity checks passed and two
OHLC containment checks generated retained warnings; no critical check failed.
The two warnings occurred on 11 December 2025 in both daily index files and were
retained without silent repair. The SET100 causal common-date cohort contained
3,360 of 3,381 SET50 reference dates (99.3789%). Twenty-one leading dates were
excluded because a completed SET100 monthly period was not yet available;
backfilling was not used. Point-in-time selection folds cover 2018-2021, and
outer folds cover 2022-2025, with label-date purging at every fold boundary.

## Data availability

The provider's historical-data pages are publicly accessible and offer a data-
download option. Access and reuse nevertheless remain subject to the provider's
terms of use; public accessibility is not represented as an open-data licence.
Accordingly, the replication package provides preprocessing code, data
contracts, checksums, integrity summaries, and point-in-time split
specifications, but not row-level provider data. Researchers may obtain the
series independently from the identified provider subject to the terms in
effect at the time of access.

## Claim boundary

SET100 is a same-exchange robustness dataset, not an independent external-market
replication. The registered benchmark was subsequently completed without
retuning: five frozen architectures, four outer folds, and five seeds produced
100/100 planned fits. Mean SET100 balanced accuracy was lower than its paired
SET50 value for all five architectures by 0.95--2.17 percentage points, and no
cross-index contrast survived Holm correction. This result is reported as
negative within-SET transfer evidence, not as external validation or an
improvement on SET100.
