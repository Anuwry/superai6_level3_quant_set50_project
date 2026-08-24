# Public market-data access amendment — 2026-08-04

Status: **CLAIM CORRECTED AND FROZEN**

## Reason for amendment

Earlier governance text described the SET50 and SET100 files as covered by an
institutional research-use entitlement. The author subsequently clarified that
the files were obtained from Investing.com's publicly accessible historical-
data pages. No institutional entitlement evidence was supplied, so the earlier
description is superseded rather than retained as a manuscript claim.

## Verified access claim

On 4 August 2026, the public SET50 and SET100 historical-data pages were checked.
Both pages were publicly reachable, displayed historical OHLCV observations,
and described an option to download data for analysis and reporting.

- SET50: https://www.investing.com/indices/set-50-historical-data
- SET100: https://www.investing.com/indices/set-100-historical-data
- Provider terms: https://cdn.investing.com/about-us/terms_and_conditions.pdf

The manuscript may therefore describe the primary market files as **publicly
accessible provider data downloaded through the historical-data interface**.
It must not describe them as public-domain, open-licensed, freely
redistributable, or institutionally licensed data.

## Acquisition boundary

The governance manifest records the primary SET50 and SET100 daily, weekly, and
monthly files as manual browser downloads supplied by the author. The partial-
2026 Track D SET50 extension is a documented exception: its access ledger
records retrieval through the provider's historical-data endpoint using
`curl_cffi`. This exception remains transparent and is reported as a source-
contingent forward evaluation; it is not relabelled as a manual download.

## Distribution policy

Provider-hosted row-level observations and reconstructive derivatives are
excluded from the public replication package. The public package contains code,
schemas, hashes, protocols, and non-reconstructive aggregate results. Access and
reuse remain subject to the provider terms in force at the time of access.

## Superseded records

Historical execution logs are retained for auditability, but any statement in
them claiming institutional entitlement or pending institutional licence
evidence is superseded by this amendment and the canonical paper-ready data
statement in `outputs/market_data_governance_v1/paper_data_statements.md`.
