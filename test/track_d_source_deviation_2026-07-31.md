# Track D Source Deviation — 2026-07-31

Status: **FROZEN BEFORE ALTERNATIVE FULL-SERIES ACCESS**  
Parent freeze: `outputs/track_d_q2/freeze_manifest.json`

## Trigger

The registered Yahoo chart request was first made after the parent freeze at
2026-07-31T15:18:07Z. It returned only one row (2026-07-31), no overlap rows
with the local series, and zero volume. The registered overlap and volume
gates therefore failed closed. The Yahoo response, normalized snapshot, and
access ledger remain immutable evidence and are not overwritten.

## Dated deviation

The alternative source is the public Investing.com SET 50 historical page and
its historical-data API:

```text
page = https://www.investing.com/indices/set-50-historical-data
instrument_id = 41049
endpoint = https://api.investing.com/api/financialdata/historical/41049
start-date = 2025-10-01
end-date = retrieval date
time-frame = Daily
add-missing-rows = false
domain-id = www
```

The page identifies the instrument as SET 50 (`SET50`) in Thailand and exposes
daily close, open, high, low, and volume. This source family also matches the
format of the historical local CSV through 2025-12-22.

The alternative adapter must be hashed before the first full-series API
request. It uses `curl_cffi==0.13.0` from a D-drive-only package directory.
The same gates remain mandatory: at least 20 overlapping dates, maximum close
difference <= 0.50 points, >=95% positive-volume coverage, unique/ordered/
finite OHLCV, and at least one 2026 extension row. No model, window, objective,
seed, feature, threshold, cost, or XAI setting changes.

## Claim boundary

The model and threshold protocol was frozen before any 2026 access, but the
registered source failed and one unusable Yahoo row was observed before this
alternative-source freeze. Results are therefore reported as a transparent
source-contingency forward evaluation, not as a pristine registered-source
confirmatory holdout. The full alternative 2026 series remains unseen until
this deviation document and adapter are frozen.
