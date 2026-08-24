# Track D Source-Deviation Parser Amendment V2

Status: **FROZEN BEFORE REPEATED SAME-RANGE REQUEST**  
Date: 2026-07-31

The first full Investing.com SET50 request under source-deviation V1 returned
200 rows for 2025-10-01 through 2026-07-31, but the frozen parser rejected one
row before saving the snapshot. On 2025-12-11, the source reports Open=840.26,
High=839.88, Low=825.54, Close=828.42. The same values appear in the frozen
local historical source, so this is a pre-existing source anomaly rather than
a field-mapping error or a newly observed 2026 condition.

V2 removes only the extra OHLC-containment rejection that was not part of the
registered source-deviation gates. It does not repair, clip, or impute the raw
values. Instead it records the count and maximum High shortfall/Low excess in
the forward data manifest. Completeness, finiteness, date uniqueness, positive
volume, >=20-row overlap, <=0.50 close-difference, and 2026-extension gates
remain unchanged. All model and evaluation settings remain frozen.

The full alternative series was accessed before this parser amendment. V2 is
therefore frozen before the repeated identical-range request and accepted
snapshot, not before alternative-series access. This limitation must remain in
the paper's source-deviation disclosure.
