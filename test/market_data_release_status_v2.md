# Market-data public-release status v2

Status time (UTC): `2026-08-03T17:08:00Z`  
Status: **CLEAN PACKAGE PASSED; PROVIDER TERMS APPLY**

This note updates only the release status recorded in
`outputs/market_data_governance_v1/release_gates.json`. It does not rewrite the
original governance run or its provenance/integrity findings.

Changes since v1:

- the SET100 same-exchange benchmark is complete at 100/100 registered fits;
- a clean fail-closed public replication bundle has been generated and audited;
- the original private working repository still contains restricted SET50
  history, but the separate release bundle contains no restricted row-level
  file; and
- the access claim is limited to publicly accessible provider pages at the
  recorded acquisition time; this is not an open-data licence; and
- provider-hosted row-level records remain excluded from public release.

The authoritative current machine-readable status is
`outputs/market_data_governance_v1/release_gates_v2.json`. The v1 file is
retained as a dated audit snapshot.

The clean bundle may be released because raw provider-hosted row-level records
are not included. Researchers must obtain the underlying series independently
from the identified publicly accessible provider pages and comply with the
terms in effect at their time of access.
