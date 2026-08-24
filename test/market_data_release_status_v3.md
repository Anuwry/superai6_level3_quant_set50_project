# Market-data public-release status v3

Status time (UTC): `2026-08-03T22:17:30Z`  
Supersedes status statements in v1/v2; earlier files remain audit history.

## Current gate

The clean public release is `release/public_replication_package_v3/` under
protocol `public-replication-package-v3`. Its exact file count and package
digest are authoritative only in the generated `PUBLIC_MANIFEST.json`; they
are deliberately not copied into this in-package document because that would
create a self-referential digest.

Independent verification passed exact path, byte-count, per-file SHA-256,
package-digest, restricted-path, and secret-pattern checks. There were zero
missing, unexpected, modified, size-mismatched, or secret-bearing files.
Evidence is recorded in
`outputs/public_replication_package_v3_audit.json` and
`outputs/market_data_governance_v1/release_gates_v3.json`.

## Access and redistribution boundary

The SET50 and SET100 observations came from publicly accessible provider
historical-data pages with download functionality. Provider terms apply;
public accessibility is not claimed to be an open-data licence. Raw and
prepared provider rows, temporal fold rows, and row-level predictions are not
redistributed. The release includes code, protocols, schemas, hashes,
non-reconstructive aggregates, and manuscript tables.

The paper uses Asia/Bangkok session dates, a conservative 17:00 information
cutoff, previous-completed-period alignment for weekly/monthly inputs, and
provider-published price-index levels without an added total-return, dividend,
or corporate-action adjustment.

SET100 remains a same-exchange breadth audit, not independent external-market
replication. The 252-session prospective result after 2026-07-30 is frozen but
not complete.
