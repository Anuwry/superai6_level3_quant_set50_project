# Public replication package v3

This package supports the paper's reliability-audit claims without
redistributing provider-hosted row-level SET50 or SET100 observations.

## Included

- Python model, evaluation, governance, and packaging code;
- automated tests;
- frozen protocols, manifests, and execution logs;
- environment requirement files;
- market-data schemas, provenance hashes, and non-reconstructive integrity
  summaries;
- aggregate SET100 same-exchange robustness results;
- aggregate integrated multimodal and moving-block-bootstrap results; and
- aggregate LLM compute-matched metrics, inference, runtime, and cost records;
- the frozen multimodal news-only, shuffled, lagged, and random-feature
  falsification aggregates; and
- manuscript-ready main/Supplement tables with exact source/output hashes.

Every copied file is listed by relative path, byte count, and SHA-256 digest in
`PUBLIC_MANIFEST.json`.

## Excluded by design

- raw or prepared SET50/SET100 rows;
- point-in-time fold CSVs and row-level market predictions;
- raw news text and labelled article-level data;
- private LLM responses, response identifiers, and checkpoints;
- API keys, `.env` files, and private contractual or administrative records; and
- artifacts from which the provider-hosted market series could be reconstructed.

The package builder rejects restricted paths, unsafe relative paths, symlinks,
row-level prediction filenames, unsupported file types, and OpenAI-key-like
secret patterns before it creates the destination.

## Data availability

Market-index observations were obtained from publicly accessible Investing.com
historical-data pages that offer a data-download option. Public accessibility
is not represented as an open-data licence, and use remains subject to the
provider's terms. The package therefore supplies code, schemas, checksums,
integrity summaries, and temporal split specifications but excludes row-level
provider data. Researchers must obtain the identified SET50 and SET100 series
independently from the provider and verify the supplied hashes and data
contracts before reproduction.

SET100 is a same-exchange index-breadth transfer audit, not an independent
external-market replication.

## Build

From the project root on Windows PowerShell:

```powershell
$env:PYTHONPATH=(Get-Location).Path
py -3.12 scripts/build_public_replication_package.py
```

The command creates `release/public_replication_package_v3/` only when the
destination does not already exist. This fail-closed behavior prevents a stale
or partly overwritten release. Use a new versioned destination for later
releases.

Independently verify every expected path, byte count, SHA-256 digest, package
digest, and secret scan after building:

```powershell
py -3.12 scripts/audit_public_replication_package.py
```

The audit writes `outputs/public_replication_package_v3_audit.json` and fails
closed on a missing, unexpected, modified, or secret-bearing file.

## Submission evidence

The governance record retains the provider URLs, retrieval/acquisition basis,
file hashes, timezone convention, adjustment convention, and a dated snapshot
of the applicable provider terms. The study claims public provider access only,
not an open-data licence or a separately negotiated redistribution right.
