# Reproducibility guide

## Scope

The repository distinguishes exact reproduction of persisted artifacts from a
fresh robustness rerun. Provider-hosted SET50 and SET100 rows are not included
in the public package. Researchers must obtain the identified series
independently from the publicly accessible provider pages and comply with the
terms in effect when they access the data.

## Environments

- Use `requirements-paper-py312.txt` with CPython 3.12.10 for the corrected
  point-in-time pipeline, governance checks, current scripts, and the complete
  test suite.
- Use `requirements-integrated-py311.txt` with CPython 3.11.15 only when exact
  recreation of the persisted integrated multimodal run is required. Its
  versions match that run's `run_metadata.json`.
- `requirements-paper.txt` is the current convenient full-test environment.
  `requirements-tuning.txt` contains optional tuning and heavyweight benchmark
  packages and is not required to reconstruct the paper's frozen primary
  comparisons.

Create separate virtual environments; do not install both lock files into one
environment and call the result an exact reproduction.

```powershell
py -3.12 -m venv .venv-paper
.\.venv-paper\Scripts\python.exe -m pip install -r requirements-paper-py312.txt
.\.venv-paper\Scripts\python.exe -m pytest tests -q
```

`pytest.ini` keeps test temporary files in the project-local `.pytest_tmp`
directory on drive D instead of the space-constrained Windows system drive.

## Data contract

The expected source files, source URLs, byte counts, SHA-256 hashes, date
ranges, Asia/Bangkok session convention, 17:00 information cutoff, and
price-index adjustment convention are recorded under
`outputs/market_data_governance_v1/`. A researcher-supplied file must pass the
same schema, temporal, and integrity checks before a fresh run is comparable.

## Reproduction order

1. Validate the public package manifest and secret/restricted-path gates.
2. Place independently obtained provider files at the documented local input
   paths; do not commit or redistribute them.
3. Run the market-data governance and point-in-time contract checks.
4. Reconstruct the registered fold inputs.
5. Run the frozen experiment or validate persisted aggregate artifacts.
6. Build manuscript tables from authoritative aggregate CSV/JSON files.
7. Run syntax, type, lint, test, security, and manifest verification.

Every fresh result must record the command, configuration/freeze hash, package
versions, seed, runtime, input hashes, and output directory. A retrospective
rerun over an already inspected date range is a robustness analysis, not an
untouched confirmatory holdout.

After all registered aggregate artifacts exist, build the manuscript table
bundle and its exact source/output hash manifest with:

```powershell
py -3.12 scripts/build_manuscript_artifacts.py
```

Create the fail-closed public package only after the manuscript bundle exists,
then independently re-hash every packaged file and rescan it for secrets:

```powershell
py -3.12 scripts/build_public_replication_package.py
py -3.12 scripts/audit_public_replication_package.py
```

The second command writes
`outputs/public_replication_package_v3_audit.json` and exits nonzero for a
missing, unexpected, modified, restricted, or secret-bearing file.

The typed-core gate intentionally covers newly maintained pure analysis and
packaging modules. Historical frozen runners retain their recorded source hashes
and are verified through execution tests rather than being rewritten after
result access. Run the typed-core gate with the active interpreter:

```powershell
$python=(py -3.12 -c "import sys; print(sys.executable)")
py -3.12 -m pyright --pythonpath $python -p pyrightconfig.json
```
