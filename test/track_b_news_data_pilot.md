# Track B Historical News Data Pilot

Run date: 2026-07-28 (Asia/Bangkok)

## Objective

Verify that historical SET-related news metadata can be recovered before
building the full Track B sentiment pipeline. This pilot is a data-availability
and filtering audit only. It is not used for model training or benchmarking.

The required Track B target period is 2012-2025. The 2015 start used in this
pilot is a source boundary for GDELT 2.0, not a reduction of the required
research period.

## Source and sampling

- Source: [GDELT 2.1 GKG](https://www.gdeltproject.org/data.html)
- Query date: 2025-12-19, a date present in the SET50 daily market data
- GDELT raw archive cadence: every 15 minutes (96 snapshots per UTC day)
- Pilot cadence: every 60 minutes
- Snapshots scanned: 24 of 96
- Temporal sampling fraction: 25%
- Text scope: GDELT metadata and page title only
- Full article bodies collected: no

The hourly pilot is intentionally incomplete. It is sufficient to verify the
archive, timestamp, metadata parsing, and relevance-filter path without first
downloading a full historical corpus.

## Relevance and timestamp controls

Candidate scoring uses only metadata:

- SET50, SET 50, SET Index, Thai stock market, or equivalent title terms
- `Stock Exchange of Thailand` in the organization metadata
- GDELT `ECON_STOCKMARKET` theme
- Thailand location metadata
- Thai source-domain signal

The strict threshold is a relevance score of at least 5. A candidate must also
have a page publication timestamp no more than seven days away from the GDELT
batch timestamp.

The publication-lag control was added after the pilot found a page published in
2022 that GDELT crawled again in a 2025 batch. Assigning that page to 2022 after
discovering it in 2025 could introduce look-ahead bias.

## Pilot result

- Candidate metadata rows after URL deduplication: 17
- Strict relevant rows: 1

Strict relevant example:

| Field | Value |
|---|---|
| Published (UTC) | 2025-12-19 00:43 |
| Published (Bangkok) | 2025-12-19 07:43 |
| Source | thailand-business-news.com |
| Title | Thai Stock Market Navigates Global Headwinds Ahead of 2026 |
| GDELT tone | -1.7458777885548 |
| Relevance score | 11 |
| Text collected | Metadata and title only |

The GDELT tone value is an archive-provided signal for inspection. It is not
treated as the final Track B LLM sentiment score.

## Source-access audit

The discovered publisher's
[robots.txt](https://www.thailand-business-news.com/robots.txt) was checked on
2026-07-28. It allows search/reference use but includes `ai-input=no` and
`ai-train=no`. Therefore, this pilot did not collect the article body and must
not feed that publisher's full content to the proposed LLM agents without a
separate licence or permission.

Before full collection, each source domain must receive a recorded access audit:

- robots and content signals
- licence or terms applicable to research use
- whether title/snippet, full text, or neither may be used as model input
- retrieval date and policy version

## Reproducibility

Script:

```text
models/gdelt_news_sample.py
```

Command used:

```powershell
python -m models.gdelt_news_sample `
  --date 2025-12-19 `
  --archive-dir tmp\gdelt_probe_20251219 `
  --output-dir data-news-sample\gdelt_2025-12-19_hourly `
  --step-minutes 60 `
  --min-relevance-score 5
```

For a complete UTC day, use `--step-minutes 15` and `--download`.

Outputs:

- `data-news-sample/gdelt_2025-12-19_hourly/gdelt_candidates.csv`
- `data-news-sample/gdelt_2025-12-19_hourly/gdelt_relevant.csv`
- `data-news-sample/gdelt_2025-12-19_hourly/run_metadata.json`

Tests:

```text
8 passed in tests/test_gdelt_news_sample.py
```

## Decision

Historical metadata collection is technically feasible. Full-text sentiment is
conditional on per-source rights and access rules. Do not launch the complete
2015-2025 scrape or benchmark Track B from this pilot alone. The next defensible
step is a coverage audit using several sampled trading days per year, followed
by a source-permission audit and a decision on the final common coverage period.

To satisfy the 2012-2025 requirement, collection must use one of these designs:

1. Preferred: obtain a licensed, consistent historical news export covering the
   entire period, such as SET's historical News & Financial data service.
2. Free fallback: recover 2012-2014 from Common Crawl and other permitted
   publisher archives, then use GDELT 2.0 discovery for 2015-2025.

The free fallback creates a source-regime boundary in 2015. Coverage, language,
headline availability, timestamp quality, and duplicate rate must therefore be
reported by year. The same text contract, preferably headline plus permitted
snippet and metadata, must be applied across all years to avoid making the
2015-2025 segment systematically richer than 2012-2014.

## Full-history scale estimate

Estimate recorded on 2026-07-28 from the local pilot:

- 24 hourly snapshots occupied 126.05 MB compressed.
- The same snapshots occupied 390.79 MB after expansion.
- Mean compressed snapshot size was 5.25 MB.
- 2015-02-19 through 2025-12-22 contains 3,960 calendar days.
- Complete 15-minute coverage would require approximately 380,160 snapshots.

If the pilot average were representative, a naive download of every GKG
snapshot would require approximately 1.9 TiB compressed and about 5.9 TiB when
expanded. The pilot download completed at roughly 5 MB/s with four concurrent
download batches. At that short-run speed the theoretical download time is
about 4.5 continuous days; a realistic raw-archive estimate is one to three
weeks after allowing for respectful request rates, retries, parsing, filtering,
and validation.

This estimate does not include:

- recovery of 2012-2014 articles from a different archive
- downloading permitted article bodies
- deduplication and manual relevance validation
- LLM sentiment inference

The preferred full-run design is therefore to query/filter GDELT metadata
before export, rather than downloading the entire raw GKG archive. A small
multi-year coverage audit and a 100-article LLM runtime benchmark must precede
the full run.

## One-day deadline decision

For a one-day experiment deadline, use the downloadable VISTEC
`Bilingual_StockTBSA` Thai dataset instead of constructing a 2012-2025 corpus.
The locally verified file contains 10,295 articles and 15,949 ticker-level
sentiment labels from 2018-01-03 through 2023-12-28.

The locked reporting scopes are:

- Track A: 2012-2025
- Track B intrinsic sentiment evaluation: 2018-2023
- Track A+B fusion: common-period comparison only

The detailed fast-track contract is recorded in
`test/track_b_one_day_experiment.md`.
