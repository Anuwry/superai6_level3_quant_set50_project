# Track B Gap-Data Audit: 2012-2017 and 2024-2025

Run date: 2026-07-28 (Asia/Bangkok)

## Confirmed local data

### 2018-2023 labelled primary data

`Bilingual_StockTBSA/Thai_Financial_TBSA_dataset_Updated.json` is downloaded
locally. It contains 10,295 articles and 15,949 article-ticker labels from
2018-01-03 through 2023-12-28.

### 2024-2025 official SET headline metadata

The SET public news search was verified and collected:

- 2024: 32,967 raw records, 32,966 after exact deduplication
- 2025: 36,864 raw records, 36,858 after exact deduplication
- total normalized headlines: 69,824
- download runtime: 28.88 seconds
- content contract: Thai headline, ticker, publication time, URL, and metadata
- sentiment labels: absent
- full article body: not collected

These records cover all SET security-type `S` company news. A point-in-time
SET50-membership filter is still required.

### 2015-2017 CMDF-VISTEC extraction

The CMDF-VISTEC Kaohoon raw CSV was downloaded and extracted because:

- its first verified records include March 2015 publication dates
- its source overlaps the dominant source in Bilingual Stock TBSA
- the repository is CC BY 4.0
- full article text is present

Extraction result:

- raw file size: 812,569,122 bytes
- raw download wall time: approximately 371 seconds
- raw rows scanned: 188,742
- rows with a parsed date: 188,742 (100%)
- selected rows: 68,514
- 2015: 22,287, beginning on 2015-03-24
- 2016: 26,039
- 2017: 20,188
- extraction runtime: 13.66 seconds

Limitations:

- the only raw columns are `id,text`
- publication dates must be parsed from article text
- sentiment and ticker labels are absent
- extracted dates need manual QA

Parser QA found that 68,346 of 68,514 selected dates were on line 3, 163 were
on lines 2 or 4, and five were later. The two records with dates after line 10
were manually inspected and contained duplicated content before a valid
publication-date block.

## Unresolved 2012-2014 gap

No free, ready-to-use, consistently dated and labelled Thai stock-news dataset
for 2012-2014 was found in the audited public repositories.

Possible recovery paths are:

1. Obtain SET's historical News & Financial/PSIMS service.
2. Recover permitted pages from Common Crawl or publisher archives.
3. Report 2012-2014 as unavailable for Track B and keep Track A at 2012-2025.

The free archive path is not suitable for a one-day deadline because it
requires source discovery, timestamp validation, deduplication, rights review,
ticker extraction, and sentiment inference.

## Defensible experiment scope

- Track A numerical benchmark: 2012-2025
- Primary Track B labelled benchmark: 2018-2023
- Extended headline/full-text robustness analysis:
  - 2015-2017 CMDF-VISTEC Kaohoon after date QA
  - 2024-2025 official SET headlines after point-in-time SET50 filtering
- Track A+B fusion: report results separately by source regime; do not fill
  2012-2014 with zero sentiment and call it observed news.

The extension data is useful for temporal robustness but must not be presented
as one homogeneous labelled 2012-2025 news dataset.
