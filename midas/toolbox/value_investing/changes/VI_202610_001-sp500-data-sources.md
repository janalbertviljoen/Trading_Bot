# VI_202610_001: S&P 500 data sources

| Field | Value |
|---|---|
| Status | in discussion |
| Priority | high |
| Kind | new capability |
| Depends on | S_202610_001; spec §9.1 (data budget) |
| Raised | 2026-10-31 |

## Description

The first stage of the pipeline (spec §5, stage 1 `ingest`; §6.2 `ingest.constituents`,
`ingest.prices`, `ingest.filings`): pull S&P 500 fundamentals, prices and historical
constituents through APIs and cache them as immutable snapshots, so every later stage
works point-in-time and reproducibly.

## Approach (to discuss)

- `Toolbox/value_investing/sources/`: one module per source model, behind one
  `DataProvider` abstract base class (spec §1.2), mapping vendor fields to canonical
  names; no calculations.
- **Fundamentals: SEC EDGAR XBRL** (free, authoritative). `companyfacts` per CIK gives
  every reported value with its form, fiscal period, filing date (`filed`) and accession
  number, which allows first-reported values keyed to filing date (spec §3.3, §3.5). A
  tag map (`tag_map.py`) maps us-gaap tags to canonical fields (revenue, EPS, current
  assets, …). SEC requires a declared User-Agent and at most 10 requests per second.
- **Prices and historical constituents:** EDGAR has neither. Options: a paid vendor
  (FMP or EODHD: prices, dividends and historical membership), or free sources
  (yfinance for prices, which the spec rules out for audit-grade output; Wikipedia's
  change table for membership, which is incomplete).
- Snapshots under `data/raw/<source>/<as_of>/` (git-ignored), written through
  `skatkis.io`, with a manifest (source, query, as-of date, row count, hash).
- Tests with small recorded fixtures; no network in unit tests.

## Open decisions (user)

- Spec §9.1: paid vendor budget (which vendor), or EDGAR only plus free prices for
  prototyping?
- Spec §9.5: reporting currency USD or ZAR.
- Spec §9.3: financials and REITs excluded or scored separately (affects which tags
  are mapped).
- Spec §9.6: DAG engine: stdlib `graphlib` registry (default) or Hamilton/Dagster.

## Kickoff prompt

Pick up VI_202610_001. Read spec §1, §3, §4, §6 and the tool's notes, confirm the
decisions above are made, then propose the source modules and canonical fields before
building the EDGAR client first.

## Done when

- Fundamentals for a small fixture universe are pulled, mapped to canonical names,
  keyed to filing date and cached with a manifest; tests pass without network.
