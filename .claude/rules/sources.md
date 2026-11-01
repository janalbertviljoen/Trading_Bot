---
paths:
  - "Toolbox/*/sources/**/*.py"
  - "Toolbox/*/io.py"
  - "Toolbox/skatkis/io/**"
---

# Sources, mapping and IO

- A source module owns the boundary from one external source (SEC EDGAR, a price
  vendor, a constituents list) to canonical names: the field or XBRL tag mapping,
  reshaping, and the source model's schema. It never computes analytical outputs.
- All sources implement one `DataProvider` abstract base class, so a vendor can be
  swapped without touching downstream code (spec §1.2).
- Every fundamental keeps its filing (acceptance) date, form type, fiscal period and
  accession number; first-reported values are kept, restatements are not applied
  (spec §3.3, §3.5).
- Raw pulls are cached as immutable snapshots under `data/raw/<source>/<as_of>/` with a
  manifest (source, query, as-of date, row count, hash); `data/` is git-ignored
  (`AGENTS.md`, "Prohibited").
- File reading and writing goes through `skatkis.io`. Network access only through
  `skatkis.io.api.ApiClient`, the central API handler, or in source modules (for a
  vendor library such as `yfinance`); the vendor's endpoints and headers live in the
  tool's source module.
- Raw snapshots keep the vendor's format as received (gzipped); tables from cleaning
  onwards are Parquet through `skatkis.io.tables`. Manifests hash the content, not the
  file bytes.
- API keys come from the environment (`AGENTS.md`, "Prohibited") and are passed in;
  never hardcoded, logged or written to a snapshot. Respect vendor rate limits (SEC: a
  declared User-Agent, at most 10 requests per second).
- Fail fast with diagnostics naming the ticker, date and field on a malformed response,
  missing field or unit mismatch.
- Never mutate the caller's DataFrame; drop unmapped columns.
