# VI_202610_001: S&P 500 data sources

| Field | Value |
|---|---|
| Status | agreed |
| Priority | high |
| Kind | new capability |
| Depends on | S_202610_001 |
| Raised | 2026-10-31 |

## Description

The first stage of the pipeline (spec §5, stage 1 `ingest`; §6.2 `ingest.constituents`,
`ingest.prices`, `ingest.filings`): pull S&P 500 fundamentals, prices and historical
constituents through APIs and cache them as immutable snapshots, so every later stage
works point-in-time and reproducibly.

## Agreed approach

The data-source layer follows ports and adapters: the port and the API mechanics are
in `skatkis`, one adapter per vendor is in the tool.

| Where | Part | Does |
|---|---|---|
| `Toolbox/skatkis/io/api.py` | `ApiClient` | HTTP GET with configured headers, rate limit, retries and timeout; no vendor knowledge |
| `Toolbox/skatkis/io/provider.py` | `DataProvider` (abstract base class) | The single provider interface of spec §1.2: a query and an explicit `as_of` date in, raw data out |
| `Toolbox/skatkis/io/snapshots.py` | raw snapshots | Fetch once and store under `data/raw/<source>/<as_of>/` with a manifest; reruns read the stored snapshot |
| `Toolbox/value_investing/sources/` | `edgar.py`, `constituents.py`, `prices.py`, `fx.py`, `tag_map.py` | Vendor details: endpoints, User-Agent, field mapping |
| `Toolbox/value_investing/preprocess/` | cleaning | Canonical names, units and types, standardised in the tool |
| then | validation | Declared columns only during the build |

`yfinance` is a library rather than an HTTP API, so its adapter bypasses `ApiClient` but
still stores its data through the snapshots module. Spec §6 places `DataProvider` in
the tool's `ingest/`; moving it to `skatkis` is a deliberate deviation, carried into the
spec in the documentation step.

Per source:
- **Fundamentals: SEC EDGAR XBRL** (free, authoritative). `companyfacts` per CIK gives
  every reported value with its form, fiscal period, filing date (`filed`) and accession
  number, which allows first-reported values keyed to filing date (spec §3.3, §3.5). A
  tag map (`tag_map.py`) maps us-gaap tags to canonical fields (revenue, EPS, current
  assets, …). SEC requires a declared User-Agent and at most 10 requests per second.
- **Historical constituents (free):** Wikipedia's current list (date added, GICS
  sector, CIK) and historical-components page, or the `fja05680/sp500` daily
  membership file from 1996, cross-checked against S&P DJI announcements. Joined to
  EDGAR on CIK.
- **Prices (free):** `yfinance` for daily closes, dividends and the benchmark
  `^SP500TR`. Known gap: no prices for delisted members.
- **Exchange rate (free):** FRED series `DEXSFUS`, the Federal Reserve H.10 daily
  ZAR per USD rate. Any API key is read from the environment at the command-line
  boundary.
- Snapshots under `data/raw/<source>/<as_of>/` (git-ignored), written through
  `skatkis.io`, with a manifest (source, query, as-of date, row count, hash).
- Tests with small recorded fixtures; no network in unit tests.

## Decisions

- 2026-10-31 (user): the free version (spec §9.1): EDGAR for fundamentals, Wikipedia
  or `fja05680` for membership, `yfinance` for prices. This is a quarterly update, not
  day trading. The missing prices of delisted members (survivorship bias) are accepted
  as a known limitation for now.
- 2026-10-31 (user, superseding the same day's REIT decision): REITs and financial
  companies are excluded from the stock universe (spec §9.3), so no bank, insurer or
  REIT tags are mapped. Exclusion by GICS sector (Financials, Real Estate) from the
  constituents list. Bonds are not part of this tool.
- 2026-10-31 (user): pull 20 years of history where EDGAR has it.
- 2026-10-31 (user): focus on the product first; validation stays minimal during the
  build (declared columns only), because the data comes through APIs.
- 2026-10-31 (user): the point-in-time panel is built from these sources; it is raised
  as its own change, VI_202610_003.
- 2026-10-31 (user, spec §9.5): report in ZAR where a rate is available, otherwise in
  USD, because the money is invested from the JSE and the exchange rate affects the
  result. A USD/ZAR daily rate is therefore a data source of this change.
- 2026-10-31 (user): setting up the API access and getting the data in is the
  priority. A central API handler lives in `skatkis.io` and carries out every API
  import; the tool holds the vendor-specific API details. Cleaning and preprocessing
  are standardised in the tool; validation comes only after that.
- 2026-10-31 (user): the split of the data-source layer in the agreed approach is
  confirmed.
- 2026-10-31 (user, refining the same day's currency decision): the main analysis is
  in USD. ZAR is secondary: added once the analysis is refined, and used to track the
  actual record of the sleeve. The exchange rate comes from FRED `DEXSFUS`.
- 2026-10-31 (user, spec §9.6): the DAG engine is the stdlib `graphlib` registry.

## Notes

### Source research (2026-10-31)

| Need | Option | Cost | Point-in-time | Remarks |
|---|---|---|---|---|
| Fundamentals | SEC EDGAR `companyfacts` API | free | yes: `filed` date and accession per value | 10 requests/s, declared User-Agent |
| Fundamentals, bulk | SEC Financial Statement Data Sets (quarterly zips) | free | yes: values "as filed", with the filing date in the `sub` table | filings after quarter end roll into the next posting; key on the filing date, not the posting |
| Constituents | S&P DJI index announcements (press releases) | free | yes: announcement and effective dates | the primary record; one release per change, so scraping is needed |
| Constituents | Wikipedia "List of S&P 500 companies" and "Historical components of the S&P 500" | free | partly: current list with date added and CIK; changes back to about 1990, incomplete early | CIK column links directly to EDGAR |
| Constituents | GitHub `fja05680/sp500`, `hanshof/sp500_constituents` | free | yes: daily membership from 1996 | community-maintained; tickers need mapping to CIK |
| Constituents | FMP, EODHD, Norgate | paid | yes | EODHD from 2000 |
| Prices | Yahoo Finance via `yfinance` (incl. `^SP500TR`) | free | n/a | unofficial; personal use only; no delisted tickers; the spec rules it out for audit-grade output |
| Prices | Tiingo | free tier (about 1,000 requests/day) | n/a | personal use; dividends and splits; delisted coverage unconfirmed |
| Prices | EODHD, FMP | paid | n/a | include delisted tickers |

- Yahoo Finance fundamentals are restated and cover only a few years, so they cannot
  serve as point-in-time fundamentals.
- Free route that fits the spec: EDGAR for fundamentals, Wikipedia or `fja05680` for
  membership cross-checked against S&P DJI announcements, and `yfinance` or Tiingo for
  prices while prototyping. Gap: prices of delisted members, which the free sources
  miss; survivorship-free backtests need a paid price source.

## Kickoff prompt

Pick up VI_202610_001. Read spec §1, §3, §4, §6 and the tool's notes, confirm the
decisions above are made, then propose the source modules and canonical fields before
building the EDGAR client first.

## Done when

- Fundamentals for a small fixture universe are pulled, mapped to canonical names,
  keyed to filing date and cached with a manifest; tests pass without network.
