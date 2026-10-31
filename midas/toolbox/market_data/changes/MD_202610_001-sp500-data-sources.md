# MD_202610_001: S&P 500 data sources

| Field | Value |
|---|---|
| Status | in progress |
| Branch | `feature/value-investing/vi-202610-001-sp500-data-sources` |
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
| `Toolbox/market_data/sources/` | `edgar.py`, `constituents.py`, `prices.py`, `fx.py`, `tag_map.py` | Vendor details: endpoints, User-Agent, field mapping |
| `Toolbox/market_data/preprocess/` | cleaning | Canonical names, units and types, standardised in the tool |
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

## Open decisions (user)

- Source of the imported S&P 500 P/E (benchmark): Robert Shiller's monthly data (Yale,
  Excel, needs an Excel reader) or S&P Dow Jones Indices' quarterly EPS sheet.

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
  as its own change, MD_202610_002.
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
- 2026-10-31 (user): storage. Raw snapshots keep the vendor's own format as received
  (EDGAR JSON, gzipped), so cleaning can be rebuilt without calling the API again.
  From cleaning onwards, tables are Parquet (spec §6, `io/artefacts.py`), adding
  `pyarrow` as a dependency. Manifests hash the content, not the file bytes.
- 2026-10-31 (user, decision of the same day applied to the area rules): network access
  is allowed in `skatkis.io`'s API client as well as in source modules;
  `.claude/rules/sources.md` is updated to match.
- 2026-10-31 (user, superseding "20 years where EDGAR has it"): only data from 2006
  onwards is of interest; prices and the exchange rate are requested from 2006. EDGAR
  returns a company's full document, so its values are trimmed to periods from 2006 when
  cleaned. Dividends are requested from 1986, so the 20-year dividend record (G4) can
  be evaluated from the start (user, same day).
- 2026-10-31 (user): the sources, the cleaned standard tables and the shared metrics
  (ratios such as P/E, current ratio, ROE, owned once) move out of `value_investing`
  into a new shared package between `skatkis` and the tools, so `value_investing` and
  the machine-learning tool `hand_o_midas` (HOM_202610_001) both read from it.
  `skatkis` stays free of domain knowledge. `value_investing` keeps the rule thresholds,
  gates, ranking and strategies. The package is `Toolbox/market_data/`, and this change
  moved from the `value_investing` log (VI_202610_001) to the new `market_data` log
  (user, same day).
- 2026-10-31 (user): before any feature engineering, the cleaned data is checked
  visually: plots per source showing coverage, gaps and outliers (plots from
  S_202610_007).
- 2026-10-31 (user): pulling data is an automated pipeline run by one terminal command,
  not scripts in `notebooks/`; it updates the data when EDGAR has new filings.
- 2026-10-31 (user): the pipeline as proposed: `Toolbox/market_data/update.py` with the
  command `python -m Toolbox.market_data update --as-of <date>` (not in `skatkis.io`,
  which stays the generic machinery); `--as-of` always explicit; EDGAR incremental
  through the `submissions` filing index; prices and FX in full per as-of date;
  companies and symbols as arguments until the constituents source exists; the
  `notebooks/fetch_*.py` scripts removed.
- 2026-10-31 (user): the S&P 500 P/E used for comparison is imported from an index
  source, not computed from the members. Model portfolios get their own aggregate P/E
  (and other aggregate metrics), compared against that benchmark.

## Notes

### Progress

- 2026-10-31: the data checks are a terminal command (`python -m Toolbox.market_data
  check`, `market_data/checks.py`) writing plots to `results/market_analysis/<source>/
  <as_of>/` (user: no scripts in `notebooks/`; `notebooks/check_data.py` removed,
  `results/` git-ignored). Added `market_data/clean.py` (`select_tags` with the user's
  most-recent-tag rule, `annual_first_reported`), `market_data/metrics.py` (`split_factor`,
  `price_earnings` on the latest annual EPS, filed the day before), `parse_tickers` for
  the CIK-to-ticker link (current tickers, until the constituents source exists) and
  `skatkis/io/figures.py`. The P/E behaves as expected: Coca-Cola's 2017 tax charge
  (EPS 0.29) gives a P/E of 150–175 in 2018–2019, Apple's first-reported FY2009 EPS
  (before its restatement) a P/E of 30–45, and no jumps at splits.
- 2026-10-31: first data checks (`notebooks/check_data.py`, plots from
  `skatkis.analysis`, saved under `data/interim/checks/<as_of>/`). All three
  APIs work: EDGAR has no missing values and no filing before its period end; reported
  diluted EPS equals net income / diluted shares within 5% for all 153 annual values;
  10-Q filed 18–34 days and 10-K 26–59 days after period end. Yahoo has 250–253 trading
  days a year for every symbol, no duplicates, no non-positive prices, and dividends
  and splits that match the companies' history (Apple's pause 1996–2011, Microsoft's
  first dividend 2003). FRED shows the known rand episodes (2008, end-2015, 2020). Two
  findings for MD_202610_002 are in the market_data notes: tags must be chosen per
  period, and per-share comparatives are restated for splits.
- 2026-10-31: update pipeline built (`update.py`, `__main__.py`,
  `EdgarSubmissions` and `latest_periodic_filing` in `sources/edgar.py`,
  `SnapshotStore.latest_as_of` in `skatkis.io.snapshots`). Live run: EDGAR reused all
  three companies (no filing since the snapshot), prices and FX written. The
  incremental rule checked offline: a company that filed after its snapshot is fetched
  again, one that did not is reused without a download. Plots of the run scripts move
  to S_202610_007. VS Code hides `__pycache__` and `__init__.py`
  (`.vscode/settings.json`, user's request).
- 2026-10-31: moved to `Toolbox/market_data/` (sources, `config/market_data.toml`, the
  "Data sources" section now in the package README). Prices requested daily from 2006,
  dividends and splits through monthly bars from 1986, FX from 2006; snapshots fetched
  with the earlier ranges were deleted (same as-of date, different request). Re-pulled:
  prices from 2006-01-03 for Apple, Microsoft, Coca-Cola and `^SP500TR`, dividends from
  1986, 5,214 FX rates from 2006; raw data about 1.6 MB in total.

- 2026-10-31: step 1 built. `skatkis.io` (`api.py`, `provider.py`, `snapshots.py`,
  `tables.py`, `config.py`), the EDGAR source (`sources/edgar.py`, `sources/tag_map.py`
  with 26 canonical fields), `config/value_investing.toml` and the run script
  `notebooks/fetch_edgar.py`. HTTP uses the standard library's `urllib` (`requests` has
  no type stubs for strict pyright). Ruff's D205 and D400 are switched off because they
  conflict with the module docstring heading of `.claude/rules/docstrings.md`. Checked
  offline with a synthetic payload (retries, snapshot reuse, integrity check, key
  redaction, unit and CIK errors); Black, ruff and pyright clean. Not yet run against
  SEC: it needs the user's User-Agent, and `pyarrow` still has to be installed.
- 2026-10-31: first live pull (Apple, Microsoft, Coca-Cola; User-Agent given by the
  user): 16,403 values, all 26 fields found. The pull needed two fixes in `ApiClient`:
  responses cut off mid-transfer are retried, and responses are requested gzip-encoded
  (large uncompressed responses were cut off repeatedly). Tag fallbacks added for
  revenue (`SalesRevenueGoodsNet`), long-term debt
  (`LongTermDebtAndCapitalLeaseObligations`) and D&A (`Depreciation`), and
  `liabilities_and_equity` mapped so total liabilities can be derived where a company
  does not report them. History reaches back to periods ending 2006–2007 only, which
  is short for the long Graham windows: see the tool's notes.
- 2026-10-31: prices built (`sources/prices.py`, run script `notebooks/fetch_prices.py`):
  Yahoo's chart endpoint through `ApiClient` instead of the `yfinance` library (same
  data; raw JSON snapshots and no extra dependency). Prices, dividends and splits from
  1970 for Apple, Microsoft, Coca-Cola and `^SP500TR`; total-return growth plotted.
  In the sandbox, Python downloads are cut off at random by its network proxy (`curl`
  is not), so some pulls needed several runs; outside the sandbox this does not apply.
- 2026-10-31: exchange rate built (`sources/fred.py`, run script `notebooks/fetch_fx.py`):
  FRED `DEXSFUS` through `ApiClient`, by the official API when `FRED_API_KEY` is set
  and otherwise by the keyless chart CSV, both parsed to one table. Pulled without a
  key: 11,738 daily rates from 1980-01-02 (the series starts in 1980, not 1971) to
  2026-10-23 (H.10 is published weekly, so the latest rates lag by about a week).

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

Pick up MD_202610_001. Read spec §1, §3, §4, §6 and the tool's notes, confirm the
decisions above are made, then propose the source modules and canonical fields before
building the EDGAR client first.

## Done when

- Fundamentals for a small fixture universe are pulled, mapped to canonical names,
  keyed to filing date and cached with a manifest; tests pass without network.
