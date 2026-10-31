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
- 2026-10-31 (user): the constituents source carries each member's GICS sector and
  sub-industry, the grouping for analysis and plots (S_202610_007).
- 2026-10-31 (user): each GICS sector is compared with its own sector benchmark as well
  as the S&P 500 (like with like). Built with the SPDR sector funds (`XLK`, `XLP`, ...)
  rather than the S&P sector indices on Yahoo (`^SP500-45`, ...), because the indices
  are price-only and the companies and `^SP500TR` are total return (developer's choice,
  stated to the user; the price indices remain an option).
- 2026-10-31 (user): the imported S&P 500 P/E comes from S&P Dow Jones Indices (their
  index earnings sheet; needs `openpyxl`, to be installed by the user).

## Notes

### Progress

- 2026-10-31: first check on the full EDGAR universe (388 companies, while prices were
  still loading): coverage from 2009 about 300 to 385 companies per field
  (`gross_profit` about 175 and `total_liabilities` about 250, line items many companies
  do not report). At scale, 386 of 6,039 annual EPS values differ from net income /
  diluted shares by more than 5% (share classes, minority interests, discontinued
  operations: to look into), and filings carry forward-dated values (maturities), so a
  filing's own period is now the latest period end on or before its filing date
  (median filing delay 52 days for 10-K, 32 for 10-Q).
- 2026-10-31 (user: keep it simple): the sector grouping is one mapping table, like the
  tag map: `gics_map` gives each company its GICS sector and sector fund; the checks
  merge it onto prices and P/E and `groupby` sector, normalising every group the same
  way against its own fund and the market. The separate universe module and the
  per-sector rebuilding in the checks were removed.
- 2026-10-31: constituents source built (`sources/constituents.py`): the Wikipedia
  pages "List of S&P 500 companies" (503 members, all with CIK, GICS sector and
  sub-industry, date added) and "Historical components of the S&P 500" (414 changes
  back to 1976; thin before 2007), parsed from wikitext through the MediaWiki API (no
  HTML library). The GICS map (`market_data/gics.py`, `gics_map`) tags each member with
  its sector and sector fund and leaves out Financials and Real Estate; `update --universe sp500` pulls them with the market and sector funds
  (`[benchmarks]` in `config/market_data.toml`). Failures per company or symbol are
  recorded and reported instead of stopping the run.
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

## Warden report

Run 2026-10-31, commit b20a167 with a dirty working tree, reviewed against e990e6c. One
report covers MD_202610_001 and S_202610_007; it is written into both change files.

**Header**
- Changes: MD_202610_001 (S&P 500 data sources, `Toolbox/market_data`) and S_202610_007 (market analysis plots and the LaTeX report). Both are labelled new capability.
- Date: 2026-10-31
- Commit reviewed: b20a167 on `feature/value-investing/vi-202610-001-sp500-data-sources`. The tree is dirty: staged and unstaged changes plus untracked `rankings.py`, `market_data/report.py`, `skatkis/financial_logic.py` and `skatkis/report.py`.
- Base: `main`. Scope is e990e6c to the working tree.
- Run: first run.
- Inputs used: `review.diff` and `review.checks.txt` from the session scratchpad, plus read-only reads of the repository.

### Check results as given
- Black, ruff and strict pyright are clean. Pytest: 9 passed, 1 warning. No test file is in the diff, so no test was removed.
- `regression_check` returned `unchanged`. The wallet is not touched.
- `market_data check` ran. LaTeX compilation failed in the sandbox: `booktabs.sty` could not be downloaded, so there is no PDF.
- The update pull failed for 2 of 397 companies (CIK 32604, CIK 864749) on a unit mismatch.

### Findings

**Blocking**
None found.

**Should fix**

1. **Companies with two share classes are counted twice.**
   - Where: `Toolbox/market_data/checks.py` line 163 (`annual ... .merge(gics[["cik","symbol","gics_sector"]], on="cik", how="left")`), with `gics.py` keeping every constituents row that has a CIK.
   - What happens: the Wikipedia list has 503 rows for 500 companies, because Alphabet, Fox and News Corp each have two share classes under one CIK. The merge copies each of those companies' annual rows once per symbol. As a result:
     - `fundamentals_sector_totals` and `sectors_fundamentals` add their revenue and net income twice. Communication Services is hit hardest.
     - `company_scores("fundamentals")` lists both symbols as separate companies. They can both appear in the best or worst lists, and they weigh twice in the sector median.
     - The company count in the introduction table is too high.
   - Rule broken: correctness of an owned calculation (AGENTS.md "Ownership"), and the plots are tracked in git as results.
   - Fix: give the GICS map one row per CIK for the fundamentals (for example, keep the first symbol per CIK). Keep per-symbol rows for prices and P/E. Then re-run `check`.
   - Not verified against the stored constituents table: I inferred this from the code and the 503-member count.

2. **The excluded sectors are not applied in `check`.**
   - Where: `checks.py` line 154 calls `gics_map(constituents, funds=sectors)` with no `excluded=`. `__main__._check` (line 84) never reads `[universe] excluded_sectors`.
   - Financials and Real Estate are left out today only because `update --universe sp500` did not pull their prices. A `--ticker` for a bank, or an older update, would bring them into the analysis and the report.
   - Rule broken: MD_202610_001 decision (2026-10-31), exclusion by GICS sector; AGENTS.md "Determinism", config as an explicit input.
   - Fix: pass `excluded=config["universe"]["excluded_sectors"]` through `run_checks`.

3. **EDGAR values are not trimmed to periods from 2006.**
   - The decision says EDGAR values are trimmed to periods from 2006 when cleaned. Neither `clean.py` nor `checks.py` does this. The check output shows "periods 2005-03-31 to 2026-10-29".
   - Rule broken: MD_202610_001 decision (2026-10-31, superseding "20 years").
   - Fix: trim in `market_data.clean` with the start date from `config/market_data.toml`, or have the secretary record that the decision was changed.

4. **Relative growth is calculated in four places.**
   - Where:
     - `skatkis/analysis.py` `_relative_growth` (line 635) and `plot_normalised`;
     - `market_data/market_analysis.py` `ranked_series` (line 265) and `sector_pe_and_price` (line 241);
     - `skatkis.financial_logic.annualised_relative_return`.
   - `skatkis/analysis.py` contradicts itself: its "Must not" (line 20) forbids computing returns, but its "Computes" (line 52) lists relative growth.
   - `market_analysis.py` also computes the lead-lag statistic (`pe_price_lead_lag`, line 484), although its own "Must not" says it computes no metrics.
   - Rule broken: AGENTS.md "Separation" (general finance lives in `skatkis.financial_logic` and is reused, never re-implemented) and "Ownership".
   - Fix: add growth of one unit and relative growth to `financial_logic`, call them from the plots, and move `pe_price_lead_lag` to `market_data.metrics`.

5. **Thresholds and windows are hard-coded constants instead of config.**
   - `rankings.py` lines 63–65: `MIN_YEARS_RETURN`, `MIN_YEARS_GROWTH`, `MIN_PE_DAYS`.
   - `checks.py` lines 66–71: `EPS_TOLERANCE`, `LARGE_DAILY_MOVE`, `SMALL_UNIVERSE`, `MAIN_COMPANIES`, `APPENDIX_COMPANIES`.
   - `clean.ANNUAL_DAYS`; the 365-day window in `dividend_yield`; `market_analysis.BENCHMARK = "^SP500TR"` (line 82), which duplicates `[benchmarks] market`.
   - The minimum return history is an open user question (S_202610_007 notes), so it in particular belongs in config.
   - Rule broken: AGENTS.md "Determinism"; developer.md "Coding conventions" (thresholds, windows and weights in `config/*.toml`).

6. **Unexplained `# type: ignore` and casts.**
   - `skatkis/analysis.py` line 632 has `# type: ignore[misc]` with no reason. Line 631's reason ("ListedColormap") is too terse.
   - Many `cast(...)` calls on `json.loads` results have no comment: `sources/edgar.py`, `fred.py`, `prices.py`, `constituents.py` and `skatkis/io/snapshots.py`.
   - Rule broken: warden.md checklist item 7; developer.md (no `# type: ignore` without a comment saying why).
   - Fix: one comment per module explaining that `json.loads` returns `Any`, or a small typed loader.

7. **Dead code.**
   - `market_data/metrics.py` `relative_performance` (line 196) and `sector_performers` (line 241) are no longer called anywhere; `rankings.py` replaced them.
   - They are still listed in the module's Public API. Remove them, or document why they stay.

8. **Files that should not be committed are staged.**
   - `.DS_Store` (modified, and tracked since before e990e6c) and `results/.DS_Store` (newly added) are staged.
   - `results/market_analysis/2026-10-31/report.log` is the log of the failed tectonic run.
   - Rule broken: writing.md "Human-readable layout" (no loose or generated files left lying around).
   - Fix: unstage them, `git rm --cached .DS_Store`, and add `.DS_Store` and the LaTeX byproducts (`results/**/*.log`) to `.gitignore`.

9. **Agent instructions are stale (secretary).**
   - `midas/instructions/AGENTS.md` line ~55 and the `midas/midas.md` decision log describe `results/market_analysis/<source>/<as_of>/` (per source). The code writes `results/market_analysis/<as_of>/<section>/`, as decided in S_202610_007 ("results by section").
   - `midas/instructions/quant.md` line 57 sends data notes to `midas/toolbox/value_investing/notes.md`. They now belong in `midas/toolbox/market_data/notes.md`.
   - The AGENTS.md branch table allows only the areas `skatkis`, `midas`, `value-investing` and `sandbox`. `market-data` and `hand-o-midas` are missing.
   - Rule broken: secretary.md "Decided rules go into the agent instructions"; warden.md checklist item 10.

**Notes**

10. **Sector scores cover different time windows.**
    - A sector's total and relative return is scored on its fund's whole history. XLC starts in 2018 and XLRE in 2015; the other funds start in 2006.
    - So the S_202610_007 result "Communication Services worst against the S&P 500 (−4.3%)" covers about 2018–2026, while the others cover 2006–2026.
    - Companies are likewise annualised from each one's own first date, with a minimum of one year (already open).
    - Suggest saying so in the report text, or scoring everything on a common window.

11. **Rankings use the whole period and current members.**
    - They are descriptive and in-sample, and the abstract states the survivorship bias. They must not be used to choose strategies.
    - Rule: AGENTS.md "Prohibited", judging a strategy on in-sample results only.

12. **`select_tags` uses later filings to choose a tag.**
    - It ranks tags by their latest `filed` date across all filings, so the choice of tag at date $t$ depends on filings after $t$. The value itself is still the first-reported one with its own `filed` date.
    - This is harmless for the analysis. MD_202610_002 should decide whether the choice must be made point-in-time.

13. **`filed` is described as the acceptance date.**
    - The `COMPANYFACTS_COLUMNS` docstring (`edgar.py` line 87) and the README say so. `companyfacts` `filed` is the filing date, which can be one business day after acceptance for filings made after hours. That is conservative, and P/E already uses EPS from the day after filing.
    - Correct the wording, or record that `filed` is a conservative stand-in for acceptance.

14. **The price adjustment basis is the real fetch day, not `as_of`.**
    - The `PRICE_COLUMNS` and `DIVIDEND_COLUMNS` docstrings (`prices.py` line 74) say prices are adjusted "as of the snapshot date". Yahoo actually adjusts as of the real fetch day.
    - With a past `--as-of`, a split between `as_of` and the fetch day is applied to prices but missing from the splits table, so `split_factor` and P/E would be wrong.
    - Correct the wording, or refuse an `--as-of` earlier than the fetch day for Yahoo.

15. **The "filed before period end 105" check line is mislabelled** (`checks.py` line 439). It counts forward-dated values such as debt maturities, not early filings. Rename it.

16. **FRED daily moves are not grouped by series.** `checks._fx` (line 506) takes `pct_change` across all series together. That is correct with only `DEXSFUS`, but it would mix series once a second one is added.

17. **The dividend-yield ×100 scaling is keyed on the column name** `"dividend_yield"` in three plot functions in `market_analysis.py`. Pass a scale argument instead.

18. **`skatkis/report.compile_pdf` (line ~211) copies `os.environ` into a subprocess, and tectonic downloads packages from the internet.**
    - Both are acceptable: the cache stays inside the repository in `data/cache/tectonic`.
    - This is a second network path outside `ApiClient`. The Guard should know about it.

19. **Some modules are missing "Computes" entries** (`.claude/rules/docstrings.md`):
    - `checks.py` (net income per share for the EPS check, filing lag);
    - `market_analysis.py` (`sector_pe_and_price` growth, sector totals in `fundamentals_sector_totals`).

20. **The report text breaks two writing rules** (`config/report_market_analysis.toml`, secretary).
    - $r$ means two different things (annualised total return and annualised relative return).
    - Annualisation is described in three sections instead of once.
    - Rule: writing.md "Formulas".

21. **Housekeeping.**
    - Neither change file says that baselines are unaffected (`regression_check` `unchanged`). Add one line to each (warden.md checklist item 2).
    - S_202610_007 still lists "Which plots first" under open decisions, although the decisions below it answer this. Its approach mentions a "distribution per period" plot that was not built. It has no Branch field.
    - The EPS cross-check result is not in `midas/toolbox/market_data/notes.md`: 387 of 6,093 annual values differ from net income divided by diluted shares by more than 5%. The progress note says "to look into".
    - VI_202610_002 says Yahoo dividends go "back to 1970". They are now requested from 1986.
    - The change-log title of S_202610_007 does not mention the LaTeX report.
    - The decision to import an index P/E from S&P DJI (which needs `openpyxl`) is not built. It is not a dependency in `pyproject.toml`. It is open, not a defect.

### Not checked
- The plots and the PDF themselves: binary files, not viewed. The PDF was never compiled.
- Whether the stored constituents table really holds the duplicate share classes (finding 1). I inferred it from the code and the 503-member count.
- Live API behaviour (retries, rate limits, the incremental EDGAR rule) and the content of the raw snapshots under `data/`.
- The individual commits 286c33f..b20a167. Only the combined diff was reviewed.
- Full pyright and pytest output: only summaries were given. Notebook state.
- Whether `.claude/settings.json`, hooks and permissions agree with the new network paths (that is Guard scope). The settings file is not in the diff.
- Who wrote `config/report_market_analysis.toml` (the secretary owns it).

### Documentation to update (after the report)
- **`Toolbox/market_data/README.md`:**
  - "Three free sources" (line 12): Wikipedia is now a fourth source.
  - History row (line 24): periods start in 2005, not 2006–2007.
  - "Until the historical S&P 500 membership is built, the companies ... are given on the command line" (line 158): `--universe sp500` now exists.
  - Add how the excluded sectors apply in `check`, the share-class handling, the fund start dates behind the sector scores, and the 2006 trim once built.
  - Define the symbols in the $g$ and $y_t$ formulas.
- **`docs/value_investing_bot_spec.md`:** the agreed deviation that places `DataProvider` in `skatkis.io` instead of the tool's `ingest/` (MD_202610_001 agreed approach); point to the shared `market_data` package.
- **Docstrings:**
  - `edgar.COMPANYFACTS_COLUMNS` (`filed`) and `prices.PRICE_COLUMNS` / `DIVIDEND_COLUMNS` (adjustment basis);
  - `skatkis/analysis.py` "Must not" against "Computes";
  - `market_analysis.py` "Owns" and "Must not";
  - `metrics.py` Public API and "Computes";
  - `financial_logic` Public API (`min_years` is missing from `annualised_relative_return`);
  - `__main__._parser` (it builds both commands, not only `update`).
- **Agent instructions (secretary):** AGENTS.md (results layout, branch areas), `midas/midas.md` decision log (results layout), `quant.md` (notes path), `secretary.md` (the report is no longer laid out as sub-sections per GICS sector; the main text shows best and worst with an appendix).

### Unit tests to write (last, per the lifecycle)
- **`skatkis/io`:** `ApiClient` retries, backoff, pacing and redaction with an injected transport and clock; snapshot reuse, the integrity error and `latest_as_of`; `write_table` / `content_hash` stability.
- **Sources:** `parse_companyfacts` (units, missing keys, unit mismatch), `latest_periodic_filing`, `parse_tickers`, `parse_chart` (dates with the UTC offset, events), `parse_fred` (JSON and CSV, missing markers), `parse_constituents` / `parse_changes` (rowspan dates, templates). All from recorded fixtures, with no network.
- **`clean`:** `select_tags` (most recent tag wins, ties go to priority, restatements kept); `annual_first_reported` (350 and 380 day boundaries, earliest 10-K kept).
- **`metrics`:**
  - `split_factor` (no splits, a split on the day itself);
  - `price_earnings`: EPS filed on day $d$ is used from $d + 1$ (a point-in-time case), EPS ≤ 0 gives NaN, no leakage across symbols;
  - `dividend_yield` (window edges, no dividends gives 0).
- **`financial_logic`:** `annualised_growth` / `annualised_relative_return` with hand-computed values, the `min_years` boundary and non-positive levels.
- **`rankings`:** best and worst direction for P/E; `MIN_PE_DAYS`; one score per company with two share classes.
- **`gics`:** exclusion and the Yahoo symbol convention.
- **`update`:** the incremental EDGAR rule (a filing on the snapshot day triggers a refetch) and failures recorded per query.
- **`skatkis/report`:** `escape` (maths, `[[label]]`, special characters) and deterministic `write_latex` output.

### Draft outcome: MD_202610_001
**Outcome (draft).** The shared data package `Toolbox/market_data/` was built on new `skatkis.io` machinery.
- **Machinery:** `ApiClient`, the `DataProvider` abstract base class, gzipped raw snapshots with content-hash manifests, Parquet tables and the config loader.
- **Sources:** SEC EDGAR `companyfacts` with 26 canonical fields in `tag_map`, and `submissions` for incremental updates; Yahoo's chart endpoint (daily prices from 2006, dividends and splits through monthly bars from 1986); FRED `DEXSFUS` from 2006; Wikipedia constituents and index changes with GICS sector and CIK.
- **Pipeline:** one terminal command, `python -m Toolbox.market_data update --as-of <date> [--universe sp500]`, with `--as-of` always explicit. The `notebooks/fetch_*.py` scripts were removed.
- **Cleaning and metrics:** `select_tags` (the company's most recently used tag) and `annual_first_reported`; shared metrics `price_earnings` (latest first-reported annual diluted EPS, put on the price's split basis, used from the day after filing) and `dividend_yield`; the `gics_map` with SPDR sector funds as sector benchmarks.
- **Alternatives considered:**
  - `yfinance`: Yahoo's chart endpoint through `ApiClient` gives the same data with raw JSON snapshots and no extra dependency.
  - `requests`: it has no type stubs for strict pyright.
  - S&P price-only sector indices: they are not total return.
  - Keeping sources in `value_investing`: the package was moved to `market_data` so `hand_o_midas` can share it and `skatkis` stays free of domain knowledge.
- **First full pull, 2026-10-31:**
  - EDGAR: 395 of 397 companies; two lost on a `dividends_per_share` unit mismatch (open note); 1,597,749 values after tag selection.
  - Yahoo: 409 symbols, 1,932,820 price rows, 34,765 dividends.
  - Constituents: 503 members and 414 changes.
  - FRED: 5,214 rates.
- **Known limits:** survivorship bias (current members only, no delisted prices); EDGAR history starts with filings in 2009.
- **Open items carried forward:**
  - a minimum history for the rankings;
  - unit-mismatch handling;
  - large daily moves still to inspect;
  - the EPS cross-check (387 of 6,093 values off by more than 5%);
  - the index P/E from S&P DJI;
  - the point-in-time panel (MD_202610_002).
- **Warden:** run 2026-10-31 on b20a167, with the tree dirty; `regression_check` `unchanged`.

### Draft outcome: S_202610_007
**Outcome (draft).**
- **Generic plots:** `skatkis.analysis` holds the layouts: series, coverage heat map, normalised and relative growth, scatter, two axes, spread and spread pair, and lag profile with `lagged_correlation`. They know no market, allow at most 30 lines per plot and keep legends outside the plot.
- **Named plots:** `market_data.market_analysis` holds the plots per source and analysis.
- **Rankings:** `market_data.rankings` gives company and sector scores per analysis: revenue growth, total return, return relative to the fund, P/E (lowest best) and dividend yield. They use `skatkis.financial_logic.annualised_growth` and `annualised_relative_return`.
- **Check command:** `python -m Toolbox.market_data check` writes plots to `results/market_analysis/<as_of>/<section>/`, tracked in git.
- **Report:** `skatkis.report` writes it as a LaTeX article, from coarse to fine: introduction with the GICS table, overview, then per analysis the summary, best and worst sector and their five best and five worst companies, data checks, and an appendix with every sector. Its text is in `config/report_market_analysis.toml`, owned by the secretary.
- **Alternatives considered:**
  - Members of a sector as individual lines: unreadable at full scale (150 lines for Information Technology).
  - Per-sector sub-sections in the main text: replaced by best and worst sectors with an appendix.
  - Seaborn styling: deferred.
- **First results, 2026-10-31 (76 plots, 31 in the main text and 45 in the appendix):**
  - Revenue growth: best Communication Services (11.9% a year), worst Utilities (2.3%).
  - Total return: best Information Technology (16.0%), worst Energy (7.9%).
  - Return against the S&P 500: best Information Technology (4.4%), worst Communication Services (−4.3%).
  - Lowest P/E: Energy.
  - The P/E–price lead-lag shows only the mechanical lag 0.
- **Not done in the sandbox:** the PDF compile (tectonic could not download packages); the user runs it once in a terminal.
- **Open:** the minimum history for return rankings (young listings lead the best lists on one year of data).
- **Warden:** run 2026-10-31 on b20a167, with the tree dirty; `regression_check` `unchanged`.

## Kickoff prompt

Pick up MD_202610_001. Read spec §1, §3, §4, §6 and the tool's notes, confirm the
decisions above are made, then propose the source modules and canonical fields before
building the EDGAR client first.

## Done when

- Fundamentals for a small fixture universe are pulled, mapped to canonical names,
  keyed to filing date and cached with a manifest; tests pass without network.
