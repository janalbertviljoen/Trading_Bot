# S_202610_007: Market analysis plots

| Field | Value |
|---|---|
| Status | in progress |
| Priority | high |
| Kind | new capability |
| Depends on | MD_202610_001 (cleaned tables to plot) |
| Raised | 2026-10-31 |

## Description

Before any feature engineering, the user wants to look at the data: callable plots that
show how a metric such as the P/E varies over time, across companies and against the
S&P 500, in the style of the existing BTC plots (`Toolbox/SandBox/btc_vwap/plot_analysis.py`:
heat maps, wallet outcomes, normalised series).

## Approach

- `skatkis.analysis`: plot functions that take a DataFrame and column names
  (date, entity, value) and know nothing about any metric: a metric over time per
  entity, its cross-sectional spread over time (median and percentile bands), a
  coverage heat map (entity × date, showing gaps), a distribution per period, and a
  series normalised to a benchmark. "P/E over time" is then the generic time-series
  plot called with the P/E column, so `skatkis` stays free of domain knowledge.
- Plots return the matplotlib figure, so notebooks display them and scripts save them.
- The first use is the visual data checks of MD_202610_001 (coverage, gaps, outliers per
  source), then the shared metrics once they exist.

## Open decisions (user)

- Which plots first, and whether any of the existing BTC plots should move here in a
  general form.

## Decisions

- 2026-10-31 (user): the report is a concise summary of the last five years; the long
  history goes to the appendix (back-testing and optimisation). Structure: introduction
  (GICS sectors, a ticker appendix by sector and ticker); per analysis (fundamentals,
  total return, relative return, P/E, P/E and value, dividend yield) a sector summary, a
  3 x 3 grid of every sector's five best and five worst companies, and an overview of
  the two best and the two worst sectors with their five best or worst companies;
  revenue and net income side by side in USD trillions; total return in three rows (5,
  10 years, full history); P/E before dividends; the P/E lead-lag and the data checks in
  the appendix; a P/E and value section showing the lowest-P/E and best-value companies
  side by side, each plot with both measures, to find companies in both lists. Panels
  are separate images joined by LaTeX (the formatting tool).

- 2026-10-31 (user): the detailed per-sector plots go to an appendix; the main text shows
  only the best and worst sector of each analysis, by that analysis's measure, then the
  five best against the five worst companies of each of the two, with references to the
  appendix. Legends on every plot; a table of contents and navigation between sections.
  The GICS sectors are summarised in an introduction before the first plot. The P/E
  section shows "P/E against value" (sectors by P/E) and "value against P/E" (sectors by
  return), with each sector's median P/E and median price growth and their bands. P/E
  comes after relative return.

- 2026-10-31 (user): the report goes from coarse to fine: section 1 the sectors against
  the S&P 500, then one section per analysis (fundamentals, total and relative return,
  dividend yield, P/E) with its sector summary first and a sub-section per GICS sector
  showing only the top and bottom performers (n about 10). Captions describe the
  specific plot; section text gives the currency and the calculations with minimal
  maths. At most 30 lines per plot (heat maps for more); colour marks the company, line
  style the metric. Trading days sorted youngest first, to show age.

- 2026-10-31 (user): sections for fundamentals (revenue and net income, the overall
  view), total and relative return, and dividends shown as paid out against price (the
  dividend yield), each with a sub-section per GICS sector; amounts in USD millions; a
  PDF report with the plots and short academic descriptions, written as LaTeX and
  compiled to PDF; the secretary owns the report's writing and formatting.

- 2026-10-31 (user): per sector, members relative to the sector benchmark (solid) and
  to the S&P 500 (dashed, same colour), or side by side with a shared y-axis; plus all
  sector benchmarks against the S&P 500. Built as one axis with solid and dashed lines.

- 2026-10-31 (user): the current plots are fine; a seaborn style may become the standard
  later.
- 2026-10-31 (user): with the full universe, plots cannot show every company, nor all in
  one figure. Companies are grouped by the standard classification (GICS sector, from
  the constituents source) and compared only within their group: per sector, the median
  of a metric with a percentile band over time; sector medians against each other;
  single companies only within a sector. Value and growth style is a separate dimension,
  to be added later.

- 2026-10-31 (user): generic plots in `skatkis.analysis`, with no market information;
  the named market plots per source in `market_data.market_analysis`, also used by the
  models. P/E is compared with price on a second axis, with the lead or lag between
  them.

- 2026-10-31 (user): a `skatkis.analysis` module of callable plots for the data
  and metrics over time, similar to the existing plots.

## Notes

- 2026-10-31: built: `FigureGroup` (lettered panels in columns) and long tables in
  `skatkis.report`; panel mode and `plot_pair_series` in `skatkis.analysis`;
  `rankings.window` (scores over the last five years, only companies with the whole
  window); `CheckSettings` and `[analysis]` in `config/market_data.toml`; one GICS row
  per company for fundamentals (warden finding 1) and the excluded sectors applied in
  `check` (finding 2); the report text rewritten (secretary). 129 plots. First P/E and
  value result: no company is in both the five lowest-P/E and the five best-value lists
  (nor in both worst lists). The PDF waits for the local settings to load (restart).

- 2026-10-31: built: `skatkis.financial_logic.annualised_growth`, `market_data.rankings`
  (company and sector scores per analysis, lowest P/E best), `plot_spread_pair` (two
  medians with bands on two axes, log P/E axis so one-year spikes do not flatten the
  median), legends outside the plot with a line-style key, `skatkis.report` with a table
  of contents, `hyperref`, labels, `[[label]]` references, tables and an appendix
  (`Document`), the introduction table from `config/report_market_analysis.toml`
  (secretary). 76 plots: 31 in the main text, 45 in the appendix. First results: best
  and worst sector by revenue growth Communication Services (11.9% a year) and
  Utilities (2.3%); by total return Information Technology (16.0%) and Energy (7.9%); by
  return against the S&P 500 Information Technology (4.4%) and Communication Services
  (-4.3%); lowest P/E Energy. Still open: the minimum history for return rankings (one
  year lets young listings such as SanDisk and Palantir lead).

- 2026-10-31: built as decided: `skatkis.financial_logic.annualised_relative_return`,
  `market_data.metrics.relative_performance` and `sector_performers` (ranked once by
  annualised return relative to the sector fund, at least one year of history, so the
  same companies appear in every section), the `companies_*` plots, a 30-line limit and a
  20-colour palette in `skatkis.analysis`, line styles per metric, the youngest-first
  trading-days heat map, and caption fields per plot. 103 plots. Open: with a one-year
  minimum, young listings (SanDisk, Palantir) lead the best lists on short strong runs;
  a longer minimum (3 or 5 years) would rank on more history.

- 2026-10-31: first full-universe run (395 companies with filings, 409 price symbols): 58
  plots. Readable: sector funds against the S&P 500 (technology about 2.5 times ahead
  since 2006, energy and financials far behind), median dividend yield and P/E per
  sector, sector totals. Not readable: members per sector as lines (75 members in
  information technology give 150 lines); proposed instead: the members' median against
  the fund (solid) with a percentile band and against the market (dashed). The summary
  of 11 funds repeats a colour (10-colour cycle). Median P/E spikes where few members
  have positive earnings (energy 2021, communication services 2014). The report did not
  compile in the sandbox: tectonic cannot download its packages through the sandbox
  proxy (the same TLS issue as pip); the user runs the check command once in a terminal,
  after which the packages are cached in `data/cache/tectonic/`.

- 2026-10-31: results by section (user): `results/market_analysis/<as_of>/<section>/`
  with summary plots and one plot per GICS sector (`fundamentals`, `total_return`,
  `relative_return`, `dividend_yield`, `pe`, `data_checks`, `companies` for small
  universes). New: trailing dividend yield (`market_data.metrics.dividend_yield`), sector
  totals of revenue and net income in USD millions (the unfinished fiscal year left out),
  total return per sector, thousands separators on large axes, growth from each series'
  own first date. The report is a LaTeX article (`skatkis.report.write_latex`,
  `market_data.report.build_report`) with its text in `config/report_market_analysis.toml`
  (secretary); compiled to PDF when an engine is installed (none yet: `brew install
  tectonic`, to be run by the user).

- 2026-10-31: grouping built. Generic: `plot_relative` with a second benchmark (solid
  against the first, dashed in the same colour against the second; each series from its
  own first date, so late listings and late funds still show), `plot_spread` (median
  and percentile band) and named `thresholds` lines. Named: `sector_relative` (members
  against their sector fund and the S&P 500), `sector_benchmarks` (sector funds against
  the S&P 500), `pe_sector_medians`, `pe_sector_spread`. Per-company plots only for a
  universe of up to 10 companies; sector plots under `results/market_analysis/sectors/`.

- 2026-10-31 (user feedback): per-company coverage plots always showed 1 (one first-reported
  value per year), so coverage is one plot of the number of companies with a value per
  field and year; marker plots now draw lines through their points; new
  `plot_relative` (generic) and `yahoo_relative_return` (company total return divided by
  the S&P 500's). The coverage plot exposed three tag gaps, closed with fallbacks
  (`CostOfGoodsSold`, `PaymentsToAcquireProductiveAssets`,
  `FiniteLivedIntangibleAssetsNet`, `OtherIntangibleAssetsNet`); the gaps left are real
  (Coca-Cola reports no total liabilities, Apple no goodwill from 2018 and no dividends or
  debt before 2013). P/E against the S&P 500 waits for an index P/E source.

- 2026-10-31: split in two (user): generic layouts and statistics in `skatkis.analysis`
  (with `plot_dual_axis`,
  `plot_lag_profile`, `lagged_correlation`), and the named market plots with their
  titles and formatting per source in `market_data/market_analysis.py`, called by the
  checks and later by the modelling tools. P/E against price on two axes per company and
  a lead-lag profile of monthly P/E and price changes (12 months either side). For
  Apple, Coca-Cola and Microsoft only lag 0 stands out (0.22–0.59, mechanical: P/E is
  price over EPS); every other lag is within about 0.15, the range of chance for about
  200 months. A real test needs the trailing-twelve-month P/E, the full universe and
  P/E level against forward returns.

- 2026-10-31: first set built (user asked to look at the data):
  `skatkis/analysis.py` with `plot_series`, `plot_coverage`, `plot_normalised`
  and `plot_scatter`, each returning a `Figure` without pyplot state; used by
  `notebooks/check_data.py` for 13 data-check plots. The spread plot (percentile bands
  over time) follows with the shared metrics. A file-level pyright directive covers
  matplotlib's untyped `**kwargs`.
- 2026-10-31: the branch `feature/value-investing/vi-202610-001-sp500-data-sources` is
  merged into `main` with this change still in progress (user). Left for a follow-up
  branch before closing: the warden's should-fix findings 3 (trim EDGAR values to 2006),
  4 (relative return calculated in four places), 5 (constants to config), 6 (comments on
  type ignores and casts), 7 (dead code in `metrics.py`) and 9 (stale instructions); the
  unit tests for the new code.

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

## Done when

- The plots run on the cleaned tables of MD_202610_001 and the user has judged them.
