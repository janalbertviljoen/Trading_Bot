# VI_202610_007: Price history of the picks in the value report

| Field | Value |
|---|---|
| Status | in progress |
| Priority | high |
| Kind | new capability |
| Depends on | VI_202610_006 |
| Branch | `feature/value-investing/vi-202610-007-pick-price-history` |
| Raised | 2026-10-31 |

## Description

The value report names the next picks (2026-10-30: First Solar, Cognizant, Snap-on,
Skyworks, ResMed) with their score and weight, but not how they have done. The user
looked up the last five years of the picks' prices and would not have bought them
(2026-10-31). The report should show each pick's price history so that a reader
sees how the market has treated a pick before buying it.

The report change only; whether weak past returns should count against a pick is a
methodology question, kept in [notes.md](../notes.md) ("Picks with weak past returns:
value traps").

## Agreed approach

- **Prices.** Per company, its split-adjusted price (`close`, as a price chart shows
  it, dividends not included) over the last five years to the valuation date; one
  small panel per company. Prices only for now, no value line (user, 2026-10-31).
- **Companies.** Every company the investor who started three years ago (the 2023
  start year of the report) bought or sold, and the next picks.
- **Trades on the price.** The wallet with sells' trades of that investor as markers
  on the price line: buys in green, sales in red.
- **How much.** A heat map of the same trades, company by decision date: green for the
  USD bought, red for the USD sold.
- **Where.** A section of the value report before the next picks; plots in
  `value_investing.plots` on layouts of `skatkis.analysis`; the text in
  `config/report_value_investing.toml`; the investor and the window in
  `config/rules.toml` `[report]`.

## Decisions

- 2026-10-31 (user): prices only for now; trades as a scatter on the price, buys and
  sales; a green and red heat map of how much was bought and sold; the past picks of
  the investor who started three years ago.
- 2026-10-31 (user): the price is the stock's own price at each date, not the wallet's
  accumulated value.
- 2026-10-31 (user): the panels run from the best to the worst price change.
- 2026-11-01 (user): the next picks get the same price tiles in the next-picks section:
  five tiles side by side, the same size, the earlier buys of the company in green and
  the potential buy in orange.
- 2026-11-01 (user): if the price history runs over a page, it shows the five best and
  the five worst performers only.
- 2026-11-01 (developer, from the user's "one page"): a page is `pick_page = 20` panels
  and the ends are `pick_ends = 5`; `pick_page` also caps the wallet-contents figure at
  the 20 holdings worth most today (the S&P 400 wallet held more companies than a plot
  may draw). Both in `config/rules.toml` `[report]`.

## Notes

- 2026-10-31: built end to end. `skatkis.analysis` gained two generic layouts
  (`plot_heatmap`, `plot_small_multiples`); `value_investing.plots` has `pick_prices`
  and `trade_amounts`; the report has a section "Price history of the picks" before
  the next picks; `pick_scenario` and `pick_years` in `config/rules.toml`. Figures in
  `results/value_investing/sp500/2026-10-31/report/` (moved from
  `results/value_investing/2026-10-31/report/` with MD_202610_004) (`pick_prices.png`,
  `trade_amounts.png`). Black, ruff and pyright clean on the changed files; tests not
  run yet. First look: the 2023 investor sold only two small trims of Intel, and
  bought Skyworks six times as it fell (notes, "The wallet keeps its losers").

## Warden report

2026-11-01, follow-up run, on `fb63e81` (`feature/value-investing/vi-202610-007-pick-price-history`)
plus the working tree; covers VI_202610_007, MD_202610_004, the fixes after the first
report (VI_202610_006), the methodology slices of MD_202610_002 (tag choice) and
VI_202610_002 (G4 phase-in), and the records S_202610_008 and VI_202610_008. Recorded
unchanged:

I found 1 blocking finding: the S&P 400 and S&P 600 reports state a look-ahead claim that is false for those indices, and the change file requires it to be stated. The code is in good shape. The blocker is in the report text. The rest is records, labelling and a few stale docstrings.

### 1. Header

| Field | Value |
|---|---|
| Changes | VI_202610_007, MD_202610_004, the fixes to the first report (S2–S6, S8–S12, N2–N4, N7), MD_202610_002 (point-in-time tag choice), VI_202610_002 (G4 phase-in), records S_202610_008 and VI_202610_008 |
| Date | 2026-11-01 |
| Commit reviewed | `fb63e81` on `feature/value-investing/vi-202610-007-pick-price-history` plus the working tree. The tree is dirty (untracked `config/indices.toml`, `docs/branding/`, results for sp400 and sp600, three sp500 report PNGs). |
| Base branch | `main` |
| Run | Follow-up to the first run of 2026-10-31 (recorded in VI_202610_006) |

### 2. Findings

#### Blocking

**B1. The S&P 400 and 600 reports misstate their universe, their benchmark and their survivorship gap.**
- Where: `config/report_value_investing.toml`, `[lookahead]` text ("Three sources of forward information…") and the `abstract`; rendered at `results/value_investing/sp600/2026-10-31/report/report.tex:21` and `:431`.
- Join dates: the text says "companies join the universe only from the date they entered the index". For sp400 and sp600 that is false: `market_data.gics.fill_date_added` gives a member with no listed addition the first date of the changes table, so some members count before they joined. The MD_202610_004 decision (user confirmed) reads "a look-ahead to state in their reports". It is not stated.
- Benchmark: the abstract says "the same money in the {index} total-return index". For sp400 and sp600 the benchmark is the fund IJH or IJR, net of a 0.05% yearly fee.
- S&P 600 start and survivorship: the abstract says "since 2020" but describes a 2016 investor; "487 companies have left the index since then against 491 since 2016" is cut at the start of the record (December 2019) and is not a measure of the gap.
- Rules broken: `quant.md` rule 6; `AGENTS.md`, Point-in-time; the recorded decision of MD_202610_004.
- Fix: per-index wording in `config/indices.toml` filled into the text (benchmark description; membership note with the year the record starts); say that survivorship counts are bounded by the start of the record; regenerate the three reports.

#### Should fix

**S1. The two methodology changes were measured together, and the expected effect was stated for the wrong score.** "Expected small" (tag choice) and "can only rise" (G4) were stated; 6,035 company-dates changed, 4,892 of them G4, about 1,143 others; picks changed on 4 of 21 decisions. "Can only rise" holds for `score` only; the picks use `score_binned_weighted`, a sum of points, where a young company loses its G4 points and G4's weight and every other company's G4 bucket shift. Rule: `developer.md`, "Planning a change to financial logic". Fix: rerun with the tag fix alone, or record the direction of change in `score_binned_weighted` and the cause of the other changes, expected against actual per change.

**S2. The N2 fix changes the wallet's trade accounting and was filed with the refactors.** `holding._value_path`: a lot now carries its last price forward instead of resetting to cost. No effect on today's data; it will move results once delisted or historical members are added. Rule: `AGENTS.md`, "Methodology vs refactor". Fix: record it in VI_202610_006 as a methodology change with no effect on today's baselines, with the user's approval.

**S3. Housekeeping: the records lag the work.**
- VI_202610_006: the fixes of 2026-11-01 are not recorded; "Fixes after the report" still lists S3–S9, S11 and S12 as open.
- The two deliberate exceptions (`PRICE_TOLERANCE_DAYS` kept in code; change IDs in `config/*.toml` comments) are recorded nowhere in `midas/`.
- `notes.md`, "Follow-ups pencilled in", item 5 still lists the should-fix items as open.
- VI_202610_007: its note gives `results/value_investing/2026-10-31/report/` (now `sp500/2026-10-31/`); `pick_page = 20`, its reuse to cap the wallet-contents figure, `pick_ends`, `verdict_band`, `min_companies`, `year_tolerance_days` and `stale_days` are not recorded as decisions.
- The tag-choice and G4 decisions are dated 2026-10-31; the approval was 2026-11-01. MD_202610_002 cites "warden report N1"; the leak was in "What the developer said".
- `quant.md` rule 4 still says "the benchmark is `^SP500TR`".

**S4. The new report text hard-codes values the config sets:** "2023", "five years", "five best and five worst" in `[history]`, `[next]` and the figure captions, where `pick_scenario`, `pick_years` and `pick_ends` are in config. Fix: fill them as placeholders.

**S5. Docstrings and comments that no longer match the code:** `report.ReportInputs.compare` ("S&P 500 first"); `report.ReportInputs.peers` (garbled sentence order); `holding` module docstring ("each comparison (`^SP500TR`, `BRK-B`)"); `holding.rule_profile` ("ahead or behind the S&P 500"); `plots.wallet_contents` `compare`; the `pick_page` comment in `config/rules.toml` (it also caps `contents.png`); `diagnostics` module docstring ("less `^SP500TR`'s").

#### Notes

- N1. Two owners of forward excess remain: `inputs.window_returns` (fixed months) and `diagnostics.forward_excess` (to the next rebalance, through `skatkis.forward_excess_return`). The concepts differ, which is defensible; say so in `diagnostics`. S3 of the first run is otherwise resolved.
- N2. `EdgarTickers.fetch` ignores `as_of`: a run with a past `--as-of` stores today's list under the past date. Documented; acceptable while updates run on the day.
- N3. The sp600 truncation check on 2018-12-21 is vacuous (members start in December 2019).
- N4. The point-in-time tag choice matches its decision; both strict xfail marks were removed with the fix, as their reasons required.
- N5. G4 phase-in matches its decision; `listings` is not cut in the truncation check, which is safe.
- N6. `index_universe` is generic and would fit `market_data` if a second tool needs it. No change now.
- N7. No API keys, e-mail addresses or vendor data in code, config or records; `SEC_USER_AGENT` is read only at the CLI boundary.
- N8. The deliberate items are acceptable once recorded (S3).
- N9. The record references are gone from the code; the verdict-band `noqa` is gone and the bands are in config.

### 3. Not checked

- `regression_check` was not run; the refactor evidence is a one-line summary of a before/after CSV diff.
- The numbers of the methodology effect and the MD_202610_004 return table (no commands run).
- The results files (197 files, a count only), the PDFs (layout, page count, overfull boxes).
- Whether the look-ahead test ran (15 passed, no skips implies it did).
- The new code paths have no tests (`_header`, `_cik`, the `parse_changes` continuation rule, `ticker_ciks` matching): expected at this stage.
- The market analysis (`check`) with `[analysis] index` was not run.

### 4. Documentation to update after the report

- Spec / tool README: section 4 of the first report; universes per index with the join-date approximation and its look-ahead; IJH and IJR as benchmarks (funds, net of fees); G4 phase-in; the point-in-time tag choice (spec §3.3, §3.5); `year_tolerance_days`, `stale_days`, `min_companies`.
- `market_data` README: `fill_date_added`; `EdgarTickers`, `parse_ticker_ciks`; `update_constituents(index, pages, ticker_ciks)`; the `listings` table; `--index` and the per-index folders; the new `annual_as_of(facts, fields, …)` signature.
- skatkis reference: `plot_heatmap`, `plot_small_multiples`, `rank_correlation`.
- The docstrings in S5; `quant.md` rule 4.
- Unit tests, in addition to the first report's list: `parse_constituents` with and without the CIK and "Date added" columns and a missing required header; `parse_changes` continuation rows; `fill_date_added`; the CIK fill from the ticker list (`BRK.B` to `BRK-B`); `_listings`; `members_on_date`; `index_universe`; `window_returns`; `evaluation.short_path`; `book_weights` when empty; `diagnostics.survivorship`; G4 at the window-start boundary and with a missing listing; `annual_as_of` with the tag switching between two dates; the `plots.pick_prices` page cut; `_value_path` carrying the last price forward.

### 5. Draft outcome

- VI_202610_007: the "Price history of the picks" section (split-adjusted price over `pick_years` of every company the `pick_scenario` investor traded and of the next picks; best to worst; buys green, sales red, marker area by USD; over `pick_page` companies only the `pick_ends` best and worst), a green-red heat map of USD traded, and next-pick tiles (earlier buys green, potential buy orange). Generic layouts `plot_small_multiples`, `plot_heatmap`. The first look (Intel trims only; Skyworks bought six times as it fell) is recorded as hypotheses, not acted on.
- MD_202610_004: route A built (indices in config; CIKs from SEC's ticker list; join dates from the changes table, a stated look-ahead for sp400 and sp600). Data fetched 2026-11-01. S&P 400: the edge shrinks to about 1–4 points a year against IJH; S&P 600: negative against IJR, record from December 2019.
- Fixes after the first report (VI_202610_006): one owner per concept (`inputs.window_returns`, `skatkis.analysis.rank_correlation`, `evaluation.short_path`, `diagnostics.survivorship`, `inputs.members_on_date`); thresholds into config; docstrings corrected and record references removed; N2 (last price carried); the point-in-time tag choice and G4 phase-in, approved by the user, together moving 6,035 of 20,290 company-dates, picks on 4 of 21 decisions, wallets under 2%; effect per change on `score_binned_weighted` to record (S1).

## Kickoff prompt

Pick up VI_202610_007. Read the agreed approach and continue the build on its branch.

## Done when

- The value report shows the five-year price of each company the 2023 investor traded
  and of each next pick, with its buys and sales marked, and a heat map of the USD
  bought and sold per company and decision date.
