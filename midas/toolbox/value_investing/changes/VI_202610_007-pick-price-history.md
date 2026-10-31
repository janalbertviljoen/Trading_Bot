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

## Notes

- 2026-10-31: built end to end. `skatkis.analysis` gained two generic layouts
  (`plot_heatmap`, `plot_small_multiples`); `value_investing.plots` has `pick_prices`
  and `trade_amounts`; the report has a section "Price history of the picks" before
  the next picks; `pick_scenario` and `pick_years` in `config/rules.toml`. Figures in
  `results/value_investing/2026-10-31/report/` (`pick_prices.png`,
  `trade_amounts.png`). Black, ruff and pyright clean on the changed files; tests not
  run yet. First look: the 2023 investor sold only two small trims of Intel, and
  bought Skyworks six times as it fell (notes, "The wallet keeps its losers").

## Kickoff prompt

Pick up VI_202610_007. Read the agreed approach and continue the build on its branch.

## Done when

- The value report shows the five-year price of each company the 2023 investor traded
  and of each next pick, with its buys and sales marked, and a heat map of the USD
  bought and sold per company and decision date.
