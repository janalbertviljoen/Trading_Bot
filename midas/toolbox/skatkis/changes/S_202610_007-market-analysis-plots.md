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

- 2026-10-31 (user): generic plots in `skatkis.analysis`, with no market information;
  the named market plots per source in `market_data.market_analysis`, also used by the
  models. P/E is compared with price on a second axis, with the lead or lag between
  them.

- 2026-10-31 (user): a `skatkis.analysis` module of callable plots for the data
  and metrics over time, similar to the existing plots.

## Notes

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

## Done when

- The plots run on the cleaned tables of MD_202610_001 and the user has judged them.
