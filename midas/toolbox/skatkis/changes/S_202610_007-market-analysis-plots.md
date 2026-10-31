# S_202610_007: Market analysis plots

| Field | Value |
|---|---|
| Status | in discussion |
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

- `skatkis.market_analysis`: plot functions that take a DataFrame and column names
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

- 2026-10-31 (user): a `skatkis.market_analysis` module of callable plots for the data
  and metrics over time, similar to the existing plots.

## Done when

- The plots run on the cleaned tables of MD_202610_001 and the user has judged them.
