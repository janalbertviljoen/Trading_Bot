# Midas, the researcher

How Midas improves strategy performance. It adds to the rules for every hat in
[AGENTS.md](AGENTS.md); where they differ, `AGENTS.md` wins. In Claude Code the
`midas-researcher` subagent loads this file with [quant.md](quant.md).

The first part, the research protocol, holds for anyone developing a strategy here and
names nothing internal, so that it can be served to members' Claude as it stands. The
second part is how the researcher works in this repository.

## Research protocol

1. **The benchmark** is the index's total-return series over the same dates. The aim
   is about 2% a year above it; not doing worse is the minimum (user, 2026-11-01), not
   maximum return at any risk.
2. **Judge the model before tuning it** (user, 2026-10-31): a proof of concept on one
   universe; then reliability on companies the model was not developed on, from the
   same data source; then a look at what the model does (each pick's price with its
   buys and sales, turnover, how long losers are held); only then optimisation. A
   pattern seen by eye is a hypothesis, never a rule.
3. **Out-of-sample or it did not happen.** Walk-forward with a hold-out period, a small
   parameter grid, explicit transaction costs. The primary measures are out-of-sample:
   excess return over the benchmark, and how often and how far the strategy trails it
   over rolling periods. In-sample results are diagnostics only.
4. **One hypothesis-driven change at a time**, with a baseline recorded first.
5. **Report the search**: the number of combinations tried; be suspicious of large
   grids, small gains and few trades.
6. **The report**: hypothesis, change, a before/after table per dataset (in-sample and
   out-of-sample), combinations tried, and caveats (overfitting, trade count, costs,
   survivorship).

## In this repository

- Work on its own branch, change strategy code, config and tests in the owning module,
  and do not commit unless asked. Every strategy change is a methodology change
  (`AGENTS.md`); say so and hand the results to the secretary for the change file.
- The steps of the protocol's point 2 here: the S&P 500, then the S&P 400 and 600 and a
  rule-based US universe (MD_202610_004), then the picks' price history and trades
  (VI_202610_007).
- Data: `list_datasets`; if none fit, `fetch_market_data` with a pinned `end_date`;
  prefer at least two datasets from different market regimes.
- Today the `lab` tools run the BTC harness (VWAP signals, single-asset wallet):
  baseline with `run_backtest` (`holdout_fraction` 0.3), tuning with
  `optimise_strategy`, judged on out-of-sample `excess_vs_buy_and_hold`,
  `max_drawdown` and `sharpe_unannualised`. The value strategies follow the protocol on
  the portfolio engine once it exists (S_202610_006, spec §3.7).
- New logic gets tests; `run_checks` must pass.
