# S_202610_006: Multi-asset portfolio backtest engine

| Field | Value |
|---|---|
| Status | in discussion |
| Priority | high |
| Kind | new capability |
| Depends on | S_202610_001; spec §9.2 (bottom ten) and §9.4 (wallet contract) |
| Raised | 2026-10-31 |

## Description

The wallet (`trade_wallet.update_wallet`) simulates one asset, row by row, buying or
selling a fraction of cash or volume on each signal, with no costs. The value-investing
strategies (spec §5) need a portfolio: ten names held at once, rebalanced quarterly,
held until they drop out of the top N, or sold after a 50% gain or two years, with
explicit transaction costs.

## Objective (user, 2026-10-31)

The user holds mostly the S&P 500 and wants to put about 10% of new investments into a
value-investing sleeve. The sleeve only needs to beat the S&P 500 by a margin, so the
benchmark is the S&P 500 itself (total return, e.g. SPY with dividends) and the primary
measure is out-of-sample excess return over it, with drawdown and turnover alongside.
The wallet tracks many strategy combinations side by side so they can be compared and
optimised on that measure.

## Approach (to discuss)

A new engine in `Toolbox/skatkis/backtest/`, alongside the existing single-asset wallet
(which stays, unchanged, for the sandbox):

- input contract: a ticker × date panel of target weights (or orders) produced by a
  strategy, plus a price panel;
- pure, vectorised accounting: positions, cash, turnover, costs, portfolio value, and
  the S&P 500 total-return benchmark over the same dates and cash flows;
- quarterly trades only (rebalance dates from config), replacing the VWAP deviation
  signals of the BTC wallet;
- many strategy combinations run side by side, each summarised by excess return over
  the S&P 500 in-sample and out-of-sample (walk-forward, spec §3.7);
- strategies as a registry in the tool (`Toolbox/value_investing/strategies.py`), each a
  function from the quarterly ranking to target weights;
- performance metrics (return, drawdown, Sharpe, turnover) in `skatkis.financial_logic`,
  shared with the existing `summarise`.

## Proposed by the user (2026-10-31)

- **Relative wealth against the index.** Results are normalised to what the same
  deposits would be worth in the S&P 500 total-return index, for example
  $W_t = V_t / V^{\text{S\&P}}_t$, where $V_t$ is the portfolio value on day $t$ and
  $V^{\text{S\&P}}_t$ the value of the same deposits, on the same dates, invested in
  `^SP500TR`. Above 1 the strategy beat an index fund. Both value investing and
  `hand_o_midas` strategies are compared on it.

## Open decisions (user)

- Spec §9.4: wallet input contract: signals, target weights (proposed) or orders.
- Spec §9.2: bottom ten as avoid/sell only, or shorted?
- Cost model: fixed basis points per trade (proposed), or per-share commission plus
  spread?
- Confirm the temporary dependency of `skatkis.midas.server` on the sandbox harness
  (from S_202610_001), which this change removes when the lab tools move to the
  portfolio engine.

## Done when

- Agreed and designed (`design/S_202610_006-portfolio-backtest-engine.md`) before any
  code.
