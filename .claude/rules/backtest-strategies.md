---
paths:
  - "Toolbox/skatkis/backtest/**"
  - "Toolbox/*/strategies.py"
  - "config/strategies.toml"
---

# Backtest engine and strategy registry

- The engine in `skatkis.backtest` is domain-agnostic: it turns a strategy's target
  weights (or signals) and a price panel into positions, cash, costs and portfolio
  value. It knows nothing about value-investing rules.
- Strategies live in the tool's registry (`strategies.py`): one function per strategy,
  registered under a lowercase identifier, from the quarterly ranking to target
  weights. Add a strategy by writing and registering a new function, never by editing
  an existing one or adding `if` branches to the engine.
- Every strategy parameter and seed is explicit, in `config/strategies.toml`.
- Trades happen only at rebalance dates and at that date's close; no information after
  the rebalance date is used (no look-ahead).
- Transaction costs are explicit and always applied; a result without costs is labelled
  as such.
- Every result is benchmarked against the S&P 500 total-return index over the same
  dates; the primary measure is out-of-sample excess return over it.
- Optimisation is walk-forward with a hold-out period and a small grid; report the
  number of combinations tried.
- Any change to trade accounting, costs or the benchmark is a methodology change.
