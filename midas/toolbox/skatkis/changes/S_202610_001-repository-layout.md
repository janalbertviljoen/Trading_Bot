# S_202610_001: Repository layout

| Field | Value |
|---|---|
| Status | in discussion |
| Priority | high |
| Kind | refactor |
| Depends on | none |
| Branch | `feature/value-investing-sp500-pipeline` (or its own, user's choice) |
| Raised | 2026-10-31 |

## Description

All code sits at the repository root (`trade_wallet.py`, `promised_land.py`, …) with the
MCP server in `mcp_server/`. The agreed target layout (`midas/instructions/AGENTS.md`,
"What this repository is") separates shared infrastructure, the value-investing tool,
exploratory code and notebooks, so ownership is visible from the tree and the index can
be configured by path.

## Approach

A pure move, no behaviour change; `regression_check` must stay `unchanged`.

| Today | Target |
|---|---|
| `trade_wallet.py` | `Toolbox/skatkis/backtest/wallet.py` |
| `mcp_server/backtest.py` | `Toolbox/skatkis/backtest/harness.py` |
| `mcp_server/server.py` | `Toolbox/skatkis/midas/server.py` (lab tools; index added in S_202610_002) |
| `trade_perforpance.py` (`detrend_series`, FFT) | `Toolbox/SandBox/btc_vwap/` (cycle analysis is BTC research) |
| `trade_indicators.py`, `read_data.py`, `promised_land.py`, `analysis.py`, `plot_analysis.py`, `milkandhoney27.py` | `Toolbox/SandBox/btc_vwap/` |
| `run_bot.ipynb`, `optimise_strategies.ipynb` | `notebooks/` |
| `_init_.py` (empty, misnamed) | removed |
| `tests/test_*.py` | `tests/skatkis/backtest/…`, `tests/sandbox/…` mirroring `Toolbox/` |
| — | `Toolbox/value_investing/` (empty package with README stub; filled by VI changes) |
| — | `pyproject.toml` with Black, ruff, pyright and pytest config (spec §1.4): added 2026-10-31 with the virtual environment, scoped to `Toolbox/`; widen if the move changes paths |

Imports in the moved modules, the notebooks, tests and `.mcp.json` are updated. Whether
the wallet belongs in `skatkis` (shared engine) or in the tool is the main design
question: it is proposed for `skatkis` because every strategy and future tool backtests
through it.

## Open decisions (user)

- Wallet in `Toolbox/skatkis/backtest/` (proposed) or in the tool?
- Do the BTC modules move to the sandbox (proposed), or stay a second tool
  (`Toolbox/btc_vwap/`) under the standard?
- Own branch `feature/skatkis/s-202610-001-repository-layout`, or the current branch?

## Kickoff prompt

Pick up S_202610_001. Record the regression baseline on a pinned dataset, move the
modules as in the approach table with `git mv`, update imports in code, tests,
notebooks and `.mcp.json`, add `pyproject.toml`, then show `regression_check` and
pytest output.

## Done when

- The tree matches the target table; nothing is left at the root but configuration.
- `regression_check` is `unchanged` and pytest passes.
- The notebooks run from `notebooks/` with the new imports.
