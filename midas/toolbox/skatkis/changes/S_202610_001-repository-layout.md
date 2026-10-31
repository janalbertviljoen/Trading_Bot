# S_202610_001: Repository layout

| Field | Value |
|---|---|
| Status | in progress |
| Priority | high |
| Kind | refactor |
| Depends on | none |
| Branch | `feature/value-investing-sp500-pipeline` |
| Raised | 2026-10-31 |

## Description

All code sits at the repository root (`trade_wallet.py`, `promised_land.py`, …) with the
MCP server in `mcp_server/`. The agreed target layout (`midas/instructions/AGENTS.md`,
"What this repository is") separates shared infrastructure, the value-investing tool,
exploratory code and notebooks, so ownership is visible from the tree and the index can
be configured by path.

## Agreed approach

A pure move, no behaviour change; `regression_check` must stay `unchanged`.

| Today | Target |
|---|---|
| `trade_wallet.py` | `Toolbox/skatkis/backtest/wallet.py` |
| `mcp_server/backtest.py` | `Toolbox/SandBox/btc_vwap/harness.py` (see Decisions) |
| `mcp_server/server.py` | `Toolbox/skatkis/midas/server.py` (lab tools; index added in S_202610_002) |
| `trade_perforpance.py` (`detrend_series`, FFT) | `Toolbox/SandBox/btc_vwap/` (cycle analysis is BTC research) |
| `trade_indicators.py`, `read_data.py`, `promised_land.py`, `analysis.py`, `plot_analysis.py`, `milkandhoney27.py` | `Toolbox/SandBox/btc_vwap/` |
| `run_bot.ipynb`, `optimise_strategies.ipynb` | `notebooks/` |
| `_init_.py` (empty, misnamed) | removed |
| `tests/test_*.py` | `tests/skatkis/backtest/test_wallet.py`, `tests/sandbox/btc_vwap/test_harness.py`, `tests/sandbox/btc_vwap/test_trade_indicators.py` |
| — | `Toolbox/value_investing/` (empty package with README stub; filled by VI changes) |
| — | `pyproject.toml` with Black, ruff, pyright and pytest config (spec §1.4): added 2026-10-31 with the virtual environment, scoped to `Toolbox/`; widen if the move changes paths |

Imports in the moved modules, the notebooks, tests and `.mcp.json` are updated. Whether
the wallet belongs in `skatkis` (shared engine) or in the tool is the main design
question: it is proposed for `skatkis` because every strategy and future tool backtests
through it.

## Decisions

- 2026-10-31 (user): carry out the layout; the proposals stand: the wallet goes to
  `Toolbox/skatkis/backtest/`, the BTC modules to `Toolbox/SandBox/btc_vwap/`, and the
  work stays on the current branch.
- 2026-10-31 (developer, deviation from the table): the harness (`mcp_server/backtest.py`)
  runs the BTC VWAP strategy, so it went to `Toolbox/SandBox/btc_vwap/harness.py`
  instead of `skatkis`, which must not depend on tool or sandbox code. The server in
  `Toolbox/skatkis/midas/` imports it for the lab tools; this one dependency of
  `skatkis` on the sandbox stays until S_202610_006 provides the portfolio engine.

## Notes

- 2026-10-31: moved with `git mv`; imports use `Toolbox.<package>` from the repository
  root, with aliases keeping the old names (`wallet as trade_wallet`, `harness as
  backtest`) so calling code is unchanged. Notebooks add the root to `sys.path`.
  `_init_.py` removed; `pyproject.toml` scope updated (`Toolbox/**/__init__.py` exempt
  from N999 because `Toolbox` is capitalised by decision).
- 2026-10-31: refactor verified: `regression_check` on
  `BTCUSDT_1h_2025-06-01_2025-10-01` is `unchanged`; pytest 9 passed; Black, ruff and
  pyright (strict) clean; the server loads from its new path with all six lab tools.
  The notebooks were not executed.

## Kickoff prompt

Pick up S_202610_001. Record the regression baseline on a pinned dataset, move the
modules as in the approach table with `git mv`, update imports in code, tests,
notebooks and `.mcp.json`, add `pyproject.toml`, then show `regression_check` and
pytest output.

## Done when

- The tree matches the target table; nothing is left at the root but configuration.
- `regression_check` is `unchanged` and pytest passes.
- The notebooks run from `notebooks/` with the new imports.
