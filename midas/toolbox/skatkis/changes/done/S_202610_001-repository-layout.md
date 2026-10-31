# S_202610_001: Repository layout

| Field | Value |
|---|---|
| Status | done (2026-10-31) |
| Priority | high |
| Kind | refactor |
| Depends on | none |
| Branch | `feature/value-investing-sp500-pipeline` |
| Raised | 2026-10-31 |

## Description

All code sat at the repository root (`trade_wallet.py`, `promised_land.py`, …) with the
MCP server in `mcp_server/`. The agreed target layout (`midas/instructions/AGENTS.md`,
"What this repository is") separates shared infrastructure, the value-investing tool,
exploratory code and notebooks, so ownership is visible from the tree and the index can
be configured by path.

## Outcome

Done 2026-10-31, commit 87a6883 on `feature/value-investing-sp500-pipeline`, with the
warden's fixes in the closing commit. Warden review 2026-10-31 of 87a6883 against
3ebb40f: no blocking findings. Closed by the user on 2026-10-31.

**Conclusion.** The code follows the layout agreed on 2026-10-31:

| Before | After |
|---|---|
| `trade_wallet.py` | `Toolbox/skatkis/backtest/wallet.py` |
| `mcp_server/server.py` | `Toolbox/skatkis/midas/server.py` (`.mcp.json` updated) |
| `mcp_server/backtest.py` | `Toolbox/SandBox/btc_vwap/harness.py` |
| the BTC VWAP modules at the root | `Toolbox/SandBox/btc_vwap/` |
| `run_bot.ipynb`, `optimise_strategies.ipynb` | `notebooks/` |
| `tests/test_*.py` | `tests/skatkis/backtest/`, `tests/sandbox/btc_vwap/` |
| `_init_.py` | removed |
| — | `Toolbox/value_investing/`, an empty package with a README stub |

Code is imported from the repository root as `Toolbox.<package>`, with aliases
(`wallet as trade_wallet`, `harness as backtest`) so calling code is unchanged.
`pyproject.toml` was rescoped to `Toolbox/` and `tests/`; `Toolbox/SandBox` and
`notebooks` are outside the lint and type-check standard, as agreed, so the harness the
lab tools run is no longer type-checked.

**Key results.** A pure refactor: five modules moved byte for byte, the rest changed
only in imports, root-path depth and docstrings. `regression_check` on
`BTCUSDT_1h_2025-06-01_2025-10-01` returned `unchanged`; pytest passed 9 tests; Black,
ruff and pyright (strict) were clean. In `notebooks/run_bot.ipynb` the new imports, the
data fetch, the wallet and the plots work; its other failures were already in 3ebb40f
and are logged in `midas/toolbox/skatkis/notes.md`. The `optimise_strategies.ipynb`
grid run had not finished at closing.

**Warden's fixes, in the closing commit.** Change IDs removed from the server and
package docstrings; old module names updated in the legacy agents
(`.claude/agents/midas.md`, `reviewer.md`, `strategy-researcher.md`); leftover
`mcp_server/` and test caches deleted.

**Left open.**
- The harness sits in the sandbox because it runs the BTC strategy, so
  `skatkis.midas.server` (and the wallet test) depend on `Toolbox/SandBox` until
  S_202610_006 provides the portfolio engine. This is a developer deviation from the
  table; it was closed with the change but awaits the user's explicit confirmation,
  tracked in S_202610_006.
- The regression baseline makes no trades, so it does not exercise the wallet's buy
  and sell paths; a baseline with trades is noted in `midas/toolbox/skatkis/notes.md`.
- Warden notes left for later: the doubled import-path setup in `tests/conftest.py`,
  the module docstring structure (S_202610_003), the lower-case `tests/sandbox/`
  mirror (S_202610_005), and Midas records cited in `pyproject.toml` comments.

**Alternatives considered.** The wallet in the value-investing tool: rejected, because
every strategy and future tool backtests through it. The BTC modules as a second tool
under the standard: rejected in favour of the sandbox. A separate branch: the user kept
the current branch. The harness in `skatkis`, as first planned: rejected, because it
runs the BTC strategy.

## Done when

- The tree matches the target table; nothing is left at the root but configuration.
- `regression_check` is `unchanged` and pytest passes.
- The notebooks run from `notebooks/` with the new imports.
