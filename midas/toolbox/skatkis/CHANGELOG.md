# skatkis change log

Planned changes to the shared core (`Toolbox/skatkis/`), Midas's index server and the
repository setup. Prefix `S`. Naming and workflow: `midas/instructions/secretary.md`.

## Open

| ID | Title | Kind | Status | Priority | Depends on |
|---|---|---|---|---|---|
| [S_202610_002](changes/S_202610_002-midas-index-server.md) | Midas index server with `index` and `lab` profiles | tooling | ready | high | S_202610_001 |
| [S_202610_003](changes/S_202610_003-numpydoc-docstrings.md) | Docstrings to the numpydoc structure (Owns, Must not, Public API, Computes) | documentation | ready | medium | S_202610_001 |
| [S_202610_004](changes/S_202610_004-claude-code-guard.md) | Claude Code permissions, protected files and post-edit hook | tooling | in discussion | medium | none |
| [S_202610_005](changes/S_202610_005-repository-sop.md) | Repository SOP and README | documentation | ready | low | S_202610_001 |
| [S_202610_007](changes/S_202610_007-market-analysis-plots.md) | Market analysis plots: metrics over time, coverage and spread, generic on any column | new capability | in discussion | high | MD_202610_001 |
| [S_202610_006](changes/S_202610_006-portfolio-backtest-engine.md) | Multi-asset portfolio backtest engine (wallet) | new capability | in discussion | high | S_202610_001, spec §9.4 |

## Done

| ID | Title | Done |
|---|---|---|
| [S_202610_001](changes/done/S_202610_001-repository-layout.md) | Repository layout: `Toolbox/`, `skatkis`, sandbox, notebooks | 2026-10-31 |
