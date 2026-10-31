# market_data change log

Planned changes to the shared market data package (`Toolbox/market_data/`): sources,
cleaned standard tables and shared metrics, read by every tool. Prefix `MD`. Naming and
workflow: `midas/instructions/secretary.md`.

## Open

| ID | Title | Kind | Status | Priority | Depends on |
|---|---|---|---|---|---|
| [MD_202610_001](changes/MD_202610_001-sp500-data-sources.md) | S&P 500 data sources: EDGAR fundamentals, prices and dividends, FX, historical constituents | new capability | in progress | high | S_202610_001; spec §9.1 |
| [MD_202610_002](changes/MD_202610_002-point-in-time-fundamentals.md) | Point-in-time fundamentals panel: filing-date keying, first-reported values, TTM | new capability | ready | high | S_202610_001, MD_202610_001 |

## Done

| ID | Title | Done |
|---|---|---|
