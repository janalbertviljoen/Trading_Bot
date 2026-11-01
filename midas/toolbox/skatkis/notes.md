# skatkis notes

## Planning (preliminary)

### Origin of the layout (2026-10-31, → S_202610_001)

The layout and governance follow the example repository in `docs/attachments.zip`
(Steward, `skatkis`, `fair_value`). Names and parts that do not apply here: client
folders, the Tutor role, accreditation sign-off documents, Word reports, Teams hosting.

### Wallet framework as found (2026-10-31, → S_202610_003, S_202610_006)

Single asset, row-by-row loop, no costs, mutates its input, needs a `RangeIndex` from
0. Documented in its docstrings on 2026-10-31; known defects listed in S_202610_003.

### Column names in the member-facing data rules (2026-11-01, open)

The data rules in `midas/instructions/quant.md` name this repository's columns
(`period_start`, `period_end`, `fiscal_year`, `fiscal_period`, `adj_close`, the splits
table). They are fine for members if the member engine exposes the same schema; decide
when the member profile is built (S_202610_002). Raised by the warden on S_202611_001.

## Sandbox and test infrastructure

### Notebook defects found when moving them (2026-10-31, open)

Found when executing the notebooks after S_202610_001; all predate the move.
`notebooks/run_bot.ipynb` uses `dict_read_data` in cell 3 before cell 7 defines it,
calls `PromisedLandTrader` without `dict_detect_range` and `dict_trade_range`, and uses
`bb` in the Fourier cells without importing it. `notebooks/optimise_strategies.ipynb`
uses `df_time_top` in cell 54 and `highest_comb` in cells 54 and 62, which no cell
defines; everything else, including the full parameter grid, runs.

### The regression baseline makes no trades (2026-10-31, open)

`benchmarks/baseline.json` for `BTCUSDT_1h_2025-06-01_2025-10-01` records no buys and no
sells, so `regression_check` does not exercise the wallet's trade paths. A second
pinned dataset or configuration that trades is needed before the wallet is refactored;
adding it is the user's decision.
