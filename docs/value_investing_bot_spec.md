# Value-Investing Screener & Backtest — Design Specification

> Status: **DRAFT — design open.** Items marked `DECISION REQUIRED` must be resolved by the owner before implementation. Do not guess; stop and ask.

## 0. Objective

Build a deterministic Python pipeline that:

1. Extracts fundamentals and prices for S&P 500 constituents via API.
2. Every quarter, computes value/quality metrics derived from Benjamin Graham (*The Intelligent Investor*), Warren Buffett and Charlie Munger.
3. Produces a **top 10** (buy/hold candidates) and **bottom 10** per quarter.
4. Feeds an existing trading wallet interface that evaluates profits under different trading strategies and optimises them.

## 1. Implementation rules (non-negotiable)

### 1.1 General

- Python ≥ 3.11, full type hints on every function, argument and return.
- Deterministic: no implicit dates (`today()`), no unseeded randomness, no hidden IO, no hardcoded paths. All dates, seeds and paths are explicit inputs (config or function arguments).
- One owner per calculated concept; never recompute an upstream result downstream.
- Fail fast at boundaries (API responses, schema, units) with clear diagnostics naming the ticker, date and field.
- Simple design: plain functions over DataFrames, minimal dataclasses, little nesting.
- Order of work: code → unit tests → documentation (documentation only once code is finalised; do not regenerate docs after every change).
- When instructed, change only the files the instruction affects.
- Never edit reference values or tolerances to make tests pass.
- Any change to a formula or threshold in §2 is a **methodology change** — flag it, do not treat it as a refactor.

### 1.2 Modularity and scalability

- One module = one concept (e.g. `metrics/graham.py` owns G1–G7 only). Modules stay small and single-purpose.
- Pure functions for all calculation: inputs in, DataFrame out; no IO, no global state, no mutation of inputs (return copies).
- IO is isolated in `io/` and `ingest/`; calculation modules never call APIs or touch the filesystem.
- Data providers sit behind a single protocol (`DataProvider`) so EDGAR, FMP or EODHD can be swapped without touching downstream code.
- New rules, factors or strategies are added by writing a new function and registering it — never by editing existing ones.
- Vectorised over the full ticker × date panel (no per-ticker Python loops in hot paths) so the universe can scale beyond the S&P 500.
- Thresholds, weights and windows live in versioned config (`config/*.toml`), never in code.

### 1.3 Docstrings (mandatory on every public and private function)

Google style. Every function docstring must contain:

```python
def current_ratio(balance: pd.DataFrame) -> pd.Series:
    """Compute the Graham current ratio (rule G2a).

    Formula:
        current_ratio = total_current_assets / total_current_liabilities

    Args:
        balance: Point-in-time balance-sheet panel indexed by (ticker, as_of_date)
            with columns ``total_current_assets`` and ``total_current_liabilities``
            in USD.

    Returns:
        Series indexed like ``balance``; dimensionless ratio, bounds [0, inf).

    Assumptions:
        Balance-sheet values are the latest quarter filed on or before as_of_date.

    Edge cases:
        Liabilities == 0 -> NaN and a flag in the diagnostics table (never inf).
        Negative or missing inputs -> raises ``DataValidationError``.

    Raises:
        DataValidationError: Required columns missing or negative values present.
    """
```

Required sections: one-line summary, **Formula** (for any calculation), **Args** (with units), **Returns** (units and bounds), **Assumptions**, **Edge cases**, **Raises**. Modules and packages also carry a module docstring describing their single responsibility.

### 1.4 Lint and type checking (must pass in CI)

`pyproject.toml` must contain at minimum:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP", "N", "D", "ANN", "SIM", "RUF", "PL", "PD", "NPY"]
ignore = ["D203", "D213"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["D", "ANN", "PLR2004"]

[tool.pyright]
typeCheckingMode = "strict"
pythonVersion = "3.11"
include = ["src", "tests"]
```

- `ruff check`, `ruff format --check` and `pyright` must report zero errors. No blanket `# type: ignore` or `# noqa`; any targeted suppression needs a comment explaining why.
- Use `pandas-stubs` for typed pandas.
- Pre-commit hook runs Ruff and pyright; CI runs Ruff, pyright and pytest (with coverage).

### 1.5 Tests

- pytest, mirroring the `src/` structure one-to-one.
- Every calculation covers: normal, zero, boundary (exact threshold, e.g. P/E = 15.0), failure (missing/invalid input) and input non-mutation.
- Hand-computed reference values in fixtures, with the derivation in a comment.
- DAG tests: graph is acyclic, every node's declared inputs are produced upstream, and a full run on a small fixture universe is bit-for-bit reproducible.

## 2. Rule catalogue

### 2.1 Graham — Defensive investor (Intelligent Investor, Ch. 14)

| ID | Rule | Formula / threshold | Data |
|---|---|---|---|
| G1 | Adequate size | Inflation-adjusted sales floor (redundant for S&P 500; implement as configurable, default off) | Revenue |
| G2a | Current ratio | Current assets / current liabilities ≥ 2.0 | Balance sheet |
| G2b | Debt cover | Long-term debt ≤ net current assets (current assets − current liabilities) | Balance sheet |
| G3 | Earnings stability | EPS > 0 in each of the last 10 fiscal years | 10y income statement |
| G4 | Dividend record | Uninterrupted dividends for 20 years | Dividend history |
| G5 | Earnings growth | 3-yr average EPS (latest) ≥ 1.33 × 3-yr average EPS (10 years earlier) | 13y EPS |
| G6 | Moderate P/E | Price / 3-yr average EPS ≤ 15 | Price, EPS |
| G7 | Moderate price-to-assets | P/B ≤ 1.5 **or** (P/E × P/B) ≤ 22.5 | Book value |

### 2.2 Graham — Enterprising investor (Ch. 15)

| ID | Rule | Threshold |
|---|---|---|
| E1 | Current ratio | ≥ 1.5 |
| E2 | Debt | ≤ 110% of net current assets |
| E3 | Earnings stability | EPS > 0 for last 5 years |
| E4 | Dividend | Some current dividend |
| E5 | Growth | EPS > EPS four years ago |
| E6 | Price | ≤ 120% of net tangible assets |

### 2.3 Graham — derived measures

| ID | Measure | Formula |
|---|---|---|
| GN | Graham number | √(22.5 × EPS × BVPS); undefined (NaN, flagged) if EPS ≤ 0 or BVPS ≤ 0 |
| NN | Net-net | Price ≤ ⅔ × NCAV per share, NCAV = current assets − total liabilities (rarely triggers in S&P 500) |

### 2.4 Buffett / Munger — quality at a fair price

| ID | Rule | Formula |
|---|---|---|
| B1 | High, consistent ROE, unlevered | ROE ≥ 15% in most of last 10 years (count configurable) **and** debt/equity ≤ cap (configurable) |
| B2 | Owner earnings | Net income + D&A − capex (total capex used as conservative proxy for maintenance capex) |
| B3 | Moat proxy | Gross-margin level and 10-yr stability (standard deviation) |
| B4 | Retained-earnings test | Δ market cap over 5 yrs ≥ cumulative retained earnings over the same 5 yrs |
| B5 | Margin of safety | Price ≤ (1 − m) × intrinsic value; intrinsic value = discounted owner earnings at an explicit rate. `m` and rate are explicit inputs |
| M1 | Return on capital | ROIC sustained above an explicit hurdle |

### 2.5 Systematic benchmarks (well-documented, backtested)

| ID | Method | Formula |
|---|---|---|
| Q1 | Greenblatt Magic Formula | Rank on EBIT/EV and EBIT/(net working capital + net PP&E) |
| Q2 | Piotroski F-score | 9 binary tests (profitability, leverage/liquidity, operating efficiency) |
| Q3 | Novy-Marx gross profitability | Gross profit / total assets |

## 3. Methodology decisions

1. **Strict Graham screen yields too few names.** In today's S&P 500, G2 and G7 eliminate most companies; there may not be ten passes. Proposed two layers:
   - **Hard gates:** solvency and earnings stability (subset of G2/G3/E1–E3; configurable).
   - **Composite rank:** valuation + quality factors → top 10 / bottom 10 per quarter.
   - This is a documented departure from pure Graham.
2. **Financials and REITs:** current ratio / NCA are meaningless. `DECISION REQUIRED`: exclude, or separate scoring (e.g. P/B and ROE).
3. **Look-ahead bias:** key every fundamental to the SEC **filing (acceptance) date**, not period-end. 10-Q lag ≈ 40–45 days; 10-K ≈ 60–90 days. A rebalance at date *t* sees only filings accepted ≤ *t*.
4. **Survivorship bias:** use **historical** S&P 500 membership at each rebalance date.
5. **Restatements:** use first-reported (point-in-time) values, not the latest revised values.
6. **Units / time scale:**
   - Flow items (earnings, cash flow) = trailing twelve months (sum of last 4 quarters).
   - Stock items (balance sheet) = latest quarter.
   - Price = rebalance-date close, USD.
   - `DECISION REQUIRED`: reporting currency (USD or ZAR).
7. **Optimisation overfitting:** walk-forward with hold-out period, small parameter grid, net of explicit transaction costs.

## 4. Data sources

| Source | Benefit | Cost |
|---|---|---|
| SEC EDGAR XBRL API | Free, authoritative, filing dates enable point-in-time | Tag inconsistency needs a mapping layer; no prices |
| FMP / EODHD | Standardised statements, prices, historical constituents | Paid; verify point-in-time behaviour and licence |
| yfinance | Quick prototyping | Unofficial, non-reproducible; **not** for audit-grade output |

Recommendation: EDGAR for fundamentals + one paid vendor for prices and historical constituents. `DECISION REQUIRED`: data budget.

## 5. Architecture

Each concept has exactly one owning module. Stages are plain functions over DataFrames.

| Stage | Responsibility | Output |
|---|---|---|
| 1. `ingest` | Raw API pulls, cached as immutable snapshots tagged by as-of date | Raw tables |
| 2. `fundamentals` | Point-in-time TTM and 10–20y history per ticker per as-of date | Fundamentals panel |
| 3. `metrics` | All ratios in §2 (sole owner of each ratio) | Metrics panel |
| 4. `ranking` | Gates → composite score → top 10 / bottom 10 per quarter | Ranking table |
| 5. `backtest` / wallet adapter | Converts ranking to wallet input; runs strategies | Strategy results |

Candidate strategies (all parameters explicit, seed explicit):

- Equal-weight top 10, rebalanced quarterly.
- Hold until a stock drops out of the top N (buffer reduces turnover).
- Graham's later rule of thumb: sell after 50% gain or after 2 years.

## 6. DAG pipeline

### 6.1 Node contract

The pipeline is a directed acyclic graph of **nodes**. Each node is a pure function registered with:

| Field | Meaning |
|---|---|
| `name` | Unique output artefact name (e.g. `metrics.graham`) — the one owner of that concept |
| `inputs` | Names of upstream artefacts it consumes |
| `func` | Pure function `(**inputs, config) -> pd.DataFrame` |
| `schema` | Expected output columns, dtypes and index; validated after every run |

Rules:

- Edges are inferred from `inputs`; ordering via stdlib `graphlib.TopologicalSorter` with ties broken alphabetically (deterministic).
- Cycles, missing inputs, or duplicate `name`s fail at graph build time, before any data is touched.
- Each node output is persisted to `data/artefacts/<run_id>/<name>.parquet` with a manifest (config hash, input hashes, as-of date, code version) for audit traceability and caching.
- A node re-runs only if its config or input hashes change.
- Partial runs: `run(targets=["ranking.quarterly"])` executes only the required sub-graph.

Library alternative (owner to decide): Hamilton or Dagster give DAG building, caching and lineage out of the box. Benefit: less custom code, a UI with Dagster. Cost: an extra dependency and framework conventions. Default: a stdlib `graphlib` registry (≈100 lines) unless the owner chooses otherwise.

### 6.2 Graph

```
ingest.constituents ─┐
ingest.prices ───────┼─────────────────────────────┐
ingest.filings ──────┤                             │
                     ▼                             │
         fundamentals.point_in_time                │
                     │                             │
      ┌──────────────┼──────────────┬──────────┐   │
      ▼              ▼              ▼          ▼   ▼
metrics.graham  metrics.buffett  metrics.quant  metrics.valuation
      └──────────────┴──────┬───────┴──────────┘
                            ▼
                     ranking.gates
                            ▼
                     ranking.composite
                            ▼
                     ranking.quarterly  (top 10 / bottom 10)
                            ▼
                     wallet.signals  ──►  backtest.strategies  ──►  backtest.optimise
```

## 7. Folder structure

```
value-bot/
├── pyproject.toml              # deps, ruff, pyright, pytest config
├── README.md
├── .pre-commit-config.yaml
├── config/
│   ├── pipeline.toml           # run dates, universe, paths, seed
│   ├── rules.toml              # thresholds for §2 (versioned)
│   └── strategies.toml         # strategy parameters and optimisation grid
├── src/value_bot/
│   ├── __init__.py
│   ├── dag/
│   │   ├── node.py             # node contract + registry
│   │   ├── graph.py            # build, validate, topological order
│   │   ├── runner.py           # execute, cache, manifest
│   │   └── registry.py         # registers all nodes
│   ├── io/
│   │   ├── artefacts.py        # parquet read/write + manifests
│   │   └── config.py           # load and validate TOML config
│   ├── ingest/
│   │   ├── provider.py         # DataProvider protocol
│   │   ├── edgar.py            # SEC XBRL client
│   │   ├── vendor.py           # prices / constituents vendor client
│   │   └── tag_map.py          # XBRL tag → canonical field mapping
│   ├── fundamentals/
│   │   ├── point_in_time.py    # filing-date alignment, first-reported values
│   │   └── ttm.py              # TTM flow items, latest stock items
│   ├── metrics/
│   │   ├── graham.py           # G1–G7, E1–E6, GN, NN
│   │   ├── buffett.py          # B1–B5, M1
│   │   ├── quant.py            # Q1–Q3
│   │   └── valuation.py        # shared price-based ratios (single owner)
│   ├── ranking/
│   │   ├── gates.py
│   │   ├── composite.py
│   │   └── quarterly.py
│   ├── wallet/
│   │   └── adapter.py          # ranking → wallet input contract
│   ├── backtest/
│   │   ├── strategies.py
│   │   ├── costs.py
│   │   └── optimise.py         # walk-forward, hold-out
│   ├── validation/
│   │   ├── schemas.py          # output schemas per node
│   │   └── errors.py           # DataValidationError etc.
│   └── cli.py                  # `value-bot run --config ... --as-of ...`
├── tests/                      # mirrors src/value_bot/ one-to-one
│   ├── fixtures/               # small deterministic universe + hand-computed references
│   └── ...
├── docs/                       # written only once code is finalised
│   ├── methodology.md          # every rule: formula, source, threshold, departures from Graham
│   ├── dag.md                  # node catalogue: inputs, outputs, schema, owner
│   ├── data_sources.md         # providers, field mapping, point-in-time handling
│   ├── strategies.md
│   └── changelog.md            # methodology changes flagged separately
└── data/                       # git-ignored: raw cache and run artefacts
```

## 8. Documentation (after code is finalised)

- `methodology.md`: for each rule ID in §2, give the formula, units, source (book/chapter), threshold, and any documented departure.
- `dag.md`: generated from the node registry (name, inputs, schema) so it cannot drift from the code, plus a rendered graph (Mermaid).
- `data_sources.md`: provider details, XBRL tag mapping, filing-lag handling, survivorship treatment.
- `changelog.md`: methodology changes listed separately from code changes.
- API reference generated from docstrings (e.g. mkdocstrings), so docstrings are the single source.

## 9. Open decisions (resolve before coding)

1. **Data:** paid vendor budget, or EDGAR only?
2. **Bottom ten:** sell/avoid, or short? Shorting changes wallet logic and risk.
3. **Financials/REITs:** exclude, or separate scoring?
4. **Wallet interface:** input contract (signals, target weights, or orders) and format. Read the wallet repo's interface before writing the adapter.
5. **Reporting currency:** USD or ZAR.

6. **DAG engine:** stdlib `graphlib` registry (default) or Hamilton/Dagster.

## 10. Disclaimer

Research and engineering specification only; not investment advice.
