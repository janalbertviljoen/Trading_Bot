# Value-Investing Trading Rules

> Extract of `value_investing_bot_spec.md` (§2, §3 and the strategy list from §5). The spec remains the single source; update it first and re-extract.

## 1. Rule catalogue

### 1.1 Graham — Defensive investor (Intelligent Investor, Ch. 14)

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

### 1.2 Graham — Enterprising investor (Ch. 15)

| ID | Rule | Threshold |
|---|---|---|
| E1 | Current ratio | ≥ 1.5 |
| E2 | Debt | ≤ 110% of net current assets |
| E3 | Earnings stability | EPS > 0 for last 5 years |
| E4 | Dividend | Some current dividend |
| E5 | Growth | EPS > EPS four years ago |
| E6 | Price | ≤ 120% of net tangible assets |

### 1.3 Graham — derived measures

| ID | Measure | Formula |
|---|---|---|
| GN | Graham number | √(22.5 × EPS × BVPS); undefined (NaN, flagged) if EPS ≤ 0 or BVPS ≤ 0 |
| NN | Net-net | Price ≤ ⅔ × NCAV per share, NCAV = current assets − total liabilities (rarely triggers in S&P 500) |

### 1.4 Buffett / Munger — quality at a fair price

| ID | Rule | Formula |
|---|---|---|
| B1 | High, consistent ROE, unlevered | ROE ≥ 15% in most of last 10 years (count configurable) **and** debt/equity ≤ cap (configurable) |
| B2 | Owner earnings | Net income + D&A − capex (total capex used as conservative proxy for maintenance capex) |
| B3 | Moat proxy | Gross-margin level and 10-yr stability (standard deviation) |
| B4 | Retained-earnings test | Δ market cap over 5 yrs ≥ cumulative retained earnings over the same 5 yrs |
| B5 | Margin of safety | Price ≤ (1 − m) × intrinsic value; intrinsic value = discounted owner earnings at an explicit rate. `m` and rate are explicit inputs |
| M1 | Return on capital | ROIC sustained above an explicit hurdle |

### 1.5 Systematic benchmarks (well-documented, backtested)

| ID | Method | Formula |
|---|---|---|
| Q1 | Greenblatt Magic Formula | Rank on EBIT/EV and EBIT/(net working capital + net PP&E) |
| Q2 | Piotroski F-score | 9 binary tests (profitability, leverage/liquidity, operating efficiency) |
| Q3 | Novy-Marx gross profitability | Gross profit / total assets |

## 2. Methodology decisions

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

## 3. Trading strategies

Candidate strategies (all parameters explicit, seed explicit):

- Equal-weight top 10, rebalanced quarterly.
- Hold until a stock drops out of the top N (buffer reduces turnover).
- Graham's later rule of thumb: sell after 50% gain or after 2 years.

## 4. Disclaimer

Research and engineering specification only; not investment advice.
