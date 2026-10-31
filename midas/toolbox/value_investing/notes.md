# value_investing notes

## Planning (preliminary)

### Objective (2026-10-31, open → S_202610_006)

A trading bot, not a risk model. The user holds mostly the S&P 500 and wants about 10%
of new investments in a value-investing sleeve that beats the S&P 500 by a margin.
Trades are quarterly, across S&P 500 companies; the wallet tracks combinations of
strategies, judged on out-of-sample excess return over the S&P 500 total-return index.
The API matters because backtesting and optimising the metrics needs point-in-time
fundamentals over many years.

### Open owner decisions from the spec (2026-10-31, open)

From `docs/value_investing_bot_spec.md` §9; each blocks the changes noted.

1. Data: paid vendor budget, or EDGAR only? Decided 2026-10-31: free sources (EDGAR,
   Wikipedia or `fja05680`, `yfinance`); survivorship gap accepted for now
   → VI_202610_001
2. Bottom ten: sell/avoid, or short? → S_202610_006
3. Financials and REITs: exclude, or separate scoring? Decided 2026-10-31: REITs and
   financial companies excluded; bonds are outside this tool
   → VI_202610_001, VI_202610_002, VI_202610_003
4. Wallet interface: signals, target weights or orders? → S_202610_006
5. Reporting currency: USD or ZAR? Decided 2026-10-31: main analysis in USD; ZAR
   secondary, once refined and to track the actual record (FRED `DEXSFUS`)
   → VI_202610_001
6. DAG engine: stdlib `graphlib` registry or Hamilton/Dagster? Decided 2026-10-31:
   `graphlib` → VI_202610_001

### The user's existing portfolio as a comparison (2026-10-31, open)

The user's current portfolio is a South African tax-free savings account (TFSA) of
JSE-listed ETFs and feeder funds, mostly US large-cap and technology (S&P 500, S&P 500
IT, Nasdaq 100, global indices); its statement benchmarks against the FTSE/JSE Top 40.
The statement is kept out of the repository, in the git-ignored `data/private/`.

- The account is a comparison only (user, 2026-10-31): the sleeve's results are tracked
  against these standard investments, in ZAR. It does not constrain where the sleeve is
  held.
- The weighting of the diversification matters more than the account (user,
  2026-10-31): the analysis should show how the sleeve shifts the overall weighting,
  for when the user moves money into individual firms reachable in the TFSA on the JSE.
  A JSE universe may follow as a later change.

### Spec §1.3 docstring style (2026-10-31, → S_202610_003)

Spec §1.3 prescribes Google style; the user chose the numpydoc structure of
`.claude/rules/docstrings.md`. The spec is updated in S_202610_003.

## Open methodology questions

### EDGAR history is too short for the long Graham windows (2026-10-31, open)

Found in the first live pull (VI_202610_001, Apple, Microsoft, Coca-Cola). XBRL
filings start in mid-2009; with comparative figures, values reach back to periods ending
in 2006–2007, but nothing was filed in XBRL before 2009. SEC's Financial Statement Data
Sets also start in 2009. So "20 years where EDGAR has it" gives about 17–20 years of
periods, and the rules with long windows first become computable late:

| Rule | Window | First rebalance with a full window (about) |
|---|---|---|
| G3 earnings stability, B1 ROE | 10 years | 2016–2017 |
| G5 earnings growth | 13 years | 2019–2020 |
| G4 dividend record | 20 years | 2026 or later |

A backtest with the full defensive rules would then cover only about 2017–2026, which
leaves little out-of-sample history. Options, for the user to decide:

- Dividend history (G4) from the price source instead: `yfinance` dividends go back
  decades, though not point-in-time filings.
- Shorter windows as explicit, configurable parameters: a methodology change, recorded
  as such.
- A paid source with point-in-time fundamentals before 2009.
- Rules phased in: each rule applies only from the first date its window is full.

