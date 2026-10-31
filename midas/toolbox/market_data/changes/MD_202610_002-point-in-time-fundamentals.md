# MD_202610_002: Point-in-time fundamentals panel

| Field | Value |
|---|---|
| Status | ready |
| Priority | high |
| Kind | new capability |
| Depends on | S_202610_001, MD_202610_001 |
| Raised | 2026-10-31 |

## Description

Spec §5 stage 2 (`fundamentals.point_in_time`): turn the cached API pulls of
MD_202610_001 into one panel per ticker and rebalance date that holds only what was
known on that date. Every rule in VI_202610_002 consumes this panel; without it, the
rules either look ahead or use restated values.

## Approach

- `Toolbox/market_data/layers/fundamentals.py`, one layer, sole owner of
  point-in-time alignment and the TTM convention.
- **Filing-date keying** (spec §3.3): a value counts from its EDGAR `filed` date; a
  rebalance at date $t$ sees only filings with $\text{filed} \le t$, same-day filings
  included.
- **First-reported values** (spec §3.5): per company, concept and fiscal period, keep
  the value with the earliest filing date; later amendments (10-K/A, restated
  comparatives in later filings) are ignored.
- **Time basis** (spec §3, item 6): flow items as trailing twelve months, the sum of
  the last four quarters, with Q4 derived as the annual figure less the first three
  quarters; stock items from the latest quarter; up to 20 fiscal years of history for
  G3, G4, G5 and B1.
- **Universe** (spec §3.4): the panel's rows are the historical S&P 500 members at each
  rebalance date, joined on CIK, without the Financials and Real Estate sectors.
- Output: one row per (ticker, rebalance date), canonical fundamentals plus the filing
  date and accession of each value, for traceability.
- Validation minimal during the build (declared columns only), per the user's decision
  of 2026-10-31.

## Open decisions (user)

- Confirm the rebalance reading below.

## Decisions

- 2026-10-31 (user): rebalance on the S&P quarterly rebalance dates (third Friday of
  March, June, September and December), each using the most recent filings accepted on
  or before that date. Reading to confirm: no extra trades between rebalance dates
  when a filing arrives.
- 2026-10-31 (user): a missing quarter or tag stays NaN; no filling from the annual
  filing and no interpolation until the data has been looked at.
- 2026-10-31 (user): 20 years of history where EDGAR has it; shorter histories stay as
  they are, and rules needing more years return NaN.
- 2026-10-31 (user): REITs and financial companies are excluded from the universe.

## Kickoff prompt

Pick up MD_202610_002. Confirm MD_202610_001 is done and the decisions above are made.
Read spec §3 and §5 and the tool's notes, then build the filing-date alignment on a
small fixture universe first and show a panel for one company across several
rebalance dates.

## Done when

- For a fixture universe, the panel at each rebalance date holds only values filed on
  or before it, first-reported, with TTM flows and latest-quarter stocks.
- A known restatement (one company, one period) is shown to use the original value.
