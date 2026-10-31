# MD_202610_004: A broader US universe for out-of-sample tests

| Field | Value |
|---|---|
| Status | in progress |
| Priority | high |
| Kind | new capability |
| Depends on | MD_202610_001, MD_202610_002 |
| Branch | `feature/value-investing/vi-202610-007-pick-price-history` (shared with VI_202610_007) |
| Raised | 2026-10-31 |

## Description

The value tool works as a proof of concept on the S&P 500. To test whether the rules are
reliable, they must be run on companies they were not developed on. The user chose a
broader US universe over other exchanges (2026-10-31): the same SEC EDGAR fundamentals
and Yahoo prices, so the pipeline follows the same principle and only the universe
changes. The survey of other exchanges that led to this choice is in
[notes.md](../notes.md) ("Other exchanges for out-of-sample tests").

The Dow Jones Industrial Average adds no new companies (all 30 are S&P 500 members), so
it can serve only as a benchmark. There is no "S&P 5000": the broad indices (S&P Total
Market Index, Wilshire 5000) have no free membership history.

## Agreed approach

Route A first, then route B (user, 2026-10-31). Two routes, both on EDGAR and Yahoo:

- **A. S&P Composite 1500** (S&P 500 + MidCap 400 + SmallCap 600). The 400 and 600 give
  about 1,000 companies that the rules have not seen, defined by an index with
  published rules. Wikipedia has the current members of both (symbol, GICS, CIK); the
  history of changes is only partial there, so membership at past dates needs another
  source (S&P DJI announcements, or a community file like `fja05680/sp500` for the
  500). Benchmarks: Yahoo `^SP400TR` and `^SP600TR`, or MidCap and SmallCap ETFs'
  adjusted close.
- **B. A rule-based universe**: every company that files a 10-K, filtered at each
  rebalance date by a market-capitalisation floor from point-in-time shares
  (`dei:EntityCommonStockSharesOutstanding`) times price. This comes closest to a "US
  total market" (several thousand companies), needs no membership history and is fully
  point-in-time by construction. Costs: a CIK-to-ticker history (tickers are reused
  and change), and a far larger survivorship gap, because Yahoo has no prices for
  delisted companies and small caps delist often.

Recommendation: A first. It is the smaller step and has a recognised benchmark. B can
follow if A's results hold up.

Either route reuses `sources/edgar.py` and `sources/prices.py`; the changes are a
second constituents source, a universe definition per index in `config/`, and the
benchmark per universe in the wallet's comparison.

## Open decisions (user)

- How the survivorship gap is measured and reported per universe (share of
  historical members without prices).

## Decisions

- 2026-10-31 (user): a broader US universe instead of other exchanges; route A (S&P
  1500) first, then route B (rule-based universe).
- 2026-10-31 (user): the value tool's reusable parts work for any index; each index
  is tracked separately, with one report each. Kept simple: an index is an entry in
  `config/indices.toml`, not a new layer.
- 2026-10-31 (user, confirmed): the S&P 400 and 600 are compared
  with the iShares core funds IJH and IJR, since Yahoo has no history for `^SP400TR`
  and `^SP600TR`; their adjusted close includes dividends, less a 0.05% yearly fee.
- 2026-10-31 (user, confirmed): the S&P 400 and 600 pages give no
  date added. A member's date is its latest addition in the page's changes table, or
  the table's first date (2012 for the 400, 2019 for the 600) when no addition is
  listed. The tables list selected changes only, so some members count as members
  before they joined: a look-ahead to state in their reports.

## Notes

- 2026-10-31: built.
  - `config/indices.toml`: per index its label, Wikipedia pages and benchmark.
  - `market_data update --index sp500 sp400 sp600` stores each index's members and
    changes under `data/interim/constituents/<as_of>/<index>/` and fetches EDGAR and
    prices for all of them together. Members without a CIK on the page (280 of the
    S&P 400) get it from SEC's ticker list (`company_tickers.json`).
  - The member parser reads columns by header (the pages differ); the changes parser
    no longer takes a dated six-cell row for a continuation row. That fixed one S&P 500
    row (Arconic for Alcoa, 1 November 2016, was dated 2 December with its columns
    shifted); no result used its date.
  - `value_investing buyability|evaluate|report --index <key>` (default `sp500`)
    scores that index's members against its benchmark and writes to
    `results/value_investing/<index>/<as_of>/`; the S&P 500 results moved there. The
    report names the index through `{index}` in its text.
  - Refactor check on the S&P 500, stored data of 2026-10-31: summary, trades,
    purchases, holdings and next picks unchanged. The evaluation's
    `model_comparison.csv` and `ranked_positions.csv` lost their `score_combined` rows,
    a model `main` no longer has (the committed files were stale).
  - Waiting: the S&P 400 and 600 data. The update needs `SEC_USER_AGENT`, which the
    agent's shell does not have; the user runs it.
- 2026-11-01: data fetched (user's identity for SEC, by consent): 738 new EDGAR
  companies (4 failed: 2 unit mismatches already in the notes, 2 broken downloads),
  744 new Yahoo symbols (CWEN-A failed). First reports for all three indices, with the
  point-in-time tag choice and G4 phase-in. Money-weighted return a year, value wallet
  against the index's benchmark:

  | Start | S&P 500 vs `^SP500TR` | S&P 400 vs IJH | S&P 600 vs IJR |
  |---|---|---|---|
  | 2016 | 26.0% vs 15.3% | 12.6% vs 10.6% | 8.9% vs 10.6% (from 2020) |
  | 2020 | 33.9% vs 15.9% | 15.0% vs 10.9% | 8.9% vs 10.6% |
  | 2023 | 43.7% vs 18.7% | 11.4% vs 12.0% | 7.7% vs 13.1% |
  | 2025 | 41.0% vs 15.7% | 11.3% vs 10.5% | 10.4% vs 14.8% |

  On companies the model was not developed on, the edge shrinks to about 1 to 4
  points a year on the S&P 400 and turns negative on the S&P 600. The S&P 600's first
  members date from December 2019 (its change table starts there), so its 2016
  investor starts buying in 2020. Survivorship is larger in both than in the S&P 500.

## Kickoff prompt

Pick up MD_202610_004. Read the agreed approach and the exchanges survey in
`midas/toolbox/market_data/notes.md`, start with route A, and check what
Wikipedia gives for the S&P 400 and 600 members and their changes before designing the
constituents source.

## Done when

- The chosen universe's membership at each rebalance date, EDGAR fundamentals and
  Yahoo prices are stored per snapshot.
- The value tool runs on the S&P 400 and 600 alone (out-of-sample against the
  S&P 500 work) and reports excess return over the matching total-return benchmark,
  with the survivorship gap stated.
- Route B's universe is built and run the same way once route A's results are in.
