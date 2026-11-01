# VI_202610_009: Unit tests for the value tool and the market data it reads

| Field | Value |
|---|---|
| Status | ready |
| Priority | high |
| Kind | tests |
| Depends on | VI_202610_002 to VI_202610_007, MD_202610_004 (code settled) |
| Raised | 2026-11-01 |

## Description

The value-investing pipeline runs end to end on three indices, but only 15 tests cover
it. The warden listed the tests it needs in both its reports (VI_202610_006, first run,
section 4; VI_202610_007, follow-up run, section 4). The user made them a high priority
(2026-11-01), before the documentation and a first version (V1) of the pipeline. The
open changes VI_202610_002 to 006 close once these tests and the documentation are in.

## Approach

Per `.claude/rules/tests.md`: files mirror `Toolbox/`; synthetic tickers and small
fixtures, no network; every calculation covers normal, zero, boundary, failure, input
non-mutation, group isolation and point-in-time; hand-computed reference values with
their derivation in a comment.

In order of risk:

1. Point-in-time and universe: `market_data.fundamentals.annual_as_of` (tag switching
   between two dates, gap and stale cuts, same-day filing), `fill_date_added`,
   `inputs.members_on_date`, `inputs.index_universe`, G4 at the window-start boundary
   and with a missing listing.
2. The rules: every rule in `graham`, `buffett` and `growth` at its exact threshold,
   with zero or negative denominators and with the window not yet full; `valuation`.
3. Scoring and weights: `rule_points` (ties, undefined value, direction), `score`
   (`min_available` boundary), `rule_weights`, `learned_score`, `weighting` (only closed
   windows, floor at 0).
4. Wallets: `evaluation` (`decision_dates`, `signals`, `book_weights` incl. empty and
   ties, `chained_score`, `short_path`), `inputs.window_returns`, `holding`
   (`buy_and_hold`, `buy_and_sell` at sale fractions 0, 0.4 and NaN, `_value_path`
   carrying the last price, `money_weighted_return` known IRR), `ledger`, `peers`.
5. Sources: `parse_constituents` with and without the CIK and "Date added" columns
   and a missing header, `parse_changes` continuation rows, the CIK fill from the
   ticker list (`BRK.B` to `BRK-B`), `parse_ticker_ciks`, `_listings`, `split_factors`.
6. Shared layouts and statistics in `tests/skatkis/`: `rank_correlation`,
   `plot_heatmap`, `plot_small_multiples` (order, titles, page cut), `skatkis.report`
   tables (both branches).
7. Diagnostics: `survivorship`, `information_coefficient`, `lookahead` on a fixture.

Unit tests may be committed directly, without another warden run (user, 2026-10-31).

## Kickoff prompt

Pick up VI_202610_009. Read the test lists in sections 4 of the two warden reports
(VI_202610_006 and VI_202610_007) and `.claude/rules/tests.md`, then write the tests in
the order of the approach, running `pytest` after each module.

## Done when

- Every item of both warden lists has a test, the suite passes, and VI_202610_002 to
  006 can close once the documentation is in.
