# S_202610_008: Midas's wrath: scenario testing of a model

| Field | Value |
|---|---|
| Status | agreed |
| Priority | high |
| Kind | new capability |
| Depends on | MD_202610_004; the market scenarios later on `hand_o_midas` |
| Raised | 2026-10-31 |

## Description

Out-of-sample backtests judge a model on the one history that happened. The user wants
a second way of testing it: take the real data, change the inputs to create scenarios
(a higher P/E, a lower book value, a market fall) and see how the model and its wallet
respond. The work would run as a new agent, `midas-wrath`. Two hard parts, named by the
user: keeping the link between changed inputs and the prices that follow believable,
and recomputing the benchmark under a scenario (possible for the S&P 500, which is an
index of its members; not for Berkshire Hathaway). With the S&P 400 and 600 added
(MD_202610_004), companies can also be followed as they move between the indices,
which allows the missing-index (survivorship) problem to be simulated.

Scenario results test robustness and failure modes; they never prove an edge and are
never used to tune parameters. The judge stays the out-of-sample excess return on real
data (`midas/instructions/researcher.md`).

## Agreed approach

Three kinds of test, in the order of how much they depend on a market model. Built now:
the input tests and the survivorship experiment, the survivorship experiment first in
importance. The market scenarios are postponed (see Decisions).

1. **Input tests (no market model).** Change one input of real data and check that the
   model responds in the expected direction, and by how much. Examples: a lower price
   (higher earnings yield) must not lower the score; earnings cut by a fifth for one
   company must lower its rank; a filing delayed by a quarter must not change any
   earlier score (point-in-time); small noise in the inputs should change few picks.
   Measured by the share of picks that change (turnover) and the score's change per
   unit of input change. These are metamorphic tests: they need no prediction of
   prices.
2. **The survivorship experiment (no market model).** The S&P 400 and 600 hold many
   companies that left the S&P 500, and their change tables record the moves between
   the indices. On the S&P 400 and 600, run the model twice: once on today's members
   only (as the S&P 500 is run today), once with the companies known to have been
   members on each date. The difference measures how much the missing members flatter
   the result, and gives an estimate for the S&P 500. Companies that fell from the
   S&P 500 into the 400 can also be scored after they left, closing part of the
   S&P 500's gap directly.
3. **Market scenarios (needs a return model).** Change fundamentals and prices
   together, with links estimated from the data, not assumed:
   - an earnings change moves the price by the elasticity estimated from the panel
     (the return over the next year against the earnings change, per sector), plus a
     residual drawn from history;
   - market-wide scenarios by block bootstrap (whole past periods resampled for all
     companies at once, so the correlations between companies are kept) or by replaying
     a past episode (2008, 2020, 2022) on the current holdings;
   - the benchmark recomputed from its members: price times shares outstanding (the
     cover-page count from EDGAR), weighted by market capitalisation. The S&P 500 is
     weighted by free-float market capitalisation, not an average, so the rebuilt
     index is first checked against `^SP500TR` on real data; the gap measures both
     the float adjustment and the missing members. Berkshire Hathaway has no members
     to rebuild and is left out of scenario comparisons.

Where the code goes: the scenario engine (shocks, bootstrap, rebuilt index) in
`skatkis` as general machinery; the value-specific checks (expected direction per rule)
beside the value tool. `midas-wrath` is a hat of its own, modelled on the warden: it
runs the scenarios and reports, and writes no code.

## Decisions

- 2026-10-31 (user): `midas-wrath` is a hat of its own, like the warden: it runs the
  scenarios and reports, and writes no code.
- 2026-10-31 (user): the market scenarios are postponed until more reasonable ones
  can be simulated, from observed correlations and machine-learning models
  (`hand_o_midas`); when they come, both the block bootstrap and replayed episodes.
- 2026-10-31 (user): the survivorship experiment is important.
- 2026-10-31 (user): Berkshire Hathaway is left out of scenario comparisons.

## Kickoff prompt

Pick up S_202610_008. Read the agreed approach and decisions, then design the
survivorship experiment and the input tests on their own branch, once the S&P 400 and
600 data of MD_202610_004 are in.

## Done when

- The survivorship experiment runs on the S&P 400 and 600 and reports how much the
  missing members flatter the result, with the estimate carried to the S&P 500.
- The input tests run on any index of `config/indices.toml` and report how the picks
  and the score respond to each change.
- `midas-wrath` exists as a hat: in the hats table of `AGENTS.md`, its own
  instructions file and `.claude/agents/midas-wrath.md`.
- The market scenarios remain a follow-up change, raised when `hand_o_midas` can model
  the links between inputs and prices.
