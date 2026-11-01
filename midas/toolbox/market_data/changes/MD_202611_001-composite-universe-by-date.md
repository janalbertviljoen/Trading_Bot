# MD_202611_001: Composite universe by date: tracking companies across the three indices

| Field | Value |
|---|---|
| Status | in discussion |
| Priority | medium |
| Kind | new capability (a larger refactor, postponed by the user) |
| Depends on | MD_202610_004 |
| Branch | not yet |
| Raised | 2026-11-01 |

## Description

Each index is scored on its own: a company is in an index's table only if it is a member of
that index today, from its `date_added`, or from the first date of the index's changes list
(2012 for the S&P 400, 2019 for the 600). The prices and the EDGAR facts are fetched for all
three indices together, so every company's data exists whichever index it is in. What is
missing is one membership by date across the three: a company that grew from the 600 to the
400 to the 500 is scored only in its current index, so the smaller indices lose their winners
and the 500 keeps them from the date they joined (212 symbols are in both the 500 and the 400
change histories; VI_202611_001, check 20). With one membership by date, a pick that moved
up, or dropped out of an index, keeps its performance and can be followed after the fact
(user, 2026-11-01: "test it in post production"), and a short on a company that moved down
could be tested in the same way.

## Approach

To be agreed with the user.

- A table of the index each company belonged to on each date, built from the three
  changes lists (the 500's from 1976, the 400's from 2012, the 600's from 2019) and today's
  members. The 400 and 600 lists give selected changes only (`indices.toml`), so their
  early history stays incomplete.
- One composite universe (the S&P 1500 on each date) and a per-company flag for the index it
  was in; the rules and the ranking run on it, and the benchmark is chosen per date or as one
  composite series.
- Companies that left all three indices and have no Yahoo prices (delisted, acquired,
  bankrupt) stay missing: the survivorship gap is reported, not closed (MD_202610_002).

## Open decisions (user)

- When to build it; the user postponed it as a larger refactor (2026-11-01).
- The benchmark of the composite universe.

## Decisions

- 2026-11-01 (user): tracking across indices is wanted; it is a larger refactor to
  postpone, after the current results.

## Kickoff prompt

Pick up MD_202611_001. Read how `index_universe` and `members_on_date` build one index's
membership, then propose the composite membership table before changing code.

## Done when

- The membership of every company on every date across the three indices is a table, a
  pick that moved index keeps its history, and the biases of the result are stated.
