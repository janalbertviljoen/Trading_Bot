# Accreditation register

The tests the value model must pass before launch, with the result of each run and where
the evidence is. The protocol they follow is `instructions/quant.md`, "Testing and
optimising a model"; the numbers are in
[VI_202611_001](toolbox/value_investing/changes/VI_202611_001-model-calibration.md). A test
that fails is kept, with its date, not removed. Code state of the runs: the working tree
on top of commit 01832d6, uncommitted; data as of 2026-10-31.

## Status

Not cleared. AT-1 passes on the S&P 500 and fails on the S&P 400 and 600; AT-4 and AT-6 are
open until the composite universe exists (MD_202611_001).

## Tests

| ID | Test | Pass when | Status |
|---|---|---|---|
| AT-1 | Heavy hitters removed (stress) | The value wallet still beats its benchmark, for every start year, with the 3 largest contributors removed from the universe and the rest regraded; the result with 5 removed is reported | S&P 500 pass; S&P 400 fail; S&P 600 fail |
| AT-2 | Company sampling bands | 12 stratified draws, each dropping 10% of the companies; the spread is reported and the wallet beats the benchmark in every draw | S&P 500 pass (12 of 12); 400 and 600: not in every draw |
| AT-3 | Unseen test companies, once | A frozen specification scored once on a seeded, stratified 10% never used; reported whatever it shows | S&P 500 done, positive, 99% from two companies |
| AT-4 | Independent indices | The frozen configuration beats random books of its own universe on the S&P 400 and 600 | Open: not reproduced; the universe is biased (migration) |
| AT-5 | Two periods | The edge appears in 2009 to 2016 and in 2017 to 2026 | S&P 500 pass |
| AT-6 | Survivorship | The gap from companies that left the index is measured and stated | Open (MD_202610_002, MD_202611_001) |
| AT-8 | Scenario tests (Midas's wrath, S_202610_008) | Input tests respond in the expected direction; the survivorship experiment measures what the missing members flatter | Open |
| AT-9 | Missed winners | The ex-post winners, the share the score picks and their score percentile are reported for every index, with what they scored low on | Done for the three indices (check 24); to enter the report |
| AT-7 | Reconcile with the real world | Results compared with market events and institutional logic, discrepancies named (`quant.md`, step 14) | Open |

## Rerun on the default of ten rules (2026-11-01)

AT-1 and AT-2 below were first run on the 16-rule score. On the ten default rules (G6, G2A, G7,
G2B, E2, NN, E1, G5, B1, E4; 10 picks):

- S&P 500, value wallet minus `^SP500TR` in points a year, deterministic run (draws mean, draws
  minimum), 2016, 2020, 2023 and 2025 starts. None removed: 9.6 (8.9, 6.0), 13.0 (11.8, 4.9),
  16.2 (15.0, 6.7), 11.3 (14.0, 8.4). Largest contributors: MU 24% of the 2016 investor's gain, LRCX
  16%, AMAT 7%, NVDA 6%, NUE 5%. Without AMAT, LRCX and MU: 3.7 (5.0, 1.3), 1.9 (3.0, 1.3), 2.1 (2.4,
  0.9), 8.9 (9.2, 3.1): **pass**. Also without NUE and NVDA: 2.3, 0.3, 1.2 and 3.5 (minimum -1.6).
- S&P 500 sampling (AT-2): the wallet beats the index in 12 of 12 draws for every start year; return
  a year 24.2% (standard deviation 1.5), 27.7% (3.2), 33.7% (3.9) and 29.6% (3.9), against 15.3%,
  15.9%, 18.7% and 15.7%; a pick that stays in the universe is picked again in 97% of the draws.
- S&P 400, against IJH: none removed 1.8, 3.1, 4.6 and 20.9; without AVT, CMC and RS 0.4, -0.4, -0.1
  and 13.6; also without ITT and SNX -0.7, -0.3, 0.0 and 14.2: **fail** for the 2016 to 2023 starts.
- S&P 600, against IJR (the 2016 and 2020 starts coincide, membership dates begin in 2019): none
  removed 6.3, 6.3, -5.6 and -3.5; without BHE, GVA and SIG 6.4, 6.4, -7.6 and -2.1: **fail**, the
  2023 and 2025 starts trail the fund with or without removal.
- Not yet rerun on ten rules: the unseen test companies (AT-3) and the two periods (AT-5), both run on the
  strict scores of the 16-rule table (VI_202611_001, checks 12 and 19).

## AT-1, heavy hitters removed (first run, 16-rule score)

Run 2026-11-01. The largest contributors to the 2016 investor's gain are removed from the
universe as if they had never existed, the rest is regraded from scratch, and the report's
wallet (10 picks, score-weighted, bought each January and July, never sold) is run again;
12 stratified 10% draws sit on top of each case. Value wallet minus the benchmark, points a
year, deterministic run (draws mean, draws minimum).

| Index | Removed | 2016 start | 2020 start | 2023 start | 2025 start |
|---|---|---|---|---|---|
| S&P 500, against `^SP500TR` | none | 14.4 (13.1, 7.2) | 13.4 (12.3, 5.3) | 16.3 (14.9, 6.2) | 15.8 (17.9, 12.4) |
| | NVDA, MU, LRCX | 4.6 (4.5, 2.9) | 3.9 (4.0, 2.5) | 4.7 (4.4, 1.5) | 15.8 (19.1, 12.4) |
| | plus AMAT, NUE | 2.8 (3.2, 1.9) | 0.1 (1.4, 0.0) | 1.7 (2.7, 0.8) | 13.9 (14.6, 10.6) |
| S&P 400, against IJH | none | 2.2 (2.4, 0.9) | 3.5 (3.2, 1.3) | 2.2 (2.0, -1.1) | 11.6 (11.2, 4.7) |
| | AVT, CMC, RS | 0.3 (0.2, -0.7) | -0.6 (-0.1, -1.0) | -3.0 (-2.3, -6.0) | 3.0 (4.2, 0.2) |
| | plus ITT, SNX | -1.5 (-1.0, -1.9) | -1.6 (-0.7, -1.8) | -4.0 (-2.5, -5.1) | 2.9 (4.0, -0.1) |
| S&P 600, against IJR | none | 2.0 (2.5, 1.6) | 2.0 (2.5, 1.6) | -1.6 (-1.5, -3.9) | 4.3 (3.5, 0.7) |
| | BHE, GVA, SIG | -0.2 (0.1, -1.5) | -0.2 (0.1, -1.5) | -4.2 (-4.1, -6.9) | 0.2 (0.2, -6.3) |
| | plus GFF, MTRN | -1.6 (-0.5, -2.6) | -1.6 (-0.5, -2.6) | -3.6 (-4.0, -6.6) | 2.5 (-0.3, -6.3) |

Reading. On the S&P 500 the wallet beats the index with its three largest winners removed,
by about 4 points a year for the 2016 to 2023 investors; with five removed the margin is 0 to
3 points (0.1 for the 2020 investor). The three largest are Nvidia, Micron and Lam Research
(60% of the 2016 investor's gain). On the S&P 400 the wallet's edge over its fund is 2 to 4
points to begin with and is gone with the three largest removed; on the S&P 600 it is 2
points or less and is gone too. The 2016 and 2020 starts coincide on the 600, whose
membership dates begin in 2019. The 400 and 600 tables are biased by membership look-ahead and
by winners that graduated to a larger index (`value_investing/notes.md`, "Index migration"),
so their failure is not yet a verdict on the rules.

Placed on the grades of the other companies, a removed company is ranked in the top 10 on
18% to 51% of the dates on the 500 (NVDA 12 of 67, MU 21 of 60, LRCX 24 of 57, AMAT 30 of
68, NUE 34 of 67), 10 to 52 of 57 to 59 dates for the 400's three largest and 3 to 18 of 26
to 28 for the 600's.
