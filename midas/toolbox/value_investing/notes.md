# value_investing notes

## Planning (preliminary)

### Objective (2026-10-31, open → S_202610_006)

A trading bot, not a risk model. The user holds mostly the S&P 500 and wants about 10%
of new investments in a value-investing sleeve that beats the S&P 500 total-return
index by about 2% a year; not doing worse than the index is the minimum (user,
2026-11-01). Trades are quarterly, across S&P 500 companies; the wallet tracks
combinations of strategies, judged out-of-sample on excess return over the index and on
how often and how far they trail it over rolling periods. Reporting as CPI + x% was
dropped (user, 2026-11-01). The API matters because
backtesting and optimising the metrics needs point-in-time fundamentals over many years.

- Hypothesis (user, 2026-11-01): the sleeve leads the index more often than it trails
  it, and is safer in a correction of overvalued stocks (an AI bubble), because the
  rules exclude expensive stocks by design. To test: the same exclusion makes value
  trail during a bubble's run-up (1998–2000, 2019–2021), so the rolling comparison
  should show both phases.

### Open owner decisions from the spec (2026-10-31, open)

From `docs/value_investing_bot_spec.md` §9; each blocks the changes noted.

1. Data: paid vendor budget, or EDGAR only? Decided 2026-10-31: free sources (EDGAR,
   Wikipedia or `fja05680`, `yfinance`); survivorship gap accepted for now
   → MD_202610_001
2. Bottom ten: sell/avoid, or short? → S_202610_006
3. Financials and REITs: exclude, or separate scoring? Decided 2026-10-31: REITs and
   financial companies excluded; bonds are outside this tool
   → MD_202610_001, VI_202610_002, MD_202610_002
4. Wallet interface: signals, target weights or orders? → S_202610_006
5. Reporting currency: USD or ZAR? Decided 2026-10-31: main analysis in USD; ZAR
   secondary, once refined and to track the actual record (FRED `DEXSFUS`)
   → MD_202610_001
6. DAG engine: stdlib `graphlib` registry or Hamilton/Dagster? Decided 2026-10-31:
   `graphlib` → MD_202610_001

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

### From buyability score to a quarterly selection (2026-10-31, → S_202611_003, VI_202611_001)

The user's plan after VI_202610_002, in order:

1. Look at the buyability score on a casual sample (ten best, ten worst) to establish
   the pattern.
2. At each quarterly update, select the most likely best buys for the next quarter
   from the most recent point-in-time score: one prediction per quarter.
3. Calibrate the portfolio parameters: diversification rules and the weight bought of
   each candidate.
4. Backtest the combinations of weighting specifications and keep the one with the best
   portfolio performance out-of-sample. The optimisation follows the pattern of the
   BTC VWAP work in `notebooks/optimise_strategies.ipynb` (high-frequency triggers
   there; one quarterly decision here). → S_202610_006

### Roadmap after closing the per-index branch (2026-11-01, open)

The user's order of work (2026-11-01):

1. **Close out** the branch `feature/value-investing/vi-202610-007-pick-price-history`
   (committed; the user pushes it and opens the pull request).
2. **Optimise Midas**: the agent set-up itself (instructions, index server S_202610_002,
   guard S_202610_004); scope to agree at the start.
3. **Unit tests**, high priority: VI_202610_009.
4. **Documentation**: the tool README and the spec departures, the `market_data` README
   and the skatkis reference (sections 4 of both warden reports); then VI_202610_002 to
   006 close.
5. **A first version (V1) of the pipeline**: what V1 contains is to agree; candidates are
   the two strategies (VI_202610_008), the survivorship experiment (S_202610_008) and
   the historical membership (MD_202610_002).

Open decisions carried over: the delisted-lot accounting (VI_202610_006, N2) and the
deferral of the DAG engine (spec §9.6).

### The process so far: from proof of concept to judging a model (2026-10-31, open)

The order followed so far, and the next stage the user set (2026-10-31):

1. **Data.** S&P 500 fundamentals from EDGAR (first-reported, keyed to the filing
   date), Yahoo prices and dividends, the membership from Wikipedia (MD_202610_001,
   MD_202610_002).
2. **Rules and score.** The Graham, Buffett and Munger rules scored and binned per
   company and date, weighted by pass rate and past success (VI_202610_002 to
   VI_202610_005).
3. **Proof of concept.** The value report (VI_202610_006): yearly picks bought and
   held, or sold when the score falls, against the S&P 500 and Berkshire Hathaway.
   The wallets beat the index, with the biases stated (today's members only,
   look-ahead in the tag choice), so the result is provisional.
4. **Next: the framework for judging a model**, before any parameter is tuned:
   - reliability on companies the model has not seen, from the same sources: the S&P
     400 and 600, then a rule-based US universe (MD_202610_004); other exchanges were
     surveyed and set aside (`midas/toolbox/market_data/notes.md`);
   - a look at what the model does: each pick's price with its buys and sales, and the
     USD traded (VI_202610_007). The first look showed the user picks they would not
     buy and losers held for years ("Picks with weak past returns", "The wallet keeps
     its losers" below).
5. **Later: optimisation** of the parameters (thresholds, weights, the sell rule),
   out-of-sample, as the researcher's workflow sets out (S_202610_006).

The rule behind step 4 is recorded in `midas/instructions/researcher.md` ("Judge the
model before tuning it").

### Follow-ups pencilled in after the warden's first run (2026-10-31, open)

Put off by the user as not urgent (2026-10-31); details in VI_202610_006, "Warden
report":

1. The tag-choice fix (point-in-time label choice): done 2026-11-01 (MD_202610_002);
   the two strict-xfail tests pass.
2. Historical S&P 500 members, to remove the survivorship bias (MD_202610_002).
3. JSE data and the Top 40 comparison: returns-only first, or a paid fundamentals
   source (MD_202610_003).
4. Whether the picks are available in the EasyEquities USD account (the user checks in
   the app).
5. The warden's should-fix items S3 to S9, S11 and S12: done 2026-11-01 (VI_202610_006,
   "Fixes after the report"); the follow-up run's findings are in VI_202610_007.
6. Documentation (tool README, spec departures, skatkis reference) and the unit tests
   the warden listed.
7. Report conventions the user set, to record as standing rules for the secretary:
   tables kept whole on one page where they fit, benchmarks thick and black (S&P 500
   dashed, Berkshire dash-dot), full company names in the concluding parts, a recent
   investor as the reference.

### Spec §1.3 docstring style (2026-10-31, → S_202610_003)

Spec §1.3 prescribes Google style; the user chose the numpydoc structure of
`.claude/rules/docstrings.md`. The spec is updated in S_202610_003. §1.4 also still
asks for `ruff format --check`, where the repository uses Black (warden, S_202611_001,
2026-11-01); a spec decision for the scribe.

## Open methodology questions

### EDGAR history is too short for the long Graham windows (2026-10-31, → VI_202610_002)

Found in the first live pull (MD_202610_001, Apple, Microsoft, Coca-Cola). XBRL
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

Decided 2026-10-31, recorded in VI_202610_002: dividend history from the price source,
the windows as explicit parameters (Graham's and Buffett's values by default), and
rules phased in until their window is full. G4's phase-in was built on 2026-11-01.

### Universe looks ahead: today's members scored in the past (2026-10-31, open)

Found in VI_202610_004. The buyability table scores today's S&P 500 members on every
past date, including companies that joined the index later because they grew (TSLA
added 2020-12-21, scored and shorted in January 2020). This flatters any long book and
punishes any short book: the equal-weighted wallet of every scored company beats
`^SP500TR` by +2.9% a year, and -0.4% once each date keeps only companies already added
(`date_added` from the constituents table). The filter fixes additions only; companies
removed from the index are still missing until the historical membership of
MD_202610_002 exists. A methodology change; to be approved by the user.

### The wallet keeps its losers (2026-10-31, open)

Seen in the first price-history figure (VI_202610_007). The investor who started in
2023 sold almost nothing in three years: two small trims of Intel (about 4 USD), while
Skyworks was bought six times in a row as its price fell from about 110 to 65 USD. The
sell rule looks only at the score (sell below a 0-to-1 score of 0.4), and a company
whose price falls while its fundamentals hold scores higher, not lower, so the rule
never fires on it. Whether a price-based exit, a holding limit or a cap on repeat
purchases helps is a hypothesis for the researcher, once the framework for judging a
model is in place (`researcher.md`, "Judge the model before tuning it"); the user
noted (2026-10-31) that the parameters can be optimised later.

### Picks with weak past returns: value traps (2026-10-31, open)

The user looked up the last five years of the 2026-10-30 picks (First Solar, Cognizant,
Snap-on, Skyworks, ResMed) and would not buy them. A cheap-on-fundamentals screen
favours companies whose price has fallen, and some fall for a reason (value traps).
A price-trend or momentum condition (for example: no pick whose 12-month return lags
the S&P 500 by more than a threshold) would address it, but it is a methodology
change: a new rule, to be tested as a hypothesis out-of-sample (the researcher, on
the broader universe of MD_202610_004), not set by looking at today's picks. The price
history in the report (VI_202610_007) shows the problem first.

### Tag choice looks ahead (2026-10-31, resolved 2026-11-01 in MD_202610_002)

Found by a truncation test (VI_202610_006): cutting the facts at a decision date t
(filed <= t), prices and dividends at t, and recomputing the scores gives the same
scores on 2025-12-19, but on 2018-12-21 2 of 296 companies differ by one point and on
2022-12-16 3 differ in the plain grades and 1 in a pass/fail flag (which moves every
rule weight by about 0.01); the top five picks are the same on all three dates. Cause:
`market_data.clean.select_tags` chooses each company's tag by its most recent use over
all filings to the snapshot date, so the choice between two first-reported values
depends on later filings (Martin Marietta's intangible assets: price to tangible assets
5.07 cut, 4.23 full, on 2018-12-21). Fix: choose the tag from filings dated on or before
each decision date, and keep the truncation test as a unit test over every decision
date. A change to point-in-time handling: methodology, for the user to approve.
Approved and built 2026-11-01: the tag is chosen from the filings made by each date; the
truncation test passes on all three dates.

Other forward information the scores still carry, stated with every result: the
universe is today's S&P 500 members (companies that left the index are missing; those
that joined later are kept out only before their `date_added`); sectors are today's
GICS; and the score's design was chosen while looking at 2016 to 2026.

### The TFSA cannot hold individual shares (2026-10-31, open)

EasyEquities' own pages (2026): a tax-free savings account may hold only collective
investment schemes (JSE-listed ETFs and actively managed ETFs, unit trusts, bundles)
and SA government bonds, not individual shares, JSE or US; US shares sit in the
separate USD account, which is not tax-free. So no pick of the value wallet can be
bought in the TFSA. Consistent with the earlier note that the TFSA is a comparison
only and does not constrain where the sleeve is held. Options for the user: hold the
sleeve in the EasyEquities USD account (taxable; currency conversion and the
offshore allowance apply), or apply the score to JSE shares in the ZAR account
(a JSE universe, a new change). Sources: easyequities.co.za/tfsa-investing,
support.easyequities.co.za article 13000085076.

## Data issues

### Companies without usable EDGAR history (2026-10-31, open)

Found in the first buyability runs (VI_202610_002). Only the Yahoo rules (G4, E4) are
evaluable for these, so with the mean score they reach 0 or 1 on two rules:

- EMR, TRMB (CIK 864749) and VYLR have no rows in the 2026-10-31 `facts` snapshot.
  Check whether the EDGAR fetch failed or the CIK in the constituents table is wrong.
- XOM maps to CIK 2115436, ExxonMobil Holdings Corp, a 2026 holding company with only
  10-Q filings; Exxon's history is under its former CIK and is not linked. A
  predecessor-CIK mapping is needed for reorganisations like this.
- HONA (Honeywell Aerospace, CIK 2089271) is a 2026 spin-off with no 10-K yet: no
  history, correctly.

### A. O. Smith FY2024 10-K missing (2026-10-31, open)

AOS (CIK 91142): the FY2024 annual values appear only as comparatives in the FY2025
10-K filed 2026-02-10; the FY2024 10-K itself (due about February 2025) is not in the
snapshot. From 2025-09 to 2026-02 the latest annual year on file is FY2023, older than
550 days, so its history is dropped and its score falls from 9 to 2. Check the
companyfacts payload for the missing accession.

### Missing tags make rules unevaluable (2026-10-31, open)

Companies without debt often report no `long_term_debt` tag, and many report no
`goodwill` or `intangible_assets`. With a missing input giving NaN (decided 2026-10-31),
G2b, E2, B1 and E6 are not evaluable for them and score 0, which penalises a debt-free
company. To decide once the data has been looked at: treat an unreported debt or
goodwill tag in a 10-K that reports the balance sheet as 0, or keep NaN.

### Thin coverage at both ends of the score (2026-10-31, resolved)

With the sum score, the lowest mean scores from 2016 were companies with short or
missing history (GEV, SOLV, HONA and FDXF spin-offs, GDDY and DELL listings): mean
`n_available` 2 to 8 against a median of 13.4. The mean score (user, 2026-10-31) moves
the problem to both ends: a company evaluable on two rules scores 0 or 1 (EMR and XOM
top the 2016 sample at 1.00). A minimum `n_available` before a company is ranked would
keep the sample to companies measured on most rules. Decided 2026-10-31: at least 5
evaluable rules (VI_202610_002).

