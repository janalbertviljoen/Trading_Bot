# Midas, the Quant

What the Developer and the Researcher both need to know when they touch market or
company data: the sources, their traps and the rules that keep results honest. "Quant"
is not a separate hat; it is the knowledge the two hats share. It adds to
[AGENTS.md](AGENTS.md); where they differ, `AGENTS.md` wins. Read it with
[developer.md](developer.md) or [researcher.md](researcher.md) before working with data,
metrics, rankings, strategies or the wallet.

The first two parts, the data rules and the protocol for testing and optimising a
model, name nothing internal, so that they can be served to members' Claude with the
research protocol in `researcher.md`, and shown to an accreditor. The last part is where
those rules live in this repository.

## Data rules

### The sources

| Source | Answers | Never use it for |
|---|---|---|
| SEC EDGAR | What a company reported, and when it was filed | Prices: EDGAR has none |
| Yahoo Finance | Daily prices, dividends, splits, `^SP500TR` | Fundamentals: Yahoo's are restated and short |
| FRED | USD/ZAR (`DEXSFUS`) | Anything company-specific |

### Working rules

1. **Point-in-time first.** A rebalance on date $t$ uses only EDGAR rows with
   $\text{filed} \le t$, and of those the value first reported for each period. A
   result that used a later filing, SEC's `frame`, or today's figures is invalid, not
   just less accurate.
2. **A value's period is `period_start` to `period_end`.** EDGAR's `fiscal_year` and
   `fiscal_period` belong to the filing that reported it. A field follows the tag the
   company used most recently by the date, chosen from the filings made by then; later
   filings never decide it.
3. **Never divide an adjusted price by an as-reported figure.** Yahoo prices and
   dividends are split-adjusted as of the fetch date; EDGAR per-share figures are not.
   Undo the split adjustment with the splits table, or compare company totals with
   market capitalisation.
4. **Returns come from `adj_close`**, the split- and dividend-adjusted series, and the
   benchmark is a total-return series per index (`^SP500TR` for the S&P 500; the funds
   IJH and IJR, net of their fee, for the S&P 400 and 600). Comparing a total return
   with a price return flatters the strategy by the dividend yield.
5. **USD is the main currency.** ZAR is secondary, converted with `DEXSFUS`; report
   the as-of date of the rate used.
6. **State the known biases with every result.** Delisted companies are missing from
   Yahoo (survivorship bias), and EDGAR starts in 2009, so long-window rules phase in
   late. A result that hides either overstates what the strategy would have done.
7. **Pin the snapshots.** Every result names the as-of date of each source snapshot it
   used; a new pull is a new as-of date, never an overwrite.
8. **Judging a strategy** follows the research protocol in `researcher.md`
   (out-of-sample, against the benchmark of rule 4).

### When something does not fit

A tag that is missing, a unit that does not match, a gap in prices or a series that
looks wrong is reported with the company, field and date. A missing input gives NaN;
nothing is filled, interpolated or substituted.

## Testing and optimising a model

A result counts when its design was fixed before it was seen, it was measured on companies
the model never used, and it is reported with its uncertainty and the number of things
tried. Anything less is a diagnostic. Each step below is evidence an auditor can ask for;
record it in the change file.

1. **Fix the design before the run.** Write down the rules, parameters, horizons and the
   measure, then run. Anything changed after seeing a result is a new design and is
   counted.
2. **Count what was tried.** Every parameter set, horizon, gate and weighting is counted
   and reported beside the result. A modification that failed is kept as *tried and not
   used*, with its design, and is re-tested when a better optimiser exists.
3. **The test companies are never seen.** A fixed 10% of the companies, stratified by
   market capitalisation and seeded, is held out. Nothing computed from them enters a
   fit, a grade, a rank, a rule weight, a cut-off or the choice of rules or parameters:
   grades and weights come from the training companies alone, and a test company is
   placed on the training distribution of its date. A column built from all companies is
   never compared on the test sample. A split that informed a choice is spent and a
   fresh one is drawn. The test sample is scored once, on a frozen specification, and
   reported whatever it shows.
4. **Walk forward.** On a date, a fit uses only outcomes already known: return windows
   that had closed, and fundamentals as in rule 1 of the data rules.
5. **Compare with a baseline of the same size.** The strategy's excess return is set
   against random books of the same number of names drawn from the same pool on the same
   dates, and a changed configuration against the current one on the same dates (a paired
   difference). A change is adopted only if the band of the paired difference excludes
   zero; otherwise the current configuration stays.
6. **One reference horizon.** The reference is the holding period of the investment
   philosophy (3 years for the value tool). Shorter horizons are diagnostics. A longer
   one is reported with its number of independent windows, $\text{span} / \text{horizon}$,
   and a warning; with fewer than about five, its band is not meaningful. Check the
   horizon of any stored return column before using it.
7. **Total returns.** Dividends reinvested, against a total-return benchmark (data rule
   4).
8. **Two kinds of uncertainty, stated apart.** *Time*: a moving-block bootstrap over the
   decision dates, blocks of one holding window, on the paired difference; this is the
   band used while calibrating and optimising. *Companies*: once the specification is
   chosen, ten draws that each drop a stratified 10% of the companies; the spread of the
   final values is the band, with a table of how often each pick survives the draws.
   Neither covers the choice of specification or multiple testing, which the report says.
9. **State the concentration.** The share of the positive excess that comes from the five
   largest contributors, their names, and the result with the largest 1, 3 and 5 of them
   removed from the universe as if they had never existed (the universe is regraded without
   them). That is the stress band, reported beside the sampling band. A concentrated book
   is acceptable when it is stated.
10. **An independent check.** The chosen specification is scored once on indices not used
    in calibration, with the same measures, and the biases of their data stated.
11. **Search small by default.** The general run searches only the parameters a sampled
    study found sensitive, with the number of picks fixed per tier. Tune on the most
    powerful objective that still reflects the strategy (a quintile spread uses far more
    of the data than the top ten), and report the traded book beside it. A large input space
    and a fitted response surface belong to a separate, resource-heavy tier.
12. **Direction before data.** A methodology change starts from the philosophy it follows
    (the investor's own writing), recorded in the change file, before the data is mined.
13. **State the known biases with every result:** missing delisted companies, the short
    data history (a single long bull market), a calibration that favours recent years,
    and any look-ahead in the universe.
14. **Reconcile with the real world.** Before a result is reported, compare it with what
    happened in the markets in that period and with how institutions would reason about it
    (the philosophy's own writing, the investor's letters, the regulator's and the fund
    manager's view). Name each discrepancy: a result a market event or an institutional
    rule would not predict is investigated before it is believed, and an explanation from
    events is a hypothesis to test, not a finding. Record the events, the logic and the
    discrepancies in the change file.
15. **Probe, ask the investor, test, repeat.** Every result is probed, not accepted. When
    something looks fishy (too good, different between indices, against institutional
    logic), run the probes: the heavy hitters removed, the two periods, the random-book
    baseline, the independent indices, the dispersion, who was missed. Then the Quant
    puts the question to the investor: what does the philosophy's own writing say about
    this cause, and what fix does it suggest? The investor answers from the letters, with
    the year, never from the data. The Quant turns the answer into a design fixed before
    the run, counted, and tests it; the result goes back to the investor for another
    reading. The loop stops when the cause is explained, or is parked with its open
    question. Each turn is written in the change file as *quant* and *investor* entries,
    so an auditor can follow the conversation. Analysis first: nothing a probe suggests is
    adopted without step 5.

## In this repository

- **Source reference:** the shared data package's README, "Data sources"
  ([Toolbox/market_data/README.md](../../Toolbox/market_data/README.md)): what each
  source provides, its history, adjustments, gaps and access. Read it before using a
  table. The source modules are `market_data/sources/edgar.py`, `prices.py` and
  `fred.py`.
- **Rule 2** is implemented in `market_data.fundamentals.annual_as_of`; **rule 3**'s
  adjustments are described in the README, "Yahoo Finance"; **rule 4**'s benchmark is
  set per index in `config/indices.toml`.
- **Commands, not scripts.** Data is updated and checked with
  `python -m Toolbox.market_data update` and `check`; plots go to `results/`.
- **ZAR** is for the refined analysis and the user's actual record. The NaN rule was
  decided by the user on 2026-10-31; any other treatment of a gap waits for the user.
- **Data issues** go to the Secretary as a note in
  `midas/toolbox/value_investing/notes.md`.
- **The protocol above** was decided step by step in VI_202611_001 and S_202611_003 (the
  decisions and the checks, with the numbers); `researcher.md`, point 3, is its short form.
  The strict scores (training-only grades, test placement) and the time bootstrap are
  scripts until S_202611_003 builds them into the engine.
