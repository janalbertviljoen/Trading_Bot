# Midas, the quant

What the developer and the researcher both need to know when they touch market or
company data: the sources, their traps and the rules that keep results honest. "Quant"
is not a separate hat; it is the knowledge the two hats share. It adds to
[AGENTS.md](AGENTS.md); where they differ, `AGENTS.md` wins. Read it with
[developer.md](developer.md) or [researcher.md](researcher.md) before working with data,
metrics, rankings, strategies or the wallet.

The first part, the data rules, names nothing internal, so that it can be served to
members' Claude with the research protocol in `researcher.md`. The second part is where
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
- **Data issues** go to the secretary as a note in
  `midas/toolbox/value_investing/notes.md`.
