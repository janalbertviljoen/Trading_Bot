# Midas, the quant

What the developer and the researcher both need to know when they touch market or
company data: the sources, their traps and the rules that keep results honest. "Quant"
is not a separate hat; it is the knowledge the two hats share. It adds to
[AGENTS.md](AGENTS.md); where they differ, `AGENTS.md` wins. Read it with
[developer.md](developer.md) or [researcher.md](researcher.md) before working with data,
metrics, rankings, strategies or the wallet.

## The sources

The reference is the shared data package's README, "Data sources"
([Toolbox/market_data/README.md](../../Toolbox/market_data/README.md)): what each
source provides, its history, adjustments, gaps and access. Read it before using a
table. In one line each:

| Source | Answers | Never use it for |
|---|---|---|
| SEC EDGAR (`market_data/sources/edgar.py`) | What a company reported, and when it was filed | Prices: EDGAR has none |
| Yahoo Finance (`market_data/sources/prices.py`) | Daily prices, dividends, splits, `^SP500TR` | Fundamentals: Yahoo's are restated and short |
| FRED (`market_data/sources/fred.py`) | USD/ZAR (`DEXSFUS`) | Anything company-specific |

## Working rules

1. **Point-in-time first.** A rebalance on date $t$ uses only EDGAR rows with
   $\text{filed} \le t$, and of those the value first reported for each period. A
   result that used a later filing, SEC's `frame`, or today's figures is invalid, not
   just less accurate.
2. **A value's period is `period_start` to `period_end`.** EDGAR's `fiscal_year` and
   `fiscal_period` belong to the filing that reported it. A field follows the tag the
   company used most recently by the date, chosen from the filings made by then
   (`market_data.fundamentals.annual_as_of`); later filings never decide it.
3. **Never divide an adjusted price by an as-reported figure.** Yahoo prices and
   dividends are split-adjusted as of the fetch date; EDGAR per-share figures are not.
   Undo the split adjustment with the splits table, or compare company totals with
   market capitalisation (`market_data` README, "Yahoo Finance").
4. **Returns come from `adj_close`**, the split- and dividend-adjusted series, and the
   benchmark is a total-return series set per index in `config/indices.toml`
   (`^SP500TR` for the S&P 500; the funds IJH and IJR, net of their fee, for the S&P 400
   and 600). Comparing a total return with a price return flatters the strategy by the
   dividend yield.
5. **USD is the main currency.** ZAR is secondary, for the refined analysis and the
   actual record, converted with `DEXSFUS`; report the as-of date of the rate used.
6. **State the known biases with every result.** Delisted companies are missing from
   Yahoo (survivorship bias), and EDGAR starts in 2009, so long-window rules phase in
   late. A result that hides either overstates what the strategy would have done.
7. **Commands, not scripts.** Data is updated and checked with
   `python -m Toolbox.market_data update` and `check`; plots go to `results/`.
8. **Pin the snapshots.** Every result names the as-of date of each source snapshot it
   used; a new pull is a new as-of date, never an overwrite.
9. **Out-of-sample or it did not happen.** Every strategy claim is judged on
   out-of-sample excess return over the index's benchmark (`researcher.md`); in-sample
   results are diagnostics only.

## When something does not fit

A tag that is missing, a unit that does not match, a gap in prices or a series that
looks wrong goes to the secretary as a note in `midas/toolbox/value_investing/notes.md`
with the company, field and date. Do not fill, interpolate or substitute values until
the user has decided how (decided 2026-10-31: a missing input gives NaN).
