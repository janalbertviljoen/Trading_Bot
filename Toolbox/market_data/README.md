# market_data

Shared market data for every tool in this repository (`value_investing`, `hand_o_midas`):
the sources, the cleaned standard tables and the shared metrics, each owned once. Tools
read from here; they never fetch or clean data themselves. The package sits between
`skatkis` (domain-free infrastructure) and the tools.

## Data sources


Three free sources feed every tool, each through one module in `sources/`, and each
answers a different question. **EDGAR** says what a company reported and when;
**Yahoo** says what its shares traded at and what they paid out; **FRED** says what a
dollar was worth in rand. None of them can stand in for another: EDGAR has no market
prices, Yahoo's fundamentals are restated and short, and FRED has no company data.

| | SEC EDGAR | Yahoo Finance | FRED |
|---|---|---|---|
| Module | `sources/edgar.py`, `sources/tag_map.py` | `sources/prices.py` | `sources/fred.py` |
| Provides | Reported financial-statement values, per-share figures, share counts | Daily prices, dividends, splits; the S&P 500 total-return index `^SP500TR` | Daily USD/ZAR exchange rate (`DEXSFUS`) |
| Key | CIK (10-digit company number) | Ticker symbol in Yahoo's convention (`BRK-B`) | Series ID |
| Requested | Full documents (the API has no date filter); used from 2006 | Daily prices from 2006; dividends and splits from 1986 | From 2006 (the series starts 1980-01-02) |
| History | Filings from 2009; periods back to 2006–2007 | From listing; `^SP500TR` from 1988 | From 1980-01-02 |
| Point-in-time | Yes: every value carries its filing date and accession number | No: adjusted as of the day fetched | Yes: rates are not revised |
| Adjustments | None: values as reported | Prices and dividends split-adjusted; `adj_close` also dividend-adjusted | None |
| Gaps | No prices; tags differ between companies and over time | Delisted companies missing (survivorship bias); unofficial endpoint | US holidays have no rate; published weekly, about a week behind |
| Access | Free; a User-Agent with name and e-mail; at most 10 requests per second | Free; unofficial; kept to 1 request per second | Free; optional API key, otherwise the keyless CSV download |

Every source is fetched through `skatkis.io.api.ApiClient` and stored once per as-of
date as an immutable raw snapshot under `data/raw/<source>/<as_of>/`, in the vendor's
own format with a manifest of content hashes. Parsed tables are written as Parquet under
`data/interim/<source>/<as_of>/`. Request settings are in `config/market_data.toml`. A result is only reproducible together with the as-of
date of each source it used.

### SEC EDGAR: what was reported, and when

`EdgarCompanyFacts` fetches a company's `companyfacts` document: every value the
company has tagged in XBRL in any filing (10-K, 10-Q, their amendments, and some
others). `parse_companyfacts` keeps the tags listed in `tag_map.TAG_MAP` and returns one
row per reported value.

- **No market prices.** EDGAR holds what companies file. The only market figure is
  `dei:EntityPublicFloat`, the market value of shares held by non-affiliates on one day
  a year (the end of the second fiscal quarter), which is too coarse for pricing.
- **One row per value per filing.** The same period is reported again in later filings
  (as a comparative figure, or restated). Each repeat is a further row with its own
  `filed` date and `accession`; none is removed. The value first reported for a period
  is the one with the earliest `filed` date.
- **`filed` is the point-in-time key.** A rebalance on date $t$ may only use rows with
  $\text{filed} \le t$. Typical lags after period end: 40–45 days for a 10-Q, 60–90 days
  for a 10-K.
- **`fiscal_year` and `fiscal_period` belong to the filing, not to the value.** A 10-K
  for fiscal 2023 reports 2022 and 2021 as comparatives, all with `fiscal_year` 2023.
  The period of a value is given by `period_start` and `period_end`.
- **Flows and stocks.** Income and cash-flow values (flows) have a `period_start`;
  balance-sheet values (stocks) are instants and have none. A 10-Q reports the quarter
  and the year to date (six or nine months); there is no separate fourth quarter, so it
  follows from the annual and nine-month values,
  $$Q_4 = V_{\text{FY}} - V_{\text{9M}},$$
  where $V_{\text{FY}}$ is the fiscal-year value and $V_{\text{9M}}$ the nine-month
  year-to-date value of the same fiscal year.
- **`frame` is not point-in-time.** SEC's `frame` marks the one value it deems current
  for a calendar period, which is the latest filed, often a restatement.
- **Per-share figures are as reported.** EPS and dividends per share are not adjusted
  for later splits: Coca-Cola's EPS before its 2012 split stays on the pre-split basis.
- **Tags vary.** The same concept appears under different us-gaap tags between
  companies and over time; `TAG_MAP` lists them per field in priority order, and
  `tag_priority` and `tag` show which one a value came from. A lower-priority tag can be
  narrower than its field (`Depreciation` for `depreciation_amortization`). Not every
  company reports total liabilities; `liabilities_and_equity` allows them to be derived.
- **Units.** Monetary fields are in USD, per-share fields in USD per share, share counts
  in shares. A mapped tag reported only in another unit (a foreign-currency filer) stops
  the parse with an error naming the company and tag.
- **History.** XBRL filing began in 2009, so with comparative figures values reach back
  to periods ending in 2006–2007, and nothing earlier. Rules with long look-back windows
  (10, 13 or 20 years) can only be evaluated from later rebalance dates.

### Yahoo Finance: what the shares traded at and paid out

`YahooChart` fetches a symbol's history from Yahoo's chart endpoint (the endpoint the
`yfinance` library uses); `parse_chart` returns three tables: prices, dividends and
splits. Daily bars from 2006 give the prices. Monthly bars from 1986 give the dividends
and splits (events keep their own dates), which keeps the request small.

- **Adjusted as of the fetch date.** `close` and dividend amounts are adjusted for every
  split up to the day the data was fetched; `adj_close` is adjusted for splits and
  dividends, so it is a total-return series. Coca-Cola's close around its 2:1 split on
  2012-08-13 shows about 39.4 on both sides, although it traded at about 78 before.
- **Returns from `adj_close`.** The return over a holding period, dividends reinvested,
  is the ratio of `adj_close` at its end and start.
- **Price ratios need the traded price.** Comparing a price with an EDGAR per-share
  figure (P/E, P/B, the Graham number) needs the price on the same basis as the filing:
  the traded price, recovered from the splits table as
  $$P^{\text{traded}}_t = P^{\text{close}}_t \times \prod_{s \in S_t} \frac{n_s}{d_s},$$
  where $S_t$ is the set of splits after day $t$ up to the fetch date, and split $s$
  gives $n_s$ new shares for $d_s$ old (`numerator`, `denominator`). Alternatively,
  compare company totals with market capitalisation.
- **Dividend history.** Dividends are requested from 1986, twenty years before the
  analysis period and longer than EDGAR reaches, so the 20-year dividend record can be
  evaluated from 2006. Only whether a dividend was paid in a
  year matters there, not its split-adjusted amount.
- **The benchmark.** `^SP500TR`, the S&P 500 total-return index, from 1988.
- **Survivorship bias.** Companies that were delisted (taken over, bankrupt) are not
  available, so a backtest over historical members sees only the survivors. Symbols can
  also be reused by another company later.
- **Unofficial.** The endpoint has no published terms or stability; requests are kept
  slow and every response is stored, so a pull is not repeated.

### FRED: what a dollar was worth in rand

`FredSeries` fetches a series from FRED, the Federal Reserve Bank of St. Louis's
database; `parse_fred` returns one row per day.

- **`DEXSFUS`** is the number of South African rand per US dollar, the Federal Reserve's
  H.10 noon buying rate in New York, daily from 1980-01-02; requested from 2006.
- **Not revised**, so the series as fetched is the series as it was known.
- **Missing days.** US holidays keep their row with a missing value.
- **Published weekly**, so the latest rates lag by about a week.
- **Access.** With `FRED_API_KEY` set, the official API (JSON); without it, the CSV
  download behind FRED's charts. Both give the same table.
- The tool fetches and stores the rate; it does not convert amounts between currencies
  yet.

### How the sources join

- **Company identity.** EDGAR identifies a company by CIK, Yahoo by ticker. Tickers
  change (a rename) and are reused, so the link between them belongs to a date: it
  comes from the historical S&P 500 membership, which is not built yet.
- **Dates.** EDGAR rows are dated by filing date and period end, Yahoo rows by trading
  day on the exchange, FRED rows by US business day. Each needs aligning to the
  rebalance date on its own terms.
- **Currency.** EDGAR values and Yahoo prices of S&P 500 members are in USD (the
  `currency` column of the price tables says so); `DEXSFUS` converts USD to ZAR.

## Updating the data

One terminal command brings the stored data up to a date; run it from the repository
root:

```
export SEC_USER_AGENT="Your Name your@email"   # required for EDGAR
export FRED_API_KEY=...                       # optional
.venv/bin/python -m Toolbox.market_data update --as-of $(date +%F) \
    --cik 320193 789019 21344 --ticker AAPL MSFT KO
```

- `--as-of` is always given, so a run never depends on the clock and can be repeated.
- `--source edgar prices fx` limits the run to some sources (default: all three).
- **EDGAR is incremental.** Each company's filing index (`submissions`) is checked
  first; its `companyfacts` are fetched again only if it has filed a 10-K or 10-Q on or
  after the date of its latest stored snapshot. Otherwise that snapshot is reused, and
  the `snapshot_as_of` column of the `facts` table says which one each row came from.
- Prices, dividends, splits and the exchange rate are small and fetched in full for
  each as-of date.
- Tables are written to `data/interim/<source>/<as_of>/` and the summary printed lists,
  per source, what was fetched, what was reused and the rows written.
- Until the historical S&P 500 membership is built, the companies (`--cik`) and symbols
  (`--ticker`) are given on the command line; `^SP500TR` is always added.
- The command can be scheduled (macOS `launchd` or `cron`), for example weekly or after
  each earnings season.
