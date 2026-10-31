# market_data notes

## Planning (preliminary)

### Snapshot identity ignores request settings (2026-10-31, open)

A snapshot is keyed by source name, query and as-of date. A changed request setting
(the start date in `config/market_data.toml`) with the same as-of date silently reuses
the stored snapshot. Today the rule is to take a new as-of date; recording the request
settings in the manifest and refusing a mismatch would make it safe.

## Data issues

### Yahoo prices and dividends are split-adjusted as of the fetch date (2026-10-31, open → MD_202610_002)

Yahoo's `close` and dividend amounts are adjusted for every split up to the fetch date;
`adj_close` is also adjusted for dividends. Checked on Coca-Cola's 2:1 split of
2012-08-13: `close` is about 39.4 on both sides (about 78 actually traded before), and
2012 dividends show 0.255 (0.51 paid per pre-split share). EDGAR per-share figures are
as reported, so price ratios (P/E, P/B, Graham number) need the split adjustment undone
first, from the splits table, or must be computed from market capitalisation and
company totals instead.

### Companies switch tags over time: choose the tag per period (2026-10-31, resolved)

Found in the first data checks. Choosing one tag per company for all years loses most
of the history: Apple reported revenue as `SalesRevenueNet`, then `Revenues`, then the
ASC 606 tag. The best available tag (lowest `tag_priority`) must be chosen per company,
field and period; with that, revenue is complete for all three companies from 2007.
Resolved 2026-10-31 (user): the company's most recently used tag is canonical
(`market_data.clean.select_tags`).

### Per-share comparatives are restated for splits (2026-10-31, open → MD_202610_002)

Apple's diluted EPS for 2011–2013 is 27, 44 and 40 as first reported, before the 7:1
split of 2014; later 10-Ks repeat those years split-adjusted (about 6.3 and 5.6), and
the 4:1 split of 2020 does the same for 2018–2019 (11.9 and 2.97). So the first-reported
value of a per-share figure is on its own pre-split basis, while comparatives move to
the later basis. Rules comparing per-share values across years (G3, G5, E5) need them on
one basis, from the Yahoo splits table, applied point-in-time (only splits before the
rebalance date).

### Share count reported on two dates per 10-K (2026-10-31, open → MD_202610_002)

`shares_outstanding` comes from the cover page (`dei:EntityCommonStockSharesOutstanding`,
dated a few days before filing) and from the balance sheet
(`us-gaap:CommonStockSharesOutstanding`, dated at the period end), so a 10-K can give two
share counts with different dates. The point-in-time stage must pick one convention
(the cover-page count is the latest known at filing).

