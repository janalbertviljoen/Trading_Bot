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
