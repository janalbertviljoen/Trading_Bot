# VI_202610_008: Two strategies: the conservative picker and the unicorn hunter

| Field | Value |
|---|---|
| Status | in discussion |
| Priority | high |
| Kind | methodology |
| Depends on | MD_202610_004, S_202610_008 (survivorship experiment); VI_202610_002 to 006 closed |
| Raised | 2026-11-01 |

## Description

The user wants two value strategies side by side instead of one report per index
(2026-11-01):

- **Conservative.** The top 5 of the S&P 500, decided twice a year (January and July),
  as today. It tracks the companies that drop out of the S&P 500.
- **Unicorn hunter.** The top 20 across every company on file: the S&P 500 and the
  companies of the S&P 400 and 600 stitched together into one universe (about 1,100
  outside Financials and Real Estate), decided and recalibrated every quarter. Still
  value investing, with a wider net: underdogs with the potential to climb. The aim is
  to pick a winner more than half of the time and to raise the chance of catching an
  outsized winner. The user also considered 10 picks.

The stitched universe brings in the index problem at its bottom: companies that fall
out of the S&P 600 leave the data altogether (survivorship), which the survivorship
experiment of S_202610_008 measures.

Each pick should also show where it stood in the index when it was first bought and
where it stands today (its rank by market value), to see whether the underdogs climb.

## Approach (proposal)

- **Universe.** An index entry in `config/indices.toml` that combines others
  (`us1500 = sp500 + sp400 + sp600`); a company's join date is its first date in any of
  them; it leaves when it leaves all three.
- **Strategies as config.** Each strategy names its universe, number of picks,
  decision months and calibration schedule (`config/strategies.toml` or a `[strategy.*]`
  table in `config/rules.toml`); the report runs per strategy.
- **Calibration every quarter** for the unicorn hunter: the rule weights (pass-rate
  weights) and the picks recomputed on each quarterly rebalance date; twice a year for
  the conservative picker.
- **Hit rate as a target.** Each report states the share of picks that beat the
  benchmark since purchase, against the 50% the user expects.
- **Rank in the index.** Market value = price times shares outstanding (EDGAR cover
  page, as filed by the date); a company's rank in its universe on its first purchase
  and today, with the change.

## Open decisions (user)

- The unicorn hunter: 10 or 20 picks?
- "Quarterly calibration": the picks every quarter, the rule weights every quarter, or
  both?
- The benchmark of the stitched universe: the S&P 1500 has no free total-return series
  on Yahoo; the S&P 500 (`^SP500TR`), or a mix of the three funds weighted by their
  share of the universe's market value?
- Do the per-index reports of MD_202610_004 stay beside the two strategies?

## Kickoff prompt

Pick up VI_202610_008. Read the description and the open decisions, agree them with
the user, then design the strategy configuration.

## Done when

- Both strategies run from config and each writes its report, with the hit rate, the
  drop-outs and each pick's rank in its universe at purchase and today.
