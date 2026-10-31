# VI_202610_005: Rule weights from past return correlation, NI1 and the universe on the date

| Field | Value |
|---|---|
| Status | in progress |
| Priority | high |
| Kind | methodology |
| Depends on | VI_202610_004 |
| Branch | `feature/value-investing/vi-202610-002-rule-catalogue` |
| Raised | 2026-10-31 |

## Description

The rule weight of VI_202610_004 (1 - pass rate) measures how hard a rule is, not
whether it pays. The user wants rules weighted by how well they have predicted company
growth: a rule many companies pass still weighs high if passing it went with better
returns (P/E), and a rule that does not go with returns weighs little (debt ratios of
companies that perform well anyway). Net profit growth is added as a rule. Because the
weights are fitted to returns, the scores now learn from prices, so the universe must
not look ahead: each date keeps only companies already in the S&P 500, as a switch
beside today's members.

## Agreed approach

- **Universe.** Two views, both written and compared: `current` (today's members on
  every date, as before) and `on_date` (only companies whose `date_added` is on or
  before the date). Companies removed from the index stay missing in both until the
  historical membership of MD_202610_002 exists. A methodology change, approved by the
  user (2026-10-31).
- **NI1, net income growth.** g = (NI(lag 0) - NI(lag 1)) / |NI(lag 1)|, annual
  first-reported, point-in-time; passes when NI grew; graded in buckets like the other
  rules, higher better. Family `N`.
- **Correlation weights.** For each rule and rebalance date s, the rank correlation
  across companies of the rule's bucket points with the company's next-12-month return
  over `^SP500TR`. On date t, w_r(t) = the mean of those correlations over the last 12
  rebalance dates whose 12-month window had closed by t (trailing 3 years), and 0 when
  negative. Nothing after t is used.
- **Static weights.** `score_ic_static`: one frozen weight per rule, the mean
  correlation over every window closed by 2021-12-31 (the end of the in-sample period),
  floored at 0 and used on every date. Decisions from 2022 are a test of it; earlier
  ones use weights fitted partly on their own outcomes and are in-sample. Averaging
  over the whole period was not taken: it would use returns after the decision date.
- **Score.** `score_ic` = Σ_r w_r(t) points_ir, missing below 5 evaluable rules or when
  every weight is 0. Added as a model in the comparison, on both universes.

## Decisions

- 2026-10-31 (user): the universe on the date is approved as a switch; both views are
  kept and compared.
- 2026-10-31 (user): correlations over the trailing 3 years of closed windows.
- 2026-10-31 (user): a negative correlation gives weight 0.
- 2026-10-31 (user): net profit growth as 1-year net income growth (NI1).
- 2026-10-31 (user): the weights must use only the past. The rolling weights already
  do (windows closed by the decision date); the docstring of `weighting.py` now says so
  with an example. A frozen static set, fitted on the in-sample windows, is added as
  `score_ic_static`.

## Notes

- 2026-10-31: built: `layers/growth.py` (NI1), `weighting.py` (forward excess,
  rule correlations, correlation weights), `scoring.learned_score` (`score_ic`),
  `pipeline.Buyability`; `[universe] views`, `[rules.NI1]`, `[learning]` in
  `config/rules.toml`. `buyability` writes `buyability_<view>.parquet` per view and
  `rule_weights_<view>.csv` and `rule_weights.png`; `evaluate` compares every model on
  both views.
- 2026-10-31: top 5, annualised excess over `^SP500TR`, all / held-out / in-sample.
  `current`: binned +7.5% / +2.4% / +11.7%; binned with 1 - pass rate weights +8.1% /
  +7.1% / +8.9%; `score_ic` +10.5% / +16.8% / +5.7%. `on_date`: binned +4.4% / -1.0% /
  +9.0%; binned with 1 - pass rate weights +5.5% / +2.1% / +8.2%; `score_ic` -0.1% /
  -3.9% / +3.1%; every company equal -0.4%.
- 2026-10-31: the learned weights are small and unstable: rank correlations of a rule's
  points with 12-month excess return are 0 to 0.1, and most rules get 0 on many dates;
  on 2025-12-19 (`on_date`) only NN (0.01) and NI1 (0.055) are above 0, so the top 5
  are picked on two rules. G6 (P/E) weighs 0 to 0.06: cheap P/E did not go with higher
  returns in 2009-2025. `score_ic`'s lead on `current` comes from companies that joined
  the index later; on `on_date` it is the weakest of the binned models.
- 2026-10-31: adding NI1 moved the binned top 5 on `current` from +6.5% to +2.4%
  held-out: with nine decisions, one added rule changes the result by several percent
  a year, so differences between these models are within noise.

- 2026-10-31: `score_ic_static` (weights frozen on windows closed by 2021-12-31), all /
  held-out / in-sample: `current` +0.8% / -7.5% / +8.0%; `on_date` -0.8% / -10.8% /
  +8.1%. The frozen weights fit the in-sample years and fail on the held-out ones:
  B1 0.059, G2b and E2 0.043, G2a and E1 0.034, NN 0.025, G3 0.017, E3 0.012, G6 0.009,
  the rest 0 (`on_date`). Liquidity and low debt went with returns up to 2021, not
  after.

- 2026-10-31: ledger for verification (user): `value_investing/ledger.py`;
  `evaluate` writes `ledger.md`, `ledger.csv`, `ledger_balance.csv` and
  `ledger_balance.png` for the wallet in `[ledger]` (top 5 on `score_binned_weighted`,
  `on_date`): each year's companies, weight, invested USD, buy and sell price, price
  change, total return and value, and the balance reinvested from 100 USD against the
  same money in `^SP500TR` and Berkshire Hathaway. Sales on the next decision date (the
  earlier 12-month windows overlapped the next buy by a day). Berkshire (`BRK-B`) added
  as `[benchmarks] strategies` in `config/market_data.toml`, fetched by
  `market_data update --source prices` (1 fetched, 409 reused).
- 2026-10-31: ledger, 100 USD from 2017-01-03 to 2026-01-02: wallet 552.79, `^SP500TR`
  353.48, BRK-B 303.27. 2017-2021 (in-sample): wallet x3.36, S&P x2.33, BRK-B x1.84.
  2022-2025 (held-out): wallet x1.65, S&P x1.52, BRK-B x1.65, so Berkshire matched the
  wallet in the test years (about 13% a year each against 11% for the S&P 500).

- 2026-10-31: covered by the warden's first run on the branch; the report and its
  fixes are recorded in VI_202610_006 ("Warden report").

## Kickoff prompt

Pick up VI_202610_005. Read VI_202610_004's notes, run `buyability` and `evaluate`, and
compare `score_ic` with the binned scores on both universes.

## Done when

- `score_ic`, NI1 and both universe views run in `buyability` and `evaluate`, and the
  comparison shows every model on both universes.
