# VI_202610_003: Signal evaluation: horizon returns, paper wallet and model score

| Field | Value |
|---|---|
| Status | in progress |
| Priority | high |
| Kind | new capability |
| Depends on | VI_202610_002 |
| Branch | `feature/value-investing/vi-202610-002-rule-catalogue` |
| Raised | 2026-10-31 |

## Description

The buyability score (VI_202610_002) is known at each rebalance date, but nothing yet
measures whether acting on it pays. The user wants a test: at the start of each year
from 2017, buy 100 USD of every company whose score is above a threshold and "sell"
100 USD of every company whose score is below one, then follow each position over
several horizons (1 month, 3 months, 1 year, 2 years) against 100 USD in `^SP500TR`.
Summed, the positions form a paper wallet; the model succeeds if the wallet grows more
than the S&P 500. The rank correlation of score and forward excess return per horizon
gives the horizon at which the signal is strongest (the "correlation wavelength"), and
the model gets one final score.

Two readings of the VI_202610_002 plots do not count as evidence: the ten best and
worst were chosen on their mean score over the same years they are plotted, and
score * relative growth contains the growth itself, so it tracks performance by
construction. The test below is what decides whether the score predicts.

## Agreed approach

- **Decision dates.** The first trading day of each year from 2017, using the latest
  score known then (the December rebalance date); no later filing or price is used.
- **Signals.** Buy when `score` >= the buy threshold (0.6), sell when `score` <= the
  sell threshold (0.4); one position of 100 USD per company and signal, held to the horizon,
  no rebalancing, no costs.
- **Horizons.** 1, 3, 12 and 24 months, from config.
- **Per position.** Long: end value 100 * P(t+h) / P(t) from `adj_close`. Sell: the
  short's value 100 * (2 - P(t+h) / P(t)). Each against 100 USD in `^SP500TR` over the
  same window.
- **Wallet.** Per horizon, the sum of long, short and long-short positions against the
  same money in `^SP500TR`; hit rate (share of positions beating the benchmark), mean
  and median excess, wallet excess per decision year and in total.
- **Correlation wavelength.** Rank correlation of score with forward excess return at
  horizons 1 to 24 months (monthly steps), per decision date and averaged: the decay
  curve and its peak.
- **Where.** `value_investing/evaluation.py` (tables), plots in `plots.py`, written by
  the `buyability` command (or an `evaluate` command) to
  `results/value_investing/<as_of>/evaluation/`. Return definitions from
  `skatkis.financial_logic`. The full wallet engine stays in S_202610_006.
- **Sell excess.** A sell gains when the stock does worse than the index: its excess is
  the benchmark's return less the stock's (selling the stock and holding the index).
- **Model score.** The annualised excess return of the buy wallet over `^SP500TR` at the
  1-year horizon: each decision year, the equal-weighted 100 USD positions of every
  company with score >= 0.6, held one year; the yearly wallet returns chained into one
  growth path and compared with `^SP500TR` chained over the same windows
  (`skatkis.financial_logic.annualised_relative_return`). Reported separately for the
  decisions of 2017-2021 (in-sample) and 2022 onwards (held out); a decision whose
  window ends after the data is left out.
- **Honesty.** Thresholds and horizons fixed before the run. The 0.6 was chosen after
  seeing the 2016-2026 plots, so the held-out years are only partly out-of-sample.
  Survivorship bias stated (only today's members, so failed companies are missing:
  longs look better and shorts worse than they were).

## Decisions

- 2026-10-31 (user): sell threshold 0.4, symmetric with the buy threshold 0.6.
- 2026-10-31 (user): the model's final score is the annualised excess return of the
  1-year buy wallet over `^SP500TR`, defined under "Agreed approach".
- 2026-10-31 (user): decisions of 2017-2021 and 2022 onwards are reported apart; the
  thresholds stay fixed.
- 2026-10-31 (user): the score alone first; score * relative growth as a signal later.
- 2026-10-31 (user): "sell" is measured as a short (gain when the price falls) for this
  evaluation; it does not decide the strategy question of spec §9.2 (sell/avoid or
  short), which stays with S_202610_006.

## Notes

- 2026-10-31: built: `value_investing/evaluation.py`, plots in `plots.py`, the
  `evaluate` command (`python -m Toolbox.value_investing evaluate --as-of 2026-10-31`),
  `[evaluation]` in `config/rules.toml`, `skatkis.financial_logic.holding_return`.
  Tables and five plots in `results/value_investing/2026-10-31/evaluation/`.
- 2026-10-31: first run (decisions 2017-01-03 to 2026-01-02, prices to 2026-10-30;
  buy >= 0.6 about 23 to 39 companies a year, sell <= 0.4 about 230 to 300). Model
  score: +4.8% a year in-sample (2017-2021), -2.4% held-out (2022-2025), +1.5% over
  all; hit rate 56% and 39%. The rank correlation of score and forward excess return
  peaks at +0.024 (6 months) in-sample and is negative at every horizon held-out,
  falling to about -0.07 at 24 months. The sold stocks grew more than the buy wallet
  over the whole period at 12 months.
- 2026-10-31: survivorship check: an equal-weighted wallet of every scored company beats
  `^SP500TR` by +3.1% a year in-sample, +2.7% held-out and +2.9% over all, with no
  signal; today's members include companies that grew into the index. Against that
  baseline the score adds about +1.7% a year in-sample and loses about 5% a year
  held-out.
- 2026-10-31: sells: mean excess is negative (the sold stocks beat the index on
  average) while the median is positive at 12 and 24 months held-out: most sold stocks
  lagged, a few large winners dominate the mean.

- 2026-10-31: covered by the warden's first run on the branch; the report and its
  fixes are recorded in VI_202610_006 ("Warden report").

## Kickoff prompt

Pick up VI_202610_003. Read VI_202610_002's outcome and this file's decisions, build
`value_investing/evaluation.py` on the stored buyability table, and show the horizon
table, the wallet against `^SP500TR` and the rank-correlation decay curve.

## Done when

- For each horizon, the long, short and long-short paper wallets are compared with
  `^SP500TR`, with hit rate and excess return per decision year.
- The rank-correlation decay curve is plotted and its peak horizon reported.
- The model's final score is reported with its in-sample label and stated biases.
