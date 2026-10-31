# VI_202610_006: Value report: buy-and-hold picks, value over time, rules behind the winners

| Field | Value |
|---|---|
| Status | in progress |
| Priority | high |
| Kind | new capability |
| Depends on | VI_202610_005 |
| Branch | `feature/value-investing/vi-202610-002-rule-catalogue` |
| Raised | 2026-10-31 |

## Description

The value tool's results live in CSV files, Markdown and terminal tables. The user wants
a short report like the market analysis report (about five pages, few appendices) that
shows which companies were picked each year and how they performed, the value over
time, and which rules the well-performing picks scored on. The strategy shown is buy
and hold: every year 100 USD buys that year's picks and nothing is ever sold, so the
result compounds; shorts come later. Three investors are compared: one who started in
2016, one in 2020 and one in 2025.

## Agreed approach

- **Picks.** Each January (first trading day), the top 5 companies by
  `score_binned_weighted` in the `on_date` universe, weight w_i = s_i / Σ s, as in
  VI_202610_004 and VI_202610_005.
- **Buy and hold.** 100 USD per year split over the picks by weight, held to the data's
  last date with dividends reinvested (`adj_close`), never sold. The same 100 USD per
  year goes into `^SP500TR` and into Berkshire Hathaway (`BRK-B`) on the same dates.
- **Scenarios.** Starting in 2016, 2020 and 2025, each buying every year to 2026.
  Per scenario and holder: money put in, value today, multiple and the money-weighted
  annual return (the internal rate of return of the yearly 100 USD and today's value).
- **Rules behind the winners.** For every pick, its annualised return over the S&P 500
  since purchase; the mean points per rule of the picks that beat the index against
  those that did not.
- **Report.** `python -m Toolbox.value_investing report --as-of <date>` writes the
  figures, `report.tex` and `report.pdf` to `results/value_investing/<as_of>/report/`
  with `skatkis.report`. Text in `config/report_value_investing.toml` (secretary).
  Sections: method, picks per year, value over time, rules behind the winners,
  conclusions; biases stated up front.

## Decisions

- 2026-10-31 (user): buy and hold with 100 USD added every year and never sold; shorts
  later. Start years 2016, 2020 and 2025.
- 2026-10-31 (user): a short report like the market analysis one, about five pages,
  fewer appendices.
- 2026-10-31 (user): a second wallet sells as well as buys. Each January, a holding
  whose score on 0 to 1 (weighted points / the most it could score on its evaluable
  rules, `score_weighted_share`) is at or below 0.4 sells the share (0.4 - s) / 0.4 of
  itself, so all of it at score 0; selling the holding only, no short. The proceeds and
  the new 100 USD buy that year's top 5 at their score weights. Shown beside the
  buy-and-hold wallet.
- 2026-10-31 (user): "Rules behind the winners" explains each rule under the figure,
  and tests the rule weighting: each rule's weight against how well it told the picks
  that beat the S&P 500 from those that did not. The report may grow to ten pages.
- 2026-10-31 (user): the value-over-time table grouped by start year; the rules table
  moves to the appendix.
- 2026-10-31 (user): a second rule weight w_T for how often the rule was successful in
  the past, multiplied with the floating weight w_r (1 - pass rate on the date): w = w_r
  * w_T. Read as: on each past rebalance date whose 12-month window had closed by the
  decision date, a success is a positive rank correlation of the rule's points with the
  next 12 months' return over `^SP500TR` across all companies; w_T is the share of
  successes over all such dates (1 with no history). Built as `score_combined` and
  compared with `score_binned_weighted` before the report's wallet changes.
- 2026-10-31 (user): winners and losers weighted by dollars: each pick's value today
  less the same money in the S&P 500 ("dollars ahead"); section 5 shows the totals,
  averages and medians of the picks that beat the index and of those that lagged, and
  the payoff ratio. The rule comparison (Figure 3) and the gap of the weighting test
  weigh each pick by the size of its dollars ahead or behind.
- 2026-10-31 (user): the report's wallets decide twice a year, on the first trading
  day of January and of July, 100 USD at each decision (200 USD a year); July buys and
  checks sales as January does. The comparisons follow the same schedule. The yearly
  paper test of `evaluate` stays yearly.
- 2026-10-31 (user): a section comparing the picks with the Magnificent 7 (Apple,
  Microsoft, Amazon, Alphabet, Meta, Nvidia, Tesla): their score and rank, the rules
  that kept them out, and, for fairness, the same money in the Seven. The weighting test
  moves to the appendix. Concluding parts (next picks, sell check, conclusions) name
  companies in full.
- 2026-10-31 (user): the forward-information check goes into the report's appendix and
  into the unit tests.
- 2026-10-31: the user's message was cut off at "We can assume every puch"; read as
  "every purchase is held to today", to confirm.

## Notes

- 2026-10-31: built: `value_investing/holding.py` (buy and hold, rule profile),
  `value_investing/report.py`, the `report` command, `config/report_value_investing.toml`
  (text, rule names, short sector names), `[report]` in `config/rules.toml`,
  `skatkis.financial_logic.money_weighted_return`, `skatkis.analysis.plot_bars`.
  Output: `results/value_investing/2026-10-31/report/report.pdf`, five pages (method,
  picks per year, value over time, rules behind the winners, conclusions; appendix of
  every purchase), with `purchases.csv` and `summary.csv`.
- 2026-10-31: results to 2026-10-30, 100 USD a year, never sold. From 2016: 1,100 USD
  in, wallet 5,614, S&P 500 2,778, Berkshire 2,498 (26.5%, 15.3%, 13.6% a year,
  money-weighted). From 2020: 700 in, 2,345 / 1,267 / 1,210. From 2025: 200 in,
  326 / 244 / 229. The five Micron purchases (2019 to 2023) are 31% of the 2016
  investor's value; the 2018 and 2024 purchases trail the S&P 500. Winners (31 of 55
  picks) scored higher on E5, E6, G6 and G7; the laggards on G3 and B1.
- 2026-10-31: the user's sell rule (sell a holding whose score drops below the
  threshold) recorded for S_202610_006.
- 2026-10-31: wallet with sells built (`holding.buy_and_sell`, `score_weighted_share`
  in `scoring.score`, `[report] share_column` and `sell_threshold`). From 2016: 5,803
  USD (27.0% a year) against 5,614 for buy and hold; from 2020: 2,341 against 2,345;
  from 2025 equal (no sale yet). 19 partial sales for the 2016 investor, 180 USD in all,
  mostly WDC, QCOM, CSCO and HAS.
- 2026-10-31 (user): the report gets a section with next year's picks and a sell check.
  Built as "Picks for next year" with "Sell check": on the latest score date
  (2026-09-18) the top 5 are FSLR, CTSH, SNA, SWKS and RMD (about 20% each); the 2016
  investor's wallet would sell part of GLW (18%), CSCO (36%), INTC (1%) and HAS (38%)
  and keep 17 holdings. A preview: the January 2027 decision uses the scores known on
  its day. The report is now seven pages (six and the appendix).

- 2026-10-31: rules explained and the weighting tested in the report
  (`holding.rule_weighting`, `holding.weighting_quality`, `plots.weighting_check`,
  `[rule_texts]` in `config/report_value_investing.toml` with thresholds filled from
  `config/rules.toml`; labels on `skatkis.analysis.plot_scatter`; captions and labels
  on `skatkis.report.Table`, so the text refers to tables by label). Weight against
  gap across the rules: rank correlation +0.42, weight against correlation +0.42. E6,
  G7 and G6 are heavy and separated winners best; NN (weight 1.00, gap -0.12) and B1
  (0.79, -0.93) are heavy with a negative gap; E5 has the largest gap (+1.01) at weight
  0.25. Report ten pages.

- 2026-10-31: w_T built (`weighting.success_weights`, `score_combined`, stored as
  `success_weights_<view>.parquet`), compared in the report ("A wallet on both
  weights"). Worse than w_r alone: weight against gap +0.13 (w_r alone +0.42), against
  correlation +0.18 (+0.42); buy and hold from 2016 5,615 USD (w_r 5,614), from 2020
  2,170 (2,345), from 2025 301 (326); yearly paper test on `on_date`, all / held-out /
  in-sample, +7.0% / +1.8% / +11.3% against +5.5% / +2.1% / +8.2%. B1 has the highest
  w_T (0.78) but separated the picks worst: success across all companies is not
  success among the picks. The report's wallet keeps w_r.
- 2026-10-31: value-over-time table grouped by start year (`skatkis.report.Table`
  breaks); rules table moved to Appendix B; contents page dropped
  (`skatkis.report.Document.contents`); figure labels (`ReportFigure.label`).
- 2026-10-31 (user): wallet contents by year, with the benchmarks normalised per holding.
  Built: `Managed.contents` (each holding's value month by month), the plot
  `plots.wallet_contents` (holdings solid, `^SP500TR` and `BRK-B` divided by the number
  of holdings dashed, log scale) in the section "Wallet contents", and Appendix C, the
  value of each holding after every January with the money put in, the sale proceeds
  and the total (5,803 USD now). Report ten pages.

- 2026-10-31: dollar view built (`ahead_usd` and `index_usd` on the lots,
  `holding.dollar_view`, dollar-weighted `rule_profile` and gap in `rule_weighting`;
  Table "dollars ahead" at the head of section 5; `ReportFigure.width` to fit the
  square scatter). Winners (31) +3,165 USD ahead of the same money in the S&P 500,
  laggards (24) -329 USD, net +2,836 USD (the wallet's lead, 5,614 - 2,778); medians
  +21 and -8 USD; payoff ratio 7.4; the five largest wins (MU 2019, LRCX 2017, MU
  2023, LRCX 2019, KLAC 2019) are 61% of the winners' total. Dollar-weighted, weight
  against gap +0.47 for w_r (+0.16 with w_T). Report ten pages.

- 2026-10-31 (user): Appendix A shows the purchases of the 2016 investor's wallet with
  sells, with the USD each purchase gave up to later sales and the USD it keeps
  (`Managed.purchases`: a sale sells the same share of every unit held, so each earlier
  purchase of the company gives up that share). Reconciled: kept 5,803.11 USD = the
  wallet's value, sold 180.15 USD = the sale trades.
- 2026-10-31 (user): w_T dropped; `weighting.success_weights`, `score_combined` and the
  report's comparison removed (no trail: built and removed the same day).
- 2026-10-31: the user's idea that a low G3 marks a holding to sell, tested on the 2016
  wallet with sells (112 holding-years, each January's G3 points against the next 12
  months' return over `^SP500TR`): rank correlation -0.13; low G3 (0 to 1 points) mean
  +22.7% (median +2.7%), mid (2 to 3) +2.0% (median -4.6%). Within the value picks a
  low G3 went with doing better (cyclical recoveries: MU, LRCX, KLAC); across all
  companies a long profit record won on 61% of past dates (the w_T figure). No G3 sell
  rule; the sell rule stays on the overall score.

- 2026-10-31: twice-yearly decisions built (`evaluation.decision_dates(months=...)`,
  `[report] decision_months = [1, 7]`), labels by month and year, the contents
  appendix by January with money put in and sold to date. Benchmarks drawn thick and
  black in every value plot, the S&P 500 dashed and Berkshire dash-dot
  (`skatkis.analysis.plot_series(references=...)`, user); an `o` marks where each
  company enters the wallet (user). Report twelve pages: Table 1 and Appendix A double
  with two purchases a year.

- 2026-10-31: Magnificent 7 section built (`value_investing/peers.py`: ranks, rule
  profile, equal-weight positions; `plots.peer_profile`; `[report] peers` and
  `peer_label`). On 2026-09-18 Nvidia ranks 21 of 387, Microsoft 36, Meta 55, Alphabet
  69, Tesla 160, Amazon 220, Apple 298. The Seven score fewer points on P/E times P/B,
  P/E, price to tangible assets and current dividend, more on net income growth and
  earnings growth. The same 100 USD a decision in the Seven, equal weights: 22,613 USD
  from 2016 (39.0% a year) against 10,232 in the value wallet; from 2020 4,445 against
  4,708. Named with hindsight, said so in the text. Weighting test moved to the
  appendix; full company names in the next picks, the sell check and the conclusions.
  Report thirteen pages.

- 2026-10-31: forward-information check built: `value_investing/lookahead.py`
  (`truncation_check`: inputs cut at each date, scores compared with the full run), run
  by `report` on `[report] lookahead_dates` and shown in Appendix "Forward information"
  with the guards and the biases the test cannot see. Unit tests:
  `tests/market_data/test_fundamentals.py` (hand-made tag switch: the value on the date
  must be 100, the full run gives 80) and `tests/value_investing/test_lookahead.py`
  (stored data, skipped without it; the picks must be the same, and no score may
  differ). The two no-leak tests are marked strict xfail until the tag-choice fix
  (MD_202610_002) is approved and built; they will fail loudly once the leak is gone,
  so the mark is removed with the fix. Suite: 13 passed, 2 xfailed. Report fourteen
  pages.
- 2026-10-31: EasyEquities instrument list (investintheus.easyequities.net, linked
  spreadsheet): 718 JSE shares and ETFs of the ZAR account, stale (it lists 1time
  Holdings, delisted about 2012), no US shares; no public list of the USD account's US
  shares was found, so availability of the picks there is unconfirmed.

- 2026-10-31 (user): tables are not split across pages where it can be helped; Appendix A
  stays. Built: `skatkis.report` keeps a table of up to `MAX_UNBROKEN_ROWS` (40) rows
  whole on one page (a float placed here, moved whole to the next page when it does
  not fit); longer tables (Appendix A) still break. Figures get labels (also
  `FigureGroup.label`) and the text refers to every table and figure by label, after
  new figures had shifted the numbers in the text. Report fifteen pages.

- 2026-10-31 (user): the survivorship bias limited by following a recent investor:
  `[report] reference_start = 2020` drives the abstract, the conclusions and the
  detailed views; a 2023 investor added. The forward-information appendix counts the
  companies that left the index since each start year (all missing from the data):
  230 since 2016 (97 on market value), 125 since 2020 (75), 69 since 2023 (47), 37
  since 2025 (24). The 2020 investor: 1,400 USD in, 4,708 USD now (S&P 500 2,480,
  Berkshire 2,415); 49% of the value in Micron. JSE data and a Top 40 comparison
  assessed and raised as MD_202610_003. Report fourteen pages.

## Kickoff prompt

Pick up VI_202610_006. Run `buyability`, then `report`, and review the PDF with the
user.

## Done when

- `report` writes a PDF of about five pages with the picks, the value over time for the
  three start years against the S&P 500 and Berkshire, and the rules behind the winners.
