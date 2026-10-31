# VI_202610_004: Ranked wallet and binned score

| Field | Value |
|---|---|
| Status | in progress |
| Priority | high |
| Kind | new capability |
| Depends on | VI_202610_002, VI_202610_003 |
| Branch | `feature/value-investing/vi-202610-002-rule-catalogue` |
| Raised | 2026-10-31 |

## Description

VI_202610_003 gave a metric to compare models (the annualised excess return over
`^SP500TR`, chained over yearly decisions). Two model options are now tried: a single
wallet that trades 100 USD each year, bought and sold by rank with weights, and a score
in which each rule grades the companies into buckets instead of a pass/fail cut.

## Agreed approach

- **Binned score.** Per rule and rebalance date, the companies evaluable on the rule
  are graded into 6 equal-sized buckets by the rule's value (direction per rule in the
  registry), the best bucket earning 5 points and the worst 0; a company whose value is
  undefined (P/E on negative earnings) earns 0. `score_binned` is the sum of the points
  (0 to 75 with 15 rules), missing below 5 evaluable rules. Compared between companies
  on the same date only, as the sum grows with the rules evaluable. G3 and E3 grade the
  number of years with positive EPS, E5 the relative EPS change, so values compare
  across companies.
- **Rule weights.** w_r = 1 - the share of evaluable companies passing the rule's
  threshold on the date (everyone passes: 0, nobody: 1), independent per rule.
  `score_binned_weighted` = Σ_r w_r points_r ranks the companies and sets the wallet
  weights. The rule's mean bucket points were considered and not taken: equal-sized
  buckets average 2.5 points by construction, so the weight came out near 0.5 for most
  rules and moved only with the share of undefined values (NN 0.92, E6 0.73).
- **Wallet.** Each January (VI_202610_003's decision dates), 100 USD buys the five
  companies with the highest score, ties at the cut-off included, weight
  w_i = s_i / Σ s over the companies bought; held 12 months. The model score is this
  buy book's annualised excess over `^SP500TR`, chained over the years
  (VI_202610_003's definition). The same money sells the five lowest, equal weights,
  reported apart; not a concern for now.
- **Comparison.** One table of models: the threshold buy wallet, the top-5 wallet on
  `score` and on `score_binned` with its short book, and the equal-weighted wallet of
  every scored company as a survivorship reference. Weighting `rank` (40/20/10/5/5)
  stays as a config option for later calibration.

## Decisions

- 2026-10-31 (user): 100 USD per decision, split 50 USD long and 50 USD short.
  Superseded the same day: 100 USD buys, the short book is reported apart.
- 2026-10-31 (user): the weights involve the score: w = score / total over the chosen
  companies; ties at the cut-off extend the book with an equal split.
- 2026-10-31 (user): rule weight -ln(p) with p floored, times the share of companies
  that could be measured on the rule; rules measured on fewer than 50% of companies
  are not weighted. Superseded the same day, below.
- 2026-10-31 (user): no rule weights; instead each rule grades the companies into 6
  buckets (5 to 0 points) and the score is the sum of the points, to tell companies at
  a cut-off from clear winners. The pass/fail `score` stays for comparison.
- 2026-10-31 (user): a rule weight multiplies the points: w_r = 1 - pass rate on the
  date, and S_i = Σ_r w_r points_ir both picks and weights the companies; the plain
  binned score stays for comparison. Later optimisation may refine w_r.
- 2026-10-31 (user): the wallet buys the top 5 by the binned score, w_i = score_i /
  sum of the five scores, 100 USD; the short book is reported but not a concern.

## Notes

- 2026-10-31: built: `scoring.rule_weights` and `score_weighted`, `evaluation.Wallet`,
  `book_weights`, `ranked_positions`, `ranked_wallet`, `chained_score`;
  `[score.weights]` and `[wallet]` in `config/rules.toml`; `evaluate` writes
  `model_comparison.csv`, `ranked_positions.csv`, `rule_weights.csv` and the plots
  `model_growth.png` and `rule_weights.png`.
- 2026-10-31: first comparison (annualised excess over `^SP500TR`, 12 months, all /
  held-out / in-sample): threshold buy wallet +1.5% / -2.4% / +4.8%; every scored
  company equal +2.9% / +2.7% / +3.1%; ranked 5+5 wallets -11.9% to -13.5% over all,
  the weighted score with score weights the least bad.
- 2026-10-31: cause of the ranked wallets' losses: the short book. 39 of its 91
  positions were companies not yet in the S&P 500 on the decision date (current
  members applied to the past), and they account for -90.6 of the short book's
  -125.5 USD; TSLA shorted in January 2020 (added December 2020, +720% in the year)
  alone cost -50 USD. Restricting each date to companies already added
  (`date_added` <= date) gives: threshold buy wallet +1.4%, every company equal -0.4%,
  ranked on `score` -5.0%, ranked on `score_weighted` -1.6% over all. Not applied: a
  universe change is a methodology change, for the user to approve.

- 2026-10-31: binned score built (`scoring.rule_points`, `score_binned`,
  `<id>_points` columns, `[score] buckets`); rule weights removed. Annualised excess
  over `^SP500TR`, 12 months, all / held-out / in-sample: top 5 on `score_binned`
  +7.6% / +6.5% / +8.5%; top 5 on `score` +0.2% / -4.5% / +4.1%; their short books
  -14.8% and -40.9% over all.
- 2026-10-31: checks on the top 5 on `score_binned`. With only companies already in the
  index on each date: +4.9% / +3.8% / +5.7% (the reference wallet of every company:
  -0.4%). Picks lean to cyclical value (NUE, STLD, SNA, CMI, CSCO, INTC, semiconductor
  equipment), weights near equal (scores close). Concentrated: of 55.3 USD total
  excess, the three best positions (LRCX 2019, NUE 2021, INTC 2023) make 42.9; 2024 lost
  31.7% (no position beat the index). Nine decisions, four held out: too few to tell
  skill from luck.

- 2026-10-31: rule weights built (`scoring.rule_weights`, `score_binned_weighted`).
  Top 5 on `score_binned_weighted`, all / held-out / in-sample: +9.1% / +7.0% /
  +10.8% with today's members; +4.6% / +1.7% / +7.0% with only companies already in
  the index on each date, below the plain binned score's +4.9% / +3.8% / +5.7% there.
  The weighting's gain over the plain binned score comes from companies that joined the
  index later; on the members-on-the-date universe it does not improve the model.
  Less concentrated: 48.1 of 104.3 USD excess remain without the three best positions
  (NUE, LRCX, FSLR).

- 2026-10-31: covered by the warden's first run on the branch; the report and its
  fixes are recorded in VI_202610_006 ("Warden report").

## Kickoff prompt

Pick up VI_202610_004. Read VI_202610_003's notes, then run
`python -m Toolbox.value_investing buyability` and `evaluate` and compare the models.

## Done when

- The ranked wallet and the weighted score run in `evaluate`, and the model comparison
  table and plots are written.
