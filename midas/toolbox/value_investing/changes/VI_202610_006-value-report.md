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

## Warden report

2026-10-31, first run, on `aab9115` (`main`) plus the staged working tree of
`feature/value-investing/vi-202610-002-rule-catalogue`; covers VI_202610_002 (first
slice) to VI_202610_006. Recorded unchanged:

I found 2 blocking findings, both in the report text. The code can be committed once they are fixed and the report is regenerated.

# Warden report: VI_202610_002 (first slice), VI_202610_003, VI_202610_004, VI_202610_005, VI_202610_006

## 1. Header

| Field | Value |
|---|---|
| Changes | VI_202610_002 (first slice: buyability score), VI_202610_003, VI_202610_004, VI_202610_005, VI_202610_006. Also touched: MD_202610_002 (first slice), S_202610_006 (sell rule recorded), MD_202610_003 (raised) |
| Date | 2026-10-31 |
| Commit reviewed | `aab9115` (HEAD of `main` merge) plus the staged, uncommitted working tree on `feature/value-investing/vi-202610-002-rule-catalogue`. The tree is dirty. |
| Base branch | `main` |
| Run | First run |

Inputs read:
- the diff at `/private/tmp/claude-501/-Users-jan-albert-Documents-Personal-Trading-Bot/5f508e56-654c-4cac-9e19-ca6ee915625e/scratchpad/warden_diff.txt`;
- the checks at `/private/tmp/claude-501/-Users-jan-albert-Documents-Personal-Trading-Bot/5f508e56-654c-4cac-9e19-ca6ee915625e/scratchpad/warden_checks.txt`;
- `results/value_investing/2026-10-31/report/report.tex` and `summary.csv`;
- the spec (§2, §3, §5, §6, §9);
- `MD_202610_002`, the tool's `notes.md`, `quant.md`, `researcher.md`, and the `.claude/rules` for tests, docstrings, layers, pipeline and baselines.

Line numbers below are file lines in the working tree, worked out from the diff.

### What the developer said, checked

- **Leak through `select_tags` (confirmed).** `Toolbox/market_data/fundamentals.py:66` runs `select_tags` on the full facts table before the as-of alignment. So the tag choice uses filings made after the date.
  - The cause is a user decision recorded in MD_202610_002 (2026-10-31: "most recently used tag is canonical"). The fix therefore needs the user's approval, which is the right handling.
  - Both strict xfails are correctly marked and give a reason.
- **`pipeline.py` has no DAG or caching (confirmed).** Deferring it is legitimate, because the DAG engine is an open decision in spec §9.6. But only the "minimal gates" part is recorded as a decision (finding S5).
- **Unit tests come later.** Correct under the lifecycle. A list of what they should cover is in section 4.
- **Documentation comes later.** Correct. Section 4 lists what to update.

## 2. Findings

### Blocking (fix before commit)

**B1. The report says no forward information is used, which is false and contradicts its own appendix.**
- Where: `config/report_value_investing.toml`, `[method] text`, rendered at `report.tex:24`: "used from the day they were filed, so a decision never uses information published after it."
- The tag-choice leak is confirmed, and the report's own appendix (`report.tex` around line 430) says scores differ on 2018-12-21 and 2022-12-16.
- Rule broken: `quant.md` rule 1 says a result that used a later filing "is invalid, not just less accurate"; rule 6 says to state the known biases with every result.
- Fix: reword the method sentence to name the known tag-choice leak and point to the appendix. Then regenerate the report.

**B2. The report claims a fair held-out test that does not exist, and states two different design periods.**
- `report.tex:28` says: "The score's design was settled while looking at the years to 2021, so only the years from 2022 test it fairly."
- `report.tex:438` says the design was settled "while looking at 2016 to 2026".
- The change files show that the held-out years were used to choose the model:
  - VI_202610_003: the 0.6 threshold was chosen after seeing the 2016–2026 plots.
  - VI_202610_004 and VI_202610_005: models were compared on held-out numbers before `score_binned_weighted` was chosen for the ledger and the report.
  - VI_202610_006: decision months, the sell threshold and the start years were set while looking at the full period.
- `report.tex:257` (Conclusions) then judges the strategy: "the value picks are worth more today … for each of the three start years".
- Rules broken: `AGENTS.md`, Prohibited ("judging a strategy on in-sample results only"); `quant.md` rule 9; `researcher.md` step 4.
- Fix:
  - Say plainly that no period in the report is out-of-sample, because 2022 onwards was used for model selection.
  - Make the conclusions descriptive, not a verdict.
  - Make the method text and the appendix agree on the period.
- Also correct the same text while there:
  - "three investors" (`report.tex:103`) and "three start years" (`:257`) should be four (2016, 2020, 2023, 2025; `[report] start_years`).
  - The `[value]` text gives rates for 2016, 2020 and 2025 only.

### Should fix

**S1. A shared `skatkis` change alters existing behaviour, not just optional arguments.**
- `Toolbox/skatkis/report.py`, `_table`: every table of 40 rows or fewer now renders as a floating `table`/`tabular` instead of a `longtable`. This changes the market analysis report the next time it is built.
- `Toolbox/skatkis/analysis.py`, `plot_series`: the new `if line.empty: continue` changes which legend entries appear in the existing plots.
- The checks file says "skatkis changes add functions and optional arguments only". That is inaccurate.
- Rules: `AGENTS.md`, Separation and Methodology vs refactor (state the effect).
- Fix: rebuild the market analysis report and check it, or record the behaviour change in the change file.

**S2. Shared finance is re-implemented in the tool.**
- `Toolbox/value_investing/holding.py:201-202` computes `((1+r)/(1+b))^(1/Y) - 1` by hand. That is `skatkis.financial_logic.annualised_relative_return`.
- It also divides by zero when `years == 0`.
- `DAYS_PER_YEAR` is redefined in `holding.py` and in `market_data/fundamentals.py`.
- Rule: `AGENTS.md`, Separation ("reused, never re-implemented in a tool").

**S3. One concept has several owners.**
- Forward excess return over the benchmark is computed in four places:
  - `diagnostics.forward_excess` (`diagnostics.py:99`);
  - `weighting.forward_excess`;
  - `evaluation.holdings`;
  - `evaluation.correlation_curve`.
- `skatkis.financial_logic.forward_excess_return` also exists.
- The rank correlation (Spearman) is coded inline six times: `diagnostics`, `evaluation`, `weighting`, and `holding` twice.
- `MIN_COMPANIES = 20` is defined three times.
- Rule: `AGENTS.md`, Ownership.

**S4. Evaluation logic sits in the command-line module.**
- `Toolbox/value_investing/__main__.py:790-792` defines the short book's return (`benchmark_return + short_excess`).
- `__main__.py:616` (`_survivorship`) computes the survivorship counts.
- The module's own "Must not" says it must not compute the evaluation or diagnostics.
- `tests/value_investing/test_lookahead.py` copies the universe construction of `_load_tables` instead of calling one shared function.
- Fix: move these into `evaluation`, `diagnostics` or `inputs`.

**S5. `pipeline.py` holds universe selection logic.**
- The `on_date` filter (`pipeline.py:132-138`) is a selection branch inside the pipeline.
- Rule: `.claude/rules/pipeline.md` ("No formulas … or strategy branches").
- Move it to `inputs` or a universe function.
- The deferral of the DAG and caching is only a note in VI_202610_002. Record it as a user decision tied to spec §9.6.

**S6. Thresholds are hard-coded instead of living in config.**
- `STALE_DAYS = 550` and `YEAR_TOLERANCE_DAYS = 45` in `market_data/fundamentals.py:46-47`.
  - `STALE_DAYS` changes scores in practice: A. O. Smith's score falls from 9 to 2 (notes).
- `PRICE_TOLERANCE_DAYS = 7` in `inputs.py:67`.
- `MIN_COMPANIES` in `diagnostics`, `evaluation` and `weighting`.
- The verdict bands ±0.3 in `report.py:637-639`.
- Rule: `developer.md`, Coding conventions ("thresholds, windows and weights in `config/*.toml`").

**S7. G4 is not phased in, against a recorded decision.**
- `layers/graham.py:172`: `g4_dividend_record` is always evaluable. A company with less than 20 years of Yahoo history fails it instead of getting NaN.
- VI_202610_002, decision 3 (2026-10-31): "a rule gives NaN until its window is full."
- This affects `score` (the mean of the evaluable rules) and the pass-rate weights.
- Fix the code, or record the exception as a decision.

**S8. Docstrings and comments no longer describe the code (checklist item 8).**

| Where | What it says | What the code does |
|---|---|---|
| `flags.py:9` | "The score counts a missing flag as 0." | `score` is the mean of the flags that exist, so a missing flag is ignored. Superseded on 2026-10-31. |
| `layers/graham.py:219` (G7) | "Fails when the average EPS … is zero or negative." | With P/B ≤ 1.5, a company with negative average EPS passes. |
| `layers/graham.py:41-42` (`_debt_cover` Computes) | "fails unless LTD is 0" | With NCA < 0 and LTD = 0 the rule also fails. |
| `scoring.py:20` | "0 to 75 with 15 rules" | There are 16 rules, so 0 to 80. |
| `scoring.py:168` | families "(G, E, B)" | `N` is missing. |
| `market_data/fundamentals.py:11-12` | "a date never sees a later filing" | It does not mention that the tag choice uses later filings. |
| `weighting.py:13-15` | Example: "grades of 2020 to 2023" | 12 closed quarterly dates before a January 2025 decision are March 2021 to December 2023. |
| `holding.py:4` ("every year") and `holding.py:563` ("Each January") | Yearly decisions | Decisions are made in January and July. |
| `report.py:143` (`ReportInputs.amount`, `picks`: "per year") | Per year | Per decision. |
| `report.py:731`, `784`, `827`, `933` | "earliest start year" | They use `inputs.reference` (2020). |
| `__main__.py:77` (`main`) | "`buyability` or `evaluate`" | `report` is missing. |
| `skatkis/report.py` Public API (`write_latex` "with a table of contents") | Always has a table of contents | It is now optional. |
| `config/rules.toml:165` and `:177` (`[report]` comments) | "every January" | January and July. |
| `config/report_value_investing.toml` header comment | Start years 2016, 2020, 2025 | 2023 is missing. |
| `plots.py:383` (`held_value` title, shown in the report) | "100 USD a year" | 100 USD per decision. |

**S9. Docstrings point to Midas's records.**
- Change IDs, `midas/instructions/researcher.md` and "(user, 2026-10-31)" appear in:
  - `fundamentals.py`, `rules.py`, `evaluation.py`, `scoring.py`, `diagnostics.py`, `layers/growth.py`;
  - `skatkis/report.py`, `peers.py`, `plots.py`, `weighting.py`.
- Today no other module in `Toolbox/` does this.
- NI1 cites VI_202610_005 instead of a spec or README section.
- Rules broken: `AGENTS.md`, Authority order ("Official documentation never references Midas's records"); `docstrings.md` (reference chain).
- This can be done in the documentation step.

**S10. Casts and `noqa` without a reason.**
- `evaluation.py:463` and `:468`: `cast(pd.Series, …)` with no comment.
- `report.py:639`: the second `# noqa: PLR2004` has no reason.
- Rule: `developer.md`, Coding conventions; `warden.md` item 7.

**S11. Housekeeping: change files and notes are out of date.**
- MD_202610_002 is unchanged:
  - The CHANGELOG still says `ready`, but its first slice is built.
  - The slice was built at `market_data/fundamentals.py`, not `layers/fundamentals.py` as the approach says.
  - The tag-choice leak goes against its own 2026-10-31 tag decision, and that is not recorded there.
  - `notes.md` points to it with "→ MD_202610_002".
- VI_202610_006 "Done when" still says about five pages and three start years. The report is fourteen pages with four start years.
- The reading of "every puch" (VI_202610_006) is still unconfirmed.
- `notes.md:93` says "Decided: the first three options" (Yahoo dividends, shorter windows, a paid source). The VI_202610_002 decision lists Yahoo dividends, explicit windows and phase-in. The two disagree.
- `quant.md` rule 2 ("a field follows the company's most recently used tag") encourages the leak and conflicts with rule 1. Update it when the fix is decided.
- Rule: `secretary.md`, Duties and Working on a change.

**S12. `tests/value_investing/test_lookahead.py` reads stored vendor data.**
- It reads the 2026-10-31 snapshot from `data/` instead of a small synthetic fixture.
- Without the data it skips silently, so the strict xfail guards nothing on a clean checkout.
- Rule: `.claude/rules/tests.md` (synthetic tickers, small fixtures).
- It is acceptable as a labelled integration check, but the point-in-time case needs a fixture test. The one in `tests/market_data/test_fundamentals.py` is a good model.

### Notes

- **N1. All results on this branch are provisional.** Under `quant.md` rule 1 they stay provisional until the tag-choice fix (methodology, user approval) is built. The picks were the same on all three check dates.
- **N2. Delisted lots are treated two ways.** In `holding._value_path` (`holding.py:288`), a lot whose prices end is reset to its cost (`fillna(1.0)`). `buy_and_sell` instead carries the last price forward (`ffill`). This does no harm while the universe is today's members, but it will be wrong once historical members are added.
- **N3. Confusing variable reuse.** `__main__.py:545-547`: inside the `rules=` dict comprehension, `settings` is reused as the loop variable while the outer `settings` is still used. It is correct in Python 3, but easy to misread.
- **N4. Duplicate settings code.** `__main__.py:193-202` builds `evaluation.Evaluation` inline, duplicating `evaluation_settings`. That function is public but is not in the module's Public API.
- **N5. Departures from the spec, recorded in change files but not yet in the spec:**
  - annual first-reported values instead of TTM and latest quarter (§3.6);
  - binned and weighted scores and a top-5 wallet instead of gates, composite rank and top/bottom 10 (§3.1, §5);
  - today's members filtered by `date_added` instead of historical membership (§3.4);
  - decisions in January and July instead of quarterly;
  - NI1, which is not in the spec catalogue.
- **N6. `filed` is the filing date, not the acceptance time** of spec §3.3. This convention already exists on `main`.
- **N7. Report text that will go stale.** The `[picks]` text states fixed facts ("lean to technology hardware and semiconductors, steel…") that are not filled from the data.
- **N8. Rules checked against spec §2 with no mismatch:** G2a, G2b, G3, G5, G6, E1–E6, NN and B1 (cap 0.5 decided by the user). The rest of the reference chain is checked under S8/S9.

## 3. Not checked

- **`regression_check` was not run.** The checks file only notes that no `value_investing` baseline exists. I could not confirm that the default backtest is untouched beyond the diff showing no change to `skatkis/backtest`.
- **The checks summary:** the claim that pyright on `main` had 13 errors, and that ruff's PLR2004 was fixed before the review. I have only the summary.
- **Black** was run on `Toolbox tests` only. `ruff` and `pyright` output were not supplied in full.
- **The pytest count against `main`.** No test file was deleted in the diff, but the earlier count was not supplied.
- **The PDF**: layout, page count, and whether the market analysis report still compiles after the `_table` change.
- **The generated CSV and PNG files** under `results/`: their contents against the code, and whether the per-position Yahoo prices in `ledger.csv` and `purchases.csv` are acceptable as committed derived data.
- **`skatkis.report.Section`** (`table` and `label` fields): it is not in the diff.
- **The numbers in the change-file notes**: I could not rerun them, as I run no commands.

## 4. Documentation to update after the report

- **Spec / tool README:**
  - The buyability score, binned score, pass-rate weights, `score_ic` and `score_ic_static`, NI1, the `on_date` universe and the annual (not TTM) basis.
  - Each marked as a departure from §3.1, §3.4, §3.6 and §5, with sources (Graham ch. 14/15 by rule; NI1 and the scores as conventions of this repository).
  - G7's handling of negative EPS once the user decides it.
- **`market_data` README:** `fundamentals.annual_series` and `annual_as_of`, `split_factors`, BRK-B under `[benchmarks] strategies`.
- **skatkis reference:** `relative_growth`, `forward_excess_return`, `holding_return`, `money_weighted_return`, `plot_bars`, `references`/`labels`/`thresholds`, the `Table` caption/label/breaks, `MAX_UNBROKEN_ROWS`, `Document.contents`.
- **Every docstring in S8**, and the record references removed (S9).
- **`quant.md` rule 2** (via the secretary) once the tag fix is decided.

**Unit tests to write (tests step), per `.claude/rules/tests.md`:**
- `fundamentals` (gap cut, stale cut, same-day filing);
- `split_factors`;
- every rule in `graham`, `buffett` and `growth`, covering:
  - the exact threshold (P/E = 15.0, CR = 2.0, D/E = 0.5);
  - zero or negative denominators;
  - the window not yet full;
  - G4 phase-in;
- `valuation`;
- `scoring.rule_points`: ties, undefined value gives 0, direction;
- `score`: `min_available` boundary;
- `rule_weights`;
- `learned_score`;
- `weighting`: only closed windows used, with a fixture where a later window would change the weight; floor at 0;
- `evaluation`: `decision_dates`, `signals` (latest score date ≤ decision), `book_weights` (ties at the cut-off, rank weights, sum 0), `chained_score`;
- `holding`: `buy_and_hold`, `buy_and_sell` (sale fraction at 0, at 0.4, NaN score; reconciliation of `purchases`), `money_weighted_return` (known IRR, no-solution case);
- `ledger`;
- `peers`;
- `lookahead` on a fixture;
- `skatkis.report._table` (both branches);
- input non-mutation and no leakage between tickers throughout.

## 5. Draft outcome

### VI_202610_002 (first slice: buyability score)

Built and run end to end on 2026-10-31 (`python -m Toolbox.value_investing buyability`).
- Coverage: 396 companies, 71 rebalance dates (third Friday of March, June, September and December, 2009-03 to 2026-09).
- Rules: G2a, G2b, G3–G7, E1–E6, NN and B1, plus NI1 from VI_202610_005. They run on annual first-reported 10-K values from their filing date, through the first slice of MD_202610_002 (`market_data/fundamentals.py`), with per-share values on the price's split basis.
- The score is the mean of the evaluable flags. A company needs at least 5 evaluable rules. The score is also given per family.
- Alternatives not taken:
  - the sum of flags and "unevaluable scores 0", superseded the same day because they depend on how many rules apply;
  - the ratio score / growth.
- Results: the first in-sample check found no relation between the score and the next quarter's excess return (mean rank correlation −0.008 from 2016). Companies with EDGAR gaps (EMR, XOM, TRMB, VYLR) are recorded in the notes.
- Validation stayed minimal and the DAG was deferred (spec §9.6).
- Warden's report on 2026-10-31 (commit `aab9115` plus the working tree).

### VI_202610_003

A paper test of yearly buy and sell signals (buy at score ≥ 0.6, sell at ≤ 0.4) over 1, 3, 12 and 24 months against `^SP500TR`.

| Measure | Result |
|---|---|
| Model score, in-sample (2017–2021) | +4.8% a year |
| Model score, held out (2022–2025) | −2.4% a year |
| Rank correlation, peak in-sample | +0.024 at 6 months |
| Rank correlation, held out | negative at every horizon |

An equal-weighted wallet of every scored company beat the index by about 3% a year with no signal. That revealed the survivorship and universe bias.

### VI_202610_004

The ranked top-5 wallet and the binned score (6 buckets, 0–5 points) with pass-rate weights.
- The short book's losses came from companies not yet in the index on the decision date (for example TSLA, January 2020). That led to VI_202610_005.
- Results on companies already in the index on each date:

| Model | All | Held out | In-sample |
|---|---|---|---|
| Top 5, binned score | +4.9% | +3.8% | +5.7% |
| Top 5, weighted binned score | +4.6% | +1.7% | +7.0% |

- Nine decisions are too few to tell skill from luck.

### VI_202610_005 (methodology, approved)

- Built: the `on_date` universe as a switch beside `current`; NI1; rule weights learned from closed 12-month windows over the last 3 years (`score_ic`); frozen weights (`score_ic_static`).
- The learned weights were small and unstable.
- `score_ic_static` failed out of the fitting window, at −10.8% a year held out on `on_date`.
- The pass-rate weighted binned score was kept for the ledger and the report.

### VI_202610_006

A report of about fourteen pages from `python -m Toolbox.value_investing report`.
- Investors start in 2016, 2020, 2023 and 2025. They buy 100 USD of the top 5 by `score_binned_weighted` (`on_date`) each January and July, against `^SP500TR` and BRK-B.
- A second wallet sells part of a holding whose 0-to-1 score is at or below 0.4. The proceeds are reinvested; there is no short.
- The report also covers the Magnificent 7, a forward-information truncation check and survivorship counts.
- The 2020 investor (the reference):

| Holder | Value (USD, from 1,400 put in) | Money-weighted a year |
|---|---|---|
| Value wallet | 4,708 | 33.9% |
| S&P 500 | 2,480 | 15.9% |
| Berkshire Hathaway | 2,415 | 15.2% |

- About half of the 2020 investor's value is in Micron.
- 230 companies have left the index since 2016 and are missing from the data.
- The weight w_T (share of past dates on which a rule succeeded) was built and dropped the same day: it was worse than the pass-rate weights.
- The truncation check found that the tag choice depends on later filings. The fix is pending the user's approval under MD_202610_002, with two strict-xfail tests.
- These are in-sample research results. No period is out-of-sample, because 2022 onwards was used for model selection.

### Fixes after the report (2026-10-31)

- B1: the method text names the open tag-choice leak, points to the appendix and calls
  every result provisional.
- B2: the method text, the conclusions and the forward-information appendix now say
  that no period is out of sample (2022 onwards was used to choose between models);
  the conclusions describe the sample instead of judging the strategy; four investors
  and the 2023 rates in the text.
- S1: the market analysis report rebuilt with the new tables: it compiles (53 pages);
  its 330pt overflow at an appendix figure was already on `main`.
- S2: `holding` uses `annualised_relative_return` (no division by zero for a lot bought
  on the valuation date) and the shared `DAYS_PER_YEAR`.
- S10: reasons on the cast and on the second `noqa`.
- S8 in part: the report's panel title says each January and July.
- Open, for the documentation and tests steps: S3 to S9 (except the above), S11 and
  S12, the documentation and unit tests listed in sections 4 of the report, and the
  user's approval of the tag-choice fix (MD_202610_002).

## Kickoff prompt

Pick up VI_202610_006. Run `buyability`, then `report`, and review the PDF with the
user.

## Done when

- `report` writes a PDF of about five pages with the picks, the value over time for the
  three start years against the S&P 500 and Berkshire, and the rules behind the winners.
