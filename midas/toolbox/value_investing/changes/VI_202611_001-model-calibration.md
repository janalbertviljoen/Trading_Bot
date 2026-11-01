# VI_202611_001: Model calibration: signed rule weights, point curves, fitted cut-offs

| Field | Value |
|---|---|
| Status | agreed |
| Priority | high |
| Kind | methodology |
| Depends on | S_202611_003 |
| Branch | its own, after S_202611_003 (`feature/value-investing/vi-202611-001-model-calibration`) |
| Raised | 2026-11-01 |

## Description

The results for the S&P 400 and 600 were not what the user wants (2026-10-31 reports):
the S&P 600 wallet trails IJR for every investor, and the S&P 400 is close to IJH for
recent investors. On the S&P 500 the buy-and-hold report beats the index by a wide
margin (the 2020 investor 33.9% a year against 15.9%), while the paper evaluation of the
broad signal (every company with `score` of at least 0.6, held 12 months) is flat after
2022 (-0.4% a year). The two test different strategies: the report's outperformance
comes from five names by `score_binned_weighted` held for good, a concentrated book.

The final product is the report with its five picks (user, 2026-11-01). The model
specification is calibrated to suit the client; once the client accepts the result,
that specification is frozen as their version. A recalibration makes a new version.

## Agreed approach

Calibration has two levels. The **inner fit** is computed, not searched, on each date
from the training companies: the rule weights, the point curves, the choice of rules
and the cut-offs. The **outer search** (S_202611_003) samples the few remaining
settings: the number of buckets, how many rules are kept, the rule windows (with
floors), the learning window and its decay, the number of names.

1. **Signed rule weights.** A rule whose points correlate negatively with the forward
   excess return counts against a company instead of being floored at 0 (today
   `correlation_weights` clips at 0). Each weight is shrunk towards 0 by how unstable
   it was across the windows.
2. **Weights over time.** The weights on date $t$ come only from the dates whose return
   window had closed by $t$ (as today), averaged over about five years with more weight
   on recent windows; the decay is a calibrated setting.
3. **Best rules.** Only the best $k$ rules are used, ranked by the stability of their
   correlation, not its mean alone; $k$ is searched (the user's starting point: 10).
4. **Point curves per rule.** The fixed 0 to 5 points per equal-count bucket are
   replaced by a curve per rule: the mean forward excess return of each bucket of that
   rule over the closed training windows, smoothed. This lets a rule reward the middle
   of its range (P/E, where the cheapest are often value traps).
5. **Fitted cut-offs.** The buy and sell cut-offs are not hard-coded: on each date they
   are fitted by one fixed rule, where the training companies' forward winners and
   losers separate best by score. The same inputs give the same cut-offs; each run
   writes them per date to its manifest. The values in `config/rules.toml` remain as
   suggested defaults for new users.
6. **Reporting**: the industry standard, annualised excess over the benchmark for
   trailing 1, 3 and 5 years and since the start, tracking error and information ratio;
   with the note in every calibration that calibration usually favours recent years.
7. **Validation**: the chosen specification rerun on 10 new test samples, with
   uncertainty bands (S_202611_003). In each rerun the test companies have a
   prediction (their score, or the expected excess from the point curves) and a
   realised forward excess on every date, which give per rerun: the rank correlation of
   the two, the hit rate of the picks, a calibration plot (mean realised excess per
   bucket of the prediction) and the wallet's excess and trailing frequency. Across the
   reruns each gets a mean and a band. The bands show the uncertainty of the company
   sample only; every run sees the same market history, and the samples overlap, which
   the report states.

## Decisions

- 2026-11-01 (user): calibration targets the strategy that is traded, the top five, with
  the paper evaluation as a diagnostic.
- 2026-11-01 (user): negative correlations are kept (signed weights), not floored:
  flooring throws away information.
- 2026-11-01 (user): calibration from forward excess returns is allowed only backwards
  from each date, from windows already closed, so it stays point-in-time.
- 2026-11-01 (user): the rule windows are tuned, with floors, because very short
  windows overfit.
- 2026-11-01 (user): train and test by company, 10% test, replacing the time hold-out
  from 2022 (`[evaluation] held_out_from`); all the data is used, recent volatility
  included. Recorded in the research protocol (`researcher.md`, point 3).
- 2026-11-01 (user): the reported measure is the industry standard, unweighted; the
  recent-years note goes with every calibration.

- 2026-11-01 (user): the smoothing of the point curves comes first; the fixed rule for
  the cut-offs is chosen once results show what makes sense.

- 2026-11-01 (user): `score_ic` stays as the basis of the time-window weights.
  `score_ic_static` is removed: it served the time hold-out (weights frozen from windows
  to 2021, tested from 2022), which the company test sample replaces, and before 2022
  it used returns not yet known. Removed with it: `static_weights`, `[learning]
  static_until` and its entry in `[wallet] score_columns`.

- 2026-11-01 (user): the weights stay floored at 0 for now. Signed weights are pencilled
  in for later investigation, priority low, together with other weighting strategies,
  simple and complex (`value_investing/notes.md`, "Rule weighting strategies"). First
  look (one 40-company test split, weights learned on the other companies): the mean
  rank correlation of every rule with the 3-month forward excess (mislabelled 12-month at first) lies between -0.018
  and +0.012, and floored, signed and shrunk-signed weights all gave a score
  correlation within 0.01 of zero; no variant showed signal.
- 2026-11-01 (user): the test sample is never seen anywhere. Nothing computed from
  the test companies may enter a fit, a bucket edge, a smoothing, a cut-off, the choice
  of rules or the choice between settings; columns built from all companies (the stored
  `score_ic`) are not compared on the test sample. Leakage is a violation of the
  protocol (`researcher.md`, point 3). Disclosed: the first weighting look above
  compared three variants on a test split, and the choice to keep the floor was made
  after seeing it, so that split is spent; later work draws fresh seeded samples.

- 2026-11-01 (user): the philosophy is Buffett's buy and hold, so the calibration
  horizon is 5 years (underdog picks need that long to show), not 12 months. A
  2.5-year look re-evaluates whether to sell (a pick may crash out sooner); that is a
  separate sell analysis, later. The 1-, 3-, 6- and 12-month horizons become
  diagnostics. The letters support it: "our favorite holding period is forever" and "if
  you aren't willing to own a stock for ten years, don't even think about owning it for
  ten minutes" (`data/interim/berkshire_letters`). The configured
  `score_horizon_months = 12` is therefore a candidate methodology change.

- 2026-11-01 (user): G3 (earnings stability) is reduced from 10 to 3 years for lack of
  data, which also corresponds to the investment horizon. A methodology change
  (`[rules.G3] years` in `config/rules.toml`, and the baselines move), not yet applied:
  first tested as a design in a script, then written to the config with the user's
  approval. Rules that need long history (G4 20 years, G5 13, B1 10) remain a data
  limit (`notes.md`, "EDGAR history is too short").

- 2026-11-01 (user): back to the plain score, no gates; weighting and gating are
  investigated later (`notes.md`, "Rule weighting strategies"). A concentrated book is
  acceptable, since underdog picking should have only a few winners, as long as the
  concentration is stated in every result (the share of the positive excess from the
  five largest contributors, and their names).
- 2026-11-01 (user): the report's reference holding period is 3 years. A 5-year result
  is reported with the number of independent windows and a warning that it rests on few
  of them, never as the headline.

- 2026-11-01 (user): the optimisation is tested on all three indices (S&P 500, 400 and
  600); only the results and the checks are produced, no report for the S&P 400 and 600.
  The number of names starts at 5, the same as the picks (`[wallet] names`).

## Open decisions (user)

- The smoothing of the point curves (number of buckets, monotone or free shape): high
  priority.
- The fixed rule for the cut-offs (the score where the expected excess crosses zero, or
  the one that separates winners and losers best): after the first results.

## Notes

- 2026-11-01, first baseline checks on the S&P 500 (`buyability_on_date`, training
  companies only, 3-month forward excess (the stored `forward_excess` is 3-month, corrected below); scripts in the session scratchpad, nothing
  tracked regenerated):
  1. Rule weighting: floored, signed and shrunk-signed weights all gave a score
     correlation within 0.01 of zero (one 40-company test split, seed 0, now spent).
  2. Bucket curves (seed 1, test companies dropped before any computation): per rule the
     top-minus-bottom quintile spread is at most about 1 percentage point, |t| at most
     1.8, and the early and late halves mostly disagree in sign. P/E and P/B show no
     mid-range sweet spot. G2A equals E1, G2B equals E2 and G6 equals `pe_average`.
  3. Top 5 by `score_binned` over 66 dates: +0.76% a year against +0.45% for random
     five-name books, an advantage of +0.30% with a 95% band of [-0.85, 1.53]; 50% of
     dates beat the training mean; five names give 54% of the positive excess.
     `score_binned_weighted` was excluded: its weights were learned on all companies.
- 2026-11-01, universe look-ahead: check 3 used `buyability_on_date`, which already
  keeps only companies on or after their `date_added`. On the unfiltered
  `buyability_current` the random five-name book returns +1.08% a year against +0.45%
  filtered, so the look-ahead from additions is worth about +0.6 points a year on a
  five-name book; the top-5 by `score_binned` is -0.55% against random there (band
  [-1.79, 0.73]), no edge in either table. Removed companies are still missing in both
  (survivorship), which flatters every result; its size needs the historical membership
  (MD_202610_002).
- 2026-11-01, correction: the stored `forward_excess` in `buyability_*.parquet` is the
  3-month excess return, not 12-month (correlation 0.997 with a recomputation at 3
  months, 0.45 at 12). Checks 1 to 3 and the universe comparison above therefore
  measure the 3-month horizon, not the 12-month horizon the score is configured for
  (`score_horizon_months = 12`). Check 1 used 12-month closed windows, which is
  conservative, not a leak. To rerun at 12 months, take the excess from
  `window_returns`. Open: why the column is 3-month.
- 2026-11-01, check 4, top 5 by `score_binned` against random five-name books on the
  same dates, excess over `^SP500TR` per window (not annualised), training companies
  only (seed 1), horizons fixed beforehand (4 tried), the test companies dropped
  first:

  | Horizon | Dates | Top 5 | Random 5 | Advantage | 95% band | Beat index |
  |---|---|---|---|---|---|---|
  | 1 month | 67 | 0.83% | 0.10% | 0.72% | [-0.04, 1.54] | 60% |
  | 3 months | 66 | 0.77% | 0.48% | 0.29% | [-0.89, 1.53] | 52% |
  | 6 months | 65 | 2.10% | 0.79% | 1.31% | [-0.66, 3.52] | 49% |
  | 12 months | 63 | 4.85% | 1.64% | 3.21% | [0.43, 6.20] | 67% |

  At 12 months the advantage is 0.80% in the early half and 5.55% in the late half. The
  bands treat dates as independent, but the 6- and 12-month windows overlap, so they
  are too narrow; four horizons were tried, so the 12-month result is indicative only.
- 2026-11-01, check 5, timing: rank correlation of `score_binned` with the 12-month
  excess for windows starting L months from the pick date (training companies, seed 1,
  recomputed from `window_returns`). L = -12: -0.119 (top quintile -4.8 points);
  -6: -0.071; -3: -0.037; 0: -0.004 (top quintile -0.06 points); +3: -0.001; +6: +0.001;
  +12: +0.010. After the pick date there is no broad signal at any lag, and the early
  half is slightly positive (+0.02) while the late half is slightly negative (-0.02 at
  L = 0). Before the pick date the high scorers have lagged the index (cheap because the
  price fell). Dividends are included: the returns use Yahoo `adj_close` (KO 4.6x price
  growth, 8.6x adjusted over 2006 to 2026) against the total-return index `^SP500TR`.
  Reading: the 12-month advantage of the top five (check 4) is a tail effect, not a
  broad ordering; the top quintile shows none.
- 2026-11-01, check 6, multi-horizon (user: 12, 18, 24 and 36 months to tighten the
  bands, then the 5-year view), top 5 by `score_binned` against random five-name books on
  the same dates, S&P 500 training companies (seed 1), returns with dividends, excess
  over `^SP500TR`, moving-block bootstrap with blocks of one window length; five
  horizons tried; the per-year figure is the cumulative advantage divided by the years:

  | Horizon | Dates | Independent windows | Top 5 | Random 5 | Advantage a year | 95% band | Beat random | Beat index |
  |---|---|---|---|---|---|---|---|---|
  | 12 months | 63 | 16.5 | 4.8% | 1.6% | 3.2% | [-0.8, 6.5] | 57% | 67% |
  | 18 months | 61 | 11.0 | 7.8% | 2.3% | 3.6% | [-0.5, 8.3] | 59% | 64% |
  | 24 months | 59 | 8.3 | 12.9% | 3.1% | 4.9% | [0.9, 11.0] | 62% | 69% |
  | 36 months | 55 | 5.5 | 35.5% | 6.0% | 9.8% | [1.6, 14.4] | 64% | 76% |
  | 60 months | 47 | 3.3 | 26.4% | 16.4% | 2.0% | [-1.0, 8.8] | 55% | 64% |

  The point estimate is positive at every horizon; the bands exclude zero at 24 and 36
  months only. Concentration: five names give 81% of the positive excess at 36 months
  (MU, NVDA, NUE, ALB, BMY, QCOM) and 75% at 60 months (NVDA, MSFT, NUE, SYK, QCOM,
  CSCO). The 5-year advantage of 2.0% a year equals the sleeve target, but with three
  independent windows and a band of [-1.0, 8.8].
- 2026-11-01, letters reading (`AGENTS.md`, "Methodology vs refactor"; letters in
  `data/interim/berkshire_letters`; commodity, cigar-butt, owner-earnings, holding and
  earnings-range passages read; the 1977 to 2024 set not read in full). Direction, to be
  tested as hypotheses on the training companies, none yet a decision:
  1. Cyclical and commodity businesses (1982, 1985, 1987, 1989): with excess capacity
     and an undifferentiated product they earn well only in a shortage, and managers add
     capacity as soon as one appears; they do well only "if they have a cost advantage
     that is both wide and sustainable" (1982). The top-five winners include cyclicals
     (MU, NUE, ALB), so a cheap score on trough or peak earnings is a trap or a lucky
     cycle. Test: score and returns split by earnings volatility or sector cyclicality.
  2. Bargain against quality (1989): "it's far better to buy a wonderful company at a
     fair price than a fair company at a wonderful price"; the "cigar butt" bargain
     erodes at the business's low return, and "time is the friend of the wonderful
     business, the enemy of the mediocre". Our high scorers are past losers
     (check 5), cheap because the price fell. Test: whether a return-on-capital or
     margin-stability screen on top of cheapness lifts the 5-year result.
  3. Owner earnings (1986): reported earnings plus depreciation and non-cash charges,
     minus the average capital expenditure needed to hold the competitive position;
     "cash flow" without the last term is deceptive. Test: an owner-earnings yield, or
     capital intensity, as a rule input.
  4. Predictability (2013): buy only when an earnings range five years out can be
     estimated, at a reasonable price against its lower bound; otherwise "move on".
     Test: earnings-stability filter before ranking; companies that fail are not
     scored, not scored low.
  5. Holding (1988, 1996): "our favorite holding period is forever"; "if you aren't
     willing to own a stock for ten years, don't even think about owning it for ten
     minutes". Supports the 5-year horizon (decision above).
  Each is a methodology change to be tested alone, with the number of combinations
  recorded.
- 2026-11-01, check 7, the ordered analysis (user: "do what the Buff says"; design fixed
  before the run, one design tried): stage 1 G3 passes; stage 2 B1 and G2B pass (failed
  or unevaluated companies are dropped); stage 3 survivors ranked by `g6_value` (P/E on
  averaged earnings, positive, lowest first), `g7_value` as tie-break; top 5 held.
  S&P 500 training companies (seed 1), dividends included, moving-block bootstrap, only
  dates with at least 5 ranked survivors. Survivors per date: stage 1 median 104 (0 to
  194), stage 2 median 10 (0 to 18); 32 of 71 dates had fewer than 5, so the sample is 35
  dates or fewer and later in time than checks 4 and 6 (not comparable with them).

  | Horizon | Dates | Independent windows | Top 5 | Advantage a year over random 5 from the full universe | over random 5 from the survivors | Beat index |
  |---|---|---|---|---|---|---|
  | 12 months | 35 | 9.5 | -3.7% | -3.4% [-8.9, 3.1] | -3.7% [-11.1, 1.7] | 40% |
  | 24 months | 31 | 4.8 | -9.7% | -4.3% [-8.4, 1.7] | -3.8% [-8.8, -0.3] | 39% |
  | 36 months | 27 | 3.2 | -11.6% | -3.6% [-7.6, 1.2] | -2.3% [-4.4, 0.5] | 33% |
  | 60 months | 19 | 1.9 | -8.4% | -3.1% (band not meaningful) | +2.3% (not meaningful) | 42% |

  At 36 and 60 months five names give 92% of the positive excess (TSCO, TPR, QCOM, META,
  CMI). The picks trail the index at every horizon and trail random books from the same
  survivors at 12 to 36 months. Not tuned after seeing it: any change of gate or ranking
  is a new design and is counted.
- 2026-11-01, check 8, gates narrow the pool and the score ranks it (user: enhance the
  score, do not eliminate it; all the stats, with Buffett filtering). Three designs
  fixed before the run, each compared with the plain `score_binned` top 5 on the same
  dates; S&P 500 training companies (seed 1); designs tried so far on the gated side: 4
  (check 7 and D1 to D3). G3 requires positive diluted EPS in each of ten fiscal years
  and all ten present, so the gated pools are empty on early dates (EDGAR history): D1
  G3 then score (pool median 133; 40 of 68 dates with at least 5), D2 G3, B1 and G2B
  then score (median 10; 39 dates), D3 G3 and the G6 ceiling then B1 value (median 9; 39
  dates). Advantage a year, with the 95% band:

  | Design | Horizon | Over random 5, full universe | Over the plain score |
  |---|---|---|---|
  | D1 | 12 months | +2.7% [-4.2, 7.6] | -3.0% [-6.3, -0.1] |
  | D1 | 24 months | +1.9% [-1.9, 8.6] | -3.0% [-4.5, -0.8] |
  | D1 | 36 months | +0.9% [-0.2, 5.4] | -11.2% [-12.2, -1.5] |
  | D2 | 12 months | +0.6% [-2.8, 5.5] | -5.0% [-9.4, 2.1] |
  | D2 | 24 months | -2.1% [-3.1, 0.2] | -7.2% [-12.2, -3.4] |
  | D2 | 36 months | -2.3% [-3.2, -0.4] | -14.8% [-17.5, -4.9] |
  | D3 | 12 months | -4.0% [-8.7, -0.1] | -9.6% [-14.3, -3.7] |
  | D3 | 24 months | -3.6% [-4.8, -0.3] | -8.8% [-14.0, -4.3] |
  | D3 | 36 months | +2.1% [-5.0, 4.9] | -10.4% [-13.1, -4.8] |

  The 60-month rows rest on about 2 independent windows and are not meaningful. On the
  same dates the plain score returns 5.5%, 9.2% and 36.3% over 12, 24 and 36 months;
  every gated design is below it at every horizon. The plain score's winners
  (MU, NVDA, ALB) have volatile earnings; the stability gate removes them. D1's winners
  are AAPL, FAST, QCOM, NUE, PKG. Over the full universe no gated design shows a
  significant edge. Nothing tuned after seeing it.
- 2026-11-01, check 9, 3-year windows (user: cut all long-history rules for this test).
  Scored in memory from a copy of the configuration, nothing written to the repository
  (script and table in the scratchpad): G3 and G4 at 3 years, B1 at 3 years with 2 of 3
  at ROE 15% or more, G5 at 6 years (the minimum for its 3-year averages); the score
  therefore changes for every company. Pass shares of the evaluated: G3 86%, B1 19%,
  G2B 20%. Designs fixed before the run (designs tried on the gated side: 6): D5a G3 then
  score (pool median 202), D5b G3, B1 and G2B then score (pool median 25). 68 and 67 of
  68 dates have at least 5 names. Advantage a year, with the 95% band:

  | Horizon | Plain score (reference) 5-name excess | D5a over random 5 | D5a over the plain score | D5b over random 5 | D5b over the plain score |
  |---|---|---|---|---|---|
  | 12 months | 3.5% | +1.7% [-2.6, 5.6] | -0.2% [-0.8, 0.2] | +2.9% [-2.8, 8.1] | +1.0% [-2.6, 4.6] |
  | 24 months | 12.4% | +4.6% [-0.1, 12.6] | -0.1% [-0.6, 0.6] | +4.8% [-0.5, 13.2] | +0.1% [-7.2, 6.4] |
  | 36 months | 26.3% | +7.4% [2.0, 15.7] | +0.4% [-0.2, 1.6] | +4.3% [-1.7, 12.3] | -2.7% [-14.2, 6.0] |
  | 60 months | 72.9% | +11.3% [5.5, 17.2] | -0.4% [-0.7, 0.0] | +13.1% [-2.0, 24.8] | +1.4% [-13.4, 11.1] |

  Reading: with 3-year windows the plain score itself is much stronger than with the
  10-year windows (60 months: 72.9% against 16.4% before, on different picks and a
  different rule set), but the gates add nothing to it: G3 at 3 years passes 86% of
  companies, so D5a is the plain score, and D5b is not distinguishable from it. NVDA
  tops the winners (D5a at 36 months: NVDA, NUE, ALB, KLAC, BMY, CSCO give 77% of the
  positive excess), 60 months rests on 3.3 independent windows, and the removed companies
  are still missing from the universe. To check: the result without the single largest
  winner, and why the shorter windows change the picks.
- 2026-11-01, check 10, the signal of the plain score with 3-year windows (no gates;
  top 5 by `score_binned`; S&P 500 training companies, seed 1; 68 quarterly dates,
  December 2009 to September 2026; dividends included; moving-block bootstrap). The
  seed-1 test companies are still unseen. Advantage a year over random five-name books:

  | Horizon | Dates | Independent windows | Top 5 excess (cumulative) | Advantage a year over random 5 | Beat index | Five largest contributors' share of the positive excess |
  |---|---|---|---|---|---|---|
  | 12 months | 63 | 16.5 | 3.5% | +1.9% [-2.4, 5.9] | 59% | 53% (NVDA, CSCO, NUE, ON, KLAC) |
  | 24 months | 59 | 8.3 | 12.4% | +4.7% [0.0, 12.4] | 66% | 72% (NVDA, NUE, CSCO, MU, KLAC) |
  | 36 months | 55 | 5.5 | 26.3% | +7.0% [1.8, 15.2] | 64% | 76% (NVDA, NUE, KLAC, BMY, CSCO) |
  | 60 months | 47 | 3.3 | 72.9% | +11.7% [5.5, 17.7] | 77% | 82% (NVDA, MU, NUE, MSFT, QCOM) |

  Limits: the score measured is `score_binned`, not the report's
  `score_binned_weighted` (learned weights, not yet recomputed on training companies
  only); most of the bands treat blocks of one window as independent; one configuration
  chosen for data reasons and then looked at; the gains lean on a few semiconductor and
  materials names in a long bull market; removed companies are missing from the
  universe; the broad ordering of the score under the 3-year windows is not measured.
- 2026-11-01, correction: `score_binned_weighted` is not learned from returns. Its
  weights are 1 minus each rule's pass rate on the date (`scoring.py`); the learned
  score is `score_ic`. My earlier reason for excluding it from checks 3 to 10 was wrong.
  Its grades and weights still come from all companies of the date, test ones included
  (their fundamentals, not their returns), which the strict rule does not allow.
- 2026-11-01, check 11, strict scores. Grades and pass-rate weights computed from the
  training companies only; each test company is placed on the training distribution of its
  date (script in the scratchpad; the strict training `score_binned` correlates 0.997 with
  the stored one). Top 5, 3-year windows, S&P 500 training companies (seed 1), advantage
  a year over random five-name books:

  | Score | Horizon | Independent windows | Top 5 excess (cumulative) | Advantage a year | Beat index | Share of five largest (names) |
  |---|---|---|---|---|---|---|
  | `score_binned` | 12 months | 16.5 | 4.1% | +2.5% [-2.3, 7.1] | 60% | 59% (NVDA, NUE, CSCO, ON) |
  | `score_binned` | 24 months | 8.3 | 13.4% | +5.2% [-0.4, 14.4] | 68% | 76% (NVDA, NUE, CSCO, KLAC) |
  | `score_binned` | 36 months | 5.5 | 27.8% | +7.5% [2.0, 16.6] | 67% | 77% (NVDA, NUE, KLAC, BMY) |
  | `score_binned` | 60 months | 3.3 | 80.4% | +13.2% [6.7, 21.3] | 77% | 84% (NVDA, MU, NUE, MSFT) |
  | `score_binned_weighted` | 12 months | 16.5 | 3.7% | +2.2% [-1.8, 5.9] | 57% | 72% (NVDA, MU, NUE, CSCO) |
  | `score_binned_weighted` | 24 months | 8.3 | 15.7% | +6.3% [2.3, 13.8] | 69% | 78% (NVDA, NUE, WDC, MU) |
  | `score_binned_weighted` | 36 months | 5.5 | 44.0% | +12.9% [3.5, 18.6] | 80% | 83% (NVDA, MU, NUE, WDC) |
  | `score_binned_weighted` | 60 months | 3.3 | 74.9% | +12.1% [7.8, 17.2] | 72% | 85% (NVDA, MU, MSFT, NUE) |

  The weighted score is the stronger at 24 and 36 months and more concentrated.
- 2026-11-01, registered before the test run (user: test the frozen specification once on
  the unseen companies): specifications frozen, both counted: S1 `score_binned`, S2
  `score_binned_weighted`, each with the 3-year windows of check 9, grades and weights
  from the training companies (check 11). Seed-1 test companies, 10% of the companies.
  On each date with at least 25 scored test companies, the top 4 (about 10%) by score
  against random 4 from the test companies; horizons 12, 24, 36 and 60 months; advantage
  a year with a moving-block bootstrap band; plus the rank correlation of score and
  excess over the test companies, and the top quintile's excess over the mean. Reported
  whatever it shows; no change to the specification afterwards, and the split is then
  spent.
- 2026-11-01, check 12, the registered test run (seed-1 test companies: 40 of 392; the
  split is now spent). Top 4 by score against random 4 from the test companies, advantage
  a year with the 95% band; rank correlation of score and excess; top quintile's excess
  over the mean a year:

  | Score | Horizon | Independent windows | Advantage a year | Beat index | Rank corr. | Top quintile over mean |
  |---|---|---|---|---|---|---|
  | `score_binned` | 12 months | 11.5 | +11.9% [1.1, 20.8] | 65% | 0.066 | +6.8% |
  | `score_binned` | 24 months | 5.8 | +10.5% [2.9, 17.8] | 56% | 0.091 | +6.9% |
  | `score_binned` | 36 months | 3.8 | +12.0% [4.8, 17.9] | 80% | 0.096 | +6.5% |
  | `score_binned` | 60 months | 2.3 | +14.2% (band not meaningful) | 89% | 0.043 | +10.6% |
  | `score_binned_weighted` | 12 months | 11.5 | +13.9% [2.4, 23.0] | 65% | 0.088 | +10.5% |
  | `score_binned_weighted` | 24 months | 5.8 | +13.0% [4.6, 19.9] | 62% | 0.113 | +7.9% |
  | `score_binned_weighted` | 36 months | 3.8 | +13.5% [6.6, 18.7] | 80% | 0.104 | +7.2% |
  | `score_binned_weighted` | 60 months | 2.3 | +19.1% (band not meaningful) | 96% | 0.069 | +10.8% |

  Concentration (36 months, both scores): only 8 distinct names were ever picked, and
  LRCX and AMAT give 99% of the positive excess (the test companies include two
  semiconductor-equipment names that rose strongly). Without LRCX the advantage falls to
  +3.7% (`score_binned`) and +4.1% (`score_binned_weighted`) a year; without LRCX, AMAT and
  KEYS it is -3.7% and -1.2%. Reading: the test confirms the picks of the score can
  beat the index and random books when they include a large winner, but it does not show a
  general edge beyond two names; the training result has the same shape (NVDA, MU, KLAC,
  WDC, STX, QCOM). The common thread is semiconductor and hardware names, which matches
  the cyclical-business direction of the letters (note 1 above). The positive rank
  correlation (0.04 to 0.11) across all 40 test companies is small but of the right sign.
  The test split is spent; a fresh split is drawn for any later test.
- 2026-11-01, the 3-year windows applied to `config/rules.toml` (user: "recompute the
  report"; unstaged for review): G3 years 3, G4 years 3, G5 years 6, B1 years 3 with
  `min_years_passed` 2; `[schedule] history_years = 13` left as it is. Methodology change.
  Run for the S&P 500 only (`buyability`, `report`, as of 2026-10-31); the S&P 400 and 600
  results still carry the old windows. Checks: 15 tests pass, Black clean. The report's
  buy-and-hold wallet barely moves: the 2020 investor 34.2% a year (33.9% before) against
  15.9% for the index, the 2016 investor 27.2% against 15.3%, the 2023 investor 41.9%
  against 18.7%. Its next picks are unchanged (FSLR, SNA, CTSH, RMD, SWKS; market capitalisation
  $13 to 32 billion). The old results remain in git for the before and after.
- 2026-11-01, why big companies appeared in the study picks: the size profile of the top
  five is about the same under both windows (median market-cap percentile 0.51 old, 0.47
  new; largest 20%: 22% old, 24% new). What changed is which names. `score_binned` is a sum
  of points, so a rule that cannot be evaluated gives no points; under the 10-year windows
  G5 (13 years of growth) was not evaluable for NVDA and MSFT in 2019 (n_available 15, score
  40 and 45) and with the 6-year window it gives them 5 and 4 points (score 52 and 50). The
  old score favoured companies with long histories. The report's picks are unchanged since
  the current leaders are not the historical ones.
- 2026-11-01, check 13, the first study as a plain script (no cache; scratchpad
  `quick_study.py`, results `quick_study.csv`): the current configuration plus 10 Latin
  hypercube samples of `config/studies/default.toml` (seed 1), S&P 500 training companies
  only (the seed-1 test companies dropped before any score), strict training-only grades,
  `score_binned_weighted`, objective the mean 36-month excess of the top `names` a year
  (simple, cumulative divided by 3), 55 dates. 11 combinations tried. No bands, no test
  companies, no comparison with random books yet.

  | Scenario | names | buckets | average years | G3 | E3 | E5 | B1 | G4 | 36-month excess a year | Share of five largest |
  |---|---|---|---|---|---|---|---|---|---|---|
  | current | 5 | 6 | 3 | 3 | 5 | 4 | 3 | 3 | 14.7% | 83% |
  | s09 | 5 | 4 | 3 | 2 | 4 | 3 | 6 | 5 | 11.6% | 76% |
  | s01 | 5 | 6 | 2 | 5 | 5 | 5 | 7 | 7 | 10.7% | 77% |
  | s05 | 10 | 5 | 3 | 5 | 6 | 4 | 3 | 10 | 9.3% | 67% |
  | s06 | 5 | 8 | 6 | 3 | 7 | 3 | 3 | 3 | 8.7% | 70% |
  | s10 | 10 | 10 | 7 | 3 | 4 | 3 | 7 | 8 | 8.4% | 74% |
  | s02 | 10 | 10 | 5 | 6 | 7 | 5 | 6 | 9 | 8.0% | 70% |
  | s08 | 10 | 4 | 5 | 6 | 3 | 4 | 4 | 4 | 7.2% | 72% |
  | s04 | 20 | 6 | 7 | 4 | 6 | 6 | 5 | 8 | 5.8% | 68% |
  | s03 | 20 | 7 | 4 | 2 | 5 | 6 | 5 | 3 | 5.7% | 68% |
  | s07 | 20 | 9 | 3 | 4 | 3 | 5 | 4 | 6 | 5.5% | 66% |

  Reading: the number of names dominates (correlation with the objective -0.90): 5 names
  give 8.7% to 14.7% a year, 10 names 7.2% to 9.3%, 20 names 5.5% to 5.8%, all positive.
  The other parameters cannot be separated from `names` with 10 samples. The current
  configuration beats every sample, but it was the one looked at (selection). Top
  contributors in every run: NVDA, MU, NUE (MSFT and KLAC with more names or more buckets).
  Next: fix `names` per tier and search the other seven parameters within each, with more
  samples, then the rule ablation.
- 2026-11-01 (user): `names` is fixed at 10 for the current optimisation (study
  `config/studies/names_10.toml`). Reasoning: about 5% a year is more plausible than 10% a
  year from 5 companies, and a book of 5 with the higher figure belongs behind the paywall
  (Assay notes, "Tiers"); diversification answers the concentration.
- 2026-11-01, check 14, the 10-name study (scratchpad `quick_study2.py`, results
  `quick_study_names10.csv`): the current configuration plus 30 Latin hypercube samples
  of the other seven parameters; training companies only, strict grades,
  `score_binned_weighted`, top 10; 31 combinations tried here (42 with check 13). Objective
  the 36-month excess a year of the top 10, in hindsight on one history, no bands.
  The samples span 5.9% to 10.5% (mean 8.4%, standard deviation 1.1 points); the current
  configuration gives 10.1%, second of 31. The advantage over random 10-name books is
  about 8 points a year at the top and 4 at the bottom. Correlation of each parameter with
  the objective: averaging years -0.70, buckets +0.55, E5 years +0.32, E3 years -0.16, B1
  years +0.09, G3 years -0.03, G4 years -0.03. Best sample s13 (10.5%): buckets 10,
  averaging years 2, G3 2, E3 4, E5 6, B1 6, G4 8; the differences among the top ten
  (9.2% to 10.5%) are within noise. Reading: a shorter earnings average and more buckets
  help, the long-history rule windows do not matter; shorter averaging follows recent
  earnings and may favour cyclical recoveries, to be checked against the letters
  (cyclicals, note 1). Next: the response surface on these runs, a real run at its
  optimum, the rule ablation, then the 10 stratified test samples for the final pick.
- 2026-11-01, check 15, confirming run (scratchpad `confirm_run.py`; 44 combinations tried
  in all): buckets 10 and averaging years 2, the rest as the current configuration, top 10,
  36-month excess a year 10.7% against 10.1%. The paired difference over the same 55 dates
  is +0.6 points with a time-bootstrap band of [-0.5, +1.0]; at 12, 24 and 60 months the
  figure is higher by 1.5, 1.2 and 2.9 points. Rule (set before the run): a parameter
  changes only if the paired band excludes zero. It does not, so the current configuration
  stays (buckets 6, averaging years 3, G3 3, E3 5, E5 4, B1 3 of which 2, G4 3). Buckets
  10 with averaging years 2 is listed as a candidate for the studio tier.
- 2026-11-01 (user): the report shows 10 picks (`[wallet] names = 10`, unstaged). The
  picks are weighted by score (score over the sum of scores); `rank_weights` belongs to the
  rank weighting, which is off (`weightings = ["score"]`), so nothing was extended. Report
  regenerated for the S&P 500; money-weighted return a year, 10 picks (5 picks before):
  2016 investor 29.8% (27.2%), 2020 29.3% (34.2%), 2023 35.0% (41.9%), 2025 31.5% (24.7%)
  against 15.3%, 15.9%, 18.7% and 15.7% for the index. The 2020 investor's 1,400 USD is
  worth 4,002 USD (4,757 before) against 2,480 USD in the index. Formatting: the abstract
  and the tables follow the 10; two phrases still say five ("the five highest-scoring
  companies", the truncation-check caption), which are in `config/report_value_investing.toml`
  (the scribe's); LaTeX reports underfull lines in the appendix tables.
- 2026-11-01 (user): the sampling bands use 12 stratified draws (seeds 2 to 13), not 10.
  The next-picks plot is split into rows of 5 (10 picks: 2 rows by 5 columns).
- 2026-11-01 (user): the sampling bands do not expect identical picks, only similar
  returns; dropping a stratified 10% of the companies can remove a candidate that a run
  would pick. The bands therefore also carry a pick-stability table: the share of the ten
  draws in which each pick appears, and which draws removed it.
- 2026-11-01, check 16, the independent indices (user: "big test: 400 and 600 with this
  configuration"; scratchpad `index_check.py`; results and checks only, no report): the
  frozen `config/rules.toml` (10 names, `score_binned_weighted`, grades from each index's
  own companies), against IJH and IJR (dividends reinvested, net of fee), time-bootstrap
  bands, in memory. Excess return of the top 10 a year, and over random 10-name books of the
  same pool and dates:

  | Index | Horizon | Dates | Independent windows | Top 10 excess a year | Over random 10 | Beat the fund | Five largest (names) |
  |---|---|---|---|---|---|---|---|
  | S&P 400 | 12 months | 55 | 14.5 | +3.3% [-1.5, 6.5] | -0.3 [-3.7, 2.0] | 62% | 45% (CMC, RS, MTZ) |
  | S&P 400 | 24 months | 51 | 7.3 | +1.5% [-2.2, 6.4] | -1.9 [-4.3, 1.1] | 63% | 59% |
  | S&P 400 | 36 months | 47 | 4.8 | +2.1% [-1.5, 8.3] | -1.4 [-3.8, 2.5] | 57% | 59% (RS, CMC, LPX) |
  | S&P 400 | 60 months | 39 | 2.9 | +1.6% [-2.7, 5.4] | -2.8 [-5.9, -0.3] | 59% | 67% |
  | S&P 600 | 12 months | 24 | 6.7 | +8.5% [1.7, 12.9] | +2.0 [-3.1, 5.7] | 62% | 57% (SIG, BHE, HCC) |
  | S&P 600 | 24 months | 20 | 3.4 | +5.9% [2.0, 10.5] | -1.7 [-4.8, 0.0] | 75% | 65% |
  | S&P 600 | 36 months | 16 | 2.2 | +9.7% [8.5, 14.3] | +0.5 [-1.4, 2.3] | 81% | 60% (SIG, BHE, IPAR) |
  | S&P 600 | 60 months | 8 | 1.3 | +10.6% (not meaningful) | -0.1 (not meaningful) | 100% | 61% |

  Reading: on both indices the top 10 beats its fund but does not beat random 10-name
  books of the same universe (point estimates -2.8 to +2.0, every band includes zero, the
  60-month 400 band excludes it on the negative side). The S&P 500 advantage over random
  books (about 8 points a year at 36 months) is not reproduced. The excess over the fund
  comes from the universe: equal-weighted random books beat IJH by about 3.5 and IJR by
  6 to 9 points a year, which the membership look-ahead of the two lists (no entry dates)
  and equal weighting explain in part. The 600 has only 28 dates. The earlier reports of
  2026-10-31 (10-year windows, 5 names) showed the 600 wallet trailing IJR; the
  difference between those and this check is not yet explained.
- 2026-11-01, check 17, the minimal optimisation (user: search only the sensitive
  parameters; scratchpad `grid_run.py`; 52 combinations tried in all): a 3 by 3 grid of
  buckets (6, 8, 10) and averaging years (2, 3, 4), `names` fixed at 10, S&P 500 training
  companies, the rest as the current configuration (6 buckets, 3 years). Rule fixed before
  the run: adopt only if the 99% band of the paired difference at 36 months excludes zero
  (8 comparisons). Paired difference a year, 36 months, with the 99% band: buckets 6 and 2
  years +0.1 [-0.6, +0.4]; 6 and 4 -0.3 [-2.1, +0.5]; 8 and 2 -0.2 [-1.0, +0.9]; 8 and 3
  -0.4 [-1.3, +0.9]; 8 and 4 -0.8 [-2.5, +0.1]; 10 and 2 +0.6 [-0.8, +1.2]; 10 and 3 -0.4
  [-1.9, +0.2]; 10 and 4 -0.1 [-2.5, +1.3]. None excludes zero, so none is adopted. The
  surface is flat at the noise of the objective: the 36-month excess lies between 9.3% and
  10.7% in all nine cells, and 10 buckets with 2 years is the highest at 12, 24, 36 and 60
  months but not significantly. Decision: the configuration stays as in `config/rules.toml`
  (buckets 6, averaging years 3); the two parameters are insensitive within these ranges.
- 2026-11-01, check 18, power of the objective (user: why would the optimisation not work;
  scratchpad `power_run.py`; no new combinations, the same nine configurations): the same
  3 by 3 grid measured two ways, paired against the current configuration at 36 months,
  99% time-bootstrap band half-width in points a year. Top 10 objective: 0.5 to 1.9
  (differences -0.8 to +0.6, none significant). Top quintile minus the pool: 0.2 to 0.6
  (differences -0.4 to +0.4, none significant; the closest are buckets 8 with 3 years
  -0.24 [-0.36, +0.04] and buckets 6 with 4 years -0.41 [-0.89, +0.03]). The quintile
  objective resolves about 2 to 4 times finer, and still finds no effect: the parameters
  are insensitive within these ranges at about 0.4 points a year, so the flat surface is
  not only noise from a small sample. The top 10 is a noisy target (3 to 4 names carry
  it); a tuning objective should use more of the data (a quintile spread) and the traded
  book is reported beside it.
- 2026-11-01, check 19, two periods (scratchpad `period_split.py`; S&P 500 training
  companies, seed 1; strict `score_binned_weighted`, top 10, against random 10-name books;
  random books beat the index by 3.4% a year in 2009 to 2016 and by -0.1% in 2017 to
  2026). Advantage over random books a year with the time band: 2009 to 2016, 12 months
  +1.8 [-2.4, 7.7], 36 months +6.8 [0.6, 14.5] (28 dates, beat index 79%, five names give
  77%: NVDA, MSFT, NTAP); 2017 to 2026, 12 months +3.0 [-4.0, 8.7], 36 months +9.7 [3.6,
  12.1] (27 dates, beat index 70%, five names give 72%: MU, NVDA, NUE). The edge at 36
  months is present in both periods, so it is not only the flattered early years, but each
  period has about three independent windows and rests on a few names, and the 400 and 600
  did not reproduce it (check 16).
- 2026-11-01, check 20, dispersion, picks and overlap on the three indices (user: is there
  enough volatility to explain the picks; fewer picks for the 600; why the 600 does not
  reproduce the picks; scratchpad `dispersion.py`; frozen configuration; 36-month windows;
  the S&P 500 line uses all companies of the table, a diagnostic, not the strict test).
  Dispersion across companies, the standard deviation of the 36-month return: S&P 500 0.69
  (0.61 in 2009 to 2016, 0.77 in 2017 to 2026); S&P 400 0.64 (0.54, 0.71); S&P 600 1.17 (only
  2017 to 2026 has a closed window). The 10th to 90th percentile spread of the yearly
  excess is 43 points (500), 43 (400) and 66 (600). Share of companies beating the fund
  by size third within the index: 500 49%, 45%, 43% (smallest to largest); 400 66%, 47%,
  33%; 600 70%, 52%, 36%. Top 5 and top 10 against random books of the same size, advantage
  a year at 36 months: S&P 500 +13.9 [5.1, 20.3] and +10.4 [4.8, 15.5]; S&P 400 +0.8 [-3.9,
  6.2] and -1.4 [-3.9, 2.4]; S&P 600 -1.3 [-6.6, 1.2] and +0.5 [-1.4, 2.3] (16 dates).
  Picks kept from the previous quarter: 79% (500, 5 and 10 names), 74% and 79% (400), 71% and
  72% (600). Reading: dispersion is ample in all three, the 600 the most, so lack of
  volatility does not explain the 400 and 600; fewer picks for the 600 changes nothing
  (-1.3 at 5 names, +0.5 at 10); the overlap is a property of a slow-moving score (about
  three quarters of the picks stay), not of the number of picks. The smallest third of the
  400 and 600 beats its cap-weighted fund 66% to 70% of the time, a size effect that
  flatters random books there. The three indices hold different companies (the lists share
  none, three between the 400 and 600), but companies move: 212 symbols are in the change
  histories of both the 500 and the 400, so winners that graduated are missing from the 400
  and 600 tables in their small-cap years (see the notes, "Index migration").
- 2026-11-01 (user): "n picks" means exactly n picks. `book_weights` included every company
  tied with the last one chosen, so a book could hold more than n; it now takes exactly `names`
  companies and breaks a tie by symbol (`evaluation.py`, comment in `config/rules.toml`,
  unstaged). Wallet accounting, so a methodology change; with scores that are sums of
  weights times points, ties are rare and the report's 10 picks per decision were already 10.
  Black, ruff, pyright and the 15 tests pass.
- 2026-11-01, check 21, the sampling bands (scratchpad `sampling_bands.py`, results
  `sampling_bands.csv` and `sampling_stability.csv`; S&P 500, the report's wallet with 10
  picks): 12 stratified draws (seeds 2 to 13), each dropping a market-cap-stratified 10% of
  the companies, regrading the other 90% (grades and pass-rate weights from them only),
  running the buy-and-hold wallet. Money-weighted return a year of the value wallet, the
  report's run against the 12 draws (mean, standard deviation, minimum, maximum), and the
  index:

  | Start | Report | Draws mean | Standard deviation | Minimum | Maximum | Index | Wallet beats the index |
  |---|---|---|---|---|---|---|---|
  | 2016 | 29.8% | 28.4% | 2.4 | 22.5% | 30.9% | 15.3% | 12 of 12 draws |
  | 2020 | 29.3% | 28.2% | 3.2 | 21.2% | 32.0% | 15.9% | 12 of 12 |
  | 2023 | 35.0% | 33.6% | 3.6 | 24.9% | 37.2% | 18.7% | 12 of 12 |
  | 2025 | 31.5% | 33.5% | 4.6 | 28.0% | 41.4% | 15.7% | 12 of 12 |

  Excess over the index a year: mean 12.3 to 17.9 points, smallest 5.3 to 12.4. Pick
  stability over the reference picks of all decisions: a pick is removed from the universe in
  11% of the draws, by design; when it stays it is picked again in 97% of the draws (median
  per pick 100%, 10th percentile 92%); the least stable are VLO 78%, HUM 85%, BIIB 86%, PWR
  86%, NVDA 87%. Reading: the result is robust to which 10% of the companies are in the
  sample, with similar returns and mostly the same picks, as the user expected. The band
  covers the company sample only; it does not cover the choice of specification, the
  market history (one bull market) or multiple testing (check 16, 19 and 20 carry those), and
  the draws are in-sample for the configuration, which was chosen on seed-1 training
  companies.
- 2026-11-01 (user): the test sampling depends on the size of the universe. For tiny sets
  use smaller samples, consider dropping the weighting, and run fewer tests so that each can
  hold more; none of the numbers (10%, 12 draws) is fixed, and the default is 5 (read: the
  default number of picks, 5, to be confirmed with the user). It is a parameter of the
  optimisation (`wallet.names`, tiers 5, 10 and 20).
- 2026-11-01 (user): the final picks shown to a client come from the full set of
  companies, scored together; the test split is for validation only, so a best pick that sits
  in a test sample is still offered. Near-tied candidates as an unlock: `assay/notes.md`.
- 2026-11-01, check 22, heavy hitters removed (user: remove a heavy hitter as if it never
  existed, and see whether we would pick it; the bands are too good to be true; scratchpad
  `heavy_hitters.py`, results `heavy_hitters.csv`): the largest contributors to the gain of the
  2016 investor's wallet are NVDA 31% of the total positive gain, MU 18%, LRCX 11%, AMAT 5%,
  NUE 4%, STX 3.5%, PWR 3%, CMI 2%. Each case removes the top k companies from the universe,
  regrades the rest from scratch and runs the report's wallet (10 picks); the draw mean and
  minimum are over 12 stratified 10% draws on top. Value wallet minus the index, points a
  year, deterministic run (draws mean, draws minimum):

  | Removed | 2016 investor | 2020 | 2023 | 2025 |
  |---|---|---|---|---|
  | none | 14.4 (13.1, 7.2) | 13.4 (12.3, 5.3) | 16.3 (14.9, 6.2) | 15.8 (17.9, 12.4) |
  | NVDA | 9.8 (9.1, 6.9) | 13.2 (12.1, 5.2) | 16.3 (15.2, 6.0) | 15.8 (18.6, 12.4) |
  | NVDA, MU, LRCX | 4.6 (4.5, 2.9) | 3.9 (4.0, 2.5) | 4.7 (4.4, 1.5) | 15.8 (19.1, 12.4) |
  | NVDA, MU, LRCX, AMAT, NUE | 2.8 (3.2, 1.9) | 0.1 (1.4, 0.0) | 1.7 (2.7, 0.8) | 13.9 (14.6, 10.6) |

  Would the score still pick a removed company? Each was placed on the grades of everyone
  else on each date and ranked: top 10 on 12 of 67 dates for NVDA (picked in 3 of the
  report's decisions), 21 of 60 for MU (11), 24 of 57 for LRCX (9), 30 of 68 for AMAT (11) and 34
  of 67 for NUE (15). So the score finds them whether or not they were in the grading set,
  on 18% to 51% of the dates, not every time. Reading: the 2016 to 2023 investors' excess
  over the index, 13 to 16 points, falls to about 4 points without the top three and to
  0 to 3 without the top five; the 2025 investor's excess (14 to 19 points) does not depend
  on them. The headline of the report must carry this stress band beside the 12-draw band: the
  edge is concentrated in a few semiconductor winners (Nvidia, Micron, Lam Research,
  Applied Materials) and Nucor.
- 2026-11-01, check 23, the removal test on the other indices (scratchpad `index_stress.py`):
  recorded with check 22 as accreditation test AT-1 in `midas/accreditation.md`. With the
  three largest contributors removed the wallet's edge over its fund is gone on both: S&P 400
  (against IJH) 0.3, -0.6, -3.0 and +3.0 points a year for the 2016, 2020, 2023 and 2025
  investors, S&P 600 (against IJR) -0.2, -0.2, -4.2 and +0.2. The edge over the fund without
  removal is 2 to 4 points on the 400 and 2 or less on the 600; the 400 and 600 tables are
  biased (migration, membership look-ahead), so this is not yet a verdict on the rules.
- 2026-11-01, check 24, who we missed (user: which successful stocks were available in the
  S&P 400 besides the top three, and why we did not pick more; scratchpad
  `missed_winners.py`; hindsight, diagnostic only, never a rule): at each date with a closed
  36-month window, the 10 companies with the highest realised excess return ("ex-post
  winners") against the 10 we would pick.

  | Index | Dates | Ex-post winners' excess a year | Our top 10 | Universe mean | Winners in our top 10 (by chance) | Winners' mean score percentile |
  |---|---|---|---|---|---|---|
  | S&P 500 | 55 | 76% | 12.1% | 1.7% | 10% (4%) | 0.54 |
  | S&P 400 | 47 | 53% | 2.1% | 3.4% | 5% (9%) | 0.41 |
  | S&P 600 | 16 | 127% | 9.7% | 9.5% | 5% (5%) | 0.47 |

  A percentile of 0.5 means the score says nothing about the winners. S&P 400, most frequent
  winners and their average score percentile: Repligen 0.27 (26 dates in the top 10),
  Chemed 0.34, Paylocity 0.03, FormFactor 0.61, Tenet 0.11, Dick's Sporting Goods 0.52,
  Synaptics 0.58, Qualys 0.31: growth in health care, technology and consumer names. Their
  rule points against the universe's: dividends G4 and E4 -0.8, debt cover G2B and E2 -0.4,
  price to assets G7 -0.3, E6 -0.3; ROE B1 +0.1, income growth NI1 +0.1. Our picks there
  score high on exactly the rules the winners score low on (debt cover 4.25, price to assets
  4.05, P/E 4.05, against 2.1 to 2.3 for the universe). Sector mix, winners against picks:
  health care 19% against 1%, materials 3% against 19%; on the S&P 600 energy 24% against 1%.
  On the S&P 500 the score ranks the winners above the median (0.54) and catches the
  semiconductor names (NVDA 0.93, LRCX 0.97, MU 0.84, WDC 0.90, PWR 0.85) but misses the
  growth platforms (AMZN 0.06, NFLX 0.22, STZ 0.20, HWM 0.23, ADBE 0.40). Reading: the
  rules are Graham's (asset and balance-sheet value, dividends), and the winners of the
  period were growth and quality names that pay no dividend and trade at higher multiples;
  on the 400 and 600 the score carries no information about who wins, on the 500 a little.
  Hypotheses to test on fresh data with the protocol, not to adopt from this list: a
  sector-neutral grade (rank within sector), a lower weight on the dividend history rules for
  growth companies, a heavier B1 and NI1 (the letters' quality and growth), each a
  methodology change.
- 2026-11-01, letters on dividends (direction for the question whether to adjust the
  dividend rules G4 and E4; `data/interim/berkshire_letters`, 1984): earnings are retained
  while each retained dollar creates more than a dollar of market value, and distributed only
  when it cannot ("Berkshire has earned well over market rates on retained earnings,
  thereby creating over one dollar of market value for every dollar retained ... any
  distribution would have been contrary to the financial interest of shareholders"). Buffett
  does not use a dividend record as a sign of quality; Graham's G4 and E4 do. The direction
  supports testing the removal or a lower weight of G4 and E4. The test is the rule cascade
  below, on the training companies, fixed before the run, then the independent indices.
- 2026-11-01 (user): the post analysis of missed winners (check 24) is included in the
  protocol and in the report, under winning picks we have missed, as opportunities
  (`VI_202611_002`, accreditation test AT-9). The user's view, to be tested: dropping the
  dividend rule would have given better results; the cascade (drop one rule at a time) has
  not been run yet. More data to crunch is wanted eventually. Valuing a company after the
  inputs change needs assumptions about correlations; literature that models those changes
  would make the market scenarios of Midas's wrath possible (S_202610_008).
- 2026-11-01, check 25, the rule cascade (user: drop rules one by one to find the
  sensitivity; scratchpad `ablation.py`, results `ablation_<index>.csv`; 22 drops per index:
  each of the 16 rules, and the groups G4 with E4, G2A with E1, G2B with E2, G3 with E3, G5 with
  E5 and the four price rules; design fixed before the run). S&P 500 on the training companies
  (seed 1), the S&P 400 and 600 on all their companies; each drop regrades the remaining rules
  from scratch (pass-rate weights, 6 buckets); paired difference against all rules on the same
  dates at 36 months, in points a year, with a 99% time-bootstrap band, for the top 10 and
  for the top quintile minus the pool. Baseline: top 10 10.1% (500), 2.1% (400), 9.7% (600).
  Dividend rules (G4 and E4 dropped together): S&P 500 top 10 -0.03 [-1.05, +0.41], quintile
  -0.14 [-0.47, +0.18]; S&P 400 -0.26 [-1.5, +0.75], quintile -0.09; S&P 600 +0.72 [+0.28,
  +1.13], quintile -0.93 [-1.56, -0.68] (16 dates, about two independent windows, bands not
  reliable). Dropping the dividend rules therefore makes no measurable difference on the
  500 and 400; the claim that results would have been better without them is not supported.
  Largest effects, S&P 500 (change in the top 10; band): NN -3.2 [-6.9, +0.3]; the four price
  rules together -1.1 [-10.5, +10.2]; G6 +2.2 [+0.2, +4.3] but quintile -0.6; G7 +1.7 [+0.3,
  +3.1] but quintile -0.5; B1 +1.0 [-0.3, +2.9]; G5 with E5 -0.7; G5 -0.8; G2A -0.4 (quintile
  -0.54 [-1.51, -0.10]); G2B and E2 -0.4. The ten largest impacts on the 500: NN, G6, the price
  group, G7, B1, G5 with E5, G5, G2A, G2B, E2. Consistent across the three indices: NN, G2A
  and the price group hurt when dropped; dropping B1 (the 3-year version) helps or does
  nothing on all three (+1.0 on the 500, +0.7 on the quintile of the 400, +1.7 on the 600);
  dropping G6 or G7 raises the top 10 on the 500 but lowers its quintile spread, so the price
  rules help the broad ordering and hurt the extreme top, consistent with the missed winners
  (check 24). Caution: 44 tests per index at 99%, a 400 and 600 with few dates, and the same
  data was used to look at the winners; nothing is adopted, each candidate (drop B1 3-year,
  lighter price rules) is a new design for fresh data. Dropping G4 and E4 would match the
  letters (1984) at no measured cost on the 500 and 400, a simplification the user may choose.
- 2026-11-01 (user): the analysis is kicking the tyres, "testing gold by fire", and probing
  results and their causes is the standard protocol (`quant.md`, step 15): when something
  looks fishy the quant asks the investor (the letters) for the direction of the fix, tests
  it as a new design, and the two keep a conversation until the cause is explained or parked,
  written in the change file as quant and investor entries. Example from this change: quant
  (check 24): the winners scored low on G4 and E4, the dividend rules; investor (1984): earnings
  are retained while a retained dollar creates more than a dollar of market value, and a
  dividend record is not a sign of quality; quant (check 25): dropping G4 and E4 changes the
  top 10 by -0.03 [-1.05, +0.41] on the 500; so the dividend rules are not what holds us back.
- 2026-11-01, check 26 (scratchpad `rule_types.py`; user): rules correlated with the company types they exclude, so that an
  avid user can switch rules on and off to personalise the report. Pass rates, S&P 500, 400
  and 600 pooled, of the evaluated companies, and the types that fail a rule at under 60% of its
  overall rate: G4 and E4 pass 68% and 71% overall and exclude companies without a dividend
  (2% and 0% pass); G6 (17% overall), G7 (17%) and E6 (2%) exclude health care (10%, 9%, 0%) and high
  P/E companies; B1 (17%) excludes communication services (8%), materials (9%) and utilities
  (0%); G2A, G2B, E1 and E2 exclude utilities (2%, 0%, 7%, 0%) and energy (18%, 13%, E2 16%);
  E3 and G5 exclude energy (28%, 25%); NN passes about 0% everywhere; G3, E5 and NI1 exclude
  no type. Reading: the debt and liquidity rules are a utilities and energy filter, the
  dividend rules a growth filter, and the price rules a health care and growth filter; the
  basis of the switches (`assay/notes.md`).
- 2026-11-01, check 27, only 5 or 10 of the 16 rules (user: determine the ones with the
  biggest weight and keep those; scratchpad `subset_test.py`; selection rule fixed before the
  run, on the S&P 500 training companies only: a rule's value is the fall in the
  top-quintile spread when it is dropped; random subsets of the same size are the baseline, 30
  each). Value, points a year: G6 +0.61, G2A +0.54, G7 +0.54, G2B +0.40, E2 +0.36, NN +0.26,
  E1 +0.25, G5 +0.24, B1 +0.14, E4 +0.12, E6 +0.10, E5 +0.04, G4 +0.02, NI1 -0.13, G3 -0.16,
  E3 -0.20 (G2A equals E1 and G2B equals E2, so the duplicates count twice). Kept: the top 10
  (G6, G2A, G7, G2B, E2, NN, E1, G5, B1, E4) and the top 5 (G6, G2A, G7, G2B, E2). Top 10 36
  months excess a year and the top-quintile spread a year; the share of the 30 random subsets of
  the same size the chosen subset beats, for the two measures:

  | Index | All 16 rules | Top 10 rules | Top 5 rules |
  |---|---|---|---|
  | S&P 500 (selection data) | 10.10 and 2.76 | 9.61 and 2.65 (beats 40% and 93%) | 5.88 and 2.13 (13% and 20%) |
  | S&P 400 | 2.06 and -3.04 | 1.47 and -1.83 (53% and 77%) | 2.32 and -0.91 (47% and 60%) |
  | S&P 600 | 9.71 and 3.61 | 15.70 and 5.24 (80% and 57%) | 10.49 and 4.45 (23% and 63%) |

  Reading: ten rules keep the S&P 500 result (9.6 against 10.1, the spread 2.65 against 2.76)
  and do not hurt the other two (the 600 is higher, 15.7); five rules lose a third of the
  500's result and are no better than random subsets of five. The chosen subsets are mostly
  not distinguishable from random ones (the 500 quintile, 93%, is in-sample). The top 5 are
  really four concepts (P/E, current ratio, price to assets, debt cover), two of them
  counted twice. A 10-rule score is a simplification at no measured cost, not a gain, so by
  the adoption rule nothing changes; it is a candidate for the speed and the switches.
- 2026-11-01 (user): the ten rules (G6, G2A, G7, G2B, E2, NN, E1, G5, B1, E4) are the default;
  the other six (G3, E3, E5, G4, E6, NI1) are commented out in `config/rules.toml` with the
  reason, to switch on (unstaged; methodology change). The user offers the rule subsets to the
  optimisation as a segment-selected approach (rule sets chosen per company type), a studio
  tier option (`notes.md`, "Optimisation candidates", 7). Run for the S&P 500: `buyability`
  and `report` work, Black and the 15 tests pass. Money-weighted return a year of the value
  wallet, 10 rules (16 rules): 2016 investor 24.9% (29.8%), 2020 28.9% (29.3%), 2023 34.9%
  (35.0%), 2025 27.0% (31.5%), against the index 15.3%, 15.9%, 18.7% and 15.7%. The next
  picks keep FSLR, SNA, CTSH, RMD, EW, SWKS, ON, PODD and FAST at the top. The report's
  method text still says sixteen rules (`config/report_value_investing.toml`, the scribe's).
- 2026-11-01, check 28, the conversation on price (quant and investor; scratchpad
  `investor_q.py`; designs fixed before the run, three tried; on the ten default rules).
  Quant: dropping the price rules G6 and G7 raised the top 10 and lowered the broad quintile
  spread on the 500 (check 25), and the missed winners were growth names (check 24); is
  paying for quality the fix? Investor (1992 letter): "value" and "growth" are "joined at the
  hip: Growth is always a component in the calculation of value"; "a low ratio of price to
  book value, a low price-earnings ratio, or a high dividend yield ... are far from
  determinative", and "opposite characteristics - a high ratio of price to book value, a high
  price-earnings ratio, and a low dividend yield - are in no way inconsistent with a 'value'
  purchase"; the test is "value at least sufficient to justify the amount paid", and growth
  alone "tells us little about value". 1994 and 2011: book value is meaningless as a measure
  of intrinsic value at most companies. 2013, Graham's adage: "Price is what you pay, value is
  what you get." The investor's direction: do not use low P/E and low P/B as the test of
  value; measure price against what the business is worth, growth and quality included.
  Quant: drop G6 and G7 together, change in the top 10 at 36 months, points a year with the
  99% band (quintile spread in brackets): S&P 500 +5.3 [+1.2, +11.7] (-0.9 [-2.8, +1.4]); S&P
  400 -0.6 [-3.3, +2.6] (+0.1); S&P 600 -2.7 (16 dates, unreliable). G6 alone: +2.2
  [-0.4, +6.7], -0.2 and +3.1; G7 alone: +1.1, -0.7 and -5.3. Result: the direction holds on the
  500 only and is carried by the top 10, that is by the growth winners (the heavy hitters
  of check 22); the 400 shows nothing, the 600 contradicts, and the broad spread does not
  improve. Not adopted. Quant's proposal to the investor, a new design to fix before the run:
  replace the absolute cheapness tests by price against earnings growth and quality (a
  growth-adjusted P/E, later owner earnings), scored on the 500 training companies and checked
  on the independent indices. Parked until the data allows it; the question stays open.
- 2026-11-01, check 29, the growth-adjusted P/E against G6 and G7 separately (user: do the
  new feature first and see if it is more sensitive than them separate; scratchpad
  `peg_test.py`; designs fixed before the run, three tried, 49 combinations counted from check 13
  on; on the ten default rules). Feature: PEG = P/E on averaged earnings divided by 100 times the
  annual growth of averaged EPS (G5's ratio to the power 1/3, minus 1), lower is better, growth
  of zero or less gives no value and zero points, passes at 1.0 or less (a value for 44% to
  59% of company-dates, a pass for 17% to 23% of those evaluated). Paired difference against the
  ten rules at 36 months, points a year, 99% band; top 10 and, in brackets, the top-quintile
  spread:

  | Design | S&P 500 | S&P 400 | S&P 600 (16 dates) |
  |---|---|---|---|
  | A: PEG replaces G6 | -0.5 [-1.7, +3.9] (+0.2) | +0.1 [-0.8, +1.4] (+0.1) | +1.6 (+0.3) |
  | B: PEG replaces G6 and G7 | +1.8 [-0.05, +8.6] (-0.6) | -0.8 [-3.0, +1.6] (+0.6) | -4.4 (-0.8) |
  | C: ten rules plus PEG | -0.6 [-1.4, +0.1] (+0.27 [-0.14, +0.99]) | -0.5 [-1.4, +0.2] (+0.45 [-0.31, +1.23]) | -1.4 (+1.33 [+0.59, +2.90]) |

  Sensitivity, the change in the objective that the rule adds inside C (top 10, quintile):
  S&P 500: PEG -0.61 and +0.27, G6 -0.12 and +0.07, G7 +1.16 and +0.16; S&P 400: PEG -0.48
  and +0.45, G6 -0.54 and +0.39, G7 -0.58 and +0.52; S&P 600: PEG -1.43 and +1.33, G6 -2.99 and
  +1.01, G7 +3.09 and +1.67; every band on the 500 and 400 includes zero. Result: PEG is not
  distinguishably more sensitive than G6 and G7 separately; its effect on the broad quintile
  spread is a little larger than G6's and G7's on the 500, G7 has the larger effect on the top
  10. The one consistent pattern is the sign when PEG is added (C): the broad spread rises on
  all three indices (+0.27, +0.45, +1.33) and the top 10 falls on all three (-0.6, -0.5,
  -1.4), small and mostly within the bands. Quant to the investor: the broad ordering improves
  slightly with growth in the price test, the extreme top does not; investor (1992): "business
  growth, per se, tells us little about value", the airline example of profitless growth, so
  growth must come with quality and durability. Next question for the investor, parked: PEG
  only where B1 (return on equity) passes, or growth only where it has been steady, a new design
  to fix before the run. Not adopted.
- 2026-11-01, check 30, routing by a quality gate (user: "if the gate says this, use this rule
  or that"; scratchpad `route_test.py`; one design fixed before the run, 50 combinations counted
  from check 13 on; on the ten default rules). Design R: the price slot is graded by PEG where the
  quality gate B1 passes and by G6 elsewhere, each within its own segment on the date; G7 and the
  other rules as in the ten. B1 passes on 15%, 13% and 7% of company-dates (S&P 500, 400, 600).
  Paired difference against the ten rules at 36 months, points a year, 99% band:

  | Index | Top 10 | Top-quintile spread |
  |---|---|---|
  | S&P 500 | +0.96 [-1.92, +3.60] | +0.33 [-0.00, +0.96] |
  | S&P 400 | +0.61 [-0.63, +1.80] | +0.38 [+0.08, +0.70] |
  | S&P 600 (16 dates) | -0.31 | +0.25 |

  Quant to the investor (1992: growth only counts with value behind it): PEG applied only where
  the business passes the quality gate. Result: both measures rise on the 500 and the 400, the
  broad spread on all three (+0.33, +0.38, +0.25); the 400's spread band excludes zero, the 500's
  touches it, the top 10 bands include zero. The unrouted PEG (check 29) lowered the top 10;
  routing does not. By the adoption rule (band excludes zero on the calibration index) it is not
  adopted: the 500's bands include zero. It is the strongest candidate so far, small (0.3 to 1
  point a year) and resting on the small quality group (7% to 15% of company-dates). To
  confirm on fresh data: a new stratified split and the independent indices once the composite
  universe exists; it counts as the last design before that test. Investor's open question:
  what makes growth durable (moat, return on capital) and whether B1's 3-year version is a
  good enough gate.
- 2026-11-01, check 31, what the companies reached in the S&P (user: correlate our scores with
  what S&P did with them, since it has far more information; not for the model until the
  Einsteinian line, a check of whether we missed growth; scratchpad `sp_reach.py`). Data: the
  "current" view scores every S&P 500 member on every date, including before it joined; the
  grades are among all current members of the date, ten default rules. Mean score percentile
  from 2010: members 0.511, companies that join within 36 months 0.419 (1,408 rows, 135
  companies), companies that join later 0.480; size percentile 0.559, 0.281 and 0.170. The
  probability that a random future joiner scores above a random member (AUC, 0.5 means
  nothing) averages 0.424 over 66 dates (2010 to 2015 0.376, 2016 to 2023 0.418). Within size
  terciles, joiners against members: smaller 0.460 and 0.528, middle 0.337 and 0.498, larger
  0.290 and 0.513. Reading: the score ranks the companies that S&P later admits below the
  members, more so among the larger ones: the companies that grew into the index scored low
  on price and dividend rules, consistent with the missed growth of checks 24 and 28. Both
  groups are survivors; the S&P 400 and 600 promotions need the composite universe
  (MD_202611_001), because the change histories record promotions only loosely. S&P credit
  ratings and quality rankings are paywalled and not used. A check, not a model input.
- 2026-11-01 (user): a feature that adds complexity for little pay-off is not default but a
  switch: the routing by the quality gate and PEG (check 30, +0.3 to +1 point a year) is such a
  case and is listed among the switches (`assay/notes.md`, "Rule switches by company type").
- 2026-11-01 (user): the design is solidified; the inspector runs next (`developer.md`,
  "Building and finalising").
- 2026-11-01, selling in the report finalised (VI_202611_002): the per-holding sell check is
  replaced by "The worst stocks", the lowest scores of the whole index on the latest score
  date, as many as the picks (10), with sector, score, the 0-to-1 score and a tier from the sell
  threshold 0.4 and half of it: at or below it avoid, hold it if you own it; at or below 0.2
  start thinking about selling; at 0 not worth anything. The sales of the wallet with sells
  appear only in its cash flows (Section "Selling below the threshold" and the appendix of
  purchases). On 18 September 2026 all ten fall in the middle tier (0.03 to 0.13); no company
  reaches 0 and none sits between 0.2 and 0.4. Size is not shown yet. The tiers follow the
  cut-offs in use and will follow the fitted ones.
- 2026-11-01, check 32, check 24 rerun on the ten default rules (scratchpad
  `missed_winners_ten.py`; the report's section "Winning picks we missed" uses these numbers,
  not those of check 24). Ex-post winners' excess a year, our top 10, the share of winners held (by
  chance), the winners' mean score percentile: S&P 500 76% / 11.2% / 10% (4%) / 0.55; S&P 400 50.5% /
  1.5% / 5% (9%) / 0.42; S&P 600 126.5% / 15.7% / 9% (5%) / 0.51. Score percentile of the frequent
  winners: S&P 500 Nvidia 0.91, Lam Research 0.96, Micron 0.91, PWR 0.80, Broadcom 0.54, Adobe 0.48,
  Constellation Brands 0.32, Netflix 0.23, Amazon 0.08; S&P 400 Chemed 0.34, Paylocity 0.03, Repligen
  0.16, Tenet 0.17, Qualys 0.20; S&P 600 Archrock 0.14, Brinker 0.16, SM Energy 0.19. S&P 400, rule
  points of the winners against the universe: E4 -0.72, G2B -0.36, G7 -0.32, G6 -0.08; our picks score
  4.43 on G2B (universe 2.14), 3.86 on G7 and 3.99 on G6. Sector, winners against our picks: S&P 400
  health care 17% against 3%, industrials 29% and 39%; S&P 600 energy 24% against 2%.
- Reading: the 3-month horizon shows no edge; the 12-month horizon, the one the score
  is configured for, shows a possible one, concentrated in recent years. The earlier
  "no edge" reading is withdrawn; the weighting, bucket and universe checks are to be
  rerun at 12 months. Calibration stays open.

## Inspections

| Date | Commits | Findings | Open |
|---|---|---|---|
| 2026-11-01 | 01832d6..working tree (first run, uncommitted) | 0 blocking, 5 should fix | stale "five" and "Sixteen" wording in `config/report_value_investing.toml` and the rule names G3 and G4 (the scribe's); everything else fixed |

Fixed after the run: the `Wallet` docstring on `names`; a comment that `rank_weights` is for 5 names
and used only by the rank weighting (off); the six repeated "off by default" comments in
`config/rules.toml` shortened to one block; the study drafts note that the commented rules are
switched on by an override and that the report shows 10 names. Ties from the exact-n rule on real
dates (S&P 500, ten rules): at the cut of the 10 best, 2 of 68 dates have a tie, 2 companies
tied; at the cut of the 10 worst, 8 dates, 2 companies; so the effect on the picks is negligible;
44 dates have companies with a score of exactly 0. Not checked by the inspector: the tie
effect on the plots, `lookahead.truncation_check` (compares the sets of the `top` best, so it is
unaffected), and `layers/*` for ratios only the removed rules produce (none found when the
pipeline and the report ran on the ten rules).

## Kickoff prompt

Pick up VI_202611_001 once S_202611_003 runs a study. Read the agreed approach, record
the baseline report summaries of the three indices, then build the inner fit one step
at a time (signed weights first), each a recorded experiment.

## Done when

- A default study calibrates the S&P 500 model and validates it with uncertainty bands;
  the chosen specification is then scored once on the S&P 400 and 600.
- The report states the specification, the combinations tried, the fitted cut-offs and
  the bands.
- Documentation to update (scribe): spec §3, decision 7 (walk-forward with a hold-out
  period), and the `optimise.py` line of §7, for the company test sample.
