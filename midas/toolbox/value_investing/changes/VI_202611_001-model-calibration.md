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

## Open decisions (user)

- The smoothing of the point curves (number of buckets, monotone or free shape): high
  priority.
- The fixed rule for the cut-offs (the score where the expected excess crosses zero, or
  the one that separates winners and losers best): after the first results.

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
