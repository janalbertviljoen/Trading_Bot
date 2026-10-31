# VI_202610_002: Rule catalogue in the tool

| Field | Value |
|---|---|
| Status | in progress |
| Priority | medium |
| Kind | new capability |
| Depends on | S_202610_001, S_202610_005, MD_202610_002 |
| Branch | `feature/value-investing/vi-202610-002-rule-catalogue` |
| Raised | 2026-10-31 |

## Description

The trading rules (G1–G7, E1–E6, GN, NN, B1–B5, M1, Q1–Q3) exist only as
documentation: spec §2–§3 and its extract `docs/trading_rules.md`. They should live in
the `value_investing` tool as financial rules built to the SOP: each rule a calculation
with one owning module, its thresholds in config, its output columns in the schema, and
its formula and source in the tool README. Until then nothing can be gated, ranked or
backtested, and two documents hold the same rules.

The tool keeps the name `value_investing` (user, 2026-10-31). The change waits for the
repository layout, the SOP and the data sources.

## Agreed approach

### First slice: buyability score (agreed 2026-10-31)

The first slice turns the rules with a pass/fail threshold into a buyability score, to
see what the data looks like before any selection or backtest.

- **Grid.** One row per company and rebalance date (third Friday of March, June,
  September and December), one column per rule flag. One long table; a company's
  history is `groupby("symbol")`, and other tables join on company and date.
- **Data basis.** Annual first-reported 10-K values, in force from their filing date
  (`market_data.clean.annual_first_reported`), through a thin as-of-filing annual panel
  in `market_data`, the first slice of MD_202610_002. The move to TTM flows and
  latest-quarter balance sheets is a later methodology change. Prices are the
  rebalance-date close; per-share EDGAR values are put on the price's split basis
  (`metrics.split_factor`) before they are averaged, compared or divided into prices.
- **Rules in the slice.** G2a, G2b, G3, G4, G5, G6, G7, E1–E6, NN and B1; G1 off.
  G4 and E4 use the Yahoo dividend history. B2–B5, M1 and Q1–Q3 follow once their
  thresholds are set.
- **Score.** The share of evaluable rules passed, from 0 to 1: the mean of the flags
  that exist, so the score stays comparable over time while long-window rules phase in.
  `n_passed` and `n_available` keep the counts. Reported in total and per family
  (G, E, B). No total score below 5 evaluable rules.
- **Diagnostics.** Ten companies with the highest and ten with the lowest mean score
  from 2016, their score over time against their total return relative to
  `^SP500TR`, and the score against the next quarter's excess return; in-sample,
  labelled as diagnostics. Plots in `results/value_investing/`.
- **Kind.** New capability; no baseline exists, so none moves.

### Full catalogue

Fitted to the SOP layout (§3.3) and the spec's stage and node split (§5, §6.2):

| What | Where | SOP |
|---|---|---|
| Graham rules G1–G7, E1–E6, GN, NN (node `metrics.graham`) | `Toolbox/value_investing/layers/graham.py` | §4.7 |
| Buffett and Munger rules B1–B5, M1 (`metrics.buffett`) | `layers/buffett.py` | §4.7 |
| Systematic benchmarks Q1–Q3 (`metrics.quant`) | `layers/quant.py` | §4.7 |
| Shared price-based ratios (P/E, P/B, EV, market cap), sole owner, consumed by the three above | `layers/valuation.py` (`metrics.valuation`) | §2.4, §4.7 |
| Rule registry: supported rule IDs, an immutable context object, rule ID → callable and output column, so `ranking.gates` and `ranking.composite` select rules from config without branching on IDs | `Toolbox/value_investing/rules.py` | §4.8, §10 |
| Thresholds and windows (2.0, 1.5, 15, 22.5, 1.33, 10 and 20 years, ROE count, debt/equity cap, margin of safety $m$, discount rate) | `config/rules.toml`, versioned; values, not behaviour | §10.6 |
| One output column per rule (pass flag and, where it exists, the underlying value), with lineage and constraints | `schema.py`, `contracts.py`, `validation.py` | §4.1–§4.3, §5 |
| General finance used by the rules (discounting for B5, multi-year averages and standard deviation for G5 and B3, compounding) | `skatkis.financial_logic`, reused, not re-implemented | §3.2 |
| Every rule: formula, threshold, source (Graham, *The Intelligent Investor*, rev. ed., ch. 14 and 15; Piotroski 2000; Greenblatt; Novy-Marx 2013), conventions (TTM flows, latest-quarter stocks, filing-date keying), the documented departure from pure Graham (hard gates plus composite rank) | the tool README, "detailed formulas and assumptions" | §12.1 |

- Each layer: copy input → hydrate → input gate → owned calculation → output gate →
  return; pure helpers, one per rule, documented to SOP §12.3 (formula, inputs and
  units, bounds, zero and boundary behaviour, e.g. GN undefined and flagged when
  EPS ≤ 0 or BVPS ≤ 0).
- The layers consume the point-in-time fundamentals panel (`fundamentals.point_in_time`,
  spec §5 stage 2); they never read files or recompute fundamentals.
- Tests: one hand-computed reference per rule on a small fixture universe, including
  boundaries (exactly at threshold, missing history, negative EPS), plus a registry test
  that every supported ID has a callable and a schema column.
- **Documentation move.** Spec §2–§3 move into the tool README; `docs/trading_rules.md`
  is deleted, and the references to it in `midas/instructions/AGENTS.md`,
  `midas/instructions/developer.md` and `.claude/agents/midas.md` are updated.

## Open decisions (user)

- **Registry or plain functions.** With a fixed rule set, the rules could be plain
  functions in the layers. The registry pays off if gates and the composite rank choose
  rules from config. Recommendation: registry.
- **Hard-gate subset:** which of G2, G3 and E1–E3 are hard gates by default (spec §3,
  item 1); belongs to the ranking change but fixes which pass flags this change must
  emit.

## Decisions

- 2026-10-31 (user): first slice as in "Agreed approach": annual first-reported values
  now, TTM later as a methodology change.
- 2026-10-31 (user): a rule that cannot be evaluated scores 0: a company not measured
  on a rule has not earned its point. `n_available` is shown beside the score.
  Superseded the same day, below.
- 2026-10-31 (user): the score is the mean of the evaluable flags (0 to 1), not their
  sum, so it ignores how many rules were available on the date and gives a consistent
  view over time.
- 2026-10-31 (user): a company is scored and ranked on a date only with at least 5
  evaluable rules (`[score] min_available` in `config/rules.toml`); the family scores
  keep their value below it. From 2016 a median of 368 of 396 companies are ranked per
  date.
- 2026-10-31 (user): B1's debt-to-equity cap is 0.5.
- 2026-10-31 (user): the sample is also shown as score * growth relative to the whole
  `^SP500TR` (not the GICS sector fund), cumulative since 2016, to amplify agreement
  between the rules and the market: high for good-scoring outperformers, low for
  bad-scoring laggards. The ratio score / growth was considered and not taken: if the
  score predicts outperformance the two cancel instead of separating.
- 2026-10-31 (user): one point per rule, and the score per family as well.
- 2026-10-31 (user): the ten best and ten worst for the diagnostics are chosen on the
  mean score from 2016; a casual sample for checking the data, not a selection.
- 2026-10-31 (user): plots of a rule's metric show the rule's cut-off as a labelled
  horizontal line (for example P/E 15 for G6). Supported by `thresholds` in
  `skatkis.analysis` (`plot_series`, `plot_spread`) and the P/E plots of
  `market_data.market_analysis`; the values come from this tool's rule config.

- 2026-10-31 (user): REITs and financial companies are excluded from the universe (spec
  §9.3), so one rule set applies to every company scored; no sector-specific rules.
- 2026-10-31 (user): a missing input gives NaN for the rule; no filling or
  interpolation until the data has been looked at.
- 2026-10-31 (user): product first; validation stays minimal (declared columns only)
  until the framework runs end to end.
- 2026-10-31: the point-in-time panel this change consumes is MD_202610_002.
- 2026-10-31 (user, on the short EDGAR history; tool notes): three measures, all to be
  built.
  1. G4 (and E4) take the dividend history from the price source (Yahoo dividends,
     back to 1970), not from EDGAR. Built in MD_202610_001 (`sources/prices.py`).
  2. The look-back windows of G3, G5, G4, E3, E5 and B1 are explicit parameters in
     `config/*.toml`, defaulting to Graham's and Buffett's values (10, 13, 20, 5, 4
     and 10 years). A shorter window is a methodology change: run as a recorded
     experiment, never as the default.
  3. Rules are phased in: a rule gives NaN until its window is full at the rebalance
     date, and gates and the composite rank treat NaN as "not yet available", so the
     backtest can start before every rule applies.

## Notes

- 2026-10-31: first slice built and running end to end:
  `python -m Toolbox.value_investing buyability --as-of 2026-10-31` (about 3 s for 396
  companies and 71 rebalance dates, 2009-03 to 2026-09). Table in
  `data/interim/value_investing/<as_of>/buyability.parquet`; plots, `sample.csv` and
  `score_returns.csv` in `results/value_investing/<as_of>/buyability/`.
- 2026-10-31: modules: `market_data/fundamentals.py` (as-of annual panel, first slice of
  MD_202610_002), `market_data.metrics.split_factors`,
  `skatkis.financial_logic.relative_growth` and `forward_excess_return`;
  `value_investing/inputs.py`, `layers/valuation.py`, `layers/graham.py`,
  `layers/buffett.py`, `flags.py`, `rules.py`, `scoring.py`, `pipeline.py`,
  `diagnostics.py`, `plots.py`, `__main__.py`; `config/rules.toml`.
- 2026-10-31: checks done: Apple EPS split adjustment (FY2013 39.75 / 28 = 1.42), Apple
  G4 = 14 years (dividend resumed 2012), Coca-Cola P/B about 11 and failing B1 on debt;
  G6 pass rate jumps to 30% in the March 2020 crash; G3 and G5 phase in about 2017 and
  2020 as the notes predicted.
- 2026-10-31: first in-sample look (sum score): no monotonic relation between score and
  the next quarter's excess return; mean rank correlation from 2016 is -0.01. With the
  mean score: -0.008, and the best ten from 2016 include EMR and XOM at 1.00 on two
  evaluable rules (EDGAR gaps, tool notes); 43 companies average fewer than 8
  evaluable rules from 2016, 13 fewer than 5. The worst ten are
  mostly companies with short or missing history (spin-offs, recent listings, EDGAR
  gaps), not companies failing the rules.
- 2026-10-31: `pipeline.py` wires the stages in order; the DAG engine with node caching
  of `.claude/rules/pipeline.md` (spec §6) is not built in this slice. Contracts and
  gates stay minimal (declared columns only), as decided.

- 2026-10-31: covered by the warden's first run on the branch; the report and its
  fixes are recorded in VI_202610_006 ("Warden report").

## Kickoff prompt

Pick up VI_202610_002. Confirm S_202610_001, S_202610_005 and MD_202610_001 are done and
the decisions above are made. Read spec §2, §3, §5 and §6, the SOP sections cited in
the approach, and the tool's notes. Propose the registry interface, the context object
and the output columns, then build `layers/valuation.py` and `layers/graham.py` first.

## Done when

- Every rule in spec §2 is computed by exactly one function in the tool, registered
  under its ID, with thresholds in `config/rules.toml` and its columns gated by the
  schema.
- Hand-computed reference tests pass for every rule.
- The tool README documents every rule with formula, threshold and source;
  `docs/trading_rules.md` is gone and no file references it.
