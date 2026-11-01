# S_202611_003: Calibration infrastructure: pipeline nodes, cache, studies and searchers

| Field | Value |
|---|---|
| Status | agreed |
| Priority | high |
| Kind | new capability (moving `pipeline.buyability` onto nodes is a refactor) |
| Depends on | none; VI_202611_001 builds on it |
| Branch | `feature/skatkis/s-202611-003-calibration-infrastructure` |
| Raised | 2026-11-01 |

## Description

Parameters are tuned by editing `config/rules.toml` and rerunning the whole pipeline;
nothing records which combinations were tried or what they gave. The user's earlier
work used a cloned copy of the pipeline in which only the changed column and the layers
after it were recomputed (user, 2026-11-01). This change formalises that: a pipeline of
nodes whose results are cached by their inputs, scenarios as overrides of a base
configuration, and studies that sample and search the parameters against a fixed
objective, with the honesty rules built into the engine. The value-investing
calibration that uses it is VI_202611_001. The user may later share an industry-grade
infrastructure, so the parts are interfaces with small built-in implementations.

## Agreed approach

1. **Profile first.** Time one full run per stage (rule inputs, ratios, rules, points
   and score, learned weights, wallet, report) so the cache and the parallel runs go
   where the time is.
2. **Node registry** in `skatkis`, on stdlib `graphlib` (spec §6, decided 2026-10-31).
   Each node declares its inputs and the configuration keys it reads; each rule is its
   own node, so a change to one rule's setting recomputes that rule and what follows.
3. **Content-addressed cache.** A node's result is keyed on its code version, its slice
   of the configuration and its inputs' keys; in memory during a study, optionally
   written to `data/artefacts/`. Scenarios that share upstream results run together;
   branches that differ only in the last stages run in parallel processes.
4. **Scenario** = base configuration plus overrides by dotted path
   (`{"rules.G6.max_pe": 12}`). A **study** is a TOML file in `config/studies/`: the
   parameters with their ranges, the parameters the client fixes, the searcher, the
   objective, the sample counts and the seed.
5. **Searchers** behind one abstract base class: seeded random sampling over the ranges
   (`scipy.stats.qmc`), grid, and the response surface (a regression fitted to the
   sampled runs, solved with `scipy.optimize`, the optimum checked with real runs).
6. **Train and test split** by company: a fixed 10% of the companies is the test
   sample on every date, stratified by market capitalisation, seeded. Test companies
   never enter a fit or the choice between runs.
7. **Calibrate, then validate.** A study calibrates on 10 sampled parameter sets
   (default) and keeps the best on the test sample; the chosen specification is then
   rerun 10 times with a new stratified test sample each time, which gives uncertainty
   bands on every reported value.
8. **Manifest per run**: configuration hash, overrides, seeds, snapshot as-of dates,
   code commit, the number of combinations tried, and any value fitted per date.
9. **Shared measures** in `skatkis.financial_logic`: annualised excess over the
   benchmark, tracking error, information ratio, how often and how far a strategy
   trails the benchmark over rolling periods.
10. **Ports and adapters**: registry, cache, scenario, searcher and objective are
    interfaces, so an outside framework can replace their internals without touching
    the value tool's nodes.

## Decisions

- 2026-11-01 (user): Newton's method is not applied to the backtest directly (the
  objective is a step function of the parameters); it runs on a regression surface
  fitted to sampled runs. Random sampling and the response surface run in parallel on
  the same runs.
- 2026-11-01 (user): SciPy is added to the `core` dependencies (pinned 1.18.1) for the
  sampling and the solvers.
- 2026-11-01 (user): one calibration basis, the S&P 500; the S&P 400 and 600 are the
  independent check of a chosen specification.
- 2026-11-01 (user): the test sample is 10% of the companies throughout, stratified by
  market capitalisation only for now; designs that target a sector come later, with
  more samplings. Sector strata were rejected for now: about 27 sector and size groups
  for about 30 test companies.
- 2026-11-01 (user): random sampling within reasonable ranges is the default for
  clients who do not know what to tune; a parameter the client fixes is held and the
  rest sampled. 10 runs for the quick calibration; more samples and reruns may be a
  paid tier (Assay notes, "Tiers").
- 2026-11-01 (user): the objective is the industry-standard measure, unweighted over
  the years; the objective is never itself tuned.

- 2026-11-01 (user): the cache stays in memory for now; writing it to disk comes per
  profile, once profiles exist. Each run writes its input parameters (the specification
  and manifest) to the results folder of its as-of date, overwritten by the next run.

## Open decisions (user)

- The ranges and floors of each parameter in the default study.

## Notes

- 2026-11-01: branch created from `feature/midas/optimise-midas` at `0e6b22e`; SciPy
  installed and pinned.
- `regression_check` covers only the BTC harness, and its baseline makes no trades
  (`skatkis/notes.md`). The refactor of `pipeline.buyability` is checked by comparing
  the buyability tables and the report summaries of the three indices before and after,
  which must be identical.

## Kickoff prompt

Pick up S_202611_003. Read the agreed approach, profile one full value-investing run
per stage, then propose the node boundaries and the cache key before writing code.

## Done when

- The buyability pipeline runs on the node registry with identical outputs for the
  S&P 500, 400 and 600.
- A study file runs 10 seeded samples and 10 validation reruns, reusing cached upstream
  results, and writes a manifest per run and a summary with uncertainty bands.
- The response-surface searcher runs on the same samples.
