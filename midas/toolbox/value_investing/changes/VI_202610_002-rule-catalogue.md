# VI_202610_002: Rule catalogue in the tool

| Field | Value |
|---|---|
| Status | ready |
| Priority | medium |
| Kind | new capability |
| Depends on | S_202610_001, S_202610_005, VI_202610_001; spec §9.3 (financials and REITs) |
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

## Approach

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
- **Spec §9.3, financials and REITs:** exclude, or score separately (e.g. P/B and ROE).
  Decides whether G2, E1, E2 and NN apply to every company or through a sector-specific
  rule set.
- **Dependency gap:** stage 2 (`fundamentals.point_in_time`, TTM and 10–20 year
  history) has no change yet. Raise it separately, or fold it into VI_202610_001.
- **Hard-gate subset:** which of G2, G3 and E1–E3 are hard gates by default (spec §3,
  item 1); belongs to the ranking change but fixes which pass flags this change must
  emit.

## Kickoff prompt

Pick up VI_202610_002. Confirm S_202610_001, S_202610_005 and VI_202610_001 are done and
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
