---
paths:
  - "Toolbox/*/layers/**/*.py"
---

# Analytical layers (metrics, ranking)

- One module = one analytical stage (spec §5, §7: `graham`, `buffett`, `quant`,
  `valuation`, `gates`, `composite`, `quarterly`). Read the module docstring for what it
  owns and must not compute before editing.
- Public entry point: `df` first, registry second, configuration keyword-only, input
  copied before mutation, output at contract grain (ticker × as-of date).
- Gate sequence: copy → hydrate contract inputs → input gate → owned calculation →
  output gate → return. Never return partial results after a failed gate.
- Must not: read files or call APIs, import or call another layer (layers are wired
  only in `pipeline.py`), recompute an upstream-owned measure, or hold thresholds in
  code (they live in `config/rules.toml`).
- Vectorised over the ticker × date panel; no per-ticker Python loops in hot paths.
  Sort by ticker and date before `shift`, `diff`, rolling or cumulative operations, and
  group by ticker so nothing leaks across companies.
- Point-in-time: a value at as-of date *t* uses only filings accepted on or before *t*.
- Zero or negative denominators give NaN plus a diagnostics flag, never inf; keep
  clipping and bounds explicit.
- Any change to a formula or threshold is a methodology change: say so.
- A new or changed output column needs schema, contract, validation, tests and
  documentation updated together.
