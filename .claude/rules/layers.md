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
- Point-in-time as `midas/instructions/quant.md` defines it (rule 1).
- Zero or negative denominators give NaN plus a diagnostics flag, never inf; keep
  clipping and bounds explicit.
- A formula or threshold change is a methodology change (`AGENTS.md`, "Methodology vs
  refactor"); a new or changed output column follows `developer.md`, "Coding
  conventions".
