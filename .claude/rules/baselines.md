---
paths:
  - "benchmarks/**"
  - "tests/**/fixtures/**"
---

# Baselines and reference values

- `benchmarks/baseline.json` holds the regression baseline: the default backtest's
  metrics per frozen dataset. `regression_check` compares against it; a refactor must
  return `unchanged`.
- Hand-computed reference values in `tests/**/fixtures/` carry their derivation in a
  comment.
- Changing them: `AGENTS.md`, "Prohibited", and `developer.md`, "Designing a change"
  (a baseline is rewritten with `update_baseline=True` only with the user's recorded
  approval).
