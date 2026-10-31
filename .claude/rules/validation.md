---
paths:
  - "Toolbox/*/validation.py"
  - "Toolbox/skatkis/pipelines/validation.py"
---

# Validation

- Tool-specific gates and business rules live in the tool's `validation.py`; generic
  assertions live in `skatkis.pipelines.validation`. Layers call gates; they do not
  implement ad hoc checks.
- Prefer schema-driven checks (flags in `schema.py`); use named domain functions for
  rules metadata cannot express (filing date after period end, one value per ticker and
  as-of date, universe coverage at a rebalance date).
- Failure messages state context, rule, columns, affected row count, sample keys
  (ticker, date) and the expected constraint. Fail fast.
- Changing what a gate accepts or rejects is a behaviour change: run the regression
  check and say so.
- No calculation, file IO or orchestration in validation modules.
