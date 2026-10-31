---
paths:
  - "tests/**/*.py"
---

# Tests

- Production is authoritative. Modify only tests unless a production defect has been
  separately agreed; report contradictions instead of inventing behaviour.
- Test only behaviour the module owns. Shared `skatkis` behaviour is tested once under
  `tests/skatkis/`, never re-tested in each tool.
- Test files mirror the source tree: `Toolbox/<pkg>/<module>.py` →
  `tests/<pkg>/test_<module>.py`.
- Every calculation covers (spec §1.5): normal, zero, boundary (exact threshold, e.g.
  P/E = 15.0), failure (missing or invalid input) and input non-mutation; add group
  isolation (no leakage across tickers) and point-in-time cases (a filing accepted
  after the as-of date is not seen) where they apply.
- Hand-computed reference values live in fixtures with the derivation in a comment.
  Never edit a reference value or tolerance to make a test pass.
- Synthetic tickers (`AAA`, `BBB`) and small fixture universes; no network in unit
  tests (record API responses as fixtures).
- Keep existing docstrings and style.
