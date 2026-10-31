---
paths:
  - "notebooks/**"
  - "*.ipynb"
---

# Notebooks

- Notebooks may only set paths and config, construct and run pipelines and backtests,
  and display results. No reusable calculations or copies of production functions: new
  logic belongs in `Toolbox/`, with tests, and is imported here.
- Notebooks are UI code: not linted or formatted, and an agent does not tidy them. A
  rename of a module, function or column is still carried into every notebook.
- No API keys or vendor data in a notebook's source or saved outputs.
