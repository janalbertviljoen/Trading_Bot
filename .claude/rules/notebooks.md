---
paths:
  - "notebooks/**"
  - "*.ipynb"
---

# Notebooks

- Notebooks may only set paths and config, construct and run pipelines and backtests,
  and display results. No reusable calculations or copies of production functions: new
  logic belongs in `Toolbox/`, with tests, and is imported here.
- Tidying and renames: `developer.md`, "Coding conventions".
- No API keys or vendor data in a notebook's source or saved outputs.
