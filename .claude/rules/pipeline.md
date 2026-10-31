---
paths:
  - "Toolbox/*/pipeline.py"
  - "Toolbox/skatkis/pipelines/dag.py"
  - "Toolbox/skatkis/pipelines/registry.py"
---

# Pipeline orchestration

- `pipeline.py` owns source registration, the DAG (as data), execution order and result
  access. No formulas, source transformations, validation logic or strategy branches.
- The DAG follows spec §6: nodes declare `name`, `inputs`, `func`, `schema`; edges are
  inferred from `inputs`; order is topological with ties broken alphabetically; cycles,
  missing inputs and duplicate names fail at build time, before data is touched.
- Each node output is persisted with a manifest (config hash, input hashes, as-of date,
  code version); a node re-runs only if its config or input hashes change; partial runs
  execute only the required sub-graph.
- Construction is explicit (paths and config in, nothing implicit); results are read
  through an accessor that raises before `run`.
- A full run on the fixture universe is bit-for-bit reproducible.
