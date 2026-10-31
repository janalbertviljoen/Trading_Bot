---
paths:
  - "Toolbox/*/schema.py"
  - "Toolbox/*/contracts.py"
  - "Toolbox/skatkis/pipelines/**"
---

# Schema and layer contracts

- `schema.py` is the single source of truth for canonical columns: name, dtype, unit,
  constraint flags (`required`, `positive`, `non_negative`, `unique_key`), owner and
  derivation. No other module redefines a column. No calculations, IO or validation
  execution here.
- Layer input schemas reference the upstream entry directly to keep lineage.
- `contracts.py` holds the pipeline contracts: name, inputs, output schema, grain
  (ticker × as-of date), immutable inputs, description. Use only the fields of the
  shared contract type in `skatkis.pipelines`; a generic need is added there, with
  tests.
- Register contracts in DAG dependency order, consistent with `pipeline.py`.
- Renaming a canonical column touches code, contracts, tests, config, notebooks and
  documentation together.
