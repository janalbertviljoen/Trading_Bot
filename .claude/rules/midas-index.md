---
paths:
  - "Toolbox/skatkis/midas/**"
  - "mcp_server/**"
  - "tests/skatkis/midas/**"
  - ".mcp.json"
---

# Midas's server (index and lab)

- Read `midas/instructions/developer.md`, "Working on Midas's code", before changing the
  server; planned changes are in `midas/toolbox/skatkis/CHANGELOG.md`.
- Discuss and agree the approach before writing code; build in small steps.
- `index` profile tools are read-only queries: no pipeline runs, no file writes, no data
  values. Tools that run code or write files belong to the `lab` profile.
- Facts come from code; contradictions are reported, not reconciled. Every answer
  carries `provenance`, `commit` and `dirty`.
- A change to the server's behaviour or profiles is recorded by the secretary in
  `midas/midas.md` and its decision log.

@../../midas/instructions/developer.md
@../../midas/instructions/writing.md
