# S_202610_002: Midas index server

| Field | Value |
|---|---|
| Status | ready |
| Priority | high |
| Kind | tooling |
| Depends on | S_202610_001 |
| Raised | 2026-10-31 |

## Description

Agents today read files one by one to learn what exists. The example repository's
Steward shows the alternative: a read-only index of code, docstrings, rules and
documents, served over MCP, so an agent asks "which function computes X", "who owns
column Y", "which rules govern this file" before writing code. Midas needs the same,
merged with today's backtest tools into one server.

## Approach

Code in `Toolbox/skatkis/midas/`, built like the example (`docs/attachments.zip`,
README §19):

| Module | Role |
|---|---|
| `docstrings.py` | Splits a docstring into header, summary and numpydoc sections (`Owns`, `Must not`, `Public API`, `Computes`, `Parameters`, `Returns`, …) |
| `index.py` | Builds the index in memory from the repository with `ast` (no code is run), plus the query functions; the library |
| `server.py` | Thin MCP adapter: registers the index queries (profile `index`) and the backtest tools (profile `lab`) |

Indexed: `Toolbox/` (excluding `SandBox/`), `.claude/rules/`, `midas/`, `docs/*.md`
(the spec split at its numbered headings), test file names, and the protected-file
patterns in `.claude/settings.json`. Configured in `pyproject.toml`
(`[tool.skatkis.midas]`) so the code never names a tool.

First tools: `search`, `list_components`, `describe_module`, `describe_symbol`,
`find_calculation`, `rules_for`, `spec_section`, `document`, `list_documents`, `health`,
`rebuild_index`. Pipeline-aware tools (`describe_layer`, `find_owner`,
`describe_column`, `plan_change`) follow once the tool has a schema and contracts.
`.mcp.json` renames the server from `trading-bot` to `midas`; agents' tool lists are
updated with it.

## Open decisions (user)

- Default profile over stdio: both (`index` + `lab`, proposed), or `index` only with
  `lab` opt-in by a flag?

## Kickoff prompt

Pick up S_202610_002. Read README §19 of the example in `docs/attachments.zip`, propose
the module layout and the first tool set, then build `docstrings.py` and `index.py` in
small steps with Python-level checks before wiring the server.

## Done when

- `midas` serves the index tools read-only, with `provenance`, `commit` and `dirty` on
  every answer, and the lab tools unchanged.
- Tests in `tests/skatkis/midas/` cover the parser, index and server on a synthetic
  repository.
- `.mcp.json`, the agents' tool lists and `midas/midas.md` §3 are updated.
