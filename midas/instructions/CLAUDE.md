@AGENTS.md

## Claude Code only

The rules for every hat are in `AGENTS.md`; each hat's own rules are in its file beside
it. This file holds only what is specific to Claude Code.

- **Hats load on demand.** In the main session, read `developer.md` before writing code
  or official documentation and `secretary.md` before writing in `midas/`, each with
  `writing.md`, and `quant.md` before working with data, metrics, strategies or the
  wallet; a big change usually needs both. The researcher and the warden are
  subagents: `.claude/agents/midas-researcher.md` and `.claude/agents/midas-warden.md`
  (read-only, no Bash). Start the warden as `developer.md`, "Building and finalising",
  describes.
- **Area rules** in `.claude/rules/*.md` load automatically for matching paths:
  `midas-records.md` brings in `secretary.md` for `midas/`, `midas-index.md` brings in
  `developer.md` for Midas's server code.
- **MCP.** The server is registered in `.mcp.json` (today as `trading-bot`, from
  `Toolbox/skatkis/midas/server.py`; it becomes `midas` with change S_202610_002). Edits to its
  code reach the running server only after Claude Code restarts it.
- **Commands.** Use `.venv/bin/python -m …` for pytest, Black, ruff and pyright.
- **Data updates and the report run outside the sandbox** (user, 2026-10-31): when the
  user asks to run an update, `python -m Toolbox.market_data update` and `check` and
  `tectonic` may run, with their hosts (SEC, Yahoo, FRED, Wikipedia, tectonic's package
  server) allowed. Configured in the git-ignored `.claude/settings.local.json`
  (`sandbox.excludedCommands`, `sandbox.network.allowedDomains`); active after Claude
  Code restarts.
- **No prompts the user did not need.** Read and search with Read and Grep; keep Bash
  for plain commands; no paths outside the repository except the session scratchpad;
  do not edit `.claude/settings.json` or hooks unless asked. The reasons are in
  [guard.md](guard.md).
