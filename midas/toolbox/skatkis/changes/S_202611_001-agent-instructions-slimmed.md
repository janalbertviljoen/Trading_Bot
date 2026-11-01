# S_202611_001: Agent instructions slimmed and roles split

| Field | Value |
|---|---|
| Status | in progress |
| Priority | high |
| Kind | tooling |
| Depends on | none |
| Branch | `feature/midas/optimise-midas` |
| Raised | 2026-11-01 |

## Description

The agent instructions cost more context than the work needs. An audit on 2026-11-01
found:

- `AGENTS.md` (about 2,050 words) loads in every session and every subagent, but half
  of it (layout, hats table, lifecycle, branches, persona) serves only the main session.
- Seven agent definitions from before the layout (`midas`, `midas-builder`,
  `midas-improver`, `midas-reviewer`, `code-improver`, `reviewer`,
  `strategy-researcher`) load their descriptions into every session, point at files
  that no longer exist and contradict `AGENTS.md` (`agent/*` branches, Google
  docstrings, `ruff format`).
- The same rule is written in up to five places (point-in-time, out-of-sample,
  baselines, the lifecycle).
- The warden is one heavy review at the end, mixing code review with housekeeping, so
  big branches are reviewed late and at full cost.
- Members of Project Assay will bring their own Claude to a restricted MCP server; the
  rules they need (the research protocol, the data rules) were mixed with repository
  internals.

## Agreed approach

Agreed with the user on 2026-11-01.

- **Core `AGENTS.md`** keeps only what guards every edit (about 1,150 words): the hats,
  the lifecycle, authority, principles, methodology vs refactor, prohibited, working
  rules. The layout and branches move to `developer.md`, the persona to `CLAUDE.md`.
- **Stale agents removed.**
- **New hats:**
  - **Inspector** (subagent, Sonnet): code-only review of the commits since the last
    inspection, at build milestones on big branches; logs one line per run in the
    change file's `## Inspections`.
  - **Warden** (subagent, Opus): once per branch, before the pull request; reviews only
    uninspected commits for code and the whole branch for dead weight, documentation,
    tests to write and housekeeping.
  - **Scribe** (subagent): the official documentation after the warden, the PDF report
    text (moved from the secretary) and documents for members.
  - **Tester** (subagent): the unit tests, last.
- **Dead weight** is a check for both reviewers: unused, duplicated or stale code,
  config, rules and instructions.
- **No review** on records-only, rename-only or tests-only branches.
- **Members:** `researcher.md` opens with a research protocol that names nothing
  internal, so it can be served to members; the honesty rules for members are enforced
  by the engine, not by instructions.

## Decisions

- 2026-11-01 (user): Sonnet for the inspector, Opus for the warden: the inspector runs
  often on small diffs against a fixed checklist; the warden runs once and needs the
  judgement across files.
- 2026-11-01 (user): report writing moves from the secretary to the scribe.
- 2026-11-01 (user): documents for members are a section of `scribe.md` for now and
  become a hat of their own when the member profile is built (S_202610_002, Assay).

## Notes

- 2026-11-01: instructions rewritten; `inspector.md` and `scribe.md` added; agents
  `midas-inspector`, `midas-scribe` and `midas-tester` added; `midas-warden` on Opus.
  The sandbox refused deleting the seven stale agents; the user removes them.
- To do: run the inspector and the warden on this branch after Claude Code restarts
  (subagent definitions load at start); then trim the duplicates left in `quant.md`
  and the area rules, and make `quant.md` free of repository paths for members.

## Inspections

| Date | Commits | Findings | Open |
|---|---|---|---|

## Kickoff prompt

Pick up S_202611_001 on `feature/midas/optimise-midas`: confirm the stale agents are
gone, run `midas-inspector` on the branch, then `midas-warden`, and finish the trims
listed under Notes.

## Done when

- The stale agents are gone and every remaining agent points at an existing file.
- The inspector and the warden have each reviewed this branch with their new
  instructions.
- No rule is written in more than one instruction file without a reference instead.
