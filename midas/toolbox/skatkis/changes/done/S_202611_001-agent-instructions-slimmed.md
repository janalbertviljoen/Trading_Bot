# S_202611_001: Agent instructions slimmed and roles split

| Field | Value |
|---|---|
| Status | done (2026-11-01) |
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

## Outcome

Done 2026-11-01. The agent instructions are slimmed and the review and writing roles
split. `AGENTS.md` now keeps only what guards every edit: the hats, the lifecycle, the
authority order, the principles, methodology versus refactor, what is prohibited and
the working rules. The repository layout and branches moved to `developer.md`, the
persona to `CLAUDE.md`. The seven agent definitions from before the layout (`midas`,
`midas-builder`, `midas-improver`, `midas-reviewer`, `code-improver`, `reviewer`,
`strategy-researcher`) were removed by the user; they pointed at deleted files and
contradicted `AGENTS.md`.

Four hats were added or redefined:

- the inspector (Sonnet): code-only reviews at build milestones on big branches, one
  log line per run;
- the warden (Opus): once per branch before the pull request, building on the
  inspections;
- the scribe: official documentation after the warden, the PDF report text (moved from
  the secretary) and documents for members;
- the tester: unit tests, last, for every branch, the researcher's included.

Both reviewers check for dead weight. Neither runs on records-only, rename-only or
tests-only branches; that rule lives only in `AGENTS.md`.

`researcher.md` opens with a research protocol, and `quant.md` with data rules, that
name nothing internal, so they can be served to members of Project Assay; the
repository specifics follow in "In this repository" in each. The protocol states only
the benchmark, the total-return series of the index the strategy runs on; the sleeve's
target stays in `AGENTS.md`. The honesty rules for members are to be enforced by the
engine, not by instructions. A rule written in more than one instruction file was
replaced by a reference to its one home: the check commands in `developer.md`, the
refactor and config rules in `AGENTS.md`, and in `.claude/rules/` the lines that
restated `AGENTS.md`, `developer.md`, `researcher.md`, `quant.md` or `baselines.md`.

Decisions (user, 2026-11-01):

- Sonnet for the inspector, which runs often on small diffs against a fixed checklist;
  Opus for the warden, which runs once and needs judgement across files.
- Report writing moves to the scribe.
- Documents for members stay a section of `scribe.md` until the member profile is built
  (S_202610_002, Assay).
- `quant.md` splits like `researcher.md`, into member-safe rules and the repository
  part.
- The research protocol states only the benchmark, because a member has their own goal.
- Reporting results as CPI + x% is dropped.
- A researcher's branch lists its unit tests for the tester and writes none.

Alternatives considered: keeping one heavy review at the end, dropped because it
reviews big branches late and at full cost; keeping the target in the protocol, dropped
because the protocol is meant for members.

Reviews: one inspector run on 2026-11-01 (main..693267c plus the working tree): 0
blocking, 3 should fix, all fixed. Warden run on 2026-11-01 at 693267c with a dirty
tree: 0 blocking, 5 should fix, all fixed, with the notes fixed or recorded (the column
names for members in `toolbox/skatkis/notes.md`, spec §1.4 in
`toolbox/value_investing/notes.md`). No Python changed; the 15 existing tests pass.

Left for the user: the description of `.claude/agents/midas-researcher.md` still says
"against the S&P 500"; the sandbox blocks writes to `.claude/agents/`.

## Done when

- The stale agents are gone and every remaining agent points at an existing file.
- The inspector and the warden have each reviewed this branch with their new
  instructions.
- No rule is written in more than one instruction file without a reference instead.
