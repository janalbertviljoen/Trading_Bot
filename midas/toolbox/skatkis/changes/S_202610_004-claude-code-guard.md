# S_202610_004: Claude Code permissions, protected files and post-edit hook

| Field | Value |
|---|---|
| Status | in discussion |
| Priority | medium |
| Kind | tooling |
| Depends on | none |
| Raised | 2026-10-31 |

## Description

The rules say baselines and reference values are protected and that edited Python is
formatted and checked, but nothing enforces it. The example repository enforces both in
`.claude/settings.json` and a PostToolUse hook. Agents may not edit `.claude/settings.json`
or hooks without the user's approval, so this change is proposed for the user to apply
or approve.

## Approach

`.claude/settings.json` (shared, committed):

- `permissions.deny`: `Edit(benchmarks/baseline.json)`, `Edit(tests/**/fixtures/**)`,
  `Edit(data/**)`;
- `permissions.allow`: the Midas `index` tools (`mcp__midas__*` once S_202610_002 is
  done; `mcp__trading-bot__list_datasets` and `run_checks` meanwhile);
- `hooks.PostToolUse` on `Edit|Write`: `.claude/hooks/post_edit_check.py` runs Black
  (and ruff `--fix` and pyright once configured) on the edited `.py` file when it is
  outside `notebooks/` and `Toolbox/SandBox/`, and reports what it found.

Old agent files to remove at the same time (superseded by the Midas hats):
`.claude/agents/code-improver.md`, `strategy-researcher.md`, `reviewer.md`,
`midas.md`, `midas-builder.md`, `midas-improver.md`, `midas-reviewer.md`. (The old
`midas/README.md` and `midas/hats/` were removed on 2026-10-31.)

## Open decisions (user)

- Apply the settings yourself, or approve Midas editing `.claude/settings.json`?
- Which paths are protected beyond the three above?

## Done when

- Protected edits are denied; the hook runs on edit; the Guard's check finds the
  settings, hook and instructions in agreement.
