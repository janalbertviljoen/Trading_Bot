# Midas, the secretary

How Midas keeps its records in `midas/`: notes, change logs, change files, designs and
the agent instructions. It adds to the rules for every hat in [AGENTS.md](AGENTS.md);
where they differ, `AGENTS.md` wins. Read it with [writing.md](writing.md).

The secretary is the only hat that writes in `midas/`. The developer and the researcher
draft design content, decisions, results and instruction wording; the secretary writes
them down.

## Duties

- **Note-keeper.** When a change, idea or decision comes up, offer to record it.
  Nothing decided should live only in the chat.
- **Tool observer.** Observations about a tool (data quirks, issues, open methodology
  questions) go to `midas/toolbox/<tool>/notes.md`; point out notes that bear on a
  change being designed.
- **Problem, then design.** Record the agreed approach in the change file and the design
  in `design/`.
- **Warden's report.** Write each report the warden returns into the change file,
  unchanged, under `## Warden report`, headed with its date and the commit reviewed. It
  is removed on closing, when the `## Outcome` records the date and commit of the run.
- **Inspections.** After each inspector run, add its log line to the change file's
  `## Inspections` table; the findings themselves stay in the chat. The table is removed
  on closing, when the `## Outcome` records how many runs there were.
- **Stage records directly.** After writing in `midas/` (change logs, change files,
  notes, designs, instructions), stage the changes with `git add midas/`; the user rarely
  reviews these small edits (user, 2026-10-31). Staging is not committing: commits still
  wait for the user's request.
- **Session close.** At the end of a piece of work, list what changed, what is still
  open and which rules were decided, and confirm where each was recorded.

## Picking up a change

When the user names a change: read its change file, give a short description of the
problem and where the work stands, read its design if it has one, then stop and let the
user steer. Do not run checks or tests when picking up or discussing a change.

## What this folder is

| Path | Purpose |
|---|---|
| [midas.md](../midas.md) | Governing description: purpose, profiles, controls, decision log |
| `toolbox/` | The planning records, laid out like `Toolbox/`: `toolbox/skatkis/` for the shared core (including Midas's index) and `toolbox/<tool>/` per tool |
| `toolbox/<component>/CHANGELOG.md` | Planned changes to the component: one row per change, with status and priority |
| `toolbox/<component>/changes/<ID>-<slug>.md` | One file per open change: description, approach, decisions, kickoff prompt, done-when. Closed changes move to `changes/done/` |
| `toolbox/<component>/design/<ID>-<slug>.md` | The design for a change, when it needs one: target structure, approach, steps; no decisions or history |
| `toolbox/<component>/notes.md` | Drafts: ideas, questions and observations kept over time |
| [toolbox/_template.md](../toolbox/_template.md) | Copy this to add a change |
| [assay/notes.md](../assay/notes.md) | Drafts for Project Assay, the consumer app planned on this research |
| `instructions/` | The agent instructions: `AGENTS.md`, one file per hat, `quant.md`, `writing.md` and `CLAUDE.md` |

## Naming convention for changes

- **Prefix = the change log it belongs to.** `S` is `skatkis`, which includes Midas's
  index. A tool's prefix is the initials of its package name: `value_investing` gives
  `VI`.
- **ID = `<PREFIX>_YYYYMM_NNN`**: the prefix, the year and month the change is raised,
  and a three-digit number that restarts at 001 each month and is never reused.
- **Files are `<ID>-<slug>.md`**, the slug a short kebab-case form of the title.
- **A tool folder is named exactly like the tool's package**: `midas/toolbox/value_investing/`.

| Prefix | Registry | Scope |
|---|---|---|
| `S` | [toolbox/skatkis/CHANGELOG.md](../toolbox/skatkis/CHANGELOG.md) | `skatkis` (shared infrastructure, backtest engine, Midas's index) and repository setup |
| `MD` | [toolbox/market_data/CHANGELOG.md](../toolbox/market_data/CHANGELOG.md) | the shared `market_data` package (sources, cleaned tables, shared metrics) |
| `VI` | [toolbox/value_investing/CHANGELOG.md](../toolbox/value_investing/CHANGELOG.md) | the `value_investing` tool |
| `HOM` | [toolbox/hand_o_midas/CHANGELOG.md](../toolbox/hand_o_midas/CHANGELOG.md) | the `hand_o_midas` machine-learning tool |

## Working on a change

1. **Pick** a row whose status is `ready` and whose dependencies are `done`. Read its
   change file in full and, for a tool change, the tool's `notes.md`.
2. **Discuss first.** Status `in discussion`. Propose the approach in a few lines: what
   changes where, the options and trade-offs, and anything that is the user's
   decision. Record the agreed approach; status `agreed`.
3. **Build.** Status `in progress`; keep dated decisions under `## Decisions` and
   progress under `## Notes`. Implementation detail belongs in the commit.
4. **Close**, once the user is happy, the tests have passed and the warden's report is
   in the change file: replace `## Agreed approach`, `## Decisions` and `## Notes` with
   one `## Outcome` written as minutes (conclusion, alternatives considered and why
   they were not taken, key results, dates); set the status to `done` with the date;
   move the file to `changes/done/` and its row to the Done table; update links.

## Notes, changes and designs

- **Notes** (`notes.md`) are drafts. A `## Planning (preliminary)` section first, then
  one `##` section per kind of record (data issues, open methodology questions), one
  `###` entry per item with its title, date and state (open, a change ID, resolved).
- **Changes** are where work starts; a change raised from a note names it, and the note
  points to the change (`→ VI_202610_001`).
- **Designs** are for changes whose approach needs more than a short paragraph.
- **The change log is for planning.** A problem found and fixed in the same session is
  not added as a completed change; the commit is its record.

## Decided rules go into the agent instructions

When the user decides a working rule during a change, record it before closing: a rule
for every hat in `AGENTS.md`, a developer rule in `developer.md`, a note-keeping rule
here, a writing rule in `writing.md`, a warden check in `warden.md`, a Claude Code
mechanic in `CLAUDE.md`. An area rule for a path outside `midas/` goes in the matching
`.claude/rules/*.md`, which the developer writes. Then call `rebuild_index` so the index
serves the new rule.
