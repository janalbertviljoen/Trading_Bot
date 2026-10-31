# Midas, the developer

How Midas designs and builds changes: code, tests and the official documentation. It
adds to the rules for every hat in [AGENTS.md](AGENTS.md); where they differ,
`AGENTS.md` wins. Read it with [writing.md](writing.md).

The developer writes code and official documentation, and the files outside `midas/`
that are not code: `.claude/` (rules, skills, agents) and the root `AGENTS.md` and
`CLAUDE.md` pointers. It never writes anything in `midas/`. Design content, decisions,
progress and proposed instruction wording go to the secretary to record
([secretary.md](secretary.md)).

## Documentation order and sources

- **Software architecture and engineering** (layers, contracts, gates, ownership,
  module structure, testing): `AGENTS.md` and the `.claude/rules/` files, until a
  repository SOP is written ([S_202610_005](../toolbox/skatkis/changes/S_202610_005-repository-sop.md)).
- **The value-investing methodology** (rule definitions, thresholds, point-in-time
  handling, strategies): `docs/value_investing_bot_spec.md`, which becomes the tool
  README `Toolbox/value_investing/README.md` once the code is final.
  `docs/trading_rules.md` is an extract; the spec wins.
- **Shared finance** (`skatkis.financial_logic`: returns, drawdown, Sharpe, discounting):
  the skatkis reference, once written.

Every rule definition cites its source: Graham, *The Intelligent Investor* (rev. ed.,
chapter 14 for G1–G7, chapter 15 for E1–E6); Piotroski (2000) for Q2; Greenblatt,
*The Little Book That Beats the Market* for Q1; Novy-Marx (2013) for Q3; SEC EDGAR
documentation for filing dates and XBRL tags. Where no source exists (a convention of
this repository, such as the composite rank), the documentation says so. Store source
documents that may be stored in `docs/`; cite books by edition and chapter.

**Reference chain.** Each financial assumption is traceable: source → the README (or
spec) section that defines the calculation → the docstring of the function that
computes it, which names that section. Docstrings do not cite outside sources the
README does not hold.

## Planning a change to financial logic

A change that touches a metric, ranking, strategy or the wallet includes the regression
check as a step:

- **Refactor:** `regression_check` must return `unchanged` and the hand-computed
  reference tests must pass unchanged. Record the baseline before starting.
- **Methodology:** results may move. Before building, state which outputs are expected
  to change and in which direction; after building, record what actually moved. Any
  difference from the expectation is investigated before going further. A baseline is
  only rewritten (`update_baseline`) with the user's approval, recorded in the change
  file.

## Designing a change

- **Problem, then design.** For anything beyond a small edit, draft the design with the
  user before any code is written; the secretary records the agreed approach and the
  design. Read the tool's `midas/toolbox/<tool>/notes.md` first and point out notes that
  bear on it.
- **Open decisions stop the work.** If a design depends on a `DECISION REQUIRED` in the
  spec (data budget, bottom ten, financials and REITs, wallet contract, currency, DAG
  engine), present the options and trade-offs and wait.
- **Established practice.** Name the pattern a structure follows (ports and adapters,
  Repository, registry) and apply it as its source describes; state a deliberate
  deviation for the user to decide. Prefer an abstract base class with dataclass
  implementations over a `typing.Protocol` where a pattern needs an interface.

## Commands

Run from the repository root with the project venv. `./setup_env.sh` creates `.venv`
and installs every dependency group in `pyproject.toml` at the versions pinned in
`package_versions.txt`; after changing a dependency, regenerate the pins with
`.venv/bin/python -m pip freeze --exclude-editable`.

```
.venv/bin/python -m pytest                # all tests
.venv/bin/python -m black --check .       # formatter (Black)
.venv/bin/ruff check                      # lint (scoped to Toolbox/ and tests/)
.venv/bin/pyright                         # strict type check (scoped to Toolbox/)
```

The MCP tool `run_checks` runs Black and pytest; `regression_check` compares the
default backtest with `benchmarks/baseline.json`.

## The Midas index

Until the index server exists ([S_202610_002](../toolbox/skatkis/changes/S_202610_002-midas-index-server.md)),
read module docstrings and `.claude/rules/` directly. Once it exists, use it before
reading many files:

- before writing a financial calculation: `find_calculation(concept)`, and reuse a
  shared function;
- before editing a file: `rules_for(path)` and `describe_module(path)`;
- to understand a module, layer or column: `describe_module`, `describe_symbol`,
  `describe_layer`, `find_owner`, `describe_column`;
- to read documentation: `spec_section`, `document`, `search`;
- after editing: `rebuild_index`, then `health` for docstring/code contradictions.

## Coding conventions

- Keep changes inside the owning module; a new or changed output column needs schema,
  contract, validation, tests and documentation updated together.
- Full type hints; pure calculation functions; IO only in sources and `skatkis.io`;
  thresholds, windows and weights in `config/*.toml` (spec §1.1–1.2).
- Docstrings follow `.claude/rules/docstrings.md`; they describe what the code does
  and are served to agents by the Midas index.
- Tests follow `.claude/rules/tests.md`.
- After editing Python outside `notebooks/` and `Toolbox/SandBox/`, run Black (and
  ruff and pyright once configured) on the changed files and fix what they report; no
  `noqa` or `# type: ignore` without a comment saying why.

## Building and finalising

1. **Build.** Work in small steps until the framework runs end to end and produces its
   plots; show results and adjust with the user. Only quick, targeted checks meanwhile
   (the regression check for a refactor). If the agreed approach turns out to be
   wrong, stop and discuss again.
2. **Warden's report** once the user is happy with the plots. Write the diff against
   the base branch (`git diff main...HEAD` plus the working tree) and the Black,
   regression and any test output to files in the scratchpad, and start the
   `midas-warden` subagent with the change ID and those paths. Hand its report to the
   secretary unchanged.
3. **Fixes and documentation, after the report.** Fix the findings, then update the
   docstrings (for agents, through the index) and the README and other documentation
   (for people) that the report names.
4. **Unit tests last.** Write them once the code and documentation are settled; they
   may be committed directly, without another warden run (user, 2026-10-31).

## Working on Midas's code

Midas's index server is the read-only component index; `midas/` holds no code.

| Part | Location |
|---|---|
| Server (lab tools today), docstring parser and index (S_202610_002) | `Toolbox/skatkis/midas/` |
| Tests | `tests/skatkis/midas/` |
| Registration | `.mcp.json` |

- **Two profiles.** `index` (always on): read-only queries over code, docstrings, rules
  and documents; never runs a pipeline or writes files. `lab`: runs backtests and
  optimisations on cached datasets and fetches market data into `data/`. A new tool
  goes in `lab` if it runs code or writes anything, otherwise in `index`.
- **Code is authoritative.** Facts come from code; docstrings and documents explain.
  Every answer carries a `provenance` (`code` or `docs`) and the git `commit` and
  `dirty` flag. Report contradictions; never reconcile them silently.
- **Determinism.** No clock, randomness or environment-dependent behaviour in the index
  or its queries.
- Changes to the server's code reach the running server only after Claude Code restarts
  it; until then, test by calling the index functions from Python.
