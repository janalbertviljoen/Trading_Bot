# S_202610_003: Docstrings to the numpydoc structure

| Field | Value |
|---|---|
| Status | ready |
| Priority | medium |
| Kind | documentation |
| Depends on | S_202610_001 |
| Raised | 2026-10-31 |

## Description

On 2026-10-31 the wallet-framework modules received Google-style docstrings (inputs,
dtypes, formulas, outputs, edge cases). The user then chose the example's numpydoc
structure so the Midas index can parse module ownership and calculations
(`.claude/rules/docstrings.md`). The modules need converting, and spec §1.3 updating.

## Approach

After the move in S_202610_001 (the module header is the path from `Toolbox/`, so it
must follow the move):

- each module docstring gets the path header, summary, `Owns`, `Must not`,
  `Public API`, `Computes` (for calculation modules) and `Dependency rule`;
- function docstrings move from `Args`/`Returns`/`Edge cases` to `Parameters`/
  `Returns`/`Raises`/`Notes`, keeping the content already written (dtypes, units,
  formulas, mutation and edge-case behaviour);
- spec §1.3 is rewritten to the same structure, with the example docstring converted.

Docstring-only: an AST comparison with docstrings stripped must show no code change.

## Known issues found while documenting (2026-10-31)

Recorded here for the improver work, not fixed by this change:

- `PromisedLandExplorer.pre_process` takes `highest_coor` from the unsorted strategic
  points, so it is the first five runs, not the five best.
- `make_trades`, `trade_direction` and `rolling_vwap` mutate their input;
  `optimise_strategy` overwrites the caller's config dicts.
- `update_wallet` needs a `RangeIndex` from 0.
- `optimise(rank_by="max_drawdown")` sorts ascending, putting the deepest drawdown
  first.

## Done when

- Every module under `Toolbox/` (except `SandBox/`) follows `.claude/rules/docstrings.md`.
- Spec §1.3 matches.
- The AST check shows no code change; Black and pytest pass.
