# skatkis notes

## Planning (preliminary)

### Origin of the layout (2026-10-31, → S_202610_001)

The layout and governance follow the example repository in `docs/attachments.zip`
(Steward, `skatkis`, `fair_value`). Names and parts that do not apply here: client
folders, the Tutor role, accreditation sign-off documents, Word reports, Teams hosting.

### Wallet framework as found (2026-10-31, → S_202610_003, S_202610_006)

Single asset, row-by-row loop, no costs, mutates its input, needs a `RangeIndex` from
0. Documented in its docstrings on 2026-10-31; known defects listed in S_202610_003.
