# value_investing change log

Planned changes to the value-investing tool (`Toolbox/value_investing/`). Prefix `VI`.
Naming and workflow: `midas/instructions/secretary.md`. The methodology is in
`docs/value_investing_bot_spec.md`.

## Open

| ID | Title | Kind | Status | Priority | Depends on |
|---|---|---|---|---|---|
| [VI_202610_002](changes/VI_202610_002-rule-catalogue.md) | Rule catalogue in the tool: G, E, B, M and Q rules as SOP layers, registry and config | new capability | in progress | medium | S_202610_001, S_202610_005, MD_202610_002 |
| [VI_202610_003](changes/VI_202610_003-signal-evaluation.md) | Signal evaluation: horizon returns, paper wallet and model score | new capability | in progress | high | VI_202610_002 |
| [VI_202610_004](changes/VI_202610_004-ranked-wallet-binned-score.md) | Ranked wallet and binned score | new capability | in progress | high | VI_202610_002, VI_202610_003 |
| [VI_202610_005](changes/VI_202610_005-return-correlation-weights.md) | Rule weights from past return correlation, NI1 and the universe on the date | methodology | in progress | high | VI_202610_004 |
| [VI_202610_006](changes/VI_202610_006-value-report.md) | Value report: buy-and-hold picks, value over time, rules behind the winners | new capability | in progress | high | VI_202610_005 |

## Done

| ID | Title | Done |
|---|---|---|
