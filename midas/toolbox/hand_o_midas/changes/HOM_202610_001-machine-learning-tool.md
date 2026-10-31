# HOM_202610_001: Machine-learning tool

| Field | Value |
|---|---|
| Status | in discussion |
| Priority | medium |
| Kind | new capability |
| Depends on | the shared data package (MD_202610_001), MD_202610_002 (point-in-time panel), S_202610_006 (portfolio wallet) |
| Raised | 2026-10-31 |

## Description

Besides the rule-based value strategies of `value_investing`, the user wants a
machine-learning approach: models that take the point-in-time metrics as input features
and predict a target, turned into a strategy that runs through the same wallet and
optimiser so both approaches can be compared against the S&P 500.

## Approach

- A new tool `Toolbox/hand_o_midas/`, on the same infrastructure as `value_investing`:
  it reads the cleaned tables and shared metrics from the shared data package and
  backtests through `skatkis.backtest`.
- **Several sub-models**, each with its own inputs and target (user, 2026-10-31). The
  first predicts the price; later ones may predict the P/E or other metrics and try
  other model types. A random forest is the first candidate.
- **Training and test sets by time.** Walk-forward splits on rebalance dates, never a
  random split, so no model is trained on information from after the date it predicts.
- Each sub-model's predictions become target weights for the wallet, judged like every
  strategy on out-of-sample excess return over `^SP500TR`.

## Open decisions (user)

- **Price target.** A random forest cannot predict outside the range of its training
  targets, and prices drift upwards, so a model trained on earlier prices cannot reach
  later ones. Proposed: predict the return to the horizon, from which the price follows
  as $P_{t+h} = P_t (1 + r)$, where $P_t$ is the price at the rebalance date, $h$ the
  horizon and $r$ the predicted return.
- Prediction horizon: one quarter (the rebalance interval), or longer.
- Which metrics enter as features for the first sub-model.

## Decisions

- 2026-10-31 (user): a separate tool, `hand_o_midas`, using the same infrastructure as
  `value_investing`; it reads the shared data package and has access to the wallet and
  optimiser.
- 2026-10-31 (user): several sub-models with different inputs and targets; the first
  predicts the price, later ones the P/E and others, with different model types.

## Kickoff prompt

Pick up HOM_202610_001. Read `midas/instructions/quant.md` and the tool README's "Data
sources", confirm the open decisions, then propose the first sub-model's features,
target, horizon and walk-forward scheme before writing code.

## Done when

- Agreed and designed (`design/HOM_202610_001-machine-learning-tool.md`) before any code.
