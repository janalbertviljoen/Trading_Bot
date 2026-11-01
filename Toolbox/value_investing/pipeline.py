"""
value_investing/pipeline.py
---------------------------
Wires the layers: rule inputs, valuation ratios, the rule flags and the score.

``buyability`` is the one place the layers meet (``.claude/rules/layers.md``): it builds
the rule inputs, computes the shared ratios, runs every rule in the config through the
registry and scores the flags. Its output is the long table of the first slice: one row
per company and rebalance date, one column per rule value and flag, then the score.

Owns
----
The order of the stages and the columns of the buyability table.

Must not
--------
Compute ratios, flags or scores itself (``value_investing.layers``,
``value_investing.scoring``); read files (``value_investing.__main__``).

Public API
----------
Buyability(table, correlations, weights)
    The buyability table and the learned rule weights.
buyability(tables, config, *, end, view, benchmark)
    The buyability table of one universe view from the first rebalance date to ``end``.

Dependency rule
---------------
pandas and ``value_investing``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from Toolbox.value_investing.inputs import (
    Tables,
    members_on_date,
    rebalance_dates,
    rule_inputs,
    window_returns,
)
from Toolbox.value_investing.layers.valuation import valuation
from Toolbox.value_investing.rules import RULES, pass_column, value_column
from Toolbox.value_investing.scoring import learned_score, rule_points, score
from Toolbox.value_investing.weighting import correlation_weights, rule_correlations


@dataclass(frozen=True)
class Buyability:
    """The buyability table and the learned rule weights behind ``score_ic``.

    Parameters
    ----------
    table : pd.DataFrame
        One row per company and rebalance date (see ``buyability``).
    correlations : pd.DataFrame
        Per date and rule, the rank correlation of points and forward excess return.
    weights : pd.DataFrame
        Per date and rule, the learned weight used on that date.
    """

    table: pd.DataFrame
    correlations: pd.DataFrame
    weights: pd.DataFrame


def buyability(
    tables: Tables,
    config: Mapping[str, Any],
    *,
    end: pd.Timestamp,
    view: str,
    benchmark: str,
) -> Buyability:
    """Return the rule values, flags and scores per company and rebalance date.

    Parameters
    ----------
    tables : Tables
        The stored tables.
    config : Mapping[str, Any]
        ``config/rules.toml``: ``[schedule]``, ``[valuation]``, ``[score]`` and
        ``[rules.<ID>]``; the
        rules run are those with a table there.
    end : pd.Timestamp
        Last rebalance date considered (the data's as-of date).
    view : str
        ``on_date`` keeps on each date only companies whose ``date_added`` (in
        ``tables.universe``) is on or before it; ``current`` keeps every company.
    benchmark : str
        Total-return benchmark for the learned weights (``^SP500TR``).

    Returns
    -------
    Buyability
        ``table``: columns ``symbol``, ``date``, ``gics_sector``, ``close``, the
        valuation ratios, ``<id>_value``, ``<id>_pass`` and ``<id>_points`` per rule,
        ``score``, ``score_<F>``, ``score_binned``, ``score_binned_weighted``,
        ``score_weighted_share``, ``score_ic``, ``n_passed`` and ``n_available``;
        sorted by symbol and date. With the correlations and the weights behind
        ``score_ic``.

    Notes
    -----
    The grades, the pass-rate weights and the learned weights are all taken across the
    companies of the view on each date, so the two views give different scores for the
    same company.
    """
    schedule = config["schedule"]
    dates = rebalance_dates(
        pd.Timestamp(schedule["start"]), end, list(schedule["months"])
    )
    inputs = rule_inputs(
        tables,
        dates,
        depth=int(schedule["history_years"]),
        tolerance_days=int(schedule["year_tolerance_days"]),
        stale_days=int(schedule["stale_days"]),
    )
    ratios = valuation(inputs, average_years=int(config["valuation"]["average_years"]))
    columns: dict[str, pd.Series] = {"close": inputs.price}
    families: dict[str, str] = {}
    for rule_id, thresholds in config["rules"].items():
        rule = RULES[rule_id]
        result = rule.compute(inputs, ratios, **thresholds)
        columns[value_column(rule_id)] = result["value"]
        columns[pass_column(rule_id)] = result["passed"]
        families[pass_column(rule_id)] = rule.family
    table = pd.concat([pd.DataFrame(columns), ratios], axis=1)
    if view == "on_date":
        table = table.loc[
            members_on_date(
                pd.MultiIndex.from_frame(table.index.to_frame()), tables.universe
            )
        ]
    points = rule_points(
        table,
        {rule_id: RULES[rule_id].higher_is_better for rule_id in config["rules"]},
        buckets=int(config["score"]["buckets"]),
    )
    scores = score(
        table,
        families,
        points,
        min_available=int(config["score"]["min_available"]),
        buckets=int(config["score"]["buckets"]),
    )
    learning = config["learning"]
    forward = window_returns(
        tables.prices,
        pd.MultiIndex.from_frame(table.index.to_frame(index=False)),
        benchmark=benchmark,
        months=int(learning["months"]),
    )
    correlations = rule_correlations(
        points,
        forward["excess"],
        min_companies=int(config["score"]["min_companies"]),
    )
    ends = forward.groupby(level="date")["end"].first()
    weights = correlation_weights(correlations, ends, window=int(learning["window"]))
    scores["score_ic"] = learned_score(
        points,
        weights,
        scores["n_available"],
        min_available=int(config["score"]["min_available"]),
    )
    table = pd.concat([table, points], axis=1)
    table = pd.concat([table, scores], axis=1).reset_index()
    sectors = tables.universe[["symbol", "gics_sector"]].astype({"symbol": "string"})
    table = table.astype({"symbol": "string"}).merge(sectors, on="symbol", how="left")
    front = ["symbol", "date", "gics_sector", "close", *ratios.columns]
    rest = [c for c in table.columns if c not in front]
    return Buyability(
        table=table[front + rest].sort_values(["symbol", "date"], ignore_index=True),
        correlations=correlations,
        weights=weights,
    )
