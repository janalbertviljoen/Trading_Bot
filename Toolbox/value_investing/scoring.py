"""
value_investing/scoring.py
--------------------------
The buyability scores: the share of its evaluable rules a company passes, and the
binned score that grades each rule's value against the other companies and sums the
grades.

Each rule's flag is 1 (passed), 0 (failed) or missing (not evaluable). ``score`` is the
mean of the flags that exist, from 0 to 1, so it does not depend on how many rules could
be evaluated on the date and stays comparable over time while long-window rules phase
in. A company evaluable on fewer than ``min_available`` rules on a
date has no total score there and is not ranked (5 in ``config/rules.toml``).
``n_passed`` and ``n_available`` keep the counts.

``score_binned`` replaces the pass/fail cut by a grade:
on each date, the companies evaluable on a rule are put in ``buckets`` (6) equal-sized
groups by the rule's value, from best to worst; the best group earns ``buckets - 1``
(5) points, the worst 0. A company evaluable on a rule whose value is undefined (a P/E
on negative earnings) earns 0. ``score_binned`` is the sum of the points (at most
``buckets - 1`` per rule: 0 to 80 with 16 rules), which tells companies just past a
cut-off from clear winners; it grows with the number of evaluable rules, so it is
compared between companies on the same date.
The grades come from the date's own companies, so they use nothing from later dates.

The score is given in total and per rule family. This is a convention of this
repository, not a published method; every rule counts once, so concepts tested by more
than one rule (the current ratio in G2a and E1) weigh more.

``score_binned_weighted`` weighs each rule's points by how hard the rule is to pass on
the date: w_r = 1 - the share of evaluable companies
passing its threshold, so a rule every company passes weighs 0 and one nobody passes
weighs 1. The weights are independent of each other (they need not sum to 1) and come
from the date's own companies.

Owns
----
The buyability score, its per-family parts, ``n_passed`` and ``n_available``; the rule
points, the rule weights, the binned score and the weighted binned score.

Must not
--------
Compute rule values or flags (``value_investing.layers``); choose companies or weights
(the strategies).

Public API
----------
rule_points(table, directions, *, buckets)
    Points per rule from the rank of its value among the companies on the date.
rule_weights(flags, columns)
    The weight of each rule on each date: 1 - its pass rate.
learned_score(points, weights, available, *, min_available)
    Points weighted by rule weights learned elsewhere (``value_investing.weighting``).
score(flags, families, points, *, min_available, buckets)
    Total, per-family, binned and weighted binned score, its 0-to-1 share,
    ``n_passed`` and ``n_available``.

Computes
--------
rule points, bucket grade, sextile : rule_points
    q = the percentile rank (0 to 1] of the rule's value among the companies evaluable
    on it that date, best value highest; points = ceil(q * k) - 1 for k buckets, so 0 to
    k - 1; 0 when the value is undefined; missing when the rule is not evaluable.
buyability score, share of rules passed : score
    score = n_passed / n_available, with n_passed the rules passed and n_available the
    rules with a flag; missing when n_available < min_available. score_<F> the same over
    family F, missing only when no rule of F is evaluable.
rule weight, rule difficulty : rule_weights
    w_r = 1 - p_r, p_r the share of companies with a flag for rule r on the date that
    pass it; 0 when no company has a flag.
binned buyability score, weighted binned score : score
    score_binned = Σ_r points_r over the evaluable rules, 0 to (k - 1) * rules;
    score_binned_weighted = Σ_r w_r points_r. score_weighted_share =
    score_binned_weighted / ((k - 1) Σ_r w_r) over the evaluable rules, 0 to 1: the
    share of the most the company could score. All missing when n_available <
    min_available; the share also when Σ_r w_r = 0.
learned buyability score : learned_score
    S = Σ_r w_r(t) points_r with w_r(t) the date's learned weights; missing when
    n_available < min_available or when every weight of the date is 0 or missing.

Dependency rule
---------------
numpy, pandas and ``value_investing.rules``.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from Toolbox.value_investing.rules import pass_column, points_column, value_column


def rule_weights(flags: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return the weight of each rule on each date: 1 - its pass rate.

    Parameters
    ----------
    flags : pd.DataFrame
        Pass flags (1.0, 0.0 or NaN), indexed (``symbol``, ``date``); not modified.
    columns : Sequence[str]
        The pass-flag columns.

    Returns
    -------
    pd.DataFrame
        One row per date, one column per pass-flag column: 1 - the mean flag of the
        companies evaluable on the rule, 0 when none is.
    """
    dates = flags.index.get_level_values("date")
    pass_rate = flags[list(columns)].groupby(dates).mean()
    return (1 - pass_rate).fillna(0.0).rename_axis(index="date")


def rule_points(
    table: pd.DataFrame, directions: Mapping[str, bool], *, buckets: int
) -> pd.DataFrame:
    """Return each rule's points on each row, from its value's rank on the date.

    Parameters
    ----------
    table : pd.DataFrame
        ``<id>_value`` and ``<id>_pass`` per rule, indexed (``symbol``, ``date``); not
        modified.
    directions : Mapping[str, bool]
        Rule ID to ``True`` when a higher value is better, ``False`` when lower is.
    buckets : int
        Number of equal-sized groups (6 gives 0 to 5 points).

    Returns
    -------
    pd.DataFrame
        ``<id>_points`` per rule (float, 0 to ``buckets - 1``), on ``table``'s index.

    Notes
    -----
    Ties share the mean rank, so equal values earn equal points. Only companies
    evaluable on the rule (a flag present) are ranked; of those, a missing value earns 0
    and is left out of the ranking of the others.
    """
    dates = table.index.get_level_values("date")
    result: dict[str, pd.Series] = {}
    for rule_id, higher in directions.items():
        flag = table[pass_column(rule_id)]
        value = table[value_column(rule_id)].where(flag.notna())
        rank = value.groupby(dates).rank(pct=True, ascending=higher)
        points = np.ceil(rank * buckets) - 1
        result[points_column(rule_id)] = points.where(
            value.notna(), pd.Series(0.0, index=table.index).where(flag.notna())
        )
    return pd.DataFrame(result, index=table.index)


def score(
    flags: pd.DataFrame,
    families: Mapping[str, str],
    points: pd.DataFrame,
    *,
    min_available: int,
    buckets: int,
) -> pd.DataFrame:
    """Return the buyability scores per row.

    Parameters
    ----------
    flags : pd.DataFrame
        One column per rule pass flag (1.0, 0.0 or NaN); not modified.
    families : Mapping[str, str]
        Pass-flag column to family (``G``, ``E``, ``B``, ``N``).
    points : pd.DataFrame
        ``rule_points`` output on the same index.
    min_available : int
        Fewest evaluable rules for a total score.
    buckets : int
        The number of buckets the points were graded in (6: 0 to 5 points).

    Returns
    -------
    pd.DataFrame
        ``score`` and ``score_<F>`` per family in order of first appearance (fractions
        from 0 to 1; ``score`` missing below ``min_available`` evaluable rules,
        ``score_<F>`` where no rule of the family is evaluable), ``score_binned``
        and ``score_binned_weighted`` (sums of points, plain and weighted by
        ``rule_weights``; missing below ``min_available``), ``score_weighted_share``
        (0 to 1), then ``n_passed`` and
        ``n_available`` (integers), on ``flags``' index.
    """
    columns = list(families)
    passed = flags[columns]
    available = passed.notna().sum(axis=1)
    enough = available >= min_available
    result = pd.DataFrame(
        {"score": passed.mean(axis=1).where(enough)}, index=flags.index
    )
    for family in dict.fromkeys(families.values()):
        own = [c for c in columns if families[c] == family]
        result[f"score_{family}"] = passed[own].mean(axis=1)
    result["score_binned"] = points.sum(axis=1).where(enough)
    weights = rule_weights(flags, columns)
    per_row = weights.reindex(flags.index.get_level_values("date"))
    per_row.columns = [c.removesuffix("_pass") + "_points" for c in per_row.columns]
    weighted = points.mul(per_row[points.columns].to_numpy())
    result["score_binned_weighted"] = weighted.sum(axis=1).where(enough)
    most = (per_row[points.columns] * points.notna().to_numpy()).sum(axis=1) * (
        buckets - 1
    )
    share = weighted.sum(axis=1).to_numpy() / most.to_numpy()
    result["score_weighted_share"] = pd.Series(share, index=flags.index).where(
        enough & (most.to_numpy() > 0)
    )
    result["n_passed"] = passed.sum(axis=1).astype("int64")
    result["n_available"] = available.astype("int64")
    return result


def learned_score(
    points: pd.DataFrame,
    weights: pd.DataFrame,
    available: pd.Series,
    *,
    min_available: int,
) -> pd.Series:
    """Return Σ_r w_r(t) points_r per row, with the weights of the row's date.

    Parameters
    ----------
    points : pd.DataFrame
        ``rule_points`` output, indexed (``symbol``, ``date``).
    weights : pd.DataFrame
        One row per date, one column per points column (``value_investing.weighting``).
    available : pd.Series
        Evaluable rules per row (``n_available``).
    min_available : int
        Fewest evaluable rules for a score.

    Returns
    -------
    pd.Series
        The score on ``points``' index; a missing point counts as 0.
    """
    per_row = weights.reindex(
        index=points.index.get_level_values("date"), columns=points.columns
    )
    values = per_row.to_numpy(dtype=float)
    total = np.nansum(values, axis=1)
    weighted = np.nansum(values * points.fillna(0.0).to_numpy(dtype=float), axis=1)
    learned = pd.Series(weighted, index=points.index)
    return learned.where((available >= min_available) & (total > 0))
