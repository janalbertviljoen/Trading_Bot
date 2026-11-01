"""
value_investing/weighting.py
----------------------------
Rule weights learned from the past: how well each rule's grade went with the returns
that had already followed it by the decision date.

For each rule and earlier rebalance date s, the rank correlation across companies of the
rule's bucket points on s (``value_investing.scoring.rule_points``) with each company's
return from s to s + ``months`` less the benchmark's. On decision date t a rule's
weight is the mean of those correlations over the last ``window`` dates s whose return
window had closed by t (s + ``months`` <= t), and 0 when the mean is negative. For a
decision in January 2025 with 12-month windows and a window of 12, that is the 12
quarterly grades of March 2021 to December 2023 and their returns up to December 2024:
everything used was known at t. The weights still depend on the universe: with today's members on
past dates they learn from companies that later joined the index.

Owns
----
The rule correlations per date and the correlation weights.

Must not
--------
Grade or score (``value_investing.scoring``); compute returns
(``value_investing.inputs.window_returns``); choose companies
(``value_investing.evaluation``).

Public API
----------
rule_correlations(points, excess, *, min_companies)
    Per date and rule, the rank correlation of points and forward excess.
correlation_weights(correlations, ends, *, window)
    Per date, each rule's mean correlation over the last closed windows, floored at 0.

Computes
--------
rule information coefficient, rank IC : rule_correlations
    Spearman correlation across companies of points_r(s) and the excess return x(s)
    over the next m months (``value_investing.inputs.window_returns``), dates with at
    least ``min_companies`` companies.
correlation weight, learned rule weight : correlation_weights
    w_r(t) = max(0, mean of IC_r(s) over the last ``window`` dates s with end(s) <= t);
    missing when no window has closed.

Dependency rule
---------------
pandas and ``skatkis.analysis``.
"""

import pandas as pd

from Toolbox.skatkis.analysis import rank_correlation


def rule_correlations(
    points: pd.DataFrame, excess: pd.Series, *, min_companies: int
) -> pd.DataFrame:
    """Return, per date and rule, the Spearman correlation of points and excess.

    Parameters
    ----------
    points : pd.DataFrame
        ``<id>_points`` per rule, indexed (``symbol``, ``date``).
    excess : pd.Series
        Forward excess return on the same index.
    min_companies : int
        Fewest companies with both for a correlation on a date.

    Returns
    -------
    pd.DataFrame
        One row per date, one column per points column; missing where fewer than
        ``min_companies`` companies have both.
    """
    dates = points.index.get_level_values("date")
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for day in sorted(set(dates)):
        mask = pd.Series(dates == day, index=points.index)
        part = points.loc[mask]
        target = excess.loc[mask]
        row: dict[str, float] = {}
        for column in points.columns:
            both = pd.DataFrame({"p": part[column], "x": target}).dropna()
            if len(both) < min_companies:
                row[column] = float("nan")
                continue
            row[column] = rank_correlation(both["p"], both["x"])
        rows[pd.Timestamp(day)] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(index="date")


def correlation_weights(
    correlations: pd.DataFrame, ends: pd.Series, *, window: int
) -> pd.DataFrame:
    """Return each rule's weight per date from the correlations of closed windows.

    Parameters
    ----------
    correlations : pd.DataFrame
        ``rule_correlations`` output, one row per date s.
    ends : pd.Series
        The end of each date's return window, indexed by date s.
    window : int
        Number of most recent closed dates averaged (12: three years of quarters).

    Returns
    -------
    pd.DataFrame
        Same shape as ``correlations``: on date t, the mean correlation over the last
        ``window`` dates s with ends(s) <= t, floored at 0; missing when none closed.
    """
    weights: dict[pd.Timestamp, pd.Series] = {}
    for day in correlations.index:
        closed = ends[ends <= day].index
        recent = correlations.loc[correlations.index.isin(closed)].tail(window)
        weights[pd.Timestamp(day)] = (
            recent.mean().clip(lower=0.0) if len(recent) else recent.mean()
        )
    return pd.DataFrame.from_dict(weights, orient="index").rename_axis(index="date")
