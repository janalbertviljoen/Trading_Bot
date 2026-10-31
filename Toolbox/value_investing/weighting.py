"""
value_investing/weighting.py
----------------------------
Rule weights learned from the past: how well each rule's grade went with the returns
that had already followed it by the decision date.

For each rule and earlier rebalance date s, the rank correlation across companies of the
rule's bucket points on s (``value_investing.scoring.rule_points``) with each company's
return from s to s + ``months`` less ``^SP500TR``'s. On decision date t a rule's weight
is the mean of those correlations over the last ``window`` dates s whose return window
had closed by t (s + ``months`` <= t), and 0 when the mean is negative (user,
2026-10-31). For a decision in January 2025 with 12-month windows and a window of 12,
that is the grades of 2020 to 2023 and their returns up to December 2024: everything
used was known at t. The weights still depend on the universe: with today's members on
past dates they learn from companies that later joined the index.

``static_weights`` freezes one set of weights instead (user, 2026-10-31): each rule's
mean correlation over every window closed by ``until`` (the end of the in-sample
period), used unchanged on every date. Decisions before ``until`` then use weights
fitted partly on their own outcomes and are in-sample; decisions after it are a test.

Owns
----
The forward excess return used for learning, the rule correlations per date and the
correlation weights.

Must not
--------
Grade or score (``value_investing.scoring``); define returns
(``skatkis.financial_logic``); choose companies (``value_investing.evaluation``).

Public API
----------
forward_excess(prices, index, *, benchmark, months)
    Each company's return over the next months less the benchmark's, with the window's
    end date.
rule_correlations(points, excess)
    Per date and rule, the rank correlation of points and forward excess.
correlation_weights(correlations, ends, *, window)
    Per date, each rule's mean correlation over the last closed windows, floored at 0.
static_weights(correlations, ends, *, until)
    One frozen weight per rule from every window closed by a date, on every date.

Computes
--------
forward excess return for learning : forward_excess
    x(s) = P(s + m) / P(s) - B(s + m) / B(s), ``adj_close`` levels, m months.
rule information coefficient, rank IC : rule_correlations
    Spearman correlation across companies of points_r(s) and x(s), dates with at least
    ``MIN_COMPANIES`` companies.
correlation weight, learned rule weight : correlation_weights
    w_r(t) = max(0, mean of IC_r(s) over the last ``window`` dates s with end(s) <= t);
    missing when no window has closed.
static correlation weight, frozen rule weight : static_weights
    w_r = max(0, mean of IC_r(s) over every s with end(s) <= until), the same on every
    date.

Dependency rule
---------------
pandas, ``skatkis.financial_logic`` and ``value_investing.inputs``.
"""

import pandas as pd

from Toolbox.skatkis.financial_logic import holding_return
from Toolbox.value_investing.inputs import price_on

MIN_COMPANIES = 20


def forward_excess(
    prices: pd.DataFrame, index: pd.MultiIndex, *, benchmark: str, months: int
) -> pd.DataFrame:
    """Return each company's return over the next ``months`` less the benchmark's.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns ``symbol``, ``date``, ``adj_close``, including ``benchmark``.
    index : pd.MultiIndex
        (``symbol``, ``date``) rows to compute it for.
    benchmark : str
        Total-return benchmark symbol.
    months : int
        Length of the window.

    Returns
    -------
    pd.DataFrame
        On ``index``: ``end`` (the window's end date) and ``excess`` (fraction, missing
        where a level is missing or the window ends after the last benchmark price).
    """
    frame = index.to_frame(index=False)
    frame["end"] = frame["date"] + pd.DateOffset(months=months)
    last = prices.loc[prices["symbol"] == benchmark, "date"].max()
    days = sorted({*frame["date"], *frame["end"]})
    symbols = pd.concat([frame["symbol"], pd.Series([benchmark])]).drop_duplicates()
    levels = price_on(prices, symbols, days, column="adj_close")

    def level(symbol: pd.Series, day: pd.Series) -> pd.Series:
        keys = pd.MultiIndex.from_arrays([symbol.astype("string"), day])
        return pd.Series(levels.reindex(keys).to_numpy(dtype=float), index=frame.index)

    market = pd.Series(benchmark, index=frame.index)
    own = holding_return(
        level(frame["symbol"], frame["date"]), level(frame["symbol"], frame["end"])
    )
    index_return = holding_return(
        level(market, frame["date"]), level(market, frame["end"])
    )
    excess = (own - index_return).where(frame["end"] <= last)
    return pd.DataFrame(
        {"end": frame["end"].to_numpy(), "excess": excess.to_numpy()}, index=index
    )


def rule_correlations(points: pd.DataFrame, excess: pd.Series) -> pd.DataFrame:
    """Return, per date and rule, the Spearman correlation of points and excess.

    Parameters
    ----------
    points : pd.DataFrame
        ``<id>_points`` per rule, indexed (``symbol``, ``date``).
    excess : pd.Series
        Forward excess return on the same index.

    Returns
    -------
    pd.DataFrame
        One row per date, one column per points column; missing where fewer than
        ``MIN_COMPANIES`` companies have both.
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
            if len(both) < MIN_COMPANIES:
                row[column] = float("nan")
                continue
            # Spearman's correlation is Pearson's on the ranks (average ranks for ties).
            row[column] = float(both["p"].rank().corr(both["x"].rank()))
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


def static_weights(
    correlations: pd.DataFrame, ends: pd.Series, *, until: pd.Timestamp
) -> pd.DataFrame:
    """Return one frozen weight per rule, repeated on every date.

    Parameters
    ----------
    correlations : pd.DataFrame
        ``rule_correlations`` output, one row per date s.
    ends : pd.Series
        The end of each date's return window, indexed by date s.
    until : pd.Timestamp
        Last day a window may end to be used (the end of the in-sample period).

    Returns
    -------
    pd.DataFrame
        Same shape as ``correlations``: every row the mean correlation over the dates s
        with ends(s) <= ``until``, floored at 0.
    """
    closed = ends[ends <= until].index
    frozen = correlations.loc[correlations.index.isin(closed)].mean().clip(lower=0.0)
    return pd.DataFrame(
        [frozen.to_numpy()] * len(correlations),
        index=correlations.index,
        columns=correlations.columns,
    )
