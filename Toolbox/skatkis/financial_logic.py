"""
skatkis/financial_logic.py
--------------------------
General financial mathematics shared by every tool: return definitions.

Owns
----
The definitions of growth, annualised return, annualised relative return, relative growth
and forward excess return.

Must not
--------
Know any tool, source or column name; read or write data.

Public API
----------
annualised_growth(series, *, min_years)
    Annualised growth of a level series between its first and last values.
annualised_relative_return(series, benchmark)
    Annualised return of a series in excess of a benchmark, over their common dates.
relative_growth(series, benchmark)
    Growth of a series divided by the benchmark's, both 1 on their first common date.
forward_excess_return(series, benchmark)
    Return from each date to the next, less the benchmark's over the same step.
holding_return(start, end)
    Simple return between two levels.
money_weighted_return(flows, value, at)
    Annual internal rate of return of dated contributions and a final value.

Computes
--------
annualised growth, compound annual growth rate, CAGR : annualised_growth
    (S(T) / S(0))^(1 / Y) - 1, with Y the years between the first and last values
    (days / 365.25); missing when Y is below ``min_years`` or a value is not positive.
annualised relative return, annualised excess growth : annualised_relative_return
    With G = (S(T) / S(0)) / (B(T) / B(0)) the growth of the series S over the
    benchmark B between their first and last common dates, and Y the years between them
    (days / 365.25): G^(1 / Y) - 1. Missing when fewer than ``min_years`` years overlap.
relative growth, growth over the benchmark : relative_growth
    R(t) = (S(t) / S(t0)) / (B(t) / B(t0)), t0 the first date both have a value.
forward excess return, next-period excess return : forward_excess_return
    x(t) = S(t+1) / S(t) - B(t+1) / B(t), over consecutive index dates; missing on the
    last date.
holding return, simple return, period return : holding_return
    r = L(end) / L(start) - 1, as a fraction; missing when either level is missing or
    the start level is not positive.
money-weighted return, internal rate of return, IRR : money_weighted_return
    The rate r with Σ c_k (1 + r)^((T - t_k) / 365.25) = V, c_k the contributions at
    t_k, V the value at T; found by bisection between -99% and +1000% a year.

Dependency rule
---------------
pandas and the standard library.
"""

import pandas as pd

DAYS_PER_YEAR = 365.25


def annualised_growth(series: pd.Series, *, min_years: float = 1.0) -> float:
    """Return the annualised growth of ``series`` from its first to its last value.

    Parameters
    ----------
    series : pd.Series
        Levels indexed by date (a total-return price, revenue per fiscal year).
    min_years : float
        Shortest span for which a value is returned.

    Returns
    -------
    float
        (S(T) / S(0))^(1 / Y) - 1 as a fraction per year; NaN when the span is shorter
        than ``min_years`` or the first or last value is not positive.
    """
    values = series.dropna()
    if values.empty:
        return float("nan")
    years = (
        pd.Timestamp(values.index[-1]) - pd.Timestamp(values.index[0])
    ).days / DAYS_PER_YEAR
    first, last = float(values.iloc[0]), float(values.iloc[-1])
    if years < min_years or first <= 0 or last <= 0:
        return float("nan")
    return (last / first) ** (1 / years) - 1


def annualised_relative_return(
    series: pd.Series, benchmark: pd.Series, *, min_years: float = 1.0
) -> float:
    """Return the annualised growth of ``series`` over ``benchmark``.

    Parameters
    ----------
    series, benchmark : pd.Series
        Total-return levels (prices with dividends reinvested) indexed by date.
    min_years : float
        Shortest overlap for which a value is returned.

    Returns
    -------
    float
        (G)^(1 / Y) - 1 as a fraction per year, where G is the series' growth divided by
        the benchmark's over their common dates and Y the years covered; NaN when the
        overlap is shorter than ``min_years``.
    """
    both = pd.concat([series, benchmark], axis=1).dropna()
    if both.empty:
        return float("nan")
    years = (
        pd.Timestamp(both.index[-1]) - pd.Timestamp(both.index[0])
    ).days / DAYS_PER_YEAR
    if years < min_years:
        return float("nan")
    first, last = both.iloc[0], both.iloc[-1]
    growth = (float(last.iloc[0]) / float(first.iloc[0])) / (
        float(last.iloc[1]) / float(first.iloc[1])
    )
    return growth ** (1 / years) - 1


def relative_growth(series: pd.Series, benchmark: pd.Series) -> pd.Series:
    """Return the growth of ``series`` divided by the benchmark's growth.

    Parameters
    ----------
    series, benchmark : pd.Series
        Total-return levels indexed by date.

    Returns
    -------
    pd.Series
        (S(t) / S(t0)) / (B(t) / B(t0)) on the dates both have a value, t0 the first such
        date; above 1 the series did better. Empty when they share no date.
    """
    both = pd.concat([series, benchmark], axis=1).dropna()
    if both.empty:
        return pd.Series(dtype="float64")
    growth = both / both.iloc[0]
    return growth.iloc[:, 0] / growth.iloc[:, 1]


def forward_excess_return(series: pd.Series, benchmark: pd.Series) -> pd.Series:
    """Return, for each date, the return to the next date less the benchmark's.

    Parameters
    ----------
    series, benchmark : pd.Series
        Total-return levels on the same dates (e.g. rebalance dates), sorted.

    Returns
    -------
    pd.Series
        S(t+1) / S(t) - B(t+1) / B(t), as a fraction, on ``series``' index; missing on
        the last date and where either level is missing.
    """
    own = series.shift(-1) / series
    market = benchmark.reindex(series.index)
    return own - market.shift(-1) / market


def holding_return(start: pd.Series, end: pd.Series) -> pd.Series:
    """Return the simple return from ``start`` to ``end`` levels, element-wise.

    Parameters
    ----------
    start, end : pd.Series
        Total-return levels at the start and end of each holding, on the same index.

    Returns
    -------
    pd.Series
        end / start - 1 as a fraction; missing where either level is missing or the
        start level is not positive.
    """
    return (end / start - 1).where(start > 0)


def money_weighted_return(flows: pd.Series, value: float, at: pd.Timestamp) -> float:
    """Return the annual internal rate of return of contributions and a final value.

    Parameters
    ----------
    flows : pd.Series
        Money put in (positive), indexed by date.
    value : float
        What the holdings are worth on ``at``.
    at : pd.Timestamp
        The valuation date, on or after the last contribution.

    Returns
    -------
    float
        r such that Σ c_k (1 + r)^(Y_k) = value, Y_k the years (days / 365.25) from
        each contribution to ``at``; NaN when nothing was put in or no rate in
        [-0.99, 10] solves it.
    """
    years = [(at - pd.Timestamp(day)).days / DAYS_PER_YEAR for day in flows.index]
    amounts = [float(c) for c in flows]
    if not amounts or sum(amounts) <= 0:
        return float("nan")

    def gap(rate: float) -> float:
        grown = sum(c * (1 + rate) ** y for c, y in zip(amounts, years, strict=True))
        return grown - value

    low, high = -0.99, 10.0
    if gap(low) > 0 or gap(high) < 0:
        return float("nan")
    for _ in range(200):
        middle = (low + high) / 2
        if gap(middle) > 0:
            high = middle
        else:
            low = middle
    return (low + high) / 2
