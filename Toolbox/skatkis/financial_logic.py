"""
skatkis/financial_logic.py
--------------------------
General financial mathematics shared by every tool: return definitions.

Owns
----
The definitions of growth, annualised return and annualised relative return.

Must not
--------
Know any tool, source or column name; read or write data.

Public API
----------
annualised_growth(series, *, min_years)
    Annualised growth of a level series between its first and last values.
annualised_relative_return(series, benchmark)
    Annualised return of a series in excess of a benchmark, over their common dates.

Computes
--------
annualised growth, compound annual growth rate, CAGR : annualised_growth
    (S(T) / S(0))^(1 / Y) - 1, with Y the years between the first and last values
    (days / 365.25); missing when Y is below ``min_years`` or a value is not positive.
annualised relative return, annualised excess growth : annualised_relative_return
    With G = (S(T) / S(0)) / (B(T) / B(0)) the growth of the series S over the
    benchmark B between their first and last common dates, and Y the years between them
    (days / 365.25): G^(1 / Y) - 1. Missing when fewer than ``min_years`` years overlap.

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
