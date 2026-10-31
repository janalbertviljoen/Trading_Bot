"""
market_data/metrics.py
----------------------
Shared metrics read by every tool; each ratio is computed here and nowhere else.

The first metric is the price-to-earnings ratio on the latest annual EPS. Prices from
Yahoo are split-adjusted to the day they were fetched, while EDGAR EPS is on the share
basis of its filing date, so the EPS is first put on the fetch-date basis with the
splits table. A filing is used from the day after it was filed. The P/E on trailing
twelve-month EPS replaces the annual version once quarterly values are cleaned.

Owns
----
Shared ratios (P/E) and the split factor that puts per-share values on one basis.

Must not
--------
Choose tags or first-reported values (``market_data.clean``); apply rule thresholds
(the tools).

Public API
----------
split_factor(splits, symbol, dates)
    Product of the split ratios after each date, per symbol.
price_earnings(prices, eps, splits)
    Daily P/E per symbol on the latest annual EPS filed before the day.

Computes
--------
split factor : split_factor
    F(d) = product of numerator / denominator over the symbol's splits dated after d;
    F = 1 when there are none.
price-to-earnings ratio, P/E : price_earnings
    P/E(t) = close(t) / (EPS / F(filed)), where EPS is the latest first-reported annual
    diluted EPS filed before day t; missing when that EPS is zero or negative.

Dependency rule
---------------
numpy and pandas.
"""

import numpy as np
import pandas as pd


def split_factor(splits: pd.DataFrame, symbol: str, dates: pd.Series) -> np.ndarray:
    """Return, for each of ``dates``, the product of ``symbol``'s split ratios after it.

    Parameters
    ----------
    splits : pd.DataFrame
        Columns ``symbol``, ``date``, ``numerator``, ``denominator``.
    symbol : str
        The symbol whose splits apply.
    dates : pd.Series
        Datetimes.

    Returns
    -------
    np.ndarray
        One factor per date: dividing a per-share value from that date by it puts the
        value on the basis after the last split. 1.0 where no split follows.
    """
    own = splits[splits["symbol"] == symbol].sort_values("date")
    ratios = np.asarray(own["numerator"] / own["denominator"], dtype=float)
    # suffix[i] = product of ratios[i:], with suffix[len] = 1
    suffix = np.append(np.cumprod(ratios[::-1])[::-1], 1.0)
    split_days = np.asarray(own["date"], dtype="datetime64[ns]")
    query = np.asarray(dates, dtype="datetime64[ns]")
    positions: np.ndarray = np.searchsorted(split_days, query, side="right")
    return suffix[positions]


def price_earnings(
    prices: pd.DataFrame, eps: pd.DataFrame, splits: pd.DataFrame
) -> pd.DataFrame:
    """Return the daily P/E per symbol on the latest annual EPS filed before each day.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns ``symbol``, ``date``, ``close`` (split-adjusted to the fetch date, USD).
    eps : pd.DataFrame
        Columns ``symbol``, ``filed``, ``period_end``, ``value``: first-reported annual
        diluted EPS (USD per share, on the basis of its filing date).
    splits : pd.DataFrame
        Columns ``symbol``, ``date``, ``numerator``, ``denominator``.

    Returns
    -------
    pd.DataFrame
        Columns ``symbol``, ``date``, ``close``, ``eps`` (on the fetch-date basis),
        ``eps_period_end``, ``pe``; one row per price row that has an EPS before it.

    Notes
    -----
    An EPS filed on day d is used from day d + 1 (``allow_exact_matches=False``), as a
    filing may arrive after the close. ``pe`` is missing where the EPS is zero or
    negative, where a P/E has no meaning.
    """
    adjusted: list[pd.DataFrame] = []
    for symbol, part in eps.groupby("symbol"):
        factor = split_factor(splits, str(symbol), part["filed"])
        adjusted.append(part.assign(eps=part["value"].to_numpy(dtype=float) / factor))
    if not adjusted:
        return pd.DataFrame(
            columns=["symbol", "date", "close", "eps", "eps_period_end", "pe"]
        )
    eps_adjusted = pd.concat(adjusted).rename(columns={"period_end": "eps_period_end"})
    merged = pd.merge_asof(
        prices[["symbol", "date", "close"]]
        .astype({"symbol": "string"})
        .sort_values("date"),
        eps_adjusted[["symbol", "filed", "eps", "eps_period_end"]]
        .astype({"symbol": "string"})
        .sort_values("filed"),
        left_on="date",
        right_on="filed",
        by="symbol",
        direction="backward",
        allow_exact_matches=False,
    ).dropna(subset=["eps"])
    merged["pe"] = (merged["close"] / merged["eps"]).where(merged["eps"] > 0)
    columns = ["symbol", "date", "close", "eps", "eps_period_end", "pe"]
    return merged[columns].sort_values(["symbol", "date"]).reset_index(drop=True)
