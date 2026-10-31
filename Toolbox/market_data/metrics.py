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
    Product of the split ratios after each date, for one symbol.
split_factors(splits, frame, *, symbol, date)
    The same factor for each row of a table holding many symbols.
price_earnings(prices, eps, splits)
    Daily P/E per symbol on the latest annual EPS filed before the day.
dividend_yield(prices, dividends, *, window_days)
    Daily trailing dividend yield per symbol.
relative_performance(prices, gics, *, min_years)
    Each company's annualised return relative to its sector fund.
sector_performers(performance, n)
    The ``n`` best and ``n`` worst companies of each GICS sector.

Computes
--------
split factor : split_factor
    F(d) = product of numerator / denominator over the symbol's splits dated after d;
    F = 1 when there are none.
split factor per row : split_factors
    F(symbol, date) per row, as ``split_factor``; 1 for a symbol without splits.
price-to-earnings ratio, P/E : price_earnings
    P/E(t) = close(t) / (EPS / F(filed)), where EPS is the latest first-reported annual
    diluted EPS filed before day t; missing when that EPS is zero or negative.
trailing dividend yield : dividend_yield
    y(t) = (sum of dividends with ex-date in (t - window, t]) / close(t); dividends and
    close are both split-adjusted to the fetch date; 0 when no dividend fell in the window.

Dependency rule
---------------
numpy, pandas and ``skatkis.financial_logic``.
"""

import numpy as np
import pandas as pd

from Toolbox.skatkis.financial_logic import annualised_relative_return


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


def split_factors(
    splits: pd.DataFrame, frame: pd.DataFrame, *, symbol: str, date: str
) -> pd.Series:
    """Return the split factor of each row of ``frame``, from its symbol and date.

    Parameters
    ----------
    splits : pd.DataFrame
        Columns ``symbol``, ``date``, ``numerator``, ``denominator``.
    frame : pd.DataFrame
        Any table with a symbol column and a datetime column; not modified.
    symbol, date : str
        Names of those two columns.

    Returns
    -------
    pd.Series
        Float factor on ``frame``'s index: dividing a per-share value reported on that
        date by it, or multiplying a share count by it, puts the value on the basis of
        the prices (after the last split). 1.0 where no split follows.
    """
    factors = pd.Series(1.0, index=frame.index)
    for name, part in frame.groupby(symbol):
        factors.loc[part.index] = split_factor(splits, str(name), part[date])
    return factors


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


def dividend_yield(
    prices: pd.DataFrame, dividends: pd.DataFrame, *, window_days: int = 365
) -> pd.DataFrame:
    """Return the daily trailing dividend yield per symbol.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns ``symbol``, ``date``, ``close`` (split-adjusted to the fetch date).
    dividends : pd.DataFrame
        Columns ``symbol``, ``ex_date``, ``amount`` (per share, split-adjusted to the
        fetch date).
    window_days : int
        Length of the trailing window; 365 days by default.

    Returns
    -------
    pd.DataFrame
        Columns ``symbol``, ``date``, ``close``, ``dividends`` (paid in the window, USD
        per share), ``dividend_yield`` (fraction); one row per price row with a close.

    Notes
    -----
    A dividend counts from its ex-date, the first day the share trades without it. Both
    series are on the same split basis, so their ratio needs no further adjustment.
    """
    frames: list[pd.DataFrame] = []
    window = np.timedelta64(window_days, "D")
    for symbol, part in prices.dropna(subset=["close"]).groupby("symbol"):
        own = dividends[dividends["symbol"] == symbol].sort_values("ex_date")
        ex_dates = np.asarray(own["ex_date"], dtype="datetime64[ns]")
        paid = np.append(0.0, np.cumsum(np.asarray(own["amount"], dtype=float)))
        days = np.asarray(part["date"], dtype="datetime64[ns]")
        upto = np.searchsorted(ex_dates, days, side="right")
        since = np.searchsorted(ex_dates, days - window, side="right")
        in_window = paid[upto] - paid[since]
        frames.append(
            pd.DataFrame(
                {
                    "symbol": str(symbol),
                    "date": part["date"].to_numpy(),
                    "close": part["close"].to_numpy(dtype=float),
                    "dividends": in_window,
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=["symbol", "date", "close", "dividends", "dividend_yield"]
        )
    result = pd.concat(frames, ignore_index=True)
    result["dividend_yield"] = result["dividends"] / result["close"]
    return result.astype({"symbol": "string"}).sort_values(
        ["symbol", "date"], ignore_index=True
    )


def relative_performance(
    prices: pd.DataFrame, gics: pd.DataFrame, *, min_years: float = 1.0
) -> pd.DataFrame:
    """Return each company's annualised total return relative to its sector fund.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns ``symbol``, ``date``, ``adj_close`` (total-return levels), including the
        sector funds.
    gics : pd.DataFrame
        The GICS map (``symbol``, ``gics_sector``, ``sector_fund``).
    min_years : float
        Companies with less common history with their fund get no value.

    Returns
    -------
    pd.DataFrame
        Columns ``symbol``, ``gics_sector``, ``sector_fund``, ``relative_return``
        (fraction per year, from ``annualised_relative_return`` over the company's and
        fund's common dates); companies without a value are left out.
    """
    wide = prices.pivot_table(index="date", columns="symbol", values="adj_close")
    rows: list[dict[str, object]] = []
    triples = zip(gics["symbol"], gics["gics_sector"], gics["sector_fund"], strict=True)
    for symbol, sector, fund in ((str(a), str(b), str(c)) for a, b, c in triples):
        if symbol not in wide.columns or fund not in wide.columns:
            continue
        value = annualised_relative_return(
            wide[symbol], wide[fund], min_years=min_years
        )
        if not np.isnan(value):
            rows.append(
                {
                    "symbol": symbol,
                    "gics_sector": sector,
                    "sector_fund": fund,
                    "relative_return": value,
                }
            )
    return pd.DataFrame(
        rows, columns=["symbol", "gics_sector", "sector_fund", "relative_return"]
    )


def sector_performers(performance: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the ``n`` best and ``n`` worst companies of each GICS sector.

    Parameters
    ----------
    performance : pd.DataFrame
        From ``relative_performance``.
    n : int
        Companies per end of the ranking.

    Returns
    -------
    pd.DataFrame
        The rows of ``performance`` with ``group`` (``"top"`` or ``"bottom"``) and
        ``rank`` (1 = best for the top, 1 = worst for the bottom). In a sector with fewer
        than ``2 * n`` companies a company can be in both groups.
    """
    ranked = performance.sort_values(
        ["gics_sector", "relative_return"], ascending=[True, False]
    )
    top = ranked.groupby("gics_sector").head(n).assign(group="top")
    bottom = (
        ranked.sort_values(["gics_sector", "relative_return"])
        .groupby("gics_sector")
        .head(n)
        .assign(group="bottom")
    )
    both = pd.concat([top, bottom], ignore_index=True)
    both["rank"] = both.groupby(["gics_sector", "group"]).cumcount() + 1
    return both
