"""
market_data/market_analysis.py
------------------------------
The named market plots: what each source and metric is shown as, with its titles and units.

Each function takes a table from ``market_data`` and returns one matplotlib ``Figure``
built with the generic layouts of ``skatkis.analysis``. The data checks
(``market_data.checks``) call them, and so can the modelling tools in their own
analysis, so a plot of the same thing looks the same everywhere.

Owns
----
Plot names, titles, axis labels and formatting per source and metric; the lead-lag table
of P/E and price changes.

Must not
--------
Build plot layouts (``skatkis.analysis``); compute metrics (``market_data.metrics``) or
clean data (``market_data.clean``); save files (callers, through ``skatkis.io``).

Public API
----------
edgar_coverage(annual, entity_name)
    First-reported annual values per field and fiscal year for one company.
edgar_annual_series(annual, field)
    One field's first-reported annual values per company.
edgar_eps_check(wide)
    Reported diluted EPS against net income / diluted shares.
edgar_filing_lag(lag)
    Days from period end to filing, per form.
yahoo_total_return(prices)
    Growth of 1 USD with dividends reinvested, against ``^SP500TR``.
yahoo_trading_days(prices)
    Trading days per symbol and year.
yahoo_dividends(dividends)
    Dividends per symbol and year.
fred_series(rates, series_id) / fred_coverage(rates, series_id)
    A FRED series over time, and its observations per year.
pe_over_time(pe) / pe_eps(pe)
    P/E per symbol, and the EPS behind it.
pe_vs_price(pe, symbol)
    One symbol's P/E and close on two axes.
pe_price_lead_lag(pe, max_lag)
    Correlation of monthly P/E changes with price changes shifted by each lag.
pe_lead_lag(correlations)
    The lead-lag profile per symbol.

Computes
--------
monthly log change, lead-lag correlation of P/E and price : pe_price_lead_lag
    Per symbol, month-end P/E and close; d log P/E(t) and d log close(t) month on month;
    for each lag k, corr(d log P/E(t), d log close(t + k)). Positive k: P/E changes
    before the price.

Dependency rule
---------------
matplotlib, numpy, pandas and ``skatkis.analysis``.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from Toolbox.skatkis.analysis import (
    lagged_correlation,
    plot_coverage,
    plot_dual_axis,
    plot_lag_profile,
    plot_normalised,
    plot_scatter,
    plot_series,
)

BENCHMARK = "^SP500TR"


def edgar_coverage(annual: pd.DataFrame, entity_name: str) -> Figure:
    """Plot one company's first-reported annual values per field and fiscal year end.

    ``annual`` has columns ``entity_name``, ``field``, ``period_end``.
    """
    part = annual[annual["entity_name"] == entity_name]
    part = part.assign(year=part["period_end"].dt.year)
    title = f"{entity_name}: first-reported annual values per field and fiscal year end"
    return plot_coverage(part, row="field", column="year", title=title)


def edgar_annual_series(annual: pd.DataFrame, field: str) -> Figure:
    """Plot one field's first-reported annual values per company.

    ``annual`` has columns ``entity_name``, ``field``, ``period_end``, ``value``.
    """
    part = annual[annual["field"] == field]
    title = f"{field}: first-reported annual values (per-share values on their filing basis)"
    return plot_series(
        part,
        x="period_end",
        y="value",
        group="entity_name",
        markers=True,
        title=title,
        ylabel=field,
    )


def edgar_eps_check(wide: pd.DataFrame) -> Figure:
    """Plot reported diluted EPS against net income / diluted shares, per company.

    ``wide`` has columns ``entity_name``, ``ni_per_share``, ``eps_diluted``.
    """
    return plot_scatter(
        wide,
        x="ni_per_share",
        y="eps_diluted",
        group="entity_name",
        diagonal=True,
        title="Reported diluted EPS against net income / diluted shares",
        xlabel="net income / weighted diluted shares (USD)",
        ylabel="reported diluted EPS (USD)",
    )


def edgar_filing_lag(lag: pd.DataFrame) -> Figure:
    """Plot the days from period end to filing, per form.

    ``lag`` has columns ``period_end``, ``lag_days``, ``form``.
    """
    title = "Days from period end to filing (the point-in-time delay)"
    return plot_series(
        lag,
        x="period_end",
        y="lag_days",
        group="form",
        markers=True,
        title=title,
        ylabel="days",
    )


def yahoo_total_return(prices: pd.DataFrame) -> Figure:
    """Plot growth of 1 USD with dividends reinvested per symbol, ``^SP500TR`` in black."""
    return plot_normalised(
        prices,
        x="date",
        y="adj_close",
        group="symbol",
        highlight=BENCHMARK,
        ylabel="Growth of 1 USD, dividends reinvested (log)",
    )


def yahoo_trading_days(prices: pd.DataFrame) -> Figure:
    """Plot the trading days per symbol and year (about 252 expected)."""
    traded = prices.dropna(subset=["close"])
    traded = traded.assign(year=traded["date"].dt.year)
    title = "Trading days per symbol and year (about 252 expected)"
    return plot_coverage(traded, row="symbol", column="year", title=title)


def yahoo_dividends(dividends: pd.DataFrame) -> Figure:
    """Plot the number of dividends per symbol and year."""
    per_year = dividends.assign(year=dividends["ex_date"].dt.year)
    return plot_coverage(
        per_year,
        row="symbol",
        column="year",
        title="Dividends per symbol and year",
        annotate=False,
    )


def fred_series(rates: pd.DataFrame, series_id: str) -> Figure:
    """Plot a FRED series over time (``DEXSFUS``: ZAR per USD)."""
    part = rates[rates["series_id"] == series_id]
    ylabel = "ZAR per USD" if series_id == "DEXSFUS" else series_id
    return plot_series(
        part, x="date", y="value", title=f"{series_id} (FRED)", ylabel=ylabel
    )


def fred_coverage(rates: pd.DataFrame, series_id: str) -> Figure:
    """Plot a FRED series' observations per year (about 250 expected)."""
    part = rates[rates["series_id"] == series_id]
    part = part.assign(year=part["date"].dt.year)
    title = f"{series_id}: observations per year (about 250 expected)"
    return plot_coverage(part, row="series_id", column="year", title=title)


def pe_over_time(pe: pd.DataFrame) -> Figure:
    """Plot the P/E per symbol over time."""
    title = "P/E on the latest annual diluted EPS (filed before the day)"
    return plot_series(pe, x="date", y="pe", group="symbol", title=title, ylabel="P/E")


def pe_eps(pe: pd.DataFrame) -> Figure:
    """Plot the annual diluted EPS behind the P/E, per symbol."""
    title = "Annual diluted EPS behind the P/E (split-adjusted to the fetch date)"
    return plot_series(
        pe, x="date", y="eps", group="symbol", title=title, ylabel="EPS (USD)"
    )


def pe_vs_price(pe: pd.DataFrame, symbol: str) -> Figure:
    """Plot one symbol's P/E (left axis) and close (right axis)."""
    part = pe[pe["symbol"] == symbol]
    return plot_dual_axis(
        part,
        x="date",
        left="pe",
        right="close",
        title=f"{symbol}: P/E and price",
        left_label="P/E",
        right_label="close (USD, split-adjusted)",
    )


def pe_price_lead_lag(pe: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    """Return, per symbol, the correlation of monthly P/E changes with shifted price changes.

    Parameters
    ----------
    pe : pd.DataFrame
        Columns ``symbol``, ``date``, ``pe``, ``close`` (from ``price_earnings``).
    max_lag : int
        Lags from ``-max_lag`` to ``max_lag`` months.

    Returns
    -------
    pd.DataFrame
        Columns ``symbol``, ``lag`` (months), ``correlation``.

    Notes
    -----
    P/E is price divided by EPS, so the two move together at lag 0 by construction; the
    other lags show whether P/E changes come before (positive lag) or after (negative
    lag) price changes. Months where the P/E is missing (EPS zero or negative) are left
    out.
    """
    frames: list[pd.DataFrame] = []
    for symbol, part in pe.dropna(subset=["pe"]).groupby("symbol"):
        monthly = part.set_index("date")[["pe", "close"]].resample("ME").last()
        changes = pd.DataFrame(np.log(monthly)).diff()
        correlations = lagged_correlation(
            changes["pe"], changes["close"], range(-max_lag, max_lag + 1)
        )
        frames.append(correlations.reset_index().assign(symbol=str(symbol)))
    if not frames:
        return pd.DataFrame(columns=["symbol", "lag", "correlation"])
    return pd.concat(frames, ignore_index=True)[["symbol", "lag", "correlation"]]


def pe_lead_lag(correlations: pd.DataFrame) -> Figure:
    """Plot the lead-lag profile of P/E and price changes per symbol."""
    return plot_lag_profile(
        correlations,
        lag="lag",
        value="correlation",
        group="symbol",
        title="Monthly P/E changes against price changes k months later",
        xlabel="k (months); positive: P/E changes first",
        ylabel="correlation",
    )
