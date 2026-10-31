"""
market_data/market_analysis.py
------------------------------
The named market plots: what each source and metric is shown as, with its titles and units.

Each function takes tables from ``market_data`` and returns one matplotlib ``Figure``
built with the generic layouts of ``skatkis.analysis``. The checks and report
(``market_data.checks``) call them, and so can the modelling tools in their own analysis,
so a plot of the same thing looks the same everywhere. Plots go from coarse to fine:
sector summaries against the S&P 500, then the best and the worst sector of an analysis
(``sectors_*``), then their best and worst companies (``companies_ranked``), where colour
marks the company and line style the group or metric. No plot has more than 30 lines.

Owns
----
Plot names, titles, axis labels and formatting per source and metric; the lead-lag table
of P/E and price changes.

Must not
--------
Build plot layouts (``skatkis.analysis``); compute metrics or rankings
(``market_data.metrics``); clean data (``market_data.clean``); save files.

Public API
----------
sector_benchmarks(prices, sectors, market) / total_return_summary(prices, sectors, market)
    Sector funds relative to the S&P 500; growth of 1 USD for funds and the S&P 500.
fundamentals_sector_totals(annual, field)
    Sum of a field over each sector's companies per fiscal year (USD millions).
dividend_yield_summary(yields) / pe_sector_medians(pe)
    Median dividend yield and median P/E per sector.
sectors_total_return, sectors_fundamentals, sectors_spread, sector_pe_and_price
    The best and the worst sector of an analysis side by side; a sector's median P/E and
    price growth with their bands.
ranked_series(prices, symbols, *, relative_to)
    Growth of 1 USD per company, optionally over a benchmark, for ``companies_ranked``.
companies_ranked(table, *, value, best, worst, ...)
    The best companies (solid) against the worst (dashed), one colour per company.
pe_and_value(pe, symbols, *, title)
    Per company its P/E (solid, left axis) and price growth (dashed, right axis).
edgar_coverage, edgar_eps_check, edgar_filing_lag, yahoo_trading_days, fred_series,
fred_coverage
    Data checks.
edgar_annual_series, yahoo_total_return, yahoo_relative_return, pe_over_time,
pe_vs_price, pe_lead_lag
    Single companies, for a small universe.
pe_price_lead_lag(pe, max_lag)
    Correlation of monthly P/E changes with price changes shifted by each lag.

Computes
--------
growth of one dollar, relative growth : ranked_series
    value(t) = S(t) / S(t0), or (S(t) / S(t0)) / (B(t) / B(t0)) over a benchmark B, from
    the symbol's first date t0 (with the benchmark).
monthly log change, lead-lag correlation of P/E and price : pe_price_lead_lag
    Per symbol, month-end P/E and close; d log P/E(t) and d log close(t) month on month;
    for each lag k, corr(d log P/E(t), d log close(t + k)). Positive k: P/E changes
    before the price.

Dependency rule
---------------
matplotlib, numpy, pandas and ``skatkis.analysis``.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from Toolbox.skatkis.analysis import (
    lagged_correlation,
    plot_coverage,
    plot_dual_axis,
    plot_lag_profile,
    plot_normalised,
    plot_pair_series,
    plot_relative,
    plot_scatter,
    plot_series,
    plot_spread,
    plot_spread_pair,
)

BENCHMARK = "^SP500TR"
MILLION = 1e6
TRILLION = 1e12


# ---------------------------------------------------------------- sector summaries


def sector_benchmarks(
    prices: pd.DataFrame, sectors: Mapping[str, str], market: str, *, title: str = ""
) -> Figure:
    """Plot each sector fund's total return relative to the S&P 500's.

    ``sectors`` maps a GICS sector name to its fund symbol.
    """
    part = prices[prices["symbol"].isin([*sectors.values(), market])]
    return plot_relative(
        part,
        x="date",
        y="adj_close",
        group="symbol",
        benchmark=market,
        labels={fund: f"{name} ({fund})" for name, fund in sectors.items()},
        title=title or "GICS sector funds relative to the S&P 500",
        ylabel="fund growth / S&P 500 growth (log)",
    )


def total_return_summary(
    prices: pd.DataFrame, sectors: Mapping[str, str], market: str, *, title: str = ""
) -> Figure:
    """Plot growth of 1 USD, dividends reinvested, for each sector fund and the S&P 500."""
    part = prices[prices["symbol"].isin([*sectors.values(), market])]
    return plot_normalised(
        part,
        x="date",
        y="adj_close",
        group="symbol",
        highlight=market,
        labels={fund: f"{name} ({fund})" for name, fund in sectors.items()},
        title=title or "GICS sector funds and the S&P 500 (black): growth of 1 USD",
        ylabel="growth of 1 USD (log)",
    )


def fundamentals_sector_totals(annual: pd.DataFrame, field: str) -> Figure:
    """Plot the sum of ``field`` over each GICS sector's companies per fiscal year (USD trillions).

    ``annual`` has columns ``gics_sector``, ``field``, ``period_end``, ``value`` (USD).
    The number of companies reporting can change between years, which moves the totals.
    """
    part = annual[annual["field"] == field]
    totals = pd.DataFrame(
        part.assign(year=part["period_end"].dt.year)
        .groupby(["gics_sector", "year"], as_index=False)["value"]
        .sum()
    )
    totals["value"] = totals["value"] / TRILLION
    totals["year"] = pd.to_datetime(totals["year"].astype(str), format="%Y")
    return plot_series(
        totals,
        x="year",
        y="value",
        group="gics_sector",
        markers=True,
        title=f"{field.replace('_', ' ').capitalize()} per GICS sector",
        ylabel="USD trillions",
    )


def dividend_yield_summary(yields: pd.DataFrame, *, title: str = "") -> Figure:
    """Plot the median trailing dividend yield of each GICS sector (percent)."""
    return plot_spread(
        yields.assign(dividend_yield=yields["dividend_yield"] * 100),
        x="date",
        y="dividend_yield",
        group="gics_sector",
        band=(0.5, 0.5),
        title=title or "Median trailing 12-month dividend yield per GICS sector",
        ylabel="dividend yield (%)",
    )


def pe_sector_medians(pe: pd.DataFrame, *, title: str = "") -> Figure:
    """Plot the median P/E of each GICS sector over time (``pe`` has ``gics_sector``)."""
    return plot_spread(
        pe,
        x="date",
        y="pe",
        group="gics_sector",
        band=(0.5, 0.5),
        title=title or "Median P/E per GICS sector",
        ylabel="median P/E",
    )


# ---------------------------------------------------------------- best and worst sectors


def sectors_total_return(
    prices: pd.DataFrame, funds: Mapping[str, str], market: str, *, title: str
) -> Figure:
    """Plot growth of 1 USD for the given sector funds and the S&P 500 (black)."""
    part = prices[prices["symbol"].isin([*funds.values(), market])]
    return plot_normalised(
        part,
        x="date",
        y="adj_close",
        group="symbol",
        highlight=market,
        labels={fund: f"{name} ({fund})" for name, fund in funds.items()},
        title=title,
        ylabel="growth of 1 USD (log)",
    )


def sectors_fundamentals(
    annual: pd.DataFrame, sectors: Sequence[str], *, title: str
) -> Figure:
    """Plot total revenue (solid) and net income (dashed) of the given sectors, USD millions."""
    part = annual[
        annual["gics_sector"].isin(sectors)
        & annual["field"].isin(["revenue", "net_income"])
    ]
    totals = pd.DataFrame(
        part.assign(year=part["period_end"].dt.year)
        .groupby(["gics_sector", "field", "year"], as_index=False)["value"]
        .sum()
    )
    totals["value"] = totals["value"] / MILLION
    totals["year"] = pd.to_datetime(totals["year"].astype(str), format="%Y")
    return plot_series(
        totals,
        x="year",
        y="value",
        group="gics_sector",
        style="field",
        style_order=["revenue", "net_income"],
        markers=True,
        title=title,
        ylabel="USD millions",
    )


def sectors_spread(
    table: pd.DataFrame, value: str, sectors: Sequence[str], *, title: str, ylabel: str
) -> Figure:
    """Plot the median and 25th to 75th percentile band of ``value`` for the given sectors."""
    part = table[table["gics_sector"].isin(sectors)]
    scale = 100.0 if value == "dividend_yield" else 1.0
    return plot_spread(
        part.assign(**{value: part[value] * scale}),
        x="date",
        y=value,
        group="gics_sector",
        title=title,
        ylabel=ylabel,
    )


def sector_pe_and_price(pe: pd.DataFrame, sector: str, *, title: str) -> Figure:
    """Plot a sector's median P/E and median price growth, each with its band, on two axes.

    Price growth is each company's split-adjusted close divided by its first close in the
    data, so the band shows how the sector's prices moved while its P/E moved.
    """
    part = pe[pe["gics_sector"] == sector].dropna(subset=["pe"]).sort_values("date")
    first = part.groupby("symbol")["close"].transform("first")
    return plot_spread_pair(
        part.assign(growth=part["close"] / first),
        x="date",
        left="pe",
        right="growth",
        title=title,
        left_label="P/E (log)",
        right_label="price growth since first close (log)",
        log_left=True,
        log_right=True,
    )


# ---------------------------------------------------------------- best and worst companies


def ranked_series(
    prices: pd.DataFrame, symbols: Sequence[str], *, relative_to: str | None = None
) -> pd.DataFrame:
    """Return growth of 1 USD per symbol from its first date, optionally over a benchmark.

    Columns ``symbol``, ``date``, ``value``; with ``relative_to``, the growth is divided by
    the benchmark's growth since the same date.
    """
    wide = prices.pivot_table(
        index="date", columns="symbol", values="adj_close"
    ).sort_index()
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        if symbol not in wide.columns:
            continue
        series = wide[symbol]
        if relative_to is not None:
            both = pd.concat([series, wide[relative_to]], axis=1).dropna()
            own, bench = both.iloc[:, 0], both.iloc[:, 1]
            growth = (own / float(own.iloc[0])) / (bench / float(bench.iloc[0]))
        else:
            values = series.dropna()
            growth = values / float(values.iloc[0])
        dates = pd.DatetimeIndex(growth.index)
        frames.append(
            pd.DataFrame({"symbol": symbol, "date": dates, "value": growth.to_numpy()})
        )
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["symbol", "date", "value"])
    )


def companies_ranked(
    table: pd.DataFrame,
    *,
    value: str,
    best: Sequence[str],
    worst: Sequence[str],
    x: str = "date",
    title: str,
    ylabel: str,
    log: bool = False,
    thresholds: Mapping[str, float] | None = None,
    panel: bool = False,
) -> Figure:
    """Plot the best companies (solid) against the worst (dashed), one colour per company.

    ``table`` has a ``symbol`` column, ``x`` and ``value``; ``best`` and ``worst`` are
    symbols. Together they must stay within the 30-line limit.
    """
    rank = {**{s: "best" for s in best}, **{s: "worst" for s in worst}}
    part = table[table["symbol"].isin(list(rank))]
    part = part.assign(rank=part["symbol"].map(rank))
    if value == "dividend_yield":
        part = part.assign(dividend_yield=part["dividend_yield"] * 100)
    return plot_series(
        part,
        x=x,
        y=value,
        group="symbol",
        style="rank",
        style_order=["best", "worst"],
        markers=x != "date",
        log=log,
        thresholds=thresholds,
        title=title,
        ylabel=ylabel,
        panel=panel,
    )


def pe_and_value(
    pe: pd.DataFrame, symbols: Sequence[str], *, title: str, panel: bool = True
) -> Figure:
    """Plot each company's P/E (solid, left) and price growth (dashed, right), one colour each.

    ``pe`` has ``symbol``, ``date``, ``pe`` and ``close``; price growth is the close divided
    by the company's first close in the table, so it starts at 1.
    """
    part = (
        pe[pe["symbol"].isin(list(symbols))].dropna(subset=["pe"]).sort_values("date")
    )
    first = part.groupby("symbol")["close"].transform("first")
    return plot_pair_series(
        part.assign(growth=part["close"] / first),
        x="date",
        left="pe",
        right="growth",
        group="symbol",
        title=title,
        left_label="P/E (log)",
        right_label="price growth (log)",
        log_left=True,
        log_right=True,
        panel=panel,
    )


# ---------------------------------------------------------------- data checks


def edgar_coverage(annual: pd.DataFrame) -> Figure:
    """Plot the number of companies with a first-reported annual value, per field and year."""
    present = annual.assign(year=annual["period_end"].dt.year)
    present = present.drop_duplicates(["cik", "field", "year"])
    companies = annual["cik"].nunique()
    title = f"Companies (of {companies}) with a first-reported annual value, per field and year"
    return plot_coverage(present, row="field", column="year", title=title)


def edgar_eps_check(wide: pd.DataFrame) -> Figure:
    """Plot reported diluted EPS against net income / diluted shares."""
    return plot_scatter(
        wide,
        x="ni_per_share",
        y="eps_diluted",
        diagonal=True,
        title="Reported diluted EPS against net income / diluted shares",
        xlabel="net income / weighted diluted shares (USD)",
        ylabel="reported diluted EPS (USD)",
    )


def edgar_filing_lag(lag: pd.DataFrame) -> Figure:
    """Plot the days from period end to filing, per form."""
    return plot_scatter(
        lag,
        x="period_end",
        y="lag_days",
        group="form",
        title="Days from period end to filing (the point-in-time delay)",
        xlabel="period end",
        ylabel="days",
    )


def yahoo_trading_days(prices: pd.DataFrame) -> Figure:
    """Plot the trading days per symbol and year, the youngest symbols at the top."""
    traded = prices.dropna(subset=["close"])
    traded = traded.assign(year=traded["date"].dt.year)
    youngest_first = (
        traded.groupby("symbol")["date"]
        .min()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return plot_coverage(
        traded,
        row="symbol",
        column="year",
        row_order=youngest_first,
        title="Trading days per symbol and year, youngest first (about 252 expected)",
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


# ---------------------------------------------------------------- single companies


def edgar_annual_series(annual: pd.DataFrame, field: str) -> Figure:
    """Plot one field's first-reported annual values per company (USD millions or per share)."""
    part = annual[annual["field"] == field]
    per_share = field.startswith(("eps", "dividends_per_share"))
    if not per_share:
        part = part.assign(value=part["value"] / MILLION)
    unit = "USD per share" if per_share else "USD millions"
    return plot_series(
        part,
        x="period_end",
        y="value",
        group="entity_name",
        markers=True,
        title=f"{field}: first-reported annual values ({unit})",
        ylabel=unit,
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


def yahoo_relative_return(prices: pd.DataFrame) -> Figure:
    """Plot each company's total return divided by the S&P 500's (above 1: beat the index)."""
    return plot_relative(
        prices,
        x="date",
        y="adj_close",
        group="symbol",
        benchmark=BENCHMARK,
        title="Total return relative to the S&P 500 (above 1: beat an index fund)",
        ylabel="company growth / S&P 500 growth (log)",
    )


def pe_over_time(
    pe: pd.DataFrame, thresholds: Mapping[str, float] | None = None
) -> Figure:
    """Plot the P/E per symbol over time, with optional rule cut-offs as lines."""
    return plot_series(
        pe,
        x="date",
        y="pe",
        group="symbol",
        title="P/E on the latest annual diluted EPS (filed before the day)",
        ylabel="P/E",
        thresholds=thresholds,
    )


def pe_vs_price(pe: pd.DataFrame, symbol: str) -> Figure:
    """Plot one symbol's P/E (left axis) and close (right axis)."""
    return plot_dual_axis(
        pe[pe["symbol"] == symbol],
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


def pe_lead_lag(correlations: pd.DataFrame, group: str = "symbol") -> Figure:
    """Plot the lead-lag profile of P/E and price changes per ``group`` (symbol or sector)."""
    return plot_lag_profile(
        correlations,
        lag="lag",
        value="correlation",
        group=group,
        title="Monthly P/E changes against price changes k months later",
        xlabel="k (months); positive: P/E changes first",
        ylabel="correlation",
    )
