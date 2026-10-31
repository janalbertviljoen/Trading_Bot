"""
skatkis/analysis.py
-------------------
Generic analysis: plot layouts and statistics for any table, with no market knowledge.

Every plot function takes a long table (one row per observation) and the names of its
columns, and returns a matplotlib ``Figure`` without showing or saving it. The functions
know nothing about any metric or source: the named market plots, with their titles and
formatting per source, are in ``market_data.market_analysis``, which calls these.

Owns
----
The plot layouts for series over time, coverage, normalised growth, scatter checks, two
series on two axes and lag profiles; the lagged correlation.

Must not
--------
Know a metric, source or column name (``market_data.market_analysis``); compute metrics
or returns (``market_data``, ``skatkis.financial_logic``); read or write files; use
pyplot's global state.

Public API
----------
plot_series(df, *, x, y, group, ...)
    A value over time, one line (or marker series) per group.
plot_coverage(df, *, row, column, ...)
    A heat map of the number of observations per row and column, showing gaps.
plot_normalised(df, *, x, y, group, highlight, ...)
    Growth of one unit per group from their first common date.
plot_scatter(df, *, x, y, group, diagonal, ...)
    One value against another, optionally with the line y = x.
plot_dual_axis(df, *, x, left, right, ...)
    Two series over the same horizontal axis, each on its own vertical axis.
plot_lag_profile(df, *, lag, value, group, ...)
    A statistic per lag, one line per group, with the zero lag marked.
lagged_correlation(a, b, lags)
    Correlation of ``a`` with ``b`` shifted by each lag.

Computes
--------
growth of one unit, normalised series : plot_normalised
    Each group's value divided by its value on the first date all groups have a value.
lagged correlation, cross-correlation, lead-lag : lagged_correlation
    For each lag k, the Pearson correlation of a(t) with b(t + k) over the positions where
    both exist; k > 0 means ``a`` leads ``b`` by k steps.

Dependency rule
---------------
matplotlib, numpy, pandas and the standard library.
"""

# matplotlib annotates its plotting methods with untyped **kwargs, which strict pyright
# reports on every call; the arguments passed here are all typed.
# pyright: reportUnknownMemberType=false

from collections.abc import Iterator

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

WIDE = (10.0, 4.5)


def plot_series(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str | None = None,
    title: str = "",
    ylabel: str = "",
    log: bool = False,
    markers: bool = False,
) -> Figure:
    """Plot ``y`` against ``x``, one line per value of ``group``.

    Parameters
    ----------
    df : pd.DataFrame
        Long table; not modified.
    x, y : str
        Columns for the horizontal axis (usually a date) and the value.
    group : str, optional
        Column whose values get one line each.
    title, ylabel : str
        Labels.
    log : bool
        Logarithmic vertical axis.
    markers : bool
        Draw points instead of lines (for sparse data such as annual values).
    """
    figure = Figure(figsize=WIDE)
    axes = figure.add_subplot()
    for name, unsorted in _groups(df, group):
        part = unsorted.sort_values(x)
        if markers:
            axes.plot(part[x], part[y], label=name, ls="none", marker="o", ms=3)
        else:
            axes.plot(part[x], part[y], label=name, lw=1.2)
    _finish(axes, title=title, ylabel=ylabel, legend=group is not None, log=log)
    figure.tight_layout()
    return figure


def plot_coverage(
    df: pd.DataFrame,
    *,
    row: str,
    column: str,
    title: str = "",
    annotate: bool = True,
) -> Figure:
    """Plot the number of rows of ``df`` per ``row`` and ``column`` value as a heat map.

    Empty cells (no observation) show as gaps. Rows and columns are sorted.

    Parameters
    ----------
    df : pd.DataFrame
        Long table; not modified.
    row, column : str
        Columns giving the heat map's rows (e.g. a field) and columns (e.g. a year).
    title : str
        Title.
    annotate : bool
        Write the count in each cell.
    """
    counts = df.pivot_table(
        index=row, columns=column, aggfunc="size", fill_value=0
    ).sort_index()
    counts = counts.reindex(sorted(counts.columns), axis=1)
    values = counts.to_numpy(dtype=float)
    height = max(2.5, 0.3 * len(counts.index) + 1.5)
    figure = Figure(figsize=(max(6.0, 0.45 * len(counts.columns) + 3), height))
    axes = figure.add_subplot()
    image = axes.imshow(
        np.where(values > 0, values, np.nan), aspect="auto", cmap="viridis"
    )
    axes.set_xticks(
        range(len(counts.columns)), [str(c) for c in counts.columns], rotation=90
    )
    axes.set_yticks(range(len(counts.index)), [str(i) for i in counts.index])
    if annotate:
        for i, j in zip(*np.nonzero(values), strict=True):
            axes.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", fontsize=6)
    figure.colorbar(image, ax=axes, label="observations")
    axes.set_title(title)
    figure.tight_layout()
    return figure


def plot_normalised(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    highlight: str | None = None,
    title: str = "",
    ylabel: str = "Growth of one unit",
    log: bool = True,
) -> Figure:
    """Plot each group's ``y`` divided by its value on the first common ``x``.

    Parameters
    ----------
    df : pd.DataFrame
        Long table; not modified.
    x, y, group : str
        Date, value and group columns.
    highlight : str, optional
        A group drawn thick and black, such as a benchmark.
    title, ylabel : str
        Labels.
    log : bool
        Logarithmic vertical axis.

    Notes
    -----
    Only dates on which every group has a value are kept, so all lines start at 1 on
    the same date.
    """
    wide = df.pivot_table(index=x, columns=group, values=y).dropna()
    growth = wide / wide.iloc[0]
    figure = Figure(figsize=WIDE)
    axes = figure.add_subplot()
    for name in growth.columns:
        thick = name == highlight
        axes.plot(
            growth.index,
            growth[name],
            label=str(name),
            lw=2.5 if thick else 1.2,
            color="black" if thick else None,
        )
    start = pd.Timestamp(growth.index[0]).date() if len(growth) else "-"
    _finish(
        axes,
        title=title or f"Growth since {start}",
        ylabel=ylabel,
        legend=True,
        log=log,
    )
    figure.tight_layout()
    return figure


def plot_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str | None = None,
    diagonal: bool = False,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> Figure:
    """Plot ``y`` against ``x`` as points, one colour per group.

    Parameters
    ----------
    df : pd.DataFrame
        Long table; not modified.
    x, y : str
        Columns to compare.
    group : str, optional
        Column whose values get one colour each.
    diagonal : bool
        Draw the line y = x, for checking two measures of the same quantity.
    title, xlabel, ylabel : str
        Labels.
    """
    figure = Figure(figsize=(6.0, 6.0))
    axes = figure.add_subplot()
    for name, part in _groups(df, group):
        axes.scatter(part[x], part[y], s=12, label=name, alpha=0.8)
    if diagonal and len(df):
        low = float(np.nanmin(df[[x, y]].to_numpy()))
        high = float(np.nanmax(df[[x, y]].to_numpy()))
        axes.plot([low, high], [low, high], color="grey", lw=1, ls="--")
    axes.set_xlabel(xlabel or x)
    _finish(axes, title=title, ylabel=ylabel or y, legend=group is not None)
    figure.tight_layout()
    return figure


def plot_dual_axis(
    df: pd.DataFrame,
    *,
    x: str,
    left: str,
    right: str,
    title: str = "",
    left_label: str = "",
    right_label: str = "",
) -> Figure:
    """Plot ``left`` and ``right`` against ``x``, each on its own vertical axis.

    Parameters
    ----------
    df : pd.DataFrame
        Table with one row per ``x``; not modified.
    x : str
        Horizontal axis, usually a date.
    left, right : str
        Columns drawn on the left and right axes, in two colours.
    title, left_label, right_label : str
        Labels; an empty axis label falls back to the column name.
    """
    data = df.sort_values(x)
    figure = Figure(figsize=WIDE)
    axes = figure.add_subplot()
    axes.plot(data[x], data[left], color="tab:blue", lw=1.2)
    axes.set_ylabel(left_label or left, color="tab:blue")
    twin = axes.twinx()
    twin.plot(data[x], data[right], color="tab:orange", lw=1.2)
    twin.set_ylabel(right_label or right, color="tab:orange")
    axes.set_title(title)
    axes.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_lag_profile(
    df: pd.DataFrame,
    *,
    lag: str,
    value: str,
    group: str | None = None,
    title: str = "",
    xlabel: str = "lag",
    ylabel: str = "correlation",
) -> Figure:
    """Plot ``value`` against ``lag``, one line with markers per group.

    A vertical line marks lag 0 and a horizontal line the value 0.
    """
    figure = Figure(figsize=WIDE)
    axes = figure.add_subplot()
    for name, unsorted in _groups(df, group):
        part = unsorted.sort_values(lag)
        axes.plot(part[lag], part[value], marker="o", ms=3, lw=1.2, label=name)
    axes.axvline(0, color="grey", lw=1, ls="--")
    axes.axhline(0, color="grey", lw=1)
    axes.set_xlabel(xlabel)
    _finish(axes, title=title, ylabel=ylabel, legend=group is not None)
    figure.tight_layout()
    return figure


def lagged_correlation(a: pd.Series, b: pd.Series, lags: range) -> pd.Series:
    """Return the correlation of ``a`` with ``b`` shifted by each lag.

    Parameters
    ----------
    a, b : pd.Series
        Equally spaced series on the same index (e.g. month ends).
    lags : range
        Lags in steps of the index; positive lags compare ``a`` now with ``b`` later.

    Returns
    -------
    pd.Series
        Pearson correlation per lag, indexed by lag; missing where fewer than three
        pairs exist.

    Notes
    -----
    corr(a(t), b(t + k)) is computed with ``b.shift(-k)``, over positions where both
    values exist.
    """
    values = {k: a.corr(b.shift(-k), min_periods=3) for k in lags}
    return pd.Series(values, name="correlation", dtype="float64").rename_axis("lag")


def _groups(
    df: pd.DataFrame, group: str | None
) -> Iterator[tuple[str | None, pd.DataFrame]]:
    """Yield ``(label, part)`` per group value in sorted order, or the whole table."""
    if group is None:
        yield None, df
        return
    for name in sorted(df[group].dropna().unique().tolist()):
        yield str(name), df[df[group] == name]


def _finish(
    axes: Axes, *, title: str, ylabel: str, legend: bool, log: bool = False
) -> None:
    """Apply the title, axis label, log scale, legend and grid shared by all plots."""
    axes.set_title(title)
    axes.set_ylabel(ylabel)
    if log:
        axes.set_yscale("log")
    if legend:
        axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
