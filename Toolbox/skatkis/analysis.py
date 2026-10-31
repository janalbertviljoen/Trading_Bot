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
plot_relative(df, *, x, y, group, benchmark, second_benchmark, ...)
    Each group's growth divided by a benchmark's (solid) and optionally a second one's
    (dashed, same colour), each from its own first common date.
plot_spread(df, *, x, y, group, band, thresholds, ...)
    Median and percentile band of a value across rows over time, per group.
plot_lag_profile(df, *, lag, value, group, ...)
    A statistic per lag, one line per group, with the zero lag marked.
lagged_correlation(a, b, lags)
    Correlation of ``a`` with ``b`` shifted by each lag.

Computes
--------
growth of one unit, normalised series : plot_normalised
    Each group's value divided by its value on the first date all groups have a value.
relative growth, relative performance : plot_relative
    Growth of one unit of each group divided by growth of one unit of the benchmark, both
    from the first date the two have a value; above 1 the group did better.
median and percentile band, cross-sectional spread : plot_spread
    Per date (and group), the median and the chosen quantiles of the value across rows.
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

from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

WIDE = (10.0, 4.5)
MAX_LEGEND = 15


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
    thresholds: Mapping[str, float] | None = None,
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
        Mark each point on the line (for sparse data such as annual values).
    thresholds : Mapping[str, float], optional
        Named horizontal lines, such as a rule's cut-off (``{"G6: P/E 15": 15.0}``).
    """
    figure = Figure(figsize=WIDE)
    axes = figure.add_subplot()
    for name, unsorted in _groups(df, group):
        part = unsorted.sort_values(x)
        if markers:
            axes.plot(part[x], part[y], label=name, lw=1, marker="o", ms=3)
        else:
            axes.plot(part[x], part[y], label=name, lw=1.2)
    _thresholds(axes, thresholds)
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


def plot_relative(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str,
    benchmark: str,
    second_benchmark: str | None = None,
    labels: Mapping[str, str] | None = None,
    title: str = "",
    ylabel: str = "Relative to the benchmark",
) -> Figure:
    """Plot each group's growth divided by the benchmark's growth, with a line at 1.

    Parameters
    ----------
    df : pd.DataFrame
        Long table including the benchmarks' rows; not modified.
    x, y, group : str
        Date, value and group columns.
    benchmark : str
        The group every other group is divided by (solid lines).
    second_benchmark : str, optional
        A second group to divide by, drawn dashed in the same colour as the solid line.
    labels : Mapping[str, str], optional
        Display names per group value.
    title, ylabel : str
        Labels.

    Notes
    -----
    Each group starts at 1 on its own first date on which it and the benchmark have a
    value, so a series that begins later (a company listed after the start) still
    shows. Above 1, the group grew more than the benchmark since its start. The legend
    is left out above ``MAX_LEGEND`` groups.
    """
    wide = df.pivot_table(index=x, columns=group, values=y).sort_index()
    names = [n for n in wide.columns if n not in (benchmark, second_benchmark)]
    figure = Figure(figsize=WIDE)
    axes = figure.add_subplot()
    for name in names:
        solid = _relative_growth(wide[name], wide[benchmark])
        label = (labels or {}).get(str(name), str(name))
        (line,) = axes.plot(solid.index, solid, label=label, lw=1.2)
        if second_benchmark is not None:
            dashed = _relative_growth(wide[name], wide[second_benchmark])
            axes.plot(dashed.index, dashed, color=line.get_color(), lw=1, ls="--")
    axes.axhline(1, color="black", lw=1)
    if not title:
        title = f"Relative to {benchmark}" + (
            f" (solid) and {second_benchmark} (dashed)" if second_benchmark else ""
        )
    _finish(axes, title=title, ylabel=ylabel, legend=len(names) <= MAX_LEGEND, log=True)
    figure.tight_layout()
    return figure


def plot_spread(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str | None = None,
    band: tuple[float, float] = (0.25, 0.75),
    thresholds: Mapping[str, float] | None = None,
    title: str = "",
    ylabel: str = "",
) -> Figure:
    """Plot the median of ``y`` across rows per ``x``, with a percentile band, per group.

    Parameters
    ----------
    df : pd.DataFrame
        Long table, many rows per ``x`` (e.g. one per company and date); not modified.
    x, y : str
        Horizontal axis and the value summarised.
    group : str, optional
        One median line and band per group value.
    band : tuple[float, float]
        Lower and upper quantile of the shaded band; ``(0.25, 0.75)`` by default.
    thresholds : Mapping[str, float], optional
        Named horizontal lines, such as a rule's cut-off.
    title, ylabel : str
        Labels.
    """
    figure = Figure(figsize=WIDE)
    axes = figure.add_subplot()
    for name, part in _groups(df, group):
        values = part.groupby(x)[y]
        low, high = values.quantile(band[0]), values.quantile(band[1])
        median = values.median()
        (line,) = axes.plot(median.index, median, lw=1.4, label=name)
        axes.fill_between(median.index, low, high, color=line.get_color(), alpha=0.15)
    _thresholds(axes, thresholds)
    _finish(axes, title=title, ylabel=ylabel, legend=group is not None)
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


def _relative_growth(series: pd.Series, benchmark: pd.Series) -> pd.Series:
    """Return the growth of ``series`` over the benchmark's, both 1 on their first common date."""
    both = pd.concat([series, benchmark], axis=1).dropna()
    if both.empty:
        return pd.Series(dtype="float64")
    growth = both / both.iloc[0]
    return growth.iloc[:, 0] / growth.iloc[:, 1]


def _thresholds(axes: Axes, thresholds: Mapping[str, float] | None) -> None:
    """Draw each named threshold as a dashed horizontal line, labelled at the right."""
    for label, value in (thresholds or {}).items():
        axes.axhline(value, color="red", lw=1, ls="--")
        axes.annotate(
            label,
            xy=(1.0, value),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize=8,
            color="red",
        )


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
