"""
skatkis/analysis.py
-------------------
Generic analysis: plot layouts and statistics for any table, with no market knowledge.

Every plot function takes a long table (one row per observation) and the names of its
columns, and returns a matplotlib ``Figure`` without showing or saving it. A line plot
holds at most ``MAX_LINES`` (30) lines and refuses more; larger sets belong in a heat map
or a selection. Colours mark groups (companies, sectors), line styles mark metrics. The functions
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
plot_spread_pair(df, *, x, left, right, band, ...)
    Median and band of two values across rows, each on its own axis.
plot_pair_series(df, *, x, left, right, group, ...)
    Two values per group on two axes (left solid, right dashed), one colour per group.
plot_lag_profile(df, *, lag, value, group, ...)
    A statistic per lag, one line per group, with the zero lag marked.
lagged_correlation(a, b, lags)
    Correlation of ``a`` with ``b`` shifted by each lag.

Computes
--------
growth of one unit, normalised series : plot_normalised
    Each group's value divided by its own first value.
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

from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter

WIDE = (10.0, 4.5)
PANEL = (5.0, 3.6)
MAX_LINES = 30
MAX_LEGEND = MAX_LINES
LINE_STYLES = ("-", "--", ":", "-.")
DENSE_ROWS = 30
THOUSANDS = 10_000


def plot_series(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    group: str | None = None,
    style: str | None = None,
    style_order: Sequence[object] | None = None,
    highlight: str | None = None,
    title: str = "",
    ylabel: str = "",
    log: bool = False,
    markers: bool = False,
    thresholds: Mapping[str, float] | None = None,
    panel: bool = False,
) -> Figure:
    """Plot ``y`` against ``x``: one colour per ``group``, one line style per ``style``.

    Parameters
    ----------
    df : pd.DataFrame
        Long table; not modified.
    x, y : str
        Columns for the horizontal axis (usually a date) and the value.
    group : str, optional
        Column whose values get one colour each (e.g. a company).
    style : str, optional
        Column whose values get one line style each in the group's colour (e.g. two
        metrics of one company: solid and dashed).
    style_order : Sequence, optional
        ``style`` values in the order solid, dashed, dotted, dash-dot; sorted otherwise.
    highlight : str, optional
        A ``group`` value drawn thick and black, such as a sector median.
    title, ylabel : str
        Labels.
    log : bool
        Logarithmic vertical axis.
    markers : bool
        Mark each point on the line (for sparse data such as annual values).
    thresholds : Mapping[str, float], optional
        Named horizontal lines, such as a rule's cut-off (``{"G6: P/E 15": 15.0}``).
    panel : bool
        A small figure (``PANEL``) with the legend inside, for grids of plots placed side
        by side in a report.

    Raises
    ------
    ValueError
        If the plot would have more than ``MAX_LINES`` lines.
    """
    groups = list(_groups(df, group))
    present: list[object] = list(df[style].dropna().unique()) if style else []
    if not style:
        styles: list[object] = [None]
    elif style_order is not None:
        styles = [v for v in style_order if v in present]
    else:
        styles = sorted(present, key=str)
    drawn = sum(
        1
        for _, part in groups
        for value in styles
        if value is None or bool((part[style] == value).any())
    )
    _check_lines(drawn)
    colours = _palette(len(groups))
    figure = Figure(figsize=PANEL if panel else WIDE)
    axes = figure.add_subplot()
    for (name, part), colour in zip(groups, colours, strict=True):
        thick = highlight is not None and name == highlight
        for index, value in enumerate(styles):
            line = part if value is None else part[part[style] == value]
            line = line.sort_values(x)
            label = name if value is None or index == 0 else None
            axes.plot(
                line[x],
                line[y],
                label=label,
                color="black" if thick else colour,
                lw=2.4 if thick else 1.2,
                ls=LINE_STYLES[index % len(LINE_STYLES)],
                marker="o" if markers else None,
                ms=3,
            )
    if style:
        _style_key(axes, [str(v) for v in styles])
    _thresholds(axes, thresholds)
    _finish(
        axes, title=title, ylabel=ylabel, legend=group is not None, log=log, panel=panel
    )
    figure.tight_layout()
    return figure


def plot_coverage(
    df: pd.DataFrame,
    *,
    row: str,
    column: str,
    title: str = "",
    annotate: bool = True,
    row_order: Sequence[object] | None = None,
) -> Figure:
    """Plot the number of rows of ``df`` per ``row`` and ``column`` value as a heat map.

    Empty cells (no observation) show as gaps. Columns are sorted; rows are sorted, or in
    ``row_order`` (top to bottom) when given. Above ``DENSE_ROWS`` rows the cells are not
    annotated and the row labels are small.

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
    row_order : Sequence, optional
        Row values from top to bottom.
    """
    counts = df.pivot_table(
        index=row, columns=column, aggfunc="size", fill_value=0
    ).sort_index()
    counts = counts.reindex(sorted(counts.columns), axis=1)
    if row_order is not None:
        counts = counts.reindex([r for r in row_order if r in counts.index])
    values = counts.to_numpy(dtype=float)
    dense = len(counts.index) > DENSE_ROWS
    height = (
        0.06 * len(counts.index) + 2
        if dense
        else max(2.5, 0.3 * len(counts.index) + 1.5)
    )
    figure = Figure(figsize=(max(6.0, 0.45 * len(counts.columns) + 3), height))
    axes = figure.add_subplot()
    image = axes.imshow(
        np.where(values > 0, values, np.nan), aspect="auto", cmap="viridis"
    )
    axes.set_xticks(
        range(len(counts.columns)), [str(c) for c in counts.columns], rotation=90
    )
    axes.set_yticks(
        range(len(counts.index)),
        [str(i) for i in counts.index],
        fontsize=3 if dense else None,
    )
    if annotate and not dense:
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
    labels: Mapping[str, str] | None = None,
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
    labels : Mapping[str, str], optional
        Display names per group value.
    title, ylabel : str
        Labels.
    log : bool
        Logarithmic vertical axis.

    Notes
    -----
    Each group starts at 1 on its own first date, so a series that begins later (a
    company listed after the start, a younger fund) still shows from its start. The
    legend is left out above ``MAX_LEGEND`` groups.
    """
    wide = df.pivot_table(index=x, columns=group, values=y).sort_index()
    _check_lines(len(wide.columns))
    colours = _palette(len(wide.columns))
    figure = Figure(figsize=WIDE)
    axes = figure.add_subplot()
    for name, colour in zip(wide.columns, colours, strict=True):
        series = wide[name].dropna()
        if series.empty:
            continue
        thick = name == highlight
        axes.plot(
            series.index,
            series / series.iloc[0],
            label=(labels or {}).get(str(name), str(name)),
            lw=2.5 if thick else 1.2,
            color="black" if thick else colour,
        )
    _finish(
        axes,
        title=title or "Growth of one unit from each series' first date",
        ylabel=ylabel,
        legend=len(wide.columns) <= MAX_LEGEND,
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
    _check_lines(len(names) * (2 if second_benchmark else 1))
    figure = Figure(figsize=WIDE)
    axes = figure.add_subplot()
    for name, colour in zip(names, _palette(len(names)), strict=True):
        solid = _relative_growth(wide[name], wide[benchmark])
        label = (labels or {}).get(str(name), str(name))
        (line,) = axes.plot(solid.index, solid, label=label, lw=1.2, color=colour)
        if second_benchmark is not None:
            dashed = _relative_growth(wide[name], wide[second_benchmark])
            axes.plot(dashed.index, dashed, color=line.get_color(), lw=1, ls="--")
    axes.axhline(1, color="black", lw=1)
    if second_benchmark is not None:
        _style_key(
            axes, [f"relative to {benchmark}", f"relative to {second_benchmark}"]
        )
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
    groups = list(_groups(df, group))
    _check_lines(len(groups))
    figure = Figure(figsize=WIDE)
    axes = figure.add_subplot()
    for (name, part), colour in zip(groups, _palette(len(groups)), strict=True):
        values = part.groupby(x)[y]
        low, high = values.quantile(band[0]), values.quantile(band[1])
        median = values.median()
        (line,) = axes.plot(median.index, median, lw=1.4, label=name, color=colour)
        axes.fill_between(median.index, low, high, color=line.get_color(), alpha=0.15)
    _thresholds(axes, thresholds)
    _finish(axes, title=title, ylabel=ylabel, legend=group is not None)
    figure.tight_layout()
    return figure


def plot_spread_pair(
    df: pd.DataFrame,
    *,
    x: str,
    left: str,
    right: str,
    band: tuple[float, float] = (0.25, 0.75),
    title: str = "",
    left_label: str = "",
    right_label: str = "",
    log_left: bool = False,
    log_right: bool = False,
) -> Figure:
    """Plot the median and band of ``left`` and of ``right`` across rows, on two axes.

    Parameters
    ----------
    df : pd.DataFrame
        Long table, many rows per ``x`` (e.g. one per company and date); not modified.
    x : str
        Horizontal axis.
    left, right : str
        Values summarised on the left and right axes, in two colours.
    band : tuple[float, float]
        Lower and upper quantile of the shaded bands.
    title, left_label, right_label : str
        Labels; an empty axis label falls back to the column name.
    log_left, log_right : bool
        Logarithmic axes, for values spanning orders of magnitude (P/E, growth).
    """
    figure = Figure(figsize=WIDE)
    axes = figure.add_subplot()
    twin = axes.twinx()
    for target, column, colour, label in (
        (axes, left, "tab:blue", left_label or left),
        (twin, right, "tab:orange", right_label or right),
    ):
        values = df.groupby(x)[column]
        low, high = values.quantile(band[0]), values.quantile(band[1])
        median = values.median()
        target.plot(median.index, median, color=colour, lw=1.4, label=f"median {label}")
        target.fill_between(median.index, low, high, color=colour, alpha=0.15)
        target.set_ylabel(label, color=colour)
    if log_left:
        axes.set_yscale("log")
    if log_right:
        twin.set_yscale("log")
    handles = axes.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = axes.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    axes.legend(
        handles,
        labels,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.08, 1.0),
        frameon=False,
    )
    axes.set_title(title)
    axes.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def plot_pair_series(
    df: pd.DataFrame,
    *,
    x: str,
    left: str,
    right: str,
    group: str,
    title: str = "",
    left_label: str = "",
    right_label: str = "",
    log_left: bool = False,
    log_right: bool = False,
    panel: bool = False,
) -> Figure:
    """Plot two values per group on two axes: ``left`` solid, ``right`` dashed, one colour each.

    Parameters
    ----------
    df : pd.DataFrame
        Long table with ``x``, ``left``, ``right`` and ``group``; not modified.
    x : str
        Horizontal axis.
    left, right : str
        Values on the left axis (solid lines) and the right axis (dashed lines).
    group : str
        Column whose values get one colour each, the same on both axes.
    title, left_label, right_label : str
        Labels; an empty axis label falls back to the column name.
    log_left, log_right : bool
        Logarithmic axes.
    panel : bool
        A small figure with the legend inside, for figures placed side by side.
    """
    groups = list(_groups(df, group))
    _check_lines(2 * len(groups))
    figure = Figure(figsize=PANEL if panel else WIDE)
    axes = figure.add_subplot()
    twin = axes.twinx()
    for (name, unsorted), colour in zip(groups, _palette(len(groups)), strict=True):
        part = unsorted.sort_values(x)
        axes.plot(part[x], part[left], color=colour, lw=1.2, label=name)
        twin.plot(part[x], part[right], color=colour, lw=1.0, ls="--")
    _style_key(axes, [left_label or left, right_label or right])
    axes.set_ylabel(left_label or left)
    twin.set_ylabel(right_label or right)
    if log_left:
        axes.set_yscale("log")
    if log_right:
        twin.set_yscale("log")
    _finish(axes, title=title, ylabel=left_label or left, legend=not panel)
    if panel:
        # Two axes leave no free corner inside the plot: the legend goes below it.
        axes.legend(
            fontsize=6,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncols=4,
            frameon=False,
        )
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
    groups = list(_groups(df, group))
    _check_lines(len(groups))
    figure = Figure(figsize=WIDE)
    axes = figure.add_subplot()
    for (name, unsorted), colour in zip(groups, _palette(len(groups)), strict=True):
        part = unsorted.sort_values(lag)
        axes.plot(
            part[lag], part[value], marker="o", ms=3, lw=1.2, label=name, color=colour
        )
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


def _style_key(axes: Axes, names: Sequence[str]) -> None:
    """Add legend entries explaining the line styles (solid, dashed, ...) in black."""
    for index, name in enumerate(names):
        style = LINE_STYLES[index % len(LINE_STYLES)]
        axes.plot(
            [],
            [],
            color="black",
            ls=style,
            lw=1.2,
            label=f"{_STYLE_NAMES[style]}: {name}",
        )


_STYLE_NAMES = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dash-dot"}


def _check_lines(count: int) -> None:
    """Refuse a plot with more than ``MAX_LINES`` lines (the readability rule)."""
    if count > MAX_LINES:
        raise ValueError(
            f"{count} lines in one plot; at most {MAX_LINES}: select fewer series or use a heat map"
        )


def _palette(count: int) -> list[tuple[float, float, float, float]]:
    """Return ``count`` distinct colours: tab10 up to 10, tab20 up to 20, then repeated."""
    name = "tab10" if count <= 10 else "tab20"  # noqa: PLR2004  palette sizes
    colours = colormaps[name].colors  # type: ignore[attr-defined]  # ListedColormap
    return [tuple(colours[i % len(colours)]) for i in range(count)]  # type: ignore[misc]


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
    axes: Axes,
    *,
    title: str,
    ylabel: str,
    legend: bool,
    log: bool = False,
    panel: bool = False,
) -> None:
    """Apply the title, axis label, scale, legend and grid shared by all plots.

    Large values (from ``THOUSANDS``) get thousands separators instead of an offset. A
    panel keeps its legend inside the plot, in two small columns.
    """
    axes.set_title(title, fontsize=10 if panel else None)
    axes.set_ylabel(ylabel)
    if log:
        axes.set_yscale("log")
    elif max((abs(v) for v in axes.get_ylim()), default=0.0) >= THOUSANDS:
        axes.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    if legend and panel:
        axes.legend(fontsize=6, loc="best", ncols=2, frameon=False)
    elif legend:
        axes.legend(
            fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False
        )
    axes.grid(alpha=0.3)
