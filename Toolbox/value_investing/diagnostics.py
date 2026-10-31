"""
value_investing/diagnostics.py
------------------------------
In-sample checks of the buyability score against what the prices did afterwards.

These tables describe the data; they are diagnostics, not evidence that a strategy
works (judged out-of-sample only, ``midas/instructions/researcher.md``). ``sample``
picks the companies with the highest and lowest mean score from a date, a casual sample
for looking at the data. ``forward_excess`` gives each company's total return from one
rebalance date to the next less ``^SP500TR``'s; ``score_returns`` and
``information_coefficient`` relate the score to it.

Owns
----
The diagnostic sample and the score-against-return tables.

Must not
--------
Compute flags or the score (``value_investing.layers``, ``value_investing.scoring``);
define returns (``skatkis.financial_logic``); plot (``value_investing.plots``).

Public API
----------
sample(table, *, mean_from, n)
    The ``n`` companies with the highest and the ``n`` with the lowest mean score.
forward_excess(prices, symbols, dates, *, benchmark)
    Next-quarter total return less the benchmark's, per company and date.
sample_paths(table, sample_table, levels, *, benchmark, start)
    Score, growth relative to the benchmark and their product over time for the sampled
    companies.
score_returns(table, *, bin_width)
    Mean next-quarter excess return per score bin.
information_coefficient(table)
    Per date, the rank correlation of the score with the next-quarter excess return.

Computes
--------
mean buyability score : sample
    Mean of ``score`` over the rebalance dates from ``mean_from``.
score times relative growth, buyability signal : sample_paths
    signal(t) = score(t) * R(t), R the growth over ``^SP500TR`` since ``start``
    (``skatkis.financial_logic.relative_growth``): high when a company scores well and
    has beaten the market, low when it scores badly and has lagged (user, 2026-10-31).
mean excess return by score : score_returns
    Mean, median and count of ``forward_excess`` per score bin [b, b + w), w the bin
    width; a score of 1 falls in the top bin.
information coefficient, rank IC : information_coefficient
    Spearman correlation across companies of ``score`` and ``forward_excess``, per date.

Dependency rule
---------------
pandas, ``skatkis.financial_logic`` and ``value_investing.inputs``.
"""

from collections.abc import Sequence

import pandas as pd

from Toolbox.skatkis.financial_logic import forward_excess_return, relative_growth
from Toolbox.value_investing.inputs import price_on

MIN_COMPANIES = 20


def sample(table: pd.DataFrame, *, mean_from: pd.Timestamp, n: int) -> pd.DataFrame:
    """Return the ``n`` best and ``n`` worst companies by mean score from ``mean_from``.

    Parameters
    ----------
    table : pd.DataFrame
        The buyability table (``symbol``, ``date``, ``gics_sector``, ``score``).
    mean_from : pd.Timestamp
        First rebalance date in the mean.
    n : int
        Companies per group.

    Returns
    -------
    pd.DataFrame
        Columns ``group`` (``best`` or ``worst``), ``symbol``, ``gics_sector``,
        ``mean_score``; best first, ties broken by symbol. Companies never scored
        from ``mean_from`` (too few evaluable rules) are left out.
    """
    recent = table[table["date"] >= mean_from]
    means = (
        recent.groupby(["symbol", "gics_sector"], dropna=False)["score"]
        .mean()
        .rename("mean_score")
        .dropna()
        .reset_index()
        .sort_values(["mean_score", "symbol"], ascending=[False, True])
    )
    best = means.head(n).assign(group="best")
    worst = means.tail(n).iloc[::-1].assign(group="worst")
    columns = ["group", "symbol", "gics_sector", "mean_score"]
    return pd.concat([best, worst], ignore_index=True)[columns]


def forward_excess(
    prices: pd.DataFrame,
    symbols: pd.Series,
    dates: Sequence[pd.Timestamp],
    *,
    benchmark: str,
) -> pd.Series:
    """Return each company's next-quarter total return less the benchmark's.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns ``symbol``, ``date``, ``adj_close`` (split- and dividend-adjusted),
        including ``benchmark``.
    symbols : pd.Series
        Companies.
    dates : Sequence[pd.Timestamp]
        Rebalance dates, sorted.
    benchmark : str
        Total-return benchmark symbol (``^SP500TR``).

    Returns
    -------
    pd.Series
        ``forward_excess`` (fraction) indexed (``symbol``, ``date``); missing on the last
        date and where a level is missing.
    """
    levels = (
        price_on(prices, symbols, dates, column="adj_close")
        .reset_index()
        .pivot_table(index="date", columns="symbol", values="adj_close")
    )
    market = price_on(prices, pd.Series([benchmark]), dates, column="adj_close").xs(
        benchmark
    )
    levels = levels.reindex(list(dates))
    market = market.reindex(list(dates))
    excess = pd.DataFrame(
        {name: forward_excess_return(levels[name], market) for name in levels.columns}
    )
    long = excess.rename_axis(index="date", columns="symbol").reset_index()
    melted = long.melt(id_vars="date", var_name="symbol", value_name="forward_excess")
    return melted.set_index(["symbol", "date"])["forward_excess"].sort_index()


def sample_paths(
    table: pd.DataFrame,
    sample_table: pd.DataFrame,
    levels: pd.DataFrame,
    *,
    benchmark: str,
    start: pd.Timestamp,
) -> pd.DataFrame:
    """Return score and growth relative to the benchmark over time, per sampled company.

    Parameters
    ----------
    table : pd.DataFrame
        The buyability table.
    sample_table : pd.DataFrame
        ``sample`` output.
    levels : pd.DataFrame
        Columns ``symbol``, ``date``, ``adj_close`` at the rebalance dates, including
        ``benchmark``.
    benchmark : str
        Total-return benchmark symbol.
    start : pd.Timestamp
        First date; relative growth is 1 there.

    Returns
    -------
    pd.DataFrame
        Columns ``group``, ``symbol``, ``date``, ``score``, ``relative_growth`` and
        ``signal`` (their product; missing where either is).
    """
    wide = levels[levels["date"] >= start].pivot_table(
        index="date", columns="symbol", values="adj_close"
    )
    frames: list[pd.DataFrame] = []
    for row in sample_table.itertuples(index=False):
        symbol = str(row.symbol)
        own = table[(table["symbol"] == symbol) & (table["date"] >= start)]
        growth = (
            relative_growth(wide[symbol], wide[benchmark])
            if symbol in wide
            else pd.Series(dtype="float64")
        )
        frames.append(
            own[["symbol", "date", "score"]].assign(
                group=str(row.group),
                relative_growth=own["date"].map(growth).to_numpy(dtype=float),
            )
        )
    paths = pd.concat(frames, ignore_index=True)
    paths["signal"] = paths["score"] * paths["relative_growth"]
    columns = ["group", "symbol", "date", "score", "relative_growth", "signal"]
    return paths[columns]


def score_returns(table: pd.DataFrame, *, bin_width: float = 0.1) -> pd.DataFrame:
    """Return the mean, median and count of ``forward_excess`` per score bin.

    ``score`` is the lower edge of each bin of width ``bin_width``; a score of exactly 1
    is put in the top bin, and rows without a score or a return are left out.
    """
    known = table.dropna(subset=["forward_excess", "score"])
    bins = round(1 / bin_width)
    lower = (known["score"] * bins).astype(int).clip(upper=bins - 1) / bins
    grouped = known.assign(score=lower).groupby("score")["forward_excess"]
    return (
        pd.DataFrame(
            {
                "mean_excess": grouped.mean(),
                "median_excess": grouped.median(),
                "count": grouped.size(),
            }
        )
        .reset_index()
        .sort_values("score", ignore_index=True)
    )


def information_coefficient(table: pd.DataFrame) -> pd.DataFrame:
    """Return, per date, the Spearman correlation of ``score`` and ``forward_excess``.

    Dates with fewer than ``MIN_COMPANIES`` companies having both are left out.
    """
    rows: list[dict[str, object]] = []
    for day, part in table.dropna(subset=["forward_excess"]).groupby("date"):
        if len(part) < MIN_COMPANIES:
            continue
        # Spearman's correlation is Pearson's on the ranks (average ranks for ties).
        ic = part["score"].rank().corr(part["forward_excess"].rank())
        rows.append({"date": day, "ic": ic, "companies": len(part)})
    return pd.DataFrame(rows, columns=["date", "ic", "companies"])
