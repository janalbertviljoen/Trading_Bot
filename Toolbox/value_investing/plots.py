"""
value_investing/plots.py
------------------------
The named plots of the buyability score: what each diagnostic is shown as.

Each function takes a table from ``value_investing.pipeline`` or
``value_investing.diagnostics`` and returns a matplotlib ``Figure`` through the generic
layouts of ``skatkis.analysis``; nothing is saved here.

Owns
----
The titles, axes and layout choice of each buyability plot.

Must not
--------
Compute scores or returns (``value_investing.scoring``, ``value_investing.diagnostics``);
save files (``value_investing.__main__``).

Public API
----------
score_spread(table)
    Median score and ``n_available`` across companies over time.
rule_rates(table, pass_columns, *, evaluable)
    Per rule, the share of companies passing (or evaluable) over time.
sample_paths(paths, group)
    Score and growth relative to ``^SP500TR`` of the sampled companies.
sample_signal(paths, group)
    Score times growth relative to ``^SP500TR`` of the sampled companies.
wallet_paths(wallet_table, *, horizon)
    Growth of the chained buy wallet, the sold stocks and ``^SP500TR``.
horizon_profile(summary, *, y, ylabel, title, threshold)
    A statistic per holding horizon, one line per side and period.
yearly_excess(wallet_table, *, horizon)
    Excess return per decision year and side.
correlation_curve(curve)
    Mean rank correlation of score and forward excess return per horizon.
model_paths(paths)
    Growth of 1 USD per model, chained over the yearly decisions, and ``^SP500TR``.
rule_weights(weights)
    Each rule's learned weight in ``score_ic`` over time.
balance(balance_table, *, title, others)
    The wallet's balance and the same money in ``^SP500TR`` and other comparisons.
held_value(paths, scenario)
    Buy and hold: value over time of the wallet and the comparisons, and money put in.
yearly_purchases(yearly, compare)
    Buy and hold: what each year's purchase is worth today, per holder.
weighting_check(check)
    Each rule's weight against the gap between winning and losing picks.
peer_profile(profile, label)
    Mean points per rule of a peer group against the picks.
wallet_contents(contents, paths, scenario, compare)
    Each holding's value over time, and the comparisons per holding.
rule_profile(profile, names)
    Mean points per rule of the picks that beat the S&P 500 and of those that did not.
score_against_return(returns)
    Mean next-quarter excess return per score.
information_coefficient(ic)
    The rank correlation of score and next-quarter excess return per date.

Dependency rule
---------------
numpy, pandas, matplotlib and ``skatkis.analysis``.
"""

# matplotlib annotates its plotting methods with untyped **kwargs, which strict pyright
# reports on every call; the arguments passed here are all typed.
# pyright: reportUnknownMemberType=false

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from Toolbox.skatkis.analysis import (
    plot_bars,
    plot_pair_series,
    plot_scatter,
    plot_series,
    plot_spread_pair,
)

# Benchmarks are drawn thick and black so they stand out (user, 2026-10-31): the S&P 500
# dashed, Berkshire Hathaway dash-dot, also when divided per holding.
REFERENCES = {
    "^SP500TR": "--",
    "^SP500TR per holding": "--",
    "BRK-B": "-.",
    "BRK-B per holding": "-.",
}


def score_spread(table: pd.DataFrame) -> Figure:
    """Plot the median score and rules evaluable across companies, interquartile band."""
    return plot_spread_pair(
        table,
        x="date",
        left="score",
        right="n_available",
        title="Buyability score across the S&P 500 (ex Financials, Real Estate)",
        left_label="Score (share of evaluable rules passed)",
        right_label="Rules evaluable",
    )


def rule_rates(
    table: pd.DataFrame, pass_columns: Sequence[str], *, evaluable: bool
) -> Figure:
    """Plot, per rule, the share of evaluable companies passing it, or of all evaluable.

    The pass rate is taken over the companies evaluable on the rule, as in the score.
    """
    flags = table[["date", *pass_columns]]
    if evaluable:
        shares = flags[list(pass_columns)].notna().groupby(flags["date"]).mean()
    else:
        shares = flags[list(pass_columns)].groupby(flags["date"]).mean()
    long = shares.reset_index().melt(
        id_vars="date", var_name="rule", value_name="share"
    )
    long["rule"] = long["rule"].str.removesuffix("_pass").str.upper()
    title = (
        "Share of companies evaluable on each rule"
        if evaluable
        else "Share of evaluable companies passing each rule"
    )
    return plot_series(
        long,
        x="date",
        y="share",
        group="rule",
        title=title,
        ylabel="Share of companies",
    )


def sample_paths(paths: pd.DataFrame, group: str) -> Figure:
    """Plot score (solid) and growth relative to ``^SP500TR`` (dashed) per company."""
    own = paths[paths["group"] == group]
    return plot_pair_series(
        own,
        x="date",
        left="score",
        right="relative_growth",
        group="symbol",
        title=f"The ten {group}-scoring companies: score and growth over ^SP500TR",
        left_label="Score (share passed)",
        right_label="Growth relative to ^SP500TR",
        log_right=True,
    )


def sample_signal(paths: pd.DataFrame, group: str) -> Figure:
    """Plot score * growth relative to ``^SP500TR`` since 2016, one line per company."""
    own = paths[paths["group"] == group]
    return plot_series(
        own,
        x="date",
        y="signal",
        group="symbol",
        title=f"The ten {group}-scoring companies: score * growth over ^SP500TR",
        ylabel="Score * growth relative to ^SP500TR",
    )


def score_against_return(returns: pd.DataFrame) -> Figure:
    """Plot the mean and median next-quarter excess return per score bin."""
    long = returns.melt(
        id_vars="score",
        value_vars=["mean_excess", "median_excess"],
        var_name="statistic",
        value_name="excess",
    )
    return plot_series(
        long,
        x="score",
        y="excess",
        group="statistic",
        markers=True,
        thresholds={"0": 0.0},
        title="Next-quarter return over ^SP500TR by score (in-sample)",
        ylabel="Excess return per quarter",
    )


def information_coefficient(ic: pd.DataFrame) -> Figure:
    """Plot the per-date rank correlation of score and next-quarter excess return."""
    return plot_series(
        ic.assign(
            **{
                "two-year mean": ic["ic"].rolling(8, min_periods=4).mean(),
                "quarter": ic["ic"],
            }
        ).melt(
            id_vars="date", value_vars=["quarter", "two-year mean"], var_name="series"
        ),
        x="date",
        y="value",
        group="series",
        thresholds={"0": 0.0},
        title="Rank correlation of score with next-quarter excess return (in-sample)",
        ylabel="Spearman correlation",
    )


def wallet_paths(wallet_table: pd.DataFrame, *, horizon: int) -> Figure:
    """Plot growth of 1 USD: the chained buy wallet, the sold stocks and ``^SP500TR``."""
    own = wallet_table[wallet_table["horizon_months"] == horizon].sort_values(
        "decision_date"
    )
    frames: list[pd.DataFrame] = []
    series = {
        "buy wallet (score >= threshold)": own[own["side"] == "buy"]["stock_return"],
        "sold stocks (score <= threshold)": own[own["side"] == "sell"]["stock_return"],
        "^SP500TR": own[own["side"] == "buy"]["benchmark_return"],
    }
    for name, returns in series.items():
        part = own.loc[returns.index]
        if part.empty:
            continue
        dates = [part["decision_date"].iloc[0], *part["end_date"]]
        growth = [1.0, *(1 + returns).cumprod()]
        frames.append(pd.DataFrame({"date": dates, "growth": growth, "series": name}))
    return plot_series(
        pd.concat(frames, ignore_index=True),
        x="date",
        y="growth",
        group="series",
        references=REFERENCES,
        markers=True,
        log=True,
        title=f"Growth of 1 USD, re-chosen each January, held {horizon} months",
        ylabel="Growth of 1 USD",
    )


def horizon_profile(
    summary: pd.DataFrame, *, y: str, ylabel: str, title: str, threshold: float
) -> Figure:
    """Plot ``y`` per holding horizon, one line per side and period."""
    return plot_series(
        summary.assign(line=summary["side"] + ", " + summary["period"]),
        x="horizon_months",
        y=y,
        group="line",
        markers=True,
        thresholds={f"{threshold:g}": threshold},
        title=title,
        ylabel=ylabel,
    )


def yearly_excess(wallet_table: pd.DataFrame, *, horizon: int) -> Figure:
    """Plot each decision year's wallet excess over ``^SP500TR`` per side."""
    own = wallet_table[wallet_table["horizon_months"] == horizon]
    return plot_series(
        own,
        x="decision_date",
        y="excess",
        group="side",
        markers=True,
        thresholds={"0": 0.0},
        title=f"Excess over ^SP500TR per decision year, held {horizon} months",
        ylabel="Excess return",
    )


def correlation_curve(curve: pd.DataFrame) -> Figure:
    """Plot the mean rank correlation of score and forward excess return per horizon."""
    mean = curve.groupby(["period", "horizon_months"])["ic"].mean().reset_index()
    return plot_series(
        mean,
        x="horizon_months",
        y="ic",
        group="period",
        markers=True,
        thresholds={"0": 0.0},
        title="Rank correlation of score with excess return over the next h months",
        ylabel="Mean Spearman correlation",
    )


def model_paths(paths: Mapping[str, pd.DataFrame]) -> Figure:
    """Plot growth of 1 USD per model and ``^SP500TR``, chained over the decisions.

    Each path has ``decision_date``, ``end_date``, ``wallet_return`` and
    ``benchmark_return``, one row per yearly decision.
    """
    frames: list[pd.DataFrame] = []
    benchmark: pd.DataFrame | None = None
    for name, path in paths.items():
        own = path.sort_values("decision_date")
        dates = [own["decision_date"].iloc[0], *own["end_date"]]
        growth = [1.0, *(1 + own["wallet_return"]).cumprod()]
        frames.append(pd.DataFrame({"date": dates, "growth": growth, "model": name}))
        if benchmark is None:
            market = [1.0, *(1 + own["benchmark_return"]).cumprod()]
            benchmark = pd.DataFrame(
                {"date": dates, "growth": market, "model": "^SP500TR"}
            )
    if benchmark is not None:
        frames.append(benchmark)
    return plot_series(
        pd.concat(frames, ignore_index=True),
        x="date",
        y="growth",
        group="model",
        references=REFERENCES,
        markers=True,
        log=True,
        title="Growth of 1 USD per model, decided each January, held 12 months",
        ylabel="Growth of 1 USD",
    )


def rule_weights(weights: pd.DataFrame) -> Figure:
    """Plot each rule's learned weight per date (``value_investing.weighting``)."""
    long = weights.reset_index().melt(id_vars="date", var_name="rule", value_name="w")
    long["rule"] = long["rule"].str.removesuffix("_points").str.upper()
    return plot_series(
        long,
        x="date",
        y="w",
        group="rule",
        thresholds={"0": 0.0},
        title="Learned rule weights: mean rank correlation with 12-month excess return",
        ylabel="Weight (trailing 3 years, floored at 0)",
    )


def balance(
    balance_table: pd.DataFrame, *, title: str, others: Mapping[str, str]
) -> Figure:
    """Plot the wallet's balance, ``^SP500TR``'s and each comparison's, in USD.

    ``others`` maps a comparison's label (``BRK-B``) to its column prefix (``brk_b``).
    """
    own = balance_table.sort_values("decision_date")
    first = own.iloc[0]
    series = [
        ("value wallet", "start_usd", "end_usd"),
        ("^SP500TR", "benchmark_start_usd", "benchmark_end_usd"),
        *((label, f"{p}_start_usd", f"{p}_end_usd") for label, p in others.items()),
    ]
    frames = [
        pd.DataFrame(
            {
                "date": [first["decision_date"], *own["sell_date"]],
                "usd": [first[start], *own[end]],
                "wallet": name,
            }
        )
        for name, start, end in series
    ]
    return plot_series(
        pd.concat(frames, ignore_index=True),
        x="date",
        y="usd",
        group="wallet",
        references=REFERENCES,
        markers=True,
        title=title,
        ylabel="Balance, USD",
    )


def held_value(paths: pd.DataFrame, scenario: int) -> Figure:
    """Plot the value of the wallet and each comparison for one start year, in USD."""
    years = pd.DatetimeIndex(paths["date"]).year
    mask = (paths["scenario"] == scenario).to_numpy() & (years >= scenario)
    own = paths.loc[pd.Series(mask, index=paths.index)]
    first = str(own["holder"].iloc[0])
    money = own.loc[own["holder"] == first].assign(holder="money put in")
    figure = plot_series(
        pd.concat(
            [own, money.assign(value_usd=money["invested_usd"])], ignore_index=True
        ),
        x="date",
        y="value_usd",
        group="holder",
        references=REFERENCES,
        title=f"100 USD each January and July from {scenario}",
        ylabel="Value, USD",
        panel=True,
    )
    figure.autofmt_xdate()
    return figure


def yearly_purchases(yearly: pd.DataFrame, compare: Sequence[str]) -> Figure:
    """Plot what each 100 USD purchase is worth today, per holder, by purchase date."""
    holders = {"value_usd": "value wallet", **{f"{s}_usd": s for s in compare}}
    long = yearly.rename(columns={"decision_date": "bought"}).melt(
        id_vars="bought", value_vars=list(holders), var_name="holder", value_name="usd"
    )
    long["holder"] = long["holder"].map(holders)
    return plot_series(
        long,
        x="bought",
        y="usd",
        group="holder",
        references=REFERENCES,
        markers=True,
        log=True,
        title="What each 100 USD purchase is worth today",
        ylabel="Value today, USD",
    )


def rule_profile(profile: pd.DataFrame, names: Mapping[str, str]) -> Figure:
    """Plot the mean points per rule (0 to 5) of winning and losing picks as bars.

    Rules are sorted by how much more the winners scored; ``names`` gives each rule ID
    a short label.
    """
    gap = profile.pivot_table(index="rule", columns="group", values="points")
    order = (gap["beat the S&P 500"] - gap["did not"]).sort_values(ascending=False)
    ranked = profile.assign(
        rule=pd.Categorical(profile["rule"], categories=list(order.index), ordered=True)
    ).sort_values(["rule", "group"])
    del names  # the rules are named in the text; the axis keeps the short IDs
    return plot_bars(
        ranked.assign(rule=ranked["rule"].astype(str)),
        x="rule",
        y="points",
        group="group",
        title="Rule points of the picks that beat the S&P 500 and of those that did not,"
        " weighted by dollars",
        ylabel="Mean points (0 to 5), dollar-weighted",
    )


def weighting_check(check: pd.DataFrame) -> Figure:
    """Plot each rule's mean weight against its points gap (winners less the others)."""
    return plot_scatter(
        check,
        x="weight",
        y="gap",
        labels="rule",
        thresholds={"no gap": 0.0},
        title="Rule weight against how well the rule told winners from losers",
        xlabel="Mean weight on the picks' dates (1 - pass rate)",
        ylabel="Points gap: picks that beat the S&P 500 less the others",
    )


def wallet_contents(
    contents: pd.DataFrame,
    paths: pd.DataFrame,
    scenario: int,
    compare: Sequence[str],
) -> Figure:
    """Plot each holding's value over time and each comparison divided by the holdings.

    Parameters
    ----------
    contents : pd.DataFrame
        ``Managed.contents``: ``scenario``, ``date``, ``symbol``, ``value_usd``.
    paths : pd.DataFrame
        ``BuyAndHold.paths`` with the comparisons as holders.
    scenario : int
        The start year shown.
    compare : Sequence[str]
        The comparisons (``^SP500TR``, ``BRK-B``).

    Notes
    -----
    A comparison holds the same money as the whole wallet; dividing it by the number of
    companies the wallet holds on the date puts it on the scale of one holding. An ``o``
    marks where each company first enters the wallet.
    """
    own = contents[(contents["scenario"] == scenario) & (contents["symbol"] != "cash")]
    held = own.groupby("date")["symbol"].nunique()
    frames = [own.assign(kind="holding")[["date", "symbol", "value_usd", "kind"]]]
    for symbol in compare:
        path = paths[(paths["scenario"] == scenario) & (paths["holder"] == symbol)]
        path = path[path["date"].isin(held.index)]
        frames.append(
            pd.DataFrame(
                {
                    "date": path["date"].to_numpy(),
                    "symbol": f"{symbol} per holding",
                    "value_usd": path["value_usd"].to_numpy()
                    / held.reindex(path["date"]).to_numpy(),
                    "kind": "benchmark per holding",
                }
            )
        )
    figure = plot_series(
        pd.concat(frames, ignore_index=True),
        x="date",
        y="value_usd",
        group="symbol",
        references=REFERENCES,
        style="kind",
        style_order=["holding", "benchmark per holding"],
        log=True,
        title=f"Wallet with sells from {scenario}: value of each holding",
        ylabel="Value, USD (log scale)",
    )
    # Mark where each company enters the wallet, in its line's colour (user, 2026-10-31).
    first = own.sort_values("date").drop_duplicates("symbol")
    entries = {
        str(symbol): (pd.Timestamp(day), float(value))
        for symbol, day, value in zip(
            first["symbol"], first["date"], first["value_usd"], strict=True
        )
    }
    axes = figure.axes[0]
    for line in list(axes.get_lines()):
        name = str(line.get_label())
        if name in entries:
            day, value = entries[name]
            axes.plot(
                np.asarray([day], dtype="datetime64[ns]"),
                [value],
                marker="o",
                ms=5,
                ls="none",
                color=line.get_color(),
                zorder=4,
            )
    return figure


def peer_profile(profile: pd.DataFrame, label: str) -> Figure:
    """Plot the mean points per rule of a peer group and of the picks, as bars.

    Rules are sorted by how many more points the picks scored than the peers.
    """
    gap = profile.pivot_table(index="rule", columns="group", values="points")
    order = (gap["picks"] - gap["peers"]).sort_values(ascending=False)
    ranked = profile.assign(
        rule=pd.Categorical(
            profile["rule"], categories=list(order.index), ordered=True
        ),
        group=profile["group"].map({"peers": label, "picks": "our picks"}),
    ).sort_values(["rule", "group"])
    return plot_bars(
        ranked.assign(rule=ranked["rule"].astype(str)),
        x="rule",
        y="points",
        group="group",
        title=f"Rule points of the {label} and of our picks",
        ylabel="Mean points (0 to 5)",
    )
