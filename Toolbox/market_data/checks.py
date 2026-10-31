"""
market_data/checks.py
---------------------
Checks of the stored data and the market analysis plots for the report.

``run_checks`` reads the tables the update wrote for an as-of date, prints numeric checks
and writes the named plots of ``market_data.market_analysis`` under
``<results>/market_analysis/<as_of>/<section>/``. The main text is a concise view of the
recent years (``recent_years``): per analysis (``TOPICS``) a sector summary, a grid of
every sector's best and worst companies, and an overview of the two best and the two
worst sectors with their best and worst companies; then P/E against value across all
companies. "Best" follows the analysis (``market_data.rankings``). The appendix holds the
full history of every sector, the P/E lead-lag, the data checks and the ticker list. Each
plot is recorded with its figure and panel, so the report can join panels into one
figure. The P/E, lead-lag, dividend-yield and score tables are stored under
``<interim>/metrics/<as_of>/``.

Owns
----
Which numeric checks describe each source, which plots each section and figure holds,
the result folder layout, and the caption fields and facts the report text uses.

Must not
--------
Define plots (``market_data.market_analysis``); compute metrics or scores
(``market_data.metrics``, ``market_data.rankings``); clean data (``market_data.clean``);
build the GICS map (``market_data.gics``); write the report (``market_data.report``).

Public API
----------
run_checks(as_of, *, storage, results, settings)
    Write the plots; return a ``CheckReport``.
CheckSettings
    Benchmarks and the analysis settings from ``config/market_data.toml``.
CheckReport
    Numeric checks, plots, facts per analysis, the sector table and the ticker list.

Dependency rule
---------------
pandas, the standard library, ``skatkis.io`` and ``market_data``.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from Toolbox.market_data import market_analysis as plots
from Toolbox.market_data.clean import annual_first_reported, select_tags
from Toolbox.market_data.gics import gics_map
from Toolbox.market_data.metrics import dividend_yield, price_earnings
from Toolbox.market_data.rankings import (
    TOPICS,
    RankingTables,
    best_and_worst,
    company_scores,
    sector_scores,
    window,
)
from Toolbox.market_data.report import Plot
from Toolbox.market_data.update import Storage
from Toolbox.skatkis.io.figures import save_figure
from Toolbox.skatkis.io.tables import read_table, write_table

EPS_TOLERANCE = 0.05
LARGE_DAILY_MOVE = {"yahoo": 0.20, "fred": 0.05}
LEAD_LAG_MONTHS = 12


@dataclass(frozen=True)
class CheckSettings:
    """Benchmarks and analysis settings (``[benchmarks]``, ``[universe]``, ``[analysis]``)."""

    market: str
    sectors: Mapping[str, str]
    excluded: Sequence[str]
    index: str = "sp500"
    recent_years: int = 5
    long_years: int = 10
    main_companies: int = 5
    appendix_companies: int = 10


@dataclass
class CheckReport:
    """What a check run produced for the report.

    Parameters
    ----------
    folder : Path
        The run's result folder.
    lines : list[str]
        Numeric checks, printed.
    plots : list[Plot]
        Every plot written, with its figure, panel, role and caption fields.
    facts : dict[str, dict[str, str]]
        Per analysis: the best and worst sectors and companies, for the section text.
    sectors : list[dict[str, str]]
        Per GICS sector: name, number of companies and fund, for the sector table.
    tickers : list[dict[str, str]]
        Per company: sector, symbol and name, for the ticker appendix.
    """

    folder: Path
    lines: list[str] = field(default_factory=list[str])
    plots: list[Plot] = field(default_factory=list[Plot])
    facts: dict[str, dict[str, str]] = field(default_factory=dict[str, dict[str, str]])
    sectors: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    tickers: list[dict[str, str]] = field(default_factory=list[dict[str, str]])

    def save(
        self,
        figure: Figure,
        topic: str,
        name: str,
        *,
        kind: str,
        fields: Mapping[str, str] | None = None,
    ) -> None:
        """Save ``figure`` as ``<folder>/<topic>/<name>.png`` and record it.

        ``kind`` is ``<figure>`` or ``<figure>:<panel>``: plots with the same figure are
        joined in the report, each with its panel as sub-caption. A topic under
        ``appendix/`` places the plot in the appendix.
        """
        path = save_figure(figure, self.folder / topic / f"{name}.png")
        group, _, panel = kind.partition(":")
        self.plots.append(
            Plot(
                topic=topic,
                name=name,
                kind=group,
                path=path,
                fields=dict(fields or {}),
                appendix=topic.startswith("appendix/"),
                figure=group,
                panel=panel,
            )
        )


@dataclass(frozen=True)
class _Data:
    """The tables of one run: full history and the recent window."""

    full: RankingTables
    recent: RankingTables
    settings: CheckSettings


def run_checks(
    as_of: date, *, storage: Storage, results: Path, settings: CheckSettings
) -> CheckReport:
    """Check the tables stored for ``as_of`` and write the plots.

    Parameters
    ----------
    as_of : date
        The as-of date whose tables are checked.
    storage : Storage
        Snapshot and table roots.
    results : Path
        Root of the result folders (``results``).
    settings : CheckSettings
        Benchmarks, excluded sectors and the analysis settings.

    Returns
    -------
    CheckReport
        Numeric checks, plots, facts, the sector table and the ticker list.
    """
    yahoo = storage.folder("yahoo", as_of)
    facts = select_tags(
        read_table(storage.folder("edgar_companyfacts", as_of) / "facts.parquet")
    )
    prices = read_table(yahoo / "prices.parquet")
    dividends = read_table(yahoo / "dividends.parquet")
    constituents = read_table(
        storage.folder("constituents", as_of) / settings.index / "constituents.parquet"
    )
    gics = gics_map(constituents, funds=settings.sectors, excluded=settings.excluded)
    gics = gics[gics["symbol"].isin(prices["symbol"])].reset_index(drop=True)
    tag = gics[["symbol", "gics_sector"]]
    one_per_company = gics.drop_duplicates("cik")[["cik", "symbol", "gics_sector"]]

    annual = pd.concat(
        [
            annual_first_reported(facts, f).assign(field=f)
            for f in sorted(facts["field"].unique())
        ]
    ).merge(one_per_company, on="cik", how="inner")
    years = pd.DatetimeIndex(annual["period_end"]).year
    complete = pd.DataFrame(
        annual.loc[years < pd.Timestamp(annual["filed"].max()).year]
    )
    eps = annual[annual["field"] == "eps_diluted"]
    pe = price_earnings(prices, eps, read_table(yahoo / "splits.parquet")).merge(
        tag, on="symbol"
    )
    lead_lag = plots.pe_price_lead_lag(pe, LEAD_LAG_MONTHS).merge(tag, on="symbol")
    yields = dividend_yield(prices, dividends).merge(tag, on="symbol")
    full = RankingTables(
        complete, prices, yields, pe, gics.drop_duplicates("cik"), settings.market
    )
    last = pd.Timestamp(prices["date"].max())
    recent = window(full, last - pd.DateOffset(years=settings.recent_years))
    data = _Data(full, recent, settings)
    metrics = storage.folder("metrics", as_of)
    for name, table in {
        "pe": pe,
        "pe_lead_lag": lead_lag,
        "dividend_yield": yields,
    }.items():
        write_table(table, metrics / f"{name}.parquet")

    report = CheckReport(folder=results / "market_analysis" / as_of.isoformat())
    for sector, members in gics.groupby("gics_sector"):
        report.sectors.append(
            {
                "sector": str(sector),
                "companies": str(members["cik"].nunique()),
                "fund": str(members["sector_fund"].iloc[0]),
            }
        )
    names = dict(
        zip(
            constituents["symbol"].astype(str),
            constituents["security"].astype(str),
            strict=True,
        )
    )
    report.tickers = [
        {
            "sector": str(r.gics_sector),
            "symbol": str(r.symbol),
            "name": names.get(str(r.symbol).replace("-", "."), ""),
        }
        for r in gics.sort_values(["gics_sector", "symbol"]).itertuples()
    ]
    _summaries(data, report)
    for topic in TOPICS:
        _analysis(topic, data, report)
        _appendix(topic, data, report)
    _pe_and_value(data, report)
    by_sector = pd.DataFrame(
        lead_lag.groupby(["gics_sector", "lag"], as_index=False)["correlation"].mean()
    )
    report.save(
        plots.pe_lead_lag(by_sector, group="gics_sector"),
        "appendix/pe",
        "lead_lag",
        kind="lead_lag",
    )
    _data_checks(facts, annual, prices, dividends, report)
    _fx(read_table(storage.folder("fred", as_of) / "observations.parquet"), report)
    return report


def _summaries(data: _Data, report: CheckReport) -> None:
    """Write the sector summary at the top of each analysis."""
    settings, prices = data.settings, data.full.prices
    available = set(prices["symbol"].unique())
    funds = {s: f for s, f in settings.sectors.items() if f in available}
    last = pd.Timestamp(prices["date"].max())
    for metric, panel in (("revenue", "Revenue"), ("net_income", "Net income")):
        figure = plots.fundamentals_sector_totals(data.full.annual, metric)
        report.save(figure, "fundamentals", metric, kind=f"summary:{panel}")
    views = (
        ("recent", f"Last {settings.recent_years} years", settings.recent_years),
        ("long", f"Last {settings.long_years} years", settings.long_years),
        ("full", "Full history", None),
    )
    for name, label, span in views:
        part = (
            prices
            if span is None
            else prices[prices["date"] >= last - pd.DateOffset(years=span)]
        )
        figure = plots.total_return_summary(part, funds, settings.market, title=label)
        report.save(figure, "total_return", f"summary_{name}", kind=f"summary:{label}")
    recent = data.recent
    report.save(
        plots.sector_benchmarks(recent.prices, funds, settings.market),
        "relative_return",
        "summary",
        kind="summary",
    )
    report.save(plots.pe_sector_medians(recent.pe), "pe", "summary", kind="summary")
    report.save(
        plots.dividend_yield_summary(recent.yields),
        "dividend_yield",
        "summary",
        kind="summary",
    )
    report.lines += [
        "METRICS",
        f"  P/E rows {len(data.full.pe):,}; symbols {data.full.pe['symbol'].nunique()}; "
        f"missing (EPS <= 0) {int(data.full.pe['pe'].isna().sum())}",
    ]


def _score_text(topic: str, score: float) -> str:
    """Format a score: P/E as a number, yield as percent, growth and returns per year."""
    if topic == "pe":
        return f"{score:.1f}"
    if topic == "dividend_yield":
        return f"{score:.2%}"
    return f"{score:.1%} a year"


@dataclass(frozen=True)
class _Picks:
    """The companies chosen for one plot: best (solid), worst (dashed), the sector fund."""

    best: Sequence[str]
    worst: Sequence[str]
    fund: str


def _companies_plot(
    topic: str, tables: RankingTables, picks: _Picks, *, title: str, panel: bool
) -> Figure:
    """Plot the best (solid) against the worst (dashed) companies for ``topic``."""
    best, worst, fund = picks.best, picks.worst, picks.fund
    if topic == "fundamentals":
        revenue = tables.annual[tables.annual["field"] == "revenue"]
        return plots.companies_ranked(
            revenue.assign(value=revenue["value"] / plots.MILLION),
            value="value",
            best=best,
            worst=worst,
            x="period_end",
            title=title,
            ylabel="revenue (USD millions)",
            panel=panel,
        )
    if topic in ("total_return", "relative_return"):
        relative = fund if topic == "relative_return" else None
        series = plots.ranked_series(
            tables.prices, [*best, *worst], relative_to=relative
        )
        label = f"growth relative to {fund}" if relative else "growth of 1 USD"
        return plots.companies_ranked(
            series,
            value="value",
            best=best,
            worst=worst,
            log=True,
            title=title,
            ylabel=f"{label} (log)",
            panel=panel,
        )
    if topic == "dividend_yield":
        return plots.companies_ranked(
            tables.yields,
            value="dividend_yield",
            best=best,
            worst=worst,
            title=title,
            ylabel="dividend yield (%)",
            panel=panel,
        )
    return plots.companies_ranked(
        tables.pe,
        value="pe",
        best=best,
        worst=worst,
        title=title,
        ylabel="P/E",
        panel=panel,
    )


def _ends(scores: pd.DataFrame, topic: str, n: int) -> tuple[list[str], list[str]]:
    """Return the symbols of the ``n`` best and ``n`` worst rows, without overlap."""
    top, bottom = best_and_worst(scores, topic, n)
    best = [str(s) for s in top["symbol"]]
    return best, [str(s) for s in bottom["symbol"] if str(s) not in best]


def _analysis(topic: str, data: _Data, report: CheckReport) -> None:
    """Write the recent view of an analysis: the sector grid, best and worst sectors."""
    tables, settings = data.recent, data.settings
    companies = company_scores(topic, tables)
    ranked = sector_scores(topic, companies, tables)
    if ranked.empty:
        return
    funds = dict(
        zip(
            ranked["gics_sector"].astype(str),
            ranked["sector_fund"].astype(str),
            strict=True,
        )
    )
    n = settings.main_companies
    for sector, own in companies.groupby("gics_sector"):
        name = str(sector)
        winners, losers = _ends(own, topic, n)
        figure = _companies_plot(
            topic, tables, _Picks(winners, losers, funds[name]), title=name, panel=True
        )
        report.save(figure, topic, f"grid_{_slug(name)}", kind=f"grid:{name}")
    top, bottom = best_and_worst(ranked, topic, 2)
    best = [str(s) for s in top["gics_sector"]]
    worst = [str(s) for s in bottom["gics_sector"]]
    report.facts[topic] = {
        "best1": best[0],
        "best2": best[-1],
        "worst1": worst[0],
        "worst2": worst[-1],
        "best1_score": _score_text(topic, float(top["score"].iloc[0])),
        "worst1_score": _score_text(topic, float(bottom["score"].iloc[0])),
        "n": str(n),
        "years": str(settings.recent_years),
        "long_years": str(settings.long_years),
    }
    for role, chosen in (("best", best), ("worst", worst)):
        for sector in chosen:
            own = companies[companies["gics_sector"] == sector]
            winners, losers = _ends(own, topic, n)
            none: list[str] = []
            picks = (
                _Picks(winners, none, funds[sector])
                if role == "best"
                else _Picks(none, losers, funds[sector])
            )
            figure = _companies_plot(topic, tables, picks, title=sector, panel=True)
            report.save(
                figure, topic, f"{role}_{_slug(sector)}", kind=f"{role}:{sector}"
            )


def _appendix(topic: str, data: _Data, report: CheckReport) -> None:
    """Write the full history of an analysis: every sector's best and worst companies."""
    tables, n = data.full, data.settings.appendix_companies
    companies = company_scores(topic, tables)
    funds = dict(
        zip(
            tables.gics["gics_sector"].astype(str),
            tables.gics["sector_fund"].astype(str),
            strict=False,
        )
    )
    for sector, own in companies.groupby("gics_sector"):
        name = str(sector)
        winners, losers = _ends(own, topic, n)
        title = f"{name}: best (solid) and worst (dashed) companies"
        figure = _companies_plot(
            topic,
            tables,
            _Picks(winners, losers, funds[name]),
            title=title,
            panel=False,
        )
        fields = {"sector": name, "n": str(len(winners)), "fund": funds[name]}
        report.save(
            figure, f"appendix/{topic}", _slug(name), kind="appendix", fields=fields
        )


def _pe_and_value(data: _Data, report: CheckReport) -> None:
    """P/E against value over the recent window, across all companies."""
    tables, n = data.recent, data.settings.main_companies
    by_pe = company_scores("pe", tables)
    by_value = company_scores("total_return", tables)
    low_pe, high_pe = _ends(by_pe, "pe", n)
    best_value, worst_value = _ends(by_value, "total_return", n)
    report.facts["pe_value"] = {
        "n": str(n),
        "years": str(data.settings.recent_years),
        "low_pe": ", ".join(low_pe),
        "high_pe": ", ".join(high_pe),
        "best_value": ", ".join(best_value),
        "worst_value": ", ".join(worst_value),
        "overlap_best": ", ".join(s for s in low_pe if s in best_value) or "none",
        "overlap_worst": ", ".join(s for s in high_pe if s in worst_value) or "none",
    }
    panels = (
        ("best", "low_pe", low_pe, f"Lowest P/E ({n} companies)"),
        ("best", "best_value", best_value, f"Best value ({n} companies)"),
        ("worst", "high_pe", high_pe, f"Highest P/E ({n} companies)"),
        ("worst", "worst_value", worst_value, f"Worst value ({n} companies)"),
    )
    for figure_name, name, symbols, label in panels:
        figure = plots.pe_and_value(tables.pe, symbols, title=label)
        report.save(figure, "pe_value", name, kind=f"{figure_name}:{label}")


def _data_checks(
    facts: pd.DataFrame,
    annual: pd.DataFrame,
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    report: CheckReport,
) -> None:
    """EDGAR and Yahoo checks: counts, EPS cross-check, filing delays, trading days."""
    forward = int((facts["filed"] < facts["period_end"]).sum())
    moves = prices.sort_values("date").groupby("symbol")["adj_close"].pct_change().abs()
    large = LARGE_DAILY_MOVE["yahoo"]
    report.lines += [
        "EDGAR",
        f"  values {len(facts):,} after one tag per value; companies {facts['cik'].nunique()}; "
        f"fields {facts['field'].nunique()}",
        f"  periods {facts['period_end'].min().date()} to {facts['period_end'].max().date()}; "
        f"missing values {int(facts['value'].isna().sum())}; forward-dated values {forward}",
        "YAHOO",
        f"  price rows {len(prices):,}; symbols {prices['symbol'].nunique()}; "
        f"duplicate (symbol, date) {int(prices.duplicated(['symbol', 'date']).sum())}; "
        f"missing close {int(prices['close'].isna().sum())}; "
        f"close <= 0 {int((prices['close'] <= 0).sum())}; "
        f"daily moves above {large:.0%} {int((moves > large).sum())}; dividends {len(dividends):,}",
    ]
    checks = "appendix/data_checks"
    report.save(plots.edgar_coverage(annual), checks, "coverage", kind="coverage")
    keys = ["cik", "entity_name", "period_end"]
    wide = annual.pivot_table(index=keys, columns="field", values="value").reset_index()
    needed = {"eps_diluted", "net_income", "weighted_average_diluted_shares"}
    if needed <= set(wide.columns):
        wide = wide.dropna(subset=list(needed))
        wide["ni_per_share"] = (
            wide["net_income"] / wide["weighted_average_diluted_shares"]
        )
        gap = (wide["eps_diluted"] / wide["ni_per_share"] - 1).abs()
        report.lines.append(
            f"  EPS check: {len(wide)} annual values; |EPS / (NI / shares) - 1| > "
            f"{EPS_TOLERANCE:.0%}: {int((gap > EPS_TOLERANCE).sum())}"
        )
        report.save(plots.edgar_eps_check(wide), checks, "eps_check", kind="eps_check")
    # A filing's own period is the latest financial-statement period end it reports on or
    # before its filing date (filings also carry forward-dated values, such as maturities).
    gaap = facts[
        (facts["taxonomy"] == "us-gaap")
        & facts["form"].isin(["10-K", "10-Q"])
        & (facts["period_end"] <= facts["filed"])
    ]
    lag = pd.DataFrame(
        gaap.groupby(["cik", "accession", "form", "filed"], as_index=False)[
            "period_end"
        ].max()
    )
    lag["lag_days"] = (lag["filed"] - lag["period_end"]).dt.days
    for form, part in lag.groupby("form"):
        report.lines.append(
            f"  {form} filed {part['lag_days'].min():.0f} to {part['lag_days'].max():.0f} days "
            f"after period end (median {part['lag_days'].median():.0f})"
        )
    report.save(plots.edgar_filing_lag(lag), checks, "filing_lag", kind="filing_lag")
    report.save(
        plots.yahoo_trading_days(prices), checks, "trading_days", kind="trading_days"
    )


def _fx(observations: pd.DataFrame, report: CheckReport) -> None:
    """FRED: coverage and large moves per series; the series and its coverage plots."""
    rates = observations.dropna(subset=["value"]).sort_values(["series_id", "date"])
    moves = rates.groupby("series_id")["value"].pct_change().abs()
    large = LARGE_DAILY_MOVE["fred"]
    report.lines += [
        "FRED",
        f"  rates {len(rates):,}; {rates['date'].min().date()} to {rates['date'].max().date()}; "
        f"daily moves above {large:.0%} {int((moves > large).sum())}",
    ]
    for series in sorted(rates["series_id"].unique()):
        name = str(series)
        report.save(
            plots.fred_series(rates, name), "appendix/data_checks", name, kind="fx"
        )
        report.save(
            plots.fred_coverage(rates, name),
            "appendix/data_checks",
            f"{name}_coverage",
            kind="fx_coverage",
        )


def _slug(name: str) -> str:
    """Return ``name`` as a file name: lower case, words joined by underscores."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
