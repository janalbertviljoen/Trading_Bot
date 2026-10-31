"""
market_data/checks.py
---------------------
Checks of the stored data and the market analysis plots, from sectors down to companies.

``run_checks`` reads the tables the update wrote for an as-of date, prints numeric checks
(missing values, duplicates, implausible values, coverage and filing delays) and writes
the named plots of ``market_data.market_analysis`` into one folder per section under
``<results>/market_analysis/<as_of>/``. The main text goes from coarse to fine: the
sector funds against the S&P 500 (``overview``); then per analysis (``TOPICS``) its
sector summary, its best and worst sector, and the best and worst companies of those
two sectors. "Best" follows the analysis (``market_data.rankings``): the highest revenue
growth, return or dividend yield, the lowest P/E. Every sector's best and worst
companies go to the appendix (``appendix/``). The P/E, lead-lag, dividend-yield and
score tables are stored under ``<interim>/metrics/<as_of>/``.

Owns
----
Which numeric checks describe each source, which plots each section holds, the result
folder layout, and the caption fields and facts the report text uses.

Must not
--------
Define plots (``market_data.market_analysis``); compute metrics or scores
(``market_data.metrics``, ``market_data.rankings``); clean data (``market_data.clean``);
build the GICS map (``market_data.gics``); write the report (``market_data.report``).

Public API
----------
run_checks(as_of, *, storage, results, market, sectors)
    Write the plots; return a ``CheckReport``.
CheckReport
    The numeric checks, the plots written, the facts per analysis and the sector table.

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
    LOWER_IS_BETTER,
    TOPICS,
    RankingTables,
    best_and_worst,
    company_scores,
    sector_scores,
)
from Toolbox.market_data.report import Plot
from Toolbox.market_data.update import Storage
from Toolbox.skatkis.io.figures import save_figure
from Toolbox.skatkis.io.tables import read_table, write_table

EPS_TOLERANCE = 0.05
LARGE_DAILY_MOVE = {"yahoo": 0.20, "fred": 0.05}
LEAD_LAG_MONTHS = 12
SMALL_UNIVERSE = 10
MAIN_COMPANIES = 5
APPENDIX_COMPANIES = 10


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
        Every plot written, with its role and caption fields.
    facts : dict[str, dict[str, str]]
        Per analysis: the best and worst sector and their scores, for the section text.
    sectors : list[dict[str, str]]
        Per GICS sector: its name, number of companies and fund, for the sector table.
    """

    folder: Path
    lines: list[str] = field(default_factory=list[str])
    plots: list[Plot] = field(default_factory=list[Plot])
    facts: dict[str, dict[str, str]] = field(default_factory=dict[str, dict[str, str]])
    sectors: list[dict[str, str]] = field(default_factory=list[dict[str, str]])

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

        ``kind`` selects the caption in the report text; ``fields`` fill it. A topic under
        ``appendix/`` places the plot in the appendix.
        """
        path = save_figure(figure, self.folder / topic / f"{name}.png")
        appendix = topic.startswith("appendix/")
        self.plots.append(Plot(topic, name, kind, path, dict(fields or {}), appendix))


def run_checks(
    as_of: date,
    *,
    storage: Storage,
    results: Path,
    market: str,
    sectors: Mapping[str, str],
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
    market : str
        The market benchmark symbol (``^SP500TR``).
    sectors : Mapping[str, str]
        GICS sector name to its sector fund symbol.

    Returns
    -------
    CheckReport
        Numeric checks, plots, facts per analysis and the sector table.
    """
    yahoo = storage.folder("yahoo", as_of)
    facts = select_tags(
        read_table(storage.folder("edgar_companyfacts", as_of) / "facts.parquet")
    )
    prices = read_table(yahoo / "prices.parquet")
    dividends = read_table(yahoo / "dividends.parquet")
    constituents = read_table(
        storage.folder("constituents", as_of) / "constituents.parquet"
    )
    gics = gics_map(constituents, funds=sectors)
    gics = gics[gics["symbol"].isin(prices["symbol"])].reset_index(drop=True)
    tag = gics[["symbol", "gics_sector"]]

    annual = pd.concat(
        [
            annual_first_reported(facts, f).assign(field=f)
            for f in sorted(facts["field"].unique())
        ]
    ).merge(gics[["cik", "symbol", "gics_sector"]], on="cik", how="left")
    years = pd.DatetimeIndex(annual["period_end"]).year
    complete = pd.DataFrame(
        annual.loc[years < pd.Timestamp(annual["filed"].max()).year]
    )
    eps = annual[annual["field"] == "eps_diluted"].dropna(subset=["symbol"])
    pe = price_earnings(prices, eps, read_table(yahoo / "splits.parquet")).merge(
        tag, on="symbol"
    )
    lead_lag = plots.pe_price_lead_lag(pe, LEAD_LAG_MONTHS).merge(tag, on="symbol")
    yields = dividend_yield(prices, dividends).merge(tag, on="symbol")
    tables = RankingTables(complete, prices, yields, pe, gics, market)
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
                "companies": str(len(members)),
                "fund": str(members["sector_fund"].iloc[0]),
            }
        )
    _overview(prices, market, sectors, report)
    _summaries(complete, yields, pe, lead_lag, report)
    for topic in TOPICS:
        companies = company_scores(topic, tables)
        ranked = sector_scores(topic, companies, tables)
        write_table(companies, metrics / f"scores_{topic}.parquet")
        _analysis(topic, companies, ranked, tables, report)
    _data_checks(facts, annual, prices, dividends, report)
    _fx(read_table(storage.folder("fred", as_of) / "observations.parquet"), report)
    if len(gics) <= SMALL_UNIVERSE:
        _companies(annual, prices, pe, lead_lag, report)
    return report


def _overview(
    prices: pd.DataFrame, market: str, sectors: Mapping[str, str], report: CheckReport
) -> None:
    """Write the sector funds against the S&P 500."""
    available = set(prices["symbol"].unique())
    funds = {s: f for s, f in sectors.items() if f in available}
    if funds and market in available:
        report.save(
            plots.sector_benchmarks(prices, funds, market),
            "overview",
            "relative",
            kind="relative",
        )
        report.save(
            plots.total_return_summary(prices, funds, market),
            "overview",
            "total",
            kind="total",
        )


def _summaries(
    complete: pd.DataFrame,
    yields: pd.DataFrame,
    pe: pd.DataFrame,
    lead_lag: pd.DataFrame,
    report: CheckReport,
) -> None:
    """Write the all-sector summary at the top of each analysis (returns: the overview)."""
    for metric in ("revenue", "net_income"):
        figure = plots.fundamentals_sector_totals(complete, metric)
        report.save(figure, "fundamentals", metric, kind=metric)
    report.save(plots.pe_sector_medians(pe), "pe", "summary", kind="summary")
    by_sector = pd.DataFrame(
        lead_lag.groupby(["gics_sector", "lag"], as_index=False)["correlation"].mean()
    )
    report.save(
        plots.pe_lead_lag(by_sector, group="gics_sector"),
        "pe",
        "lead_lag",
        kind="lead_lag",
    )
    report.save(
        plots.dividend_yield_summary(yields),
        "dividend_yield",
        "summary",
        kind="summary",
    )
    report.lines += [
        "METRICS",
        f"  P/E rows {len(pe):,}; symbols {pe['symbol'].nunique()}; "
        f"missing (EPS <= 0) {int(pe['pe'].isna().sum())}; dividend-yield rows {len(yields):,}",
    ]


def _score_text(topic: str, score: float) -> str:
    """Format a score for text: P/E as a number, the others as percent (per year)."""
    if topic == "pe":
        return f"{score:.1f}"
    if topic == "dividend_yield":
        return f"{score:.2%}"
    return f"{score:.1%} a year"


def _analysis(
    topic: str,
    companies: pd.DataFrame,
    ranked: pd.DataFrame,
    tables: RankingTables,
    report: CheckReport,
) -> None:
    """One analysis: best and worst sector, their companies, and the appendix."""
    if ranked.empty:
        return
    top, bottom = best_and_worst(ranked, topic, 1)
    best, worst = str(top["gics_sector"].iloc[0]), str(bottom["gics_sector"].iloc[0])
    funds = dict(
        zip(
            ranked["gics_sector"].astype(str),
            ranked["sector_fund"].astype(str),
            strict=True,
        )
    )
    report.facts[topic] = {
        "best": best,
        "worst": worst,
        "best_score": _score_text(topic, float(top["score"].iloc[0])),
        "worst_score": _score_text(topic, float(bottom["score"].iloc[0])),
        "direction": "lowest" if topic in LOWER_IS_BETTER else "highest",
    }
    pair = report.facts[topic]
    title = f"Best: {best}; worst: {worst}"
    if topic == "fundamentals":
        figure = plots.sectors_fundamentals(tables.annual, [best, worst], title=title)
        report.save(figure, topic, "sectors", kind="sectors", fields=pair)
    elif topic == "total_return":
        chosen = {best: funds[best], worst: funds[worst]}
        figure = plots.sectors_total_return(
            tables.prices, chosen, tables.market, title=title
        )
        report.save(figure, topic, "sectors", kind="sectors", fields=pair)
    elif topic == "relative_return":
        chosen = {best: funds[best], worst: funds[worst]}
        figure = plots.sector_benchmarks(tables.prices, chosen, tables.market)
        report.save(figure, topic, "sectors", kind="sectors", fields=pair)
    elif topic == "dividend_yield":
        figure = plots.sectors_spread(
            tables.yields,
            "dividend_yield",
            [best, worst],
            title=title,
            ylabel="dividend yield (%)",
        )
        report.save(figure, topic, "sectors", kind="sectors", fields=pair)
    else:
        for role, sector in (("best", best), ("worst", worst)):
            figure = plots.sector_pe_and_price(
                tables.pe, sector, title=f"{sector}: P/E and price"
            )
            report.save(
                figure, topic, f"sector_{role}", kind=f"sector_{role}", fields=pair
            )
        # Value against P/E: the sectors with the best and worst return relative to the market.
        value = sector_scores(
            "relative_return", company_scores("relative_return", tables), tables
        )
        value_top, value_bottom = best_and_worst(value, "relative_return", 1)
        pair["value_best"] = str(value_top["gics_sector"].iloc[0])
        pair["value_worst"] = str(value_bottom["gics_sector"].iloc[0])
        for role in ("best", "worst"):
            sector = pair[f"value_{role}"]
            figure = plots.sector_pe_and_price(
                tables.pe, sector, title=f"{sector}: P/E and price"
            )
            report.save(
                figure, topic, f"value_{role}", kind=f"value_{role}", fields=pair
            )
    for role, sector in (("best", best), ("worst", worst)):
        own = companies[companies["gics_sector"] == sector]
        winners, losers = best_and_worst(own, topic, MAIN_COMPANIES)
        figure = _companies_plot(
            topic, tables, sector, funds[sector], (winners, losers)
        )
        fields = {
            **pair,
            "sector": sector,
            "n": str(MAIN_COMPANIES),
            "fund": funds[sector],
        }
        report.save(figure, topic, f"companies_{role}", kind="companies", fields=fields)
    for sector, own in companies.groupby("gics_sector"):
        name = str(sector)
        winners, losers = best_and_worst(own, topic, APPENDIX_COMPANIES)
        figure = _companies_plot(topic, tables, name, funds[name], (winners, losers))
        fields = {"sector": name, "n": str(len(winners)), "fund": funds[name]}
        report.save(
            figure,
            f"appendix/{topic}",
            _slug(name),
            kind="appendix",
            fields=fields,
        )


def _companies_plot(
    topic: str,
    tables: RankingTables,
    sector: str,
    fund: str,
    ranks: tuple[pd.DataFrame, pd.DataFrame],
) -> Figure:
    """Plot a sector's best (solid) against its worst (dashed) companies for ``topic``."""
    winners, losers = ranks
    best = [str(s) for s in winners["symbol"]]
    worst = [str(s) for s in losers["symbol"] if s not in set(best)]
    symbols: Sequence[str] = [*best, *worst]
    title = f"{sector}: best (solid) and worst (dashed) companies"
    if topic == "fundamentals":
        revenue = tables.annual[tables.annual["field"] == "revenue"]
        revenue = revenue.assign(value=revenue["value"] / plots.MILLION)
        return plots.companies_ranked(
            revenue,
            value="value",
            best=best,
            worst=worst,
            x="period_end",
            title=title,
            ylabel="revenue (USD millions)",
        )
    if topic == "total_return":
        series = plots.ranked_series(tables.prices, symbols)
        return plots.companies_ranked(
            series,
            value="value",
            best=best,
            worst=worst,
            log=True,
            title=title,
            ylabel="growth of 1 USD (log)",
        )
    if topic == "relative_return":
        series = plots.ranked_series(tables.prices, symbols, relative_to=fund)
        return plots.companies_ranked(
            series,
            value="value",
            best=best,
            worst=worst,
            log=True,
            title=title,
            ylabel=f"growth relative to {fund} (log)",
        )
    if topic == "dividend_yield":
        return plots.companies_ranked(
            tables.yields,
            value="dividend_yield",
            best=best,
            worst=worst,
            title=title,
            ylabel="dividend yield (%)",
        )
    return plots.companies_ranked(
        tables.pe, value="pe", best=best, worst=worst, title=title, ylabel="P/E"
    )


def _data_checks(
    facts: pd.DataFrame,
    annual: pd.DataFrame,
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    report: CheckReport,
) -> None:
    """EDGAR and Yahoo checks: counts, EPS cross-check, filing delays, trading days."""
    early = int((facts["filed"] < facts["period_end"]).sum())
    moves = prices.sort_values("date").groupby("symbol")["adj_close"].pct_change().abs()
    large = LARGE_DAILY_MOVE["yahoo"]
    report.lines += [
        "EDGAR",
        f"  values {len(facts):,} after one tag per value; companies {facts['cik'].nunique()}; "
        f"fields {facts['field'].nunique()}",
        f"  periods {facts['period_end'].min().date()} to {facts['period_end'].max().date()}; "
        f"missing values {int(facts['value'].isna().sum())}; filed before period end {early}",
        "YAHOO",
        f"  price rows {len(prices):,}; symbols {prices['symbol'].nunique()}; "
        f"duplicate (symbol, date) {int(prices.duplicated(['symbol', 'date']).sum())}; "
        f"missing close {int(prices['close'].isna().sum())}; "
        f"close <= 0 {int((prices['close'] <= 0).sum())}; "
        f"daily moves above {large:.0%} {int((moves > large).sum())}; dividends {len(dividends):,}",
    ]
    report.save(
        plots.edgar_coverage(annual), "data_checks", "coverage", kind="coverage"
    )
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
        report.save(
            plots.edgar_eps_check(wide), "data_checks", "eps_check", kind="eps_check"
        )
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
    report.save(
        plots.edgar_filing_lag(lag), "data_checks", "filing_lag", kind="filing_lag"
    )
    report.save(
        plots.yahoo_trading_days(prices),
        "data_checks",
        "trading_days",
        kind="trading_days",
    )


def _fx(observations: pd.DataFrame, report: CheckReport) -> None:
    """FRED: coverage and large moves; the series and its coverage plots."""
    rates = observations.dropna(subset=["value"])
    moves = rates["value"].pct_change().abs()
    large = LARGE_DAILY_MOVE["fred"]
    report.lines += [
        "FRED",
        f"  rates {len(rates):,}; {rates['date'].min().date()} to {rates['date'].max().date()}; "
        f"daily moves above {large:.0%} {int((moves > large).sum())}",
    ]
    for series in sorted(rates["series_id"].unique()):
        name = str(series)
        report.save(plots.fred_series(rates, name), "data_checks", name, kind="fx")
        report.save(
            plots.fred_coverage(rates, name),
            "data_checks",
            f"{name}_coverage",
            kind="fx_coverage",
        )


def _companies(
    annual: pd.DataFrame,
    prices: pd.DataFrame,
    pe: pd.DataFrame,
    lead_lag: pd.DataFrame,
    report: CheckReport,
) -> None:
    """Single companies, for a small universe only."""
    for metric in ("revenue", "net_income", "eps_diluted"):
        report.save(
            plots.edgar_annual_series(annual, metric), "companies", metric, kind=metric
        )
    report.save(
        plots.yahoo_total_return(prices), "companies", "total_return", kind="total"
    )
    report.save(
        plots.yahoo_relative_return(prices),
        "companies",
        "relative_return",
        kind="relative",
    )
    report.save(plots.pe_over_time(pe), "companies", "pe", kind="pe")
    report.save(
        plots.pe_lead_lag(lead_lag), "companies", "pe_lead_lag", kind="lead_lag"
    )


def _slug(name: str) -> str:
    """Return ``name`` as a file name: lower case, words joined by underscores."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
