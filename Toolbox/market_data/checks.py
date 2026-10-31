"""
market_data/checks.py
---------------------
Checks of the stored data, as numbers and plots, and the first metric plots.

``run_checks`` reads the tables the update wrote for an as-of date, prints numeric checks
(missing values, duplicates, implausible values, coverage and filing delays) and writes
the named plots of ``market_data.market_analysis`` under
``<results>/market_analysis/<source>/<as_of>/`` (``edgar``, ``yahoo``, ``fred``,
``sectors``, ``metrics``). Companies are tagged with the GICS map (``market_data.gics``)
and grouped by sector: each group is normalised the same way, against its own sector
fund and the market. Per-company plots are drawn only for a small universe
(``SMALL_UNIVERSE`` companies or fewer). The P/E and lead-lag tables are stored under
``<interim>/metrics/<as_of>/``. The terminal command ``python -m Toolbox.market_data
check`` calls it.

Owns
----
Which numeric checks describe each source, which plots are written, and the layout of the
result folders.

Must not
--------
Define plots (``market_data.market_analysis``); compute metrics (``market_data.metrics``),
clean data (``market_data.clean``) or build the GICS map (``market_data.gics``).

Public API
----------
run_checks(as_of, *, storage, results, market, sectors)
    Write the check, sector and metric plots; return a ``CheckReport``.
CheckReport
    The numeric checks as lines of text and the paths of the plots written.

Dependency rule
---------------
pandas, the standard library, ``skatkis.io`` and ``market_data``.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from Toolbox.market_data import market_analysis as plots
from Toolbox.market_data.clean import annual_first_reported, select_tags
from Toolbox.market_data.gics import gics_map
from Toolbox.market_data.metrics import price_earnings
from Toolbox.market_data.update import Storage
from Toolbox.skatkis.io.figures import save_figure
from Toolbox.skatkis.io.tables import read_table, write_table

EPS_TOLERANCE = 0.05
LARGE_DAILY_MOVE = {"yahoo": 0.20, "fred": 0.05}
LEAD_LAG_MONTHS = 12
SMALL_UNIVERSE = 10


@dataclass
class CheckReport:
    """The numeric checks as text lines, and the plots written."""

    lines: list[str] = field(default_factory=list[str])
    figures: list[Path] = field(default_factory=list[Path])

    def save(self, figure: Figure, folder: Path, name: str) -> None:
        """Save ``figure`` as ``<folder>/<name>.png`` and record its path."""
        self.figures.append(save_figure(figure, folder / f"{name}.png"))


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
        Numeric checks per source and the paths of the plots written.
    """
    yahoo = storage.folder("yahoo", as_of)
    facts = select_tags(
        read_table(storage.folder("edgar_companyfacts", as_of) / "facts.parquet")
    )
    prices = read_table(yahoo / "prices.parquet")
    dividends = read_table(yahoo / "dividends.parquet")
    observations = read_table(storage.folder("fred", as_of) / "observations.parquet")
    constituents = read_table(
        storage.folder("constituents", as_of) / "constituents.parquet"
    )
    gics = gics_map(constituents, funds=sectors)
    small = (
        prices.loc[prices["symbol"].isin(gics["symbol"]), "symbol"].nunique()
        <= SMALL_UNIVERSE
    )

    report = CheckReport()
    root = results / "market_analysis"
    day = as_of.isoformat()
    _check_edgar(facts, small, report, root / "edgar" / day)
    _check_yahoo(prices, dividends, small, report, root / "yahoo" / day)
    _check_fred(observations, report, root / "fred" / day)
    report.lines.append("SECTORS")
    sector_folder = root / "sectors" / day
    report.save(
        plots.sector_benchmarks(prices, sectors, market), sector_folder, "funds"
    )
    _check_sectors(prices, gics, market, report, sector_folder)

    tagged = gics[["symbol", "gics_sector"]]
    eps = annual_first_reported(facts, "eps_diluted").merge(
        gics[["cik", "symbol"]], on="cik"
    )
    pe = price_earnings(prices, eps, read_table(yahoo / "splits.parquet"))
    pe = pe.merge(tagged, on="symbol", how="left")
    lead_lag = plots.pe_price_lead_lag(pe, LEAD_LAG_MONTHS).merge(
        tagged, on="symbol", how="left"
    )
    metrics = storage.folder("metrics", as_of)
    write_table(pe, metrics / "pe.parquet")
    write_table(lead_lag, metrics / "pe_lead_lag.parquet")
    _check_metrics(pe, lead_lag, small, report, root / "metrics" / day)
    return report


def _check_edgar(
    facts: pd.DataFrame, small: bool, report: CheckReport, folder: Path
) -> None:
    """EDGAR: counts, the EPS cross-check, filing delays; their plots."""
    early = int((facts["filed"] < facts["period_end"]).sum())
    report.lines += [
        "EDGAR",
        f"  values {len(facts):,} after one tag per value; companies {facts['cik'].nunique()}; "
        f"fields {facts['field'].nunique()}",
        f"  periods {facts['period_end'].min().date()} to {facts['period_end'].max().date()}; "
        f"missing values {int(facts['value'].isna().sum())}; filed before period end {early}",
    ]
    annual = pd.concat(
        [
            annual_first_reported(facts, f).assign(field=f)
            for f in sorted(facts["field"].unique())
        ]
    )
    report.save(plots.edgar_coverage(annual), folder, "coverage")
    if small:
        for metric in ("revenue", "net_income", "eps_diluted"):
            report.save(plots.edgar_annual_series(annual, metric), folder, metric)

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
        report.save(plots.edgar_eps_check(wide), folder, "eps_check")

    # A filing's own period is the latest financial-statement period end it reports.
    gaap = facts[
        (facts["taxonomy"] == "us-gaap") & facts["form"].isin(["10-K", "10-Q"])
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
    report.save(plots.edgar_filing_lag(lag), folder, "filing_lag")


def _check_yahoo(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    small: bool,
    report: CheckReport,
    folder: Path,
) -> None:
    """Yahoo: duplicates, implausible prices, large moves; their plots."""
    moves = prices.sort_values("date").groupby("symbol")["adj_close"].pct_change().abs()
    large = LARGE_DAILY_MOVE["yahoo"]
    report.lines += [
        "YAHOO",
        f"  price rows {len(prices):,}; symbols {prices['symbol'].nunique()}",
        f"  duplicate (symbol, date) {int(prices.duplicated(['symbol', 'date']).sum())}; "
        f"missing close {int(prices['close'].isna().sum())}; "
        f"close <= 0 {int((prices['close'] <= 0).sum())}; "
        f"daily moves above {large:.0%} {int((moves > large).sum())}; dividends {len(dividends):,}",
    ]
    if small:
        report.save(plots.yahoo_total_return(prices), folder, "total_return")
        report.save(plots.yahoo_relative_return(prices), folder, "relative_return")
    report.save(plots.yahoo_trading_days(prices), folder, "trading_days")
    report.save(plots.yahoo_dividends(dividends), folder, "dividends")


def _check_fred(observations: pd.DataFrame, report: CheckReport, folder: Path) -> None:
    """FRED: coverage and large moves; the series and its coverage plots."""
    rates = observations.dropna(subset=["value"])
    moves = rates["value"].pct_change().abs()
    large = LARGE_DAILY_MOVE["fred"]
    report.lines += [
        "FRED",
        f"  rows {len(observations):,}; rates {len(rates):,}; "
        f"{rates['date'].min().date()} to {rates['date'].max().date()}; "
        f"daily moves above {large:.0%} {int((moves > large).sum())}",
    ]
    for series in sorted(rates["series_id"].unique()):
        report.save(plots.fred_series(rates, str(series)), folder, str(series))
        report.save(
            plots.fred_coverage(rates, str(series)), folder, f"{series}_coverage"
        )


def _check_sectors(
    prices: pd.DataFrame,
    gics: pd.DataFrame,
    market: str,
    report: CheckReport,
    folder: Path,
) -> None:
    """Per GICS sector: the members against their sector fund and the market."""
    priced = gics[gics["symbol"].isin(prices["symbol"])]
    for (sector, fund), members in priced.groupby(["gics_sector", "sector_fund"]):
        symbols = [str(s) for s in members["symbol"]]
        report.lines.append(f"  {sector}: {len(symbols)} members, fund {fund}")
        figure = plots.sector_relative(
            prices, symbols, sector=str(sector), fund=str(fund), market=market
        )
        report.save(figure, folder, _slug(str(sector)))


def _check_metrics(
    pe: pd.DataFrame,
    lead_lag: pd.DataFrame,
    small: bool,
    report: CheckReport,
    folder: Path,
) -> None:
    """P/E: counts; plots per company for a small universe, per GICS sector always."""
    report.lines += [
        "METRICS",
        f"  P/E rows {len(pe):,}; symbols {pe['symbol'].nunique()}; "
        f"missing (EPS <= 0) {int(pe['pe'].isna().sum())}",
    ]
    if small:
        report.save(plots.pe_over_time(pe), folder, "pe")
        report.save(plots.pe_eps(pe), folder, "pe_eps")
        for symbol in sorted(pe["symbol"].unique()):
            report.save(
                plots.pe_vs_price(pe, str(symbol)), folder, f"pe_vs_price_{symbol}"
            )
        report.save(plots.pe_lead_lag(lead_lag), folder, "pe_lead_lag")
    report.save(plots.pe_sector_medians(pe), folder, "pe_sector_medians")
    for sector, _ in pe.groupby("gics_sector"):
        report.save(
            plots.pe_sector_spread(pe, str(sector)), folder, f"pe_{_slug(str(sector))}"
        )
    by_sector = pd.DataFrame(
        lead_lag.groupby(["gics_sector", "lag"], as_index=False)["correlation"].mean()
    )
    report.save(
        plots.pe_lead_lag(by_sector, group="gics_sector"), folder, "pe_lead_lag_sectors"
    )


def _slug(name: str) -> str:
    """Return ``name`` as a file name: lower case, words joined by underscores."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
