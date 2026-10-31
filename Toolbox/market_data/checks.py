"""
market_data/checks.py
---------------------
Checks of the stored data, as numbers and plots, and the first metric plots.

``run_checks`` reads the tables the update wrote for an as-of date, prints numeric checks
(missing values, duplicates, implausible values, coverage and filing delays) and writes
the named plots of ``market_data.market_analysis``: one folder per source under
``<results>/market_analysis/<source>/<as_of>/`` (``edgar``, ``yahoo``, ``fred``) and the
metric plots under ``.../metrics/<as_of>/``. The P/E table and its lead-lag table are
stored under ``<interim>/metrics/<as_of>/``. The terminal command
``python -m Toolbox.market_data check`` calls it.

Owns
----
Which numeric checks describe each source, which plots are written, and the layout of the
result folders.

Must not
--------
Define plots (``market_data.market_analysis``); compute metrics (``market_data.metrics``)
or clean data (``market_data.clean``).

Public API
----------
run_checks(as_of, *, storage, results)
    Write the check and metric plots; return a ``CheckReport``.
CheckReport
    The numeric checks as lines of text and the paths of the plots written.

Dependency rule
---------------
pandas, the standard library, ``skatkis.io`` and ``market_data``.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from Toolbox.market_data import market_analysis as plots
from Toolbox.market_data.clean import annual_first_reported, select_tags
from Toolbox.market_data.metrics import price_earnings
from Toolbox.market_data.sources.edgar import format_cik, parse_tickers
from Toolbox.market_data.update import Storage
from Toolbox.skatkis.io.figures import save_figure
from Toolbox.skatkis.io.snapshots import read_snapshot
from Toolbox.skatkis.io.tables import read_table, write_table

EPS_TOLERANCE = 0.05
LARGE_DAILY_MOVE = {"yahoo": 0.20, "fred": 0.05}
LEAD_LAG_MONTHS = 12


@dataclass
class CheckReport:
    """The numeric checks as text lines, and the plots written."""

    lines: list[str] = field(default_factory=list[str])
    figures: list[Path] = field(default_factory=list[Path])

    def save(self, figure: Figure, folder: Path, name: str) -> None:
        """Save ``figure`` as ``<folder>/<name>.png`` and record its path."""
        self.figures.append(save_figure(figure, folder / f"{name}.png"))


def run_checks(as_of: date, *, storage: Storage, results: Path) -> CheckReport:
    """Check the tables stored for ``as_of`` and write the plots.

    Parameters
    ----------
    as_of : date
        The as-of date whose tables are checked.
    storage : Storage
        Snapshot and table roots.
    results : Path
        Root of the result folders (``results``).

    Returns
    -------
    CheckReport
        Numeric checks per source and the paths of the plots written.
    """
    report = CheckReport()
    root = results / "market_analysis"
    day = as_of.isoformat()
    facts = select_tags(
        read_table(storage.folder("edgar_companyfacts", as_of) / "facts.parquet")
    )
    _check_edgar(facts, report, root / "edgar" / day)
    yahoo = storage.folder("yahoo", as_of)
    prices = read_table(yahoo / "prices.parquet")
    splits = read_table(yahoo / "splits.parquet")
    dividends = read_table(yahoo / "dividends.parquet")
    _check_yahoo(prices, dividends, splits, report, root / "yahoo" / day)
    observations = read_table(storage.folder("fred", as_of) / "observations.parquet")
    _check_fred(observations, report, root / "fred" / day)
    pe, lead_lag = _metric_tables(facts, prices, splits, storage, as_of)
    _check_metrics(pe, lead_lag, report, root / "metrics" / day)
    return report


def _check_edgar(facts: pd.DataFrame, report: CheckReport, folder: Path) -> None:
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
    for cik, part in annual.groupby("cik"):
        entity = str(part["entity_name"].iloc[0])
        report.save(plots.edgar_coverage(annual, entity), folder, f"coverage_{cik}")
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
    splits: pd.DataFrame,
    report: CheckReport,
    folder: Path,
) -> None:
    """Yahoo: duplicates, implausible prices, large moves; their plots."""
    moves = prices.sort_values("date").groupby("symbol")["adj_close"].pct_change().abs()
    large = LARGE_DAILY_MOVE["yahoo"]
    report.lines += [
        "YAHOO",
        f"  price rows {len(prices):,}; symbols {', '.join(sorted(prices['symbol'].unique()))}",
        f"  duplicate (symbol, date) {int(prices.duplicated(['symbol', 'date']).sum())}; "
        f"missing close {int(prices['close'].isna().sum())}; "
        f"close <= 0 {int((prices['close'] <= 0).sum())}; "
        f"daily moves above {large:.0%} {int((moves > large).sum())}",
        f"  dividends {len(dividends):,}; splits {len(splits)}",
    ]
    report.save(plots.yahoo_total_return(prices), folder, "total_return")
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


def _metric_tables(
    facts: pd.DataFrame,
    prices: pd.DataFrame,
    splits: pd.DataFrame,
    storage: Storage,
    as_of: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute and store the P/E table and its lead-lag table against price."""
    eps = annual_first_reported(facts, "eps_diluted")
    symbols = {cik: _ticker(storage, int(cik), as_of) for cik in eps["cik"].unique()}
    eps = eps.assign(symbol=eps["cik"].map(symbols)).dropna(subset=["symbol"])
    pe = price_earnings(prices, eps, splits)
    lead_lag = plots.pe_price_lead_lag(pe, LEAD_LAG_MONTHS)
    tables = storage.folder("metrics", as_of)
    write_table(pe, tables / "pe.parquet")
    write_table(lead_lag, tables / "pe_lead_lag.parquet")
    return pe, lead_lag


def _check_metrics(
    pe: pd.DataFrame, lead_lag: pd.DataFrame, report: CheckReport, folder: Path
) -> None:
    """P/E: counts and the strongest lead or lag against price; their plots."""
    report.lines += [
        "METRICS",
        f"  P/E rows {len(pe):,}; symbols {', '.join(sorted(pe['symbol'].unique()))}; "
        f"missing (EPS <= 0) {int(pe['pe'].isna().sum())}",
    ]
    for symbol, part in lead_lag.groupby("symbol"):
        lag0 = float(part.loc[part["lag"] == 0, "correlation"].iloc[0])
        others = part[part["lag"] != 0].sort_values(
            "correlation", key=abs, ascending=False
        )
        best_lag, best = int(others["lag"].iloc[0]), float(
            others["correlation"].iloc[0]
        )
        report.lines.append(
            f"  {symbol}: P/E vs price changes, lag 0 {lag0:.2f}; strongest other lag "
            f"{best_lag:+d} months ({best:.2f})"
        )
    report.save(plots.pe_over_time(pe), folder, "pe")
    report.save(plots.pe_eps(pe), folder, "pe_eps")
    for symbol in sorted(pe["symbol"].unique()):
        report.save(plots.pe_vs_price(pe, str(symbol)), folder, f"pe_vs_price_{symbol}")
    report.save(plots.pe_lead_lag(lead_lag), folder, "pe_lead_lag")


def _ticker(storage: Storage, cik: int, as_of: date) -> str | None:
    """Return the company's current first ticker, from its stored ``submissions`` snapshot."""
    query = format_cik(cik)
    stored = storage.snapshots.latest_as_of(
        "edgar_submissions", query, on_or_before=as_of
    )
    if stored is None:
        return None
    tickers = parse_tickers(
        read_snapshot(storage.snapshots, "edgar_submissions", query, stored)
    )
    return tickers[0] if tickers else None
