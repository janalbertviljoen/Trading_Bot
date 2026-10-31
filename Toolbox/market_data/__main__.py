r"""
market_data/__main__.py
-----------------------
The terminal commands that update and check the stored market data.

Run from the repository root::

    export SEC_USER_AGENT="Your Name your@email"   # required for EDGAR
    export FRED_API_KEY=...                       # optional
    .venv/bin/python -m Toolbox.market_data update --as-of 2026-10-31 \
        --cik 320193 789019 21344 --ticker AAPL MSFT KO
    .venv/bin/python -m Toolbox.market_data update --as-of 2026-10-31 --universe sp500
    .venv/bin/python -m Toolbox.market_data check --as-of 2026-10-31

This is the command-line boundary: it reads the environment, the config file and the
arguments, and passes them explicitly to ``market_data.update``. The as-of date is
always given (``--as-of $(date +%F)`` for today), so a run never depends on the clock.

Owns
----
The command-line interface: arguments, environment variables, the summary printed.

Must not
--------
Fetch, parse or write data itself (``market_data.update``).

Public API
----------
main(argv)
    Run ``update`` or ``check``; returns the exit code.

Dependency rule
---------------
The standard library, ``skatkis.io`` and ``market_data``.
"""

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from Toolbox.market_data.checks import CheckSettings, run_checks
from Toolbox.market_data.gics import gics_map
from Toolbox.market_data.report import ReportContext, build_report
from Toolbox.market_data.sources.constituents import WikipediaPage, wikipedia_client
from Toolbox.market_data.sources.edgar import edgar_client
from Toolbox.market_data.sources.fred import FredSeries, fred_client
from Toolbox.market_data.sources.prices import YahooChart, yahoo_client
from Toolbox.market_data.update import (
    Storage,
    UpdateResult,
    update_constituents,
    update_edgar,
    update_fx,
    update_prices,
)
from Toolbox.skatkis.io.config import load_config
from Toolbox.skatkis.io.snapshots import SnapshotStore
from Toolbox.skatkis.io.tables import read_table
from Toolbox.skatkis.report import compile_pdf

SOURCES = ("edgar", "prices", "fx")
UNIVERSES = ("sp500",)
LATEX_ENGINES = ("tectonic", "latexmk", "pdflatex")
FX_SERIES = ("DEXSFUS",)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments and run the ``update`` or ``check`` command."""
    args = _parser().parse_args(argv)
    storage = Storage(SnapshotStore(args.data_dir / "raw"), args.data_dir / "interim")
    if args.command == "check":
        return _check(args, storage)
    return _update(args, storage)


def _check(args: argparse.Namespace, storage: Storage) -> int:
    """Write the check plots and the report; compile the report if an engine is found."""
    config = load_config(args.config)
    settings = CheckSettings(
        market=config["benchmarks"]["market"],
        sectors=config["benchmarks"]["sectors"],
        excluded=config["universe"]["excluded_sectors"],
        **config.get("analysis", {}),
    )
    report = run_checks(
        args.as_of, storage=storage, results=args.results_dir, settings=settings
    )
    print("\n".join(report.lines))
    tex = build_report(
        ReportContext(report.plots, report.facts, report.sectors, report.tickers),
        load_config(args.report_text),
        folder=report.folder,
        subtitle=f"Data as of {args.as_of.isoformat()}",
    )
    print(f"\n{len(report.plots)} plots and {tex.name} in {report.folder}")
    engine = next((e for e in LATEX_ENGINES if shutil.which(e)), None)
    if engine is None:
        print(
            "No LaTeX engine found (tectonic, latexmk, pdflatex): report.tex not compiled."
        )
        print("Install one, e.g. `brew install tectonic`, and run check again.")
        return 0
    try:
        pdf = compile_pdf(
            tex,
            engine=str(shutil.which(engine)),
            cache=args.data_dir / "cache" / "tectonic",
        )
    except subprocess.CalledProcessError as error:
        detail = (
            (error.stderr or b"").decode(errors="replace").strip().splitlines()[-1:]
        )
        print(f"LaTeX ({engine}) failed: {' '.join(detail)}; report.tex is kept.")
        return 1
    print(f"Compiled {pdf}")
    return 0


def _update(args: argparse.Namespace, storage: Storage) -> int:
    """Run the requested updates and print a summary."""
    config = load_config(args.config)
    sources = config["sources"]
    benchmarks = [
        config["benchmarks"]["market"],
        *config["benchmarks"]["sectors"].values(),
        *config["benchmarks"].get("strategies", []),
    ]
    results: list[UpdateResult] = []
    ciks: list[int | str] = list(args.cik)
    tickers: list[str] = list(args.ticker)

    if args.universe == "sp500":
        wiki = WikipediaPage(wikipedia_client(**sources["wikipedia"]))
        members = update_constituents(args.as_of, storage=storage, provider=wiki)
        results.append(members)
        gics = gics_map(
            read_table(members.folder / "constituents.parquet"),
            funds=config["benchmarks"]["sectors"],
            excluded=config["universe"]["excluded_sectors"],
        )
        ciks += [int(cik) for cik in gics["cik"]]
        tickers += [str(symbol) for symbol in gics["symbol"]]

    if "edgar" in args.source and ciks:
        user_agent = os.environ.get("SEC_USER_AGENT", "")
        if not user_agent:
            print(
                'EDGAR needs SEC_USER_AGENT, e.g. export SEC_USER_AGENT="Name you@email"'
            )
            return 2
        client = edgar_client(user_agent, **sources["edgar"])
        results.append(
            update_edgar(ciks, as_of=args.as_of, storage=storage, client=client)
        )
    if "prices" in args.source:
        yahoo = dict(sources["yahoo"])
        start, events_start = yahoo.pop("start"), yahoo.pop("events_start")
        client = yahoo_client(**yahoo)
        results.append(
            update_prices(
                list(dict.fromkeys([*tickers, *benchmarks])),
                as_of=args.as_of,
                storage=storage,
                daily=YahooChart(client, start),
                monthly=YahooChart(client, events_start, interval="1mo"),
            )
        )
    if "fx" in args.source:
        fred = dict(sources["fred"])
        start = fred.pop("start")
        provider = FredSeries(
            fred_client(**fred), start, api_key=os.environ.get("FRED_API_KEY") or None
        )
        results.append(
            update_fx(FX_SERIES, as_of=args.as_of, storage=storage, provider=provider)
        )

    for result in results:
        tables = ", ".join(
            f"{name} {rows:,} rows" for name, rows in result.tables.items()
        )
        print(
            f"{result.source}: fetched {len(result.fetched)}, reused {len(result.reused)}, "
            f"failed {len(result.failed)}; {tables} -> {result.folder}"
        )
        for query, reason in list(result.failed.items())[:10]:
            print(f"  failed {query}: {reason[:160]}")
    return 0


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser of the ``update`` and ``check`` commands."""
    parser = argparse.ArgumentParser(prog="python -m Toolbox.market_data")
    commands = parser.add_subparsers(dest="command", required=True)
    update = commands.add_parser(
        "update", help="bring the stored data up to an as-of date"
    )
    update.add_argument(
        "--as-of", required=True, type=date.fromisoformat, help="YYYY-MM-DD"
    )
    update.add_argument("--source", nargs="+", choices=SOURCES, default=list(SOURCES))
    update.add_argument("--cik", nargs="*", default=[], help="EDGAR company CIKs")
    update.add_argument(
        "--ticker",
        nargs="*",
        default=[],
        help="price symbols (the market, sector and strategy benchmarks are always added)",
    )
    update.add_argument(
        "--universe",
        choices=UNIVERSES,
        help="add the index members (outside the excluded sectors) to --cik and --ticker",
    )
    update.add_argument("--data-dir", type=Path, default=Path("data"))
    update.add_argument("--config", type=Path, default=Path("config/market_data.toml"))
    check = commands.add_parser(
        "check", help="plot and check the data stored for an as-of date"
    )
    check.add_argument(
        "--as-of", required=True, type=date.fromisoformat, help="YYYY-MM-DD"
    )
    check.add_argument("--data-dir", type=Path, default=Path("data"))
    check.add_argument("--results-dir", type=Path, default=Path("results"))
    check.add_argument("--config", type=Path, default=Path("config/market_data.toml"))
    check.add_argument(
        "--report-text", type=Path, default=Path("config/report_market_analysis.toml")
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
