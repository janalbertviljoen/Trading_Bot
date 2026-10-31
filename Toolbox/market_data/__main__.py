r"""
market_data/__main__.py
-----------------------
The terminal commands that update and check the stored market data.

Run from the repository root::

    export SEC_USER_AGENT="Your Name your@email"   # required for EDGAR
    export FRED_API_KEY=...                       # optional
    .venv/bin/python -m Toolbox.market_data update --as-of 2026-10-31 \
        --cik 320193 789019 21344 --ticker AAPL MSFT KO
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
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from Toolbox.market_data.checks import run_checks
from Toolbox.market_data.sources.edgar import edgar_client
from Toolbox.market_data.sources.fred import FredSeries, fred_client
from Toolbox.market_data.sources.prices import YahooChart, yahoo_client
from Toolbox.market_data.update import (
    Storage,
    UpdateResult,
    update_edgar,
    update_fx,
    update_prices,
)
from Toolbox.skatkis.io.config import load_config
from Toolbox.skatkis.io.snapshots import SnapshotStore

SOURCES = ("edgar", "prices", "fx")
BENCHMARK = "^SP500TR"
FX_SERIES = ("DEXSFUS",)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments and run the ``update`` or ``check`` command."""
    args = _parser().parse_args(argv)
    storage = Storage(SnapshotStore(args.data_dir / "raw"), args.data_dir / "interim")
    if args.command == "check":
        report = run_checks(args.as_of, storage=storage, results=args.results_dir)
        print("\n".join(report.lines))
        print(
            f"\n{len(report.figures)} plots under {args.results_dir / 'market_analysis'}"
        )
        return 0
    return _update(args, storage)


def _update(args: argparse.Namespace, storage: Storage) -> int:
    """Run the requested updates and print a summary."""
    config = load_config(args.config)["sources"]
    results: list[UpdateResult] = []

    if "edgar" in args.source and args.cik:
        user_agent = os.environ.get("SEC_USER_AGENT", "")
        if not user_agent:
            print(
                'EDGAR needs SEC_USER_AGENT, e.g. export SEC_USER_AGENT="Name you@email"'
            )
            return 2
        client = edgar_client(user_agent, **config["edgar"])
        results.append(
            update_edgar(args.cik, as_of=args.as_of, storage=storage, client=client)
        )
    if "prices" in args.source:
        yahoo = dict(config["yahoo"])
        start, events_start = yahoo.pop("start"), yahoo.pop("events_start")
        client = yahoo_client(**yahoo)
        results.append(
            update_prices(
                [*args.ticker, BENCHMARK],
                as_of=args.as_of,
                storage=storage,
                daily=YahooChart(client, start),
                monthly=YahooChart(client, events_start, interval="1mo"),
            )
        )
    if "fx" in args.source:
        fred = dict(config["fred"])
        start = fred.pop("start")
        provider = FredSeries(
            fred_client(**fred), start, api_key=os.environ.get("FRED_API_KEY") or None
        )
        results.append(
            update_fx(
                FX_SERIES,
                as_of=args.as_of,
                storage=storage,
                provider=provider,
            )
        )

    for result in results:
        tables = ", ".join(
            f"{name} {rows:,} rows" for name, rows in result.tables.items()
        )
        print(
            f"{result.source}: fetched {len(result.fetched)}, reused {len(result.reused)}; "
            f"{tables} -> {result.folder}"
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser of the ``update`` command."""
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
        help=f"price symbols ({BENCHMARK} is always added)",
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
    return parser


if __name__ == "__main__":
    sys.exit(main())
