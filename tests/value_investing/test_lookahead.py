"""Truncation test of the scores on the stored data: no forward information.

Runs only where the stored tables of ``SNAPSHOT`` exist (``data/`` is not in git); it
scores the companies again with every input cut at each date and compares.
"""

from pathlib import Path

import pandas as pd
import pytest

from Toolbox.market_data.gics import gics_map
from Toolbox.market_data.sources.prices import yahoo_symbol
from Toolbox.skatkis.io.config import load_config
from Toolbox.skatkis.io.tables import read_table
from Toolbox.value_investing.lookahead import truncation_check
from Toolbox.value_investing.pipeline import Tables

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = "2026-10-31"
INTERIM = REPO / "data" / "interim"
FULL = INTERIM / "value_investing" / SNAPSHOT / "buyability_on_date.parquet"
DATES = [
    pd.Timestamp("2018-12-21"),
    pd.Timestamp("2022-12-16"),
    pd.Timestamp("2025-12-19"),
]
COLUMNS = ["score_binned_weighted", "score_weighted_share", "score_binned", "score"]

pytestmark = pytest.mark.skipif(
    not FULL.exists(), reason=f"stored data of {SNAPSHOT} not present"
)


@pytest.fixture(scope="module")
def check() -> pd.DataFrame:
    rules = load_config(REPO / "config" / "rules.toml")
    market = load_config(REPO / "config" / "market_data.toml")
    constituents = read_table(
        INTERIM / "constituents" / SNAPSHOT / "constituents.parquet"
    )
    added = pd.DataFrame(
        {
            "symbol": [yahoo_symbol(str(s)) for s in constituents["symbol"]],
            "date_added": constituents["date_added"].to_numpy(),
        }
    ).astype({"symbol": "string"})
    universe = gics_map(
        constituents,
        funds=market["benchmarks"]["sectors"],
        excluded=rules["universe"]["excluded_sectors"],
    ).merge(added, on="symbol", how="left")
    yahoo = INTERIM / "yahoo" / SNAPSHOT
    tables = Tables(
        facts=read_table(INTERIM / "edgar_companyfacts" / SNAPSHOT / "facts.parquet"),
        prices=read_table(yahoo / "prices.parquet"),
        dividends=read_table(yahoo / "dividends.parquet"),
        splits=read_table(yahoo / "splits.parquet"),
        universe=universe,
    )
    return truncation_check(
        tables,
        rules,
        read_table(FULL),
        DATES,
        view="on_date",
        benchmark=str(market["benchmarks"]["market"]),
        columns=COLUMNS,
        top=int(rules["wallet"]["names"]),
    )


def test_the_picks_are_the_same_with_the_inputs_cut_at_the_date(
    check: pd.DataFrame,
) -> None:
    assert check["same_top"].all()


@pytest.mark.xfail(
    strict=True,
    reason="known leak: select_tags ranks tags by their use in later filings (a few "
    "companies on 2018-12-21 and 2022-12-16); fix pending the user's approval "
    "(MD_202610_002). Remove this mark with the fix.",
)
def test_no_score_changes_with_the_inputs_cut_at_the_date(check: pd.DataFrame) -> None:
    differ = [c for c in check.columns if c.startswith("differ_")]
    assert (check[differ] == 0).all().all()
