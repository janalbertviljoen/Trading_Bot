"""Truncation test of the scores on the stored data: no forward information.

Runs only where the stored tables of ``SNAPSHOT`` exist (``data/`` is not in git); it
scores the companies again with every input cut at each date and compares.
"""

from pathlib import Path

import pandas as pd
import pytest

from Toolbox.skatkis.io.config import load_config
from Toolbox.skatkis.io.tables import read_table
from Toolbox.value_investing.inputs import index_universe
from Toolbox.value_investing.lookahead import truncation_check
from Toolbox.value_investing.pipeline import Tables

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = "2026-10-31"
INTERIM = REPO / "data" / "interim"
INDEX = "sp500"
MEMBERS = INTERIM / "constituents" / SNAPSHOT / INDEX
FULL = INTERIM / "value_investing" / SNAPSHOT / INDEX / "buyability_on_date.parquet"
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
    universe = index_universe(
        read_table(MEMBERS / "constituents.parquet"),
        read_table(MEMBERS / "changes.parquet"),
        funds=market["benchmarks"]["sectors"],
        excluded=rules["universe"]["excluded_sectors"],
    )
    yahoo = INTERIM / "yahoo" / SNAPSHOT
    tables = Tables(
        facts=read_table(INTERIM / "edgar_companyfacts" / SNAPSHOT / "facts.parquet"),
        prices=read_table(yahoo / "prices.parquet"),
        dividends=read_table(yahoo / "dividends.parquet"),
        splits=read_table(yahoo / "splits.parquet"),
        listings=read_table(yahoo / "listings.parquet"),
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


def test_no_score_changes_with_the_inputs_cut_at_the_date(check: pd.DataFrame) -> None:
    differ = [c for c in check.columns if c.startswith("differ_")]
    assert (check[differ] == 0).all().all()
