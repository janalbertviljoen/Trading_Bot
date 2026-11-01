"""The report's tier, worst-list and static tables (value_investing/report.py)."""

from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from Toolbox.value_investing.report import (
    ReportInputs,
    _static_table,
    _tier,
    _worst_table,
)


@pytest.mark.parametrize(
    ("share", "threshold", "ratio", "expected"),
    [
        (-0.1, 0.4, 0.5, "Red"),
        (0.0, 0.4, 0.5, "Red"),
        (0.1999, 0.4, 0.5, "Red"),
        (0.2, 0.4, 0.5, "Amber"),  # exactly threshold * ratio is not Red
        (0.3, 0.4, 0.5, "Amber"),
        (0.4, 0.4, 0.5, "Amber"),  # exactly the threshold is not Green
        (0.4001, 0.4, 0.5, "Green"),
        (1.0, 0.4, 0.5, "Green"),
        (0.0, 0.3, 0.25, "Red"),
        (0.074, 0.3, 0.25, "Red"),
        (0.075, 0.3, 0.25, "Amber"),
        (0.3, 0.3, 0.25, "Amber"),
        (0.31, 0.3, 0.25, "Green"),
    ],
)
def test_tier_boundaries(
    share: float, threshold: float, ratio: float, expected: str
) -> None:
    assert _tier(share, threshold, ratio) == expected


def _inputs(worst: pd.DataFrame) -> ReportInputs:
    # ReportInputs has 28 fields (BuyAndHold, Managed, ...); _worst_table reads only
    # worst, names, threshold and tier_ratio, so a stand-in carries those.
    stand_in = SimpleNamespace(
        worst=worst,
        names={"AAA": "Alpha Corp", "BBB": "Beta Inc"},
        threshold=0.4,
        tier_ratio=0.5,
    )
    return cast(ReportInputs, cast(Any, stand_in))


def test_worst_table_rows_follow_the_input_order_with_tiers() -> None:
    worst = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "gics_sector": ["Energy", "Utilities"],
            "score": [1.234, 5.0],
            "score_share": [0.0, 0.3],
            "score_date": [pd.Timestamp("2026-09-18")] * 2,
        }
    )
    table = _worst_table(_inputs(worst))
    assert [r[1] for r in table.rows] == ["Energy", "Utilities"]
    assert [r[2] for r in table.rows] == ["1.2", "5.0"]
    assert [r[3] for r in table.rows] == ["0.00", "0.30"]
    assert [r[4] for r in table.rows] == ["Red", "Amber"]
    assert "Alpha" in table.rows[0][0]
    assert len(table.header) == len(table.rows[0]) == 5
    assert len(table.columns.split()) == 5


def test_worst_table_empty_gives_a_none_row() -> None:
    worst = pd.DataFrame(columns=["symbol", "gics_sector", "score", "score_share"])
    table = _worst_table(_inputs(worst))
    assert table.rows == [["none", "", "", "", ""]]


def test_static_table_casts_cells_to_strings() -> None:
    spec = {
        "header": ["Year", "Excess"],
        "rows": [[2020, 1.5], ["2021", None]],
        "columns": "l r",
    }
    table = _static_table(spec, "A caption", "tab:x")
    assert table.header == ["Year", "Excess"]
    assert table.rows == [["2020", "1.5"], ["2021", "None"]]
    assert table.columns == "l r"
    assert table.caption == "A caption"
    assert table.label == "tab:x"
