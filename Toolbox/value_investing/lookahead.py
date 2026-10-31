"""
value_investing/lookahead.py
----------------------------
A check that the scores use no forward information: cut every input at a date and
score again.

On a date t, a score must depend only on what was known on t. ``truncation_check`` keeps
the EDGAR facts filed on or before t, the prices up to t and the dividends with an
ex-date up to t, scores the companies again with ``value_investing.pipeline`` and
compares the scores on t with those of the full run. Any difference means some input
from after t reached the score. The splits table is kept whole: it only puts per-share
figures on the basis of the split-adjusted prices, which are themselves adjusted for
every later split, so cutting it would mix two bases. The check cannot see biases in
the inputs themselves, such as a universe of today's index members.

Owns
----
The truncation check and its comparison table.

Must not
--------
Score companies itself (``value_investing.pipeline``); read files
(``value_investing.__main__``).

Public API
----------
truncation_check(tables, config, full, dates, *, view, benchmark, columns, top)
    Per date, the companies whose scores differ between the cut and the full run.

Dependency rule
---------------
pandas and ``value_investing``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pandas as pd

from Toolbox.value_investing.pipeline import Tables, buyability

TOLERANCE = 1e-9


def truncation_check(  # noqa: PLR0913  one keyword per setting of the check
    tables: Tables,
    config: Mapping[str, Any],
    full: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    *,
    view: str,
    benchmark: str,
    columns: Sequence[str],
    top: int,
) -> pd.DataFrame:
    """Return, per date, how many scores change when the inputs are cut at the date.

    Parameters
    ----------
    tables : Tables
        The stored tables, as the full run used them.
    config : Mapping[str, Any]
        ``config/rules.toml``.
    full : pd.DataFrame
        The full run's buyability table (``symbol``, ``date`` and ``columns``).
    dates : Sequence[pd.Timestamp]
        Rebalance dates to check.
    view : str
        The universe view of ``full``.
    benchmark : str
        Total-return benchmark of the run.
    columns : Sequence[str]
        The scores compared.
    top : int
        Size of the top group whose membership is compared (the picks).

    Returns
    -------
    pd.DataFrame
        One row per date: ``date``, ``companies``, ``differ_<column>`` (companies whose
        score differs by more than ``TOLERANCE`` or is missing in one run only) per
        column, and ``same_top`` (whether the ``top`` highest of the first column are
        the same companies). All differences 0 means no forward information reached
        those scores on that date.
    """
    rows: list[dict[str, object]] = []
    for value in dates:
        day = pd.Timestamp(value)
        cut = replace(
            tables,
            facts=tables.facts[tables.facts["filed"] <= day],
            prices=tables.prices[tables.prices["date"] <= day],
            dividends=tables.dividends[tables.dividends["ex_date"] <= day],
        )
        part = buyability(cut, config, end=day, view=view, benchmark=benchmark).table
        before = full.loc[full["date"] == day].set_index("symbol")[list(columns)]
        after = part.loc[part["date"] == day].set_index("symbol")[list(columns)]
        both = before.join(after, rsuffix="_cut", how="outer")
        row: dict[str, object] = {"date": day, "companies": len(both)}
        for column in columns:
            gap = (both[column] - both[f"{column}_cut"]).abs() > TOLERANCE
            one_side = both[column].isna() != both[f"{column}_cut"].isna()
            row[f"differ_{column}"] = int((gap | one_side).sum())
        first = columns[0]
        row["same_top"] = set(before[first].nlargest(top).index) == set(
            after[first].nlargest(top).index
        )
        rows.append(row)
    return pd.DataFrame(rows)
