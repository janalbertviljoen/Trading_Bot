"""
market_data/fundamentals.py
---------------------------
Annual fundamentals as known on a date: the first slice of the point-in-time panel.

``annual_series`` gives, per company and field, the first-reported 10-K value of each
fiscal year with its filing date (``market_data.clean``). ``annual_as_of`` then answers,
for each date, what the company had filed by that date: the latest fiscal year (lag 0)
and the years before it (lag 1, 2, ...). A value counts from its filing date, same-day
filings included, so a date never sees a later filing or a restatement. Flows are annual
figures and balance-sheet values are fiscal-year-end figures; trailing twelve months and
latest-quarter values follow with the full panel (MD_202610_002).

Owns
----
The point-in-time alignment of annual fundamentals: which fiscal years a date sees, and
their lag.

Must not
--------
Choose tags or first-reported values (``market_data.clean``); adjust for splits
(``market_data.metrics``); apply rule thresholds (the tools).

Public API
----------
annual_series(facts, fields)
    First-reported annual values of several fields, in one long table.
annual_as_of(series, dates, *, depth, ...)
    Per date, the last ``depth`` fiscal years filed by that date, with their lag.

Dependency rule
---------------
pandas and ``market_data.clean``.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from Toolbox.market_data.clean import annual_first_reported, select_tags

DAYS_PER_YEAR = 365.25
YEAR_TOLERANCE_DAYS = 45
STALE_DAYS = 550
AS_OF_COLUMNS = ["cik", "field", "date", "lag", "period_end", "filed", "value"]


def annual_series(facts: pd.DataFrame, fields: Sequence[str]) -> pd.DataFrame:
    """Return the first-reported annual values of ``fields`` in one long table.

    Parameters
    ----------
    facts : pd.DataFrame
        The ``facts`` table; not modified.
    fields : Sequence[str]
        Canonical fields (``tag_map.TAG_MAP``).

    Returns
    -------
    pd.DataFrame
        Columns ``cik``, ``field``, ``period_end``, ``filed``, ``value``; one row per
        company, field and fiscal year end.
    """
    selected = select_tags(facts[facts["field"].isin(list(fields))])
    frames = [
        annual_first_reported(selected, field).assign(field=field) for field in fields
    ]
    columns = ["cik", "field", "period_end", "filed", "value"]
    return pd.concat(frames, ignore_index=True)[columns].astype({"field": "string"})


def annual_as_of(
    series: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    *,
    depth: int,
    tolerance_days: int = YEAR_TOLERANCE_DAYS,
    stale_days: int = STALE_DAYS,
) -> pd.DataFrame:
    """Return, per date, the last ``depth`` fiscal years each company had filed by then.

    Parameters
    ----------
    series : pd.DataFrame
        ``annual_series`` output; not modified.
    dates : Sequence[pd.Timestamp]
        The dates to align on (rebalance dates).
    depth : int
        Number of fiscal years kept per date: lags 0 to ``depth - 1``.
    tolerance_days : int
        How far a fiscal year end may lie from whole years before the latest one.
    stale_days : int
        Oldest the latest fiscal year end may be on the date; older histories are dropped.

    Returns
    -------
    pd.DataFrame
        Columns ``cik``, ``field``, ``date``, ``lag`` (0 is the latest fiscal year),
        ``period_end``, ``filed``, ``value``.

    Notes
    -----
    A value is seen on date t when filed <= t. The history is cut at the first gap: lag k
    is kept only if its period end lies within ``tolerance_days`` of k years before
    lag 0's and every lag before it was kept, so a missing fiscal year shortens the
    history instead of shifting older years into its place. A company whose latest
    fiscal year ended more than ``stale_days`` before t has no row on t.
    """
    frames: list[pd.DataFrame] = []
    keys = ["cik", "field"]
    for day in dates:
        known = series[series["filed"] <= day].sort_values(
            [*keys, "period_end"], ascending=[True, True, False]
        )
        lag = known.groupby(keys).cumcount()
        known = known.assign(lag=lag.to_numpy())[lag.to_numpy() < depth]
        latest = known.groupby(keys)["period_end"].transform("max")
        offset = (latest - known["period_end"]).dt.days - known["lag"] * DAYS_PER_YEAR
        in_step = np.abs(offset.to_numpy(dtype=float)) <= tolerance_days
        # Once a lag is out of step, every older lag of that company and field is too.
        broken = (
            pd.Series(~in_step, index=known.index).groupby(
                [known["cik"], known["field"]]
            )
        ).cummax()
        fresh = (day - latest).dt.days <= stale_days
        keep = pd.Series(~broken.to_numpy() & fresh.to_numpy(), index=known.index)
        kept = known.loc[keep]
        frames.append(kept.assign(date=day))
    if not frames:
        return pd.DataFrame(columns=AS_OF_COLUMNS)
    return pd.concat(frames, ignore_index=True)[AS_OF_COLUMNS]
