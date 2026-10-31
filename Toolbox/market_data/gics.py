"""
market_data/gics.py
-------------------
The GICS map: each company tagged with its GICS sector and that sector's benchmark.

Like ``sources/tag_map.py`` for XBRL tags, this is one mapping table, built from the
constituents source: one row per company with its price symbol, CIK, GICS sector and the
sector fund it is compared with. Analysis merges it onto any table (prices, P/E) and
groups by ``gics_sector``; every group is normalised the same way, only against its own
``sector_fund``.

Owns
----
The company-to-sector map, the choice of companies (excluded sectors) and the date a
member joined its index where the members page does not give it.

Must not
--------
Fetch or parse sources (``market_data.sources``); plot (``market_data.market_analysis``).

Public API
----------
gics_map(constituents, *, funds, excluded)
    One row per company: ``symbol``, ``cik``, ``gics_sector``, ``sector_fund``.
fill_date_added(constituents, changes)
    The members table with each missing ``date_added`` taken from the changes.

Dependency rule
---------------
pandas, the standard library and ``market_data.sources.prices`` (the symbol convention).
"""

from collections.abc import Collection, Mapping

import pandas as pd

from Toolbox.market_data.sources.prices import yahoo_symbol


def gics_map(
    constituents: pd.DataFrame,
    *,
    funds: Mapping[str, str],
    excluded: Collection[str] = (),
) -> pd.DataFrame:
    """Return the GICS map: each member with its sector and sector fund.

    Parameters
    ----------
    constituents : pd.DataFrame
        The constituents table (``CONSTITUENT_COLUMNS``); not modified.
    funds : Mapping[str, str]
        GICS sector name to its benchmark symbol (``config/market_data.toml``).
    excluded : Collection[str]
        GICS sectors left out (Financials and Real Estate for the stock universe).

    Returns
    -------
    pd.DataFrame
        Columns ``symbol`` (Yahoo convention, ``BRK.B`` becomes ``BRK-B``), ``cik``,
        ``gics_sector``, ``gics_sub_industry``, ``sector_fund``; members without a CIK or
        in an excluded sector are left out. Sorted by symbol.
    """
    kept = constituents[
        constituents["cik"].notna() & ~constituents["gics_sector"].isin(list(excluded))
    ]
    mapping = pd.DataFrame(
        {
            "symbol": [yahoo_symbol(str(s)) for s in kept["symbol"]],
            "cik": kept["cik"].astype("int64").to_numpy(),
            "gics_sector": kept["gics_sector"].to_numpy(),
            "gics_sub_industry": kept["gics_sub_industry"].to_numpy(),
        }
    )
    mapping["sector_fund"] = mapping["gics_sector"].map(dict(funds))
    return mapping.astype({"symbol": "string", "gics_sector": "string"}).sort_values(
        "symbol", ignore_index=True
    )


def fill_date_added(constituents: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """Return the members with each missing ``date_added`` taken from the index changes.

    Parameters
    ----------
    constituents : pd.DataFrame
        The constituents table (``CONSTITUENT_COLUMNS``); not modified.
    changes : pd.DataFrame
        The index's changes (``CHANGE_COLUMNS``).

    Returns
    -------
    pd.DataFrame
        The same table. A member without ``date_added`` gets the date of its latest
        addition in ``changes``; a member never added there gets the earliest date of
        ``changes``, as it joined before the record starts. A given date is kept.

    Notes
    -----
    The S&P 400 and 600 pages list only selected changes, so a member that joined
    after the record starts but whose addition is missing is taken as a member from
    the record's start: a look-ahead for that company.
    """
    added = (
        changes.dropna(subset=["added_symbol"]).groupby("added_symbol")["date"].max()
    )
    start = changes["date"].min()
    derived = pd.Series(
        added.reindex(constituents["symbol"].astype(str)).to_numpy(),
        index=constituents.index,
    ).fillna(start)
    return constituents.assign(date_added=constituents["date_added"].fillna(derived))
