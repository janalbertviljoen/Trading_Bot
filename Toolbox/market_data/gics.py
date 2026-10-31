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
The company-to-sector map and the choice of companies (excluded sectors).

Must not
--------
Fetch or parse sources (``market_data.sources``); plot (``market_data.market_analysis``).

Public API
----------
gics_map(constituents, *, funds, excluded)
    One row per company: ``symbol``, ``cik``, ``gics_sector``, ``sector_fund``.

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
