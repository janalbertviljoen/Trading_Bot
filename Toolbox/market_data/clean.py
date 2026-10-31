"""
market_data/clean.py
--------------------
First cleaning steps for EDGAR facts: one tag per value, and first-reported annual values.

Companies report a canonical field (``tag_map.TAG_MAP``) under different XBRL tags over
time, and sometimes under two tags in the same filing. ``select_tags`` keeps one value
per company, field, period and filing: the one reported under the tag the company used
most recently, so each field follows the company's current tag. ``annual_first_reported``
then gives, per company and fiscal year, the value as first filed in a 10-K.

Owns
----
The tag choice per filing value and the first-reported annual series.

Must not
--------
Fetch or parse payloads (``market_data.sources``); compute ratios (``market_data.metrics``);
adjust for splits (``market_data.metrics``).

Public API
----------
select_tags(facts)
    One value per company, field, period and filing, under the most recently used tag.
annual_first_reported(facts, field)
    Per company and fiscal year: the value first reported in a 10-K, with its filing date.

Dependency rule
---------------
pandas.
"""

import pandas as pd

ANNUAL_DAYS = (350, 380)
_VALUE_KEYS = ["cik", "field", "period_start", "period_end", "accession"]


def select_tags(facts: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per company, field, period and filing: the most recently used tag's.

    Parameters
    ----------
    facts : pd.DataFrame
        The ``facts`` table (columns of ``COMPANYFACTS_COLUMNS``); not modified.

    Returns
    -------
    pd.DataFrame
        The same columns, with at most one row per ``cik``, ``field``, ``period_start``,
        ``period_end`` and ``accession``, sorted like the input.

    Notes
    -----
    A tag's recency is the latest ``filed`` date of any of the company's values under
    it, so the company's current tag ranks first; ties go to the lower
    ``tag_priority``. Restatements in later filings are kept: they have another
    accession.
    """
    last_used = facts.groupby(["cik", "field", "tag"])["filed"].transform("max")
    ranked = facts.assign(_last_used=last_used).sort_values(
        ["_last_used", "tag_priority"], ascending=[False, True], kind="stable"
    )
    kept = ranked.drop_duplicates(_VALUE_KEYS).drop(columns="_last_used")
    return kept.sort_index()


def annual_first_reported(facts: pd.DataFrame, field: str) -> pd.DataFrame:
    """Return the first-reported 10-K value of ``field`` per company and fiscal year.

    Parameters
    ----------
    facts : pd.DataFrame
        The ``facts`` table, normally after ``select_tags``; not modified.
    field : str
        Canonical field, e.g. ``"eps_diluted"``.

    Returns
    -------
    pd.DataFrame
        Columns ``cik``, ``entity_name``, ``period_end``, ``filed``, ``value``; one row
        per company and period end, sorted by ``cik`` and ``period_end``.

    Notes
    -----
    Flows (with a ``period_start``) count as annual when they span 350 to 380 days;
    balance-sheet values (instants) are taken at the fiscal year end. Of the 10-K rows
    for a period, the earliest filed is kept, so comparatives and restatements in later
    10-Ks are ignored. Per-share values stay on the share basis of their filing date.
    """
    rows = facts[(facts["field"] == field) & (facts["form"] == "10-K")]
    span = rows["period_end"] - rows["period_start"]
    days = pd.Series(span.dt.days, index=rows.index, dtype="float64")
    low, high = ANNUAL_DAYS
    is_annual = rows["period_start"].isna() | days.between(low, high)
    annual = pd.DataFrame(rows[is_annual])
    first = annual.sort_values("filed", kind="stable").drop_duplicates(
        ["cik", "period_end"]
    )
    columns = ["cik", "entity_name", "period_end", "filed", "value"]
    return first[columns].sort_values(["cik", "period_end"]).reset_index(drop=True)
