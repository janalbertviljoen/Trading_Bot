"""
value_investing/peers.py
------------------------
A peer group set against the picks: its score and rank, its rule points, and a wallet
that buys it.

The report compares the picks with a named group of companies, the Magnificent 7
(user, 2026-10-31): where they rank on the score, which rules give them fewer points
than the picks, and what the same money would have become in them, bought on the same
decision dates and held (``value_investing.holding.buy_and_hold``). A peer counts on a
decision date only when it is in the universe with a score, so a company that joined
the S&P 500 later is bought from then on.

Owns
----
The peer group's ranks, its rule profile against the picks, and its positions for an
equal-weight wallet.

Must not
--------
Score companies (``value_investing.scoring``); value holdings
(``value_investing.holding``); plot (``value_investing.plots``).

Public API
----------
peer_ranks(table, symbols, decisions, *, score_column, share_column)
    Each peer's latest score, its rank and its median rank on the decision dates.
peer_profile(table, symbols, picks, decisions, rule_ids)
    Mean points per rule of the peers and of the picks on the decision dates.
peer_positions(table, symbols, decisions, *, score_column)
    Equal-weight buy positions in the peers on each decision date.

Computes
--------
rank on the score : peer_ranks
    Rank 1 = highest score among the companies scored on the date; median over the
    score dates of the decisions.
peer rule profile : peer_profile
    Mean ``<id>_points`` of the peers over the score dates of the decisions, and of the
    companies picked on them.

Dependency rule
---------------
pandas and ``value_investing.rules``.
"""

from collections.abc import Sequence

import pandas as pd

from Toolbox.value_investing.rules import points_column


def _score_dates(
    table: pd.DataFrame, decisions: Sequence[pd.Timestamp]
) -> list[pd.Timestamp]:
    """Return the latest rebalance date on or before each decision date."""
    scored = sorted(pd.Timestamp(d) for d in table["date"].unique())
    return sorted({max(d for d in scored if d <= day) for day in decisions})


def peer_ranks(
    table: pd.DataFrame,
    symbols: Sequence[str],
    decisions: Sequence[pd.Timestamp],
    *,
    score_column: str,
    share_column: str,
) -> pd.DataFrame:
    """Return each peer's latest score and rank, and its median rank on the decisions.

    Parameters
    ----------
    table : pd.DataFrame
        The buyability table (``symbol``, ``date`` and the score columns).
    symbols : Sequence[str]
        The peers.
    decisions : Sequence[pd.Timestamp]
        Decision dates; each uses its latest rebalance date.
    score_column, share_column : str
        The score ranked and its 0-to-1 share.

    Returns
    -------
    pd.DataFrame
        One row per peer: ``symbol``, ``score``, ``share``, ``rank`` and ``companies``
        on the latest rebalance date, and ``median_rank`` over the decisions' score
        dates where it was scored; missing where the peer has no score.
    """
    ranked = table.dropna(subset=[score_column]).copy()
    ranked["rank"] = ranked.groupby("date")[score_column].rank(ascending=False)
    ranked["companies"] = ranked.groupby("date")[score_column].transform("count")
    is_latest = (ranked["date"] == ranked["date"].max()).to_numpy()
    newest = ranked.loc[pd.Series(is_latest, index=ranked.index)]
    latest = {str(r["symbol"]): r for r in newest.to_dict("records")}
    dates = _score_dates(table, decisions)
    past = ranked[ranked["date"].isin(dates)]
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        row = latest.get(symbol)
        rows.append(
            {
                "symbol": symbol,
                "score": None if row is None else float(row[score_column]),
                "share": None if row is None else float(row[share_column]),
                "rank": None if row is None else float(row["rank"]),
                "companies": None if row is None else float(row["companies"]),
                "median_rank": float(
                    past.loc[past["symbol"] == symbol, "rank"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def peer_profile(
    table: pd.DataFrame,
    symbols: Sequence[str],
    picks: pd.DataFrame,
    decisions: Sequence[pd.Timestamp],
    rule_ids: Sequence[str],
) -> pd.DataFrame:
    """Return the mean points per rule of the peers and of the picks.

    Parameters
    ----------
    table : pd.DataFrame
        The buyability table with ``<id>_points``.
    symbols : Sequence[str]
        The peers.
    picks : pd.DataFrame
        The companies picked: ``symbol`` and ``score_date``.
    decisions : Sequence[pd.Timestamp]
        Decision dates; the peers are taken on their score dates.
    rule_ids : Sequence[str]
        Rules to compare.

    Returns
    -------
    pd.DataFrame
        Columns ``rule``, ``group`` (``peers`` or ``picks``) and ``points`` (mean, 0 to
        5).
    """
    columns = [points_column(r) for r in rule_ids]
    dates = _score_dates(table, decisions)
    peers = table[table["symbol"].isin(list(symbols)) & table["date"].isin(dates)]
    chosen = picks[["symbol", "score_date"]].merge(
        table[["symbol", "date", *columns]],
        left_on=["symbol", "score_date"],
        right_on=["symbol", "date"],
    )
    rows: list[dict[str, object]] = []
    for group, part in (("peers", peers), ("picks", chosen)):
        for rule_id, column in zip(rule_ids, columns, strict=True):
            rows.append(
                {"rule": rule_id, "group": group, "points": float(part[column].mean())}
            )
    return pd.DataFrame(rows)


def peer_positions(
    table: pd.DataFrame,
    symbols: Sequence[str],
    decisions: Sequence[pd.Timestamp],
    *,
    score_column: str,
) -> pd.DataFrame:
    """Return equal-weight buy positions in the peers scored on each decision's date.

    Returns
    -------
    pd.DataFrame
        The columns of ``value_investing.evaluation.ranked_positions``:
        ``decision_date``, ``score_date``, ``symbol``, ``score``, ``side`` (``buy``),
        ``weight`` (1 / number of peers scored) and ``gics_sector``.
    """
    scored = sorted(pd.Timestamp(d) for d in table["date"].unique())
    frames: list[pd.DataFrame] = []
    for day in decisions:
        known = [d for d in scored if d <= day]
        if not known:
            continue
        own = table[
            (table["date"] == known[-1]) & table["symbol"].isin(list(symbols))
        ].dropna(subset=[score_column])
        if own.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "decision_date": day,
                    "score_date": known[-1],
                    "symbol": own["symbol"].to_numpy(),
                    "score": own[score_column].to_numpy(),
                    "side": "buy",
                    "weight": 1 / len(own),
                    "gics_sector": own["gics_sector"].to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
