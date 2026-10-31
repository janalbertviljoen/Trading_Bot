"""
value_investing/ledger.py
-------------------------
The wallet's ledger: what was bought each year, at which weight and price, and the
balance carried from year to year against the same money in ``^SP500TR``.

Each decision date the whole balance is spent on the companies of the buy book
(``value_investing.evaluation.ranked_positions``) at their weights, held to the next
decision date, sold there, and the proceeds reinvested the same day; the last decision
is held ``months``. The same money is followed in ``^SP500TR`` and in further
comparisons such as Berkshire Hathaway (``BRK-B``, which pays no dividend). The ledger shows each
position's price change (``close``, split-adjusted) and total return (``adj_close``,
with dividends reinvested); the balance uses the total return, like ``^SP500TR``. A
position without a price over its window keeps its money in cash. No costs or taxes.

Owns
----
The position ledger and the balance over time of the buy book and the benchmark.

Must not
--------
Choose companies or weights (``value_investing.evaluation``); define returns
(``skatkis.financial_logic``).

Public API
----------
ledger(positions, prices, *, benchmark, others, months, start_usd)
    One row per position bought, and the balance per decision.
comparison_name(symbol)
    The column prefix of a further comparison in the balance.

Computes
--------
position value, wallet balance : ledger
    a_i = balance(t) * w_i; value_i = a_i * (1 + r_i), r_i the total return from t to
    t + m; balance(t + m) = Σ value_i plus any cash; each comparison's balance grows
    by its total return over the same window. Price change = close(t + m) /
    close(t) - 1.

Dependency rule
---------------
pandas, ``skatkis.financial_logic`` and ``value_investing.inputs``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import pandas as pd

from Toolbox.skatkis.financial_logic import holding_return
from Toolbox.value_investing.inputs import price_on


@dataclass(frozen=True)
class Ledger:
    """The ledger of one wallet.

    Parameters
    ----------
    positions : pd.DataFrame
        One row per position bought: ``decision_date``, ``sell_date``, ``symbol``,
        ``gics_sector``, ``score``, ``weight``, ``invested_usd``, ``buy_price``,
        ``sell_price``, ``price_change``, ``total_return``, ``value_usd``,
        ``gain_usd``.
    balance : pd.DataFrame
        One row per decision: ``decision_date``, ``sell_date``, ``positions``,
        ``start_usd``, ``end_usd``, ``wallet_return``, ``benchmark_start_usd``,
        ``benchmark_end_usd``, ``benchmark_return``, the same three per further
        comparison (``brk_b_start_usd``, ...), and ``ahead_usd`` (wallet less
        benchmark at the end).
    """

    positions: pd.DataFrame
    balance: pd.DataFrame


def ledger(  # noqa: PLR0913  one keyword per setting of the run
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    benchmark: str,
    others: Sequence[str],
    months: int,
    start_usd: float,
) -> Ledger:
    """Return the ledger of the buy book, reinvesting the balance every decision.

    Parameters
    ----------
    positions : pd.DataFrame
        ``ranked_positions`` output; only ``side == "buy"`` rows are used, with
        ``decision_date``, ``symbol``, ``score``, ``weight`` and, when present,
        ``gics_sector``.
    prices : pd.DataFrame
        Columns ``symbol``, ``date``, ``close``, ``adj_close``, including ``benchmark``
        and ``others``.
    benchmark : str
        Total-return benchmark symbol (``^SP500TR``).
    others : Sequence[str]
        Further comparisons held the same way (``BRK-B``), from ``adj_close``.
    months : int
        Holding period of the last decision; the others are sold on the next decision
        date.
    start_usd : float
        Money at the first decision.

    Returns
    -------
    Ledger
        Positions and balance; a decision whose sale falls after the last benchmark
        price is left out.
    """
    buys = positions[positions["side"] == "buy"].copy()
    decisions = sorted(pd.Timestamp(d) for d in buys["decision_date"].unique())
    following = dict(pairwise(decisions))
    buys["sell_date"] = [
        following.get(pd.Timestamp(d), pd.Timestamp(d) + pd.DateOffset(months=months))
        for d in buys["decision_date"]
    ]
    last = prices.loc[prices["symbol"] == benchmark, "date"].max()
    buys = buys.loc[buys["sell_date"] <= pd.Timestamp(last)]
    days = sorted({*buys["decision_date"], *buys["sell_date"]})
    compared = [benchmark, *others]
    symbols = pd.concat([buys["symbol"], pd.Series(compared)]).drop_duplicates()
    close = price_on(prices, symbols, days, column="close")
    total = price_on(prices, symbols, days, column="adj_close")

    def at(levels: pd.Series, symbol: pd.Series, day: pd.Series) -> pd.Series:
        keys = pd.MultiIndex.from_arrays([symbol.astype("string"), day])
        return pd.Series(levels.reindex(keys).to_numpy(dtype=float), index=symbol.index)

    def grown(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
        one = pd.Series([symbol])
        return float(
            holding_return(
                at(total, one, pd.Series([start])), at(total, one, pd.Series([end]))
            ).iloc[0]
        )

    buys["buy_price"] = at(close, buys["symbol"], buys["decision_date"])
    buys["sell_price"] = at(close, buys["symbol"], buys["sell_date"])
    buys["price_change"] = holding_return(buys["buy_price"], buys["sell_price"])
    buys["total_return"] = holding_return(
        at(total, buys["symbol"], buys["decision_date"]),
        at(total, buys["symbol"], buys["sell_date"]),
    )
    rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    wallet_usd = start_usd
    held_usd = dict.fromkeys(compared, start_usd)
    for _, part in buys.groupby("decision_date", sort=True):
        day = pd.Timestamp(part["decision_date"].iloc[0])
        sell = pd.Timestamp(part["sell_date"].iloc[0])
        invested = wallet_usd * part["weight"]
        value = (invested * (1 + part["total_return"])).fillna(invested)
        frames.append(
            part.assign(
                invested_usd=invested, value_usd=value, gain_usd=value - invested
            )
        )
        end_usd = float(value.sum())
        row: dict[str, object] = {
            "decision_date": day,
            "sell_date": sell,
            "positions": len(part),
            "start_usd": wallet_usd,
            "end_usd": end_usd,
            "wallet_return": end_usd / wallet_usd - 1,
        }
        for symbol in compared:
            growth = grown(symbol, day, sell)
            prefix = "benchmark" if symbol == benchmark else comparison_name(symbol)
            row[f"{prefix}_start_usd"] = held_usd[symbol]
            held_usd[symbol] *= 1 + growth
            row[f"{prefix}_end_usd"] = held_usd[symbol]
            row[f"{prefix}_return"] = growth
        row["ahead_usd"] = end_usd - held_usd[benchmark]
        rows.append(row)
        wallet_usd = end_usd
    columns = [
        "decision_date",
        "sell_date",
        "symbol",
        "gics_sector",
        "score",
        "weight",
        "invested_usd",
        "buy_price",
        "sell_price",
        "price_change",
        "total_return",
        "value_usd",
        "gain_usd",
    ]
    held = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "gics_sector" not in held:
        held["gics_sector"] = pd.NA
    return Ledger(positions=held[columns], balance=pd.DataFrame(rows))


def comparison_name(symbol: str) -> str:
    """Return a column prefix for a comparison symbol, e.g. ``BRK-B`` -> ``brk_b``."""
    return symbol.lower().lstrip("^").replace("-", "_").replace(".", "_")
