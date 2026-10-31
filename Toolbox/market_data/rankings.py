"""
market_data/rankings.py
-----------------------
Scores that rank companies and GICS sectors per analysis, and the best and worst of them.

Each analysis has its own meaning of "best": the highest revenue growth, total return,
return relative to a benchmark or dividend yield, or the lowest P/E. ``company_scores``
gives every company one number per analysis; ``sector_scores`` gives every sector one,
from its fund for the return analyses and from the median of its companies otherwise.
``best_and_worst`` picks the ends of a ranking.

Owns
----
The score of a company and of a sector per analysis, and which end of a ranking is best.

Must not
--------
Define return or growth formulas (``skatkis.financial_logic``); compute the underlying
metrics (``market_data.metrics``); plot.

Public API
----------
TOPICS
    The analyses with a ranking.
LOWER_IS_BETTER
    Analyses where a lower score is better (P/E).
company_scores(topic, tables)
    One score per company: columns ``symbol``, ``gics_sector``, ``score``.
sector_scores(topic, companies, tables)
    One score per sector: columns ``gics_sector``, ``sector_fund``, ``score``.
best_and_worst(scores, topic, n)
    The ``n`` best and ``n`` worst rows.
window(tables, start)
    The tables cut to a recent window, for rankings over the same years.
RankingTables
    The tables the scores are computed from.

Computes
--------
company score, sector score : company_scores, sector_scores
    fundamentals: annualised growth of first-reported annual revenue (at least
    ``MIN_YEARS_GROWTH`` years); total_return: annualised growth of the adjusted close;
    relative_return: annualised return relative to the sector fund (companies) or to the
    market (sector funds); dividend_yield: mean trailing yield; pe: median P/E (at least
    ``MIN_PE_DAYS`` daily values). Sector scores for fundamentals, dividend yield and P/E
    are the median of the sector's company scores.

Dependency rule
---------------
pandas, the standard library and ``skatkis.financial_logic``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from Toolbox.skatkis.financial_logic import (
    DAYS_PER_YEAR,
    annualised_growth,
    annualised_relative_return,
)

TOPICS = ("fundamentals", "total_return", "relative_return", "dividend_yield", "pe")
LOWER_IS_BETTER = frozenset({"pe"})
MIN_YEARS_RETURN = 1.0
MIN_YEARS_GROWTH = 3.0
MIN_PE_DAYS = 250
_COLUMNS = ["symbol", "gics_sector", "score"]


@dataclass(frozen=True)
class RankingTables:
    """The tables scores are computed from.

    Parameters
    ----------
    annual : pd.DataFrame
        First-reported annual values with ``cik``, ``field``, ``period_end``, ``value``.
    prices : pd.DataFrame
        ``symbol``, ``date``, ``adj_close`` for companies, sector funds and the market.
    yields : pd.DataFrame
        ``symbol``, ``date``, ``dividend_yield``.
    pe : pd.DataFrame
        ``symbol``, ``date``, ``pe``.
    gics : pd.DataFrame
        The GICS map (``symbol``, ``cik``, ``gics_sector``, ``sector_fund``).
    market : str
        The market benchmark symbol.
    min_years_return : float
        Shortest history for a return score (one year; the window length when ranking a
        window, so every company is scored over the same years).
    """

    annual: pd.DataFrame
    prices: pd.DataFrame
    yields: pd.DataFrame
    pe: pd.DataFrame
    gics: pd.DataFrame
    market: str
    min_years_return: float = MIN_YEARS_RETURN


def company_scores(topic: str, tables: RankingTables) -> pd.DataFrame:
    """Return one score per company for ``topic`` (see the module's Computes section)."""
    wide = tables.prices.pivot_table(index="date", columns="symbol", values="adj_close")
    symbols = [str(s) for s in tables.gics["symbol"]]
    funds = [str(f) for f in tables.gics["sector_fund"]]
    ciks = [str(c) for c in tables.gics["cik"]]
    scores: dict[str, float] = {}
    if topic == "fundamentals":
        revenue = tables.annual[tables.annual["field"] == "revenue"]
        by_cik: dict[str, float] = {}
        for cik, part in revenue.groupby("cik"):
            series = part.set_index("period_end")["value"].sort_index()
            by_cik[str(cik)] = annualised_growth(series, min_years=MIN_YEARS_GROWTH)
        scores = {s: by_cik.get(c, np.nan) for s, c in zip(symbols, ciks, strict=True)}
    elif topic == "total_return":
        scores = {
            s: annualised_growth(wide[s], min_years=tables.min_years_return)
            for s in symbols
            if s in wide.columns
        }
    elif topic == "relative_return":
        scores = {
            s: annualised_relative_return(
                wide[s], wide[f], min_years=tables.min_years_return
            )
            for s, f in zip(symbols, funds, strict=True)
            if s in wide.columns and f in wide.columns
        }
    elif topic == "dividend_yield":
        means = tables.yields.groupby("symbol")["dividend_yield"].mean()
        scores = {str(k): float(v) for k, v in means.items()}
    elif topic == "pe":
        counts = tables.pe.groupby("symbol")["pe"].count()
        medians = tables.pe.groupby("symbol")["pe"].median()
        scores = {str(k): float(v) for k, v in medians[counts >= MIN_PE_DAYS].items()}
    else:
        raise ValueError(f"no ranking for {topic!r}; one of {TOPICS}")
    frame = pd.DataFrame({"symbol": list(scores), "score": list(scores.values())})
    frame = frame.merge(tables.gics[["symbol", "gics_sector"]], on="symbol")
    return frame.dropna(subset=["score"])[_COLUMNS].reset_index(drop=True)


def sector_scores(
    topic: str, companies: pd.DataFrame, tables: RankingTables
) -> pd.DataFrame:
    """Return one score per GICS sector for ``topic``.

    Return analyses score the sector fund (``total_return``: its annualised growth;
    ``relative_return``: its annualised return relative to the market); the others take
    the median of the sector's company scores.
    """
    funds = tables.gics.drop_duplicates("gics_sector")[["gics_sector", "sector_fund"]]
    if topic in ("total_return", "relative_return"):
        wide = tables.prices.pivot_table(
            index="date", columns="symbol", values="adj_close"
        )
        values: list[float] = []
        for fund in (str(f) for f in funds["sector_fund"]):
            if fund not in wide.columns:
                values.append(np.nan)
            elif topic == "total_return":
                values.append(
                    annualised_growth(wide[fund], min_years=tables.min_years_return)
                )
            else:
                values.append(
                    annualised_relative_return(
                        wide[fund],
                        wide[tables.market],
                        min_years=tables.min_years_return,
                    )
                )
        result = funds.assign(score=values)
    else:
        medians = companies.groupby("gics_sector", as_index=False)["score"].median()
        result = funds.merge(medians, on="gics_sector")
    return result.dropna(subset=["score"]).reset_index(drop=True)


def window(
    tables: RankingTables, start: pd.Timestamp, *, tolerance_days: int = 10
) -> RankingTables:
    """Return ``tables`` cut to dates from ``start``, requiring the whole window for returns.

    Prices, P/E and yields keep rows dated from ``start``; annual values keep fiscal years
    ending from ``start``. Return scores then need a history of the whole window, less
    ``tolerance_days`` (for holidays at the start), so a company listed during the window
    is not ranked against those with the full period.
    """
    last = pd.Timestamp(tables.prices["date"].max())
    years = ((last - start).days - tolerance_days) / DAYS_PER_YEAR
    return RankingTables(
        annual=pd.DataFrame(tables.annual[tables.annual["period_end"] >= start]),
        prices=pd.DataFrame(tables.prices[tables.prices["date"] >= start]),
        yields=pd.DataFrame(tables.yields[tables.yields["date"] >= start]),
        pe=pd.DataFrame(tables.pe[tables.pe["date"] >= start]),
        gics=tables.gics,
        market=tables.market,
        min_years_return=years,
    )


def best_and_worst(
    scores: pd.DataFrame, topic: str, n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the ``n`` best and the ``n`` worst rows of ``scores`` for ``topic``.

    Best is the highest score, except for ``LOWER_IS_BETTER`` analyses (P/E), where it is
    the lowest. Both tables are ordered from the extreme inwards.
    """
    ascending = topic in LOWER_IS_BETTER
    ordered = scores.sort_values("score", ascending=ascending, kind="stable")
    return ordered.head(n).reset_index(drop=True), ordered.tail(n).iloc[
        ::-1
    ].reset_index(drop=True)
