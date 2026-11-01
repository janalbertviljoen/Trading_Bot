"""
value_investing/inputs.py
-------------------------
The rules' inputs on each rebalance date: price, annual history and dividends.

``rule_inputs`` lines up, for every company and rebalance date, the closing price and
the annual fundamentals filed by that date (``market_data.fundamentals``), lag 0 being
the latest fiscal year. Per-share values (EPS) are divided by the split factor of their
filing date and share counts multiplied by it (``market_data.metrics.split_factors``), so
every per-share figure is on the basis of the split-adjusted prices and figures from
different years can be averaged and compared. It moves and aligns data only; the
layers compute from it.

Owns
----
The rebalance calendar, the rule inputs per company and date, the derived total
liabilities, which rows a company was already an index member on, and the forward
return over a window.

Must not
--------
Choose first-reported values or align filings to dates (``market_data.fundamentals``);
compute split factors (``market_data.metrics``); compute ratios or apply thresholds
(``value_investing.layers``).

Public API
----------
rebalance_dates(start, end, months)
    The third Friday of each rebalance month between two dates.
Tables(facts, prices, dividends, splits, listings, universe)
    The stored tables the inputs are built from.
RuleInputs
    Price, annual history and dividends per company and rebalance date.
rule_inputs(tables, dates, *, depth, tolerance_days, stale_days)
    Build ``RuleInputs`` from the stored tables.
price_on(prices, symbols, dates, *, column)
    The last price on or before each date, per symbol.
index_universe(constituents, changes, *, funds, excluded)
    The companies scored: an index's members with their CIK, sector and join date.
members_on_date(index, universe)
    Which (company, date) rows fall on or after the company joined the index.
window_returns(prices, index, *, benchmark, months)
    Each company's total return over the next months, the benchmark's and the excess.

Computes
--------
forward return, forward excess return over a window : window_returns
    r = P(s + m) / P(s) - 1 and b likewise for the benchmark (``adj_close`` levels, m
    months, ``skatkis.financial_logic.holding_return``); excess = r - b, missing when
    the window ends after the benchmark's last price.

Dependency rule
---------------
pandas, ``market_data`` and ``skatkis.financial_logic``.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from Toolbox.market_data.fundamentals import annual_as_of
from Toolbox.market_data.gics import fill_date_added, gics_map
from Toolbox.market_data.metrics import split_factors
from Toolbox.market_data.sources.prices import yahoo_symbol
from Toolbox.skatkis.financial_logic import holding_return

PER_SHARE_FIELDS = ("eps_diluted",)
SHARE_FIELDS = ("shares_outstanding",)
FIELDS = (
    "eps_diluted",
    "net_income",
    "current_assets",
    "current_liabilities",
    "long_term_debt",
    "total_liabilities",
    "liabilities_and_equity",
    "stockholders_equity",
    "goodwill",
    "intangible_assets",
    "shares_outstanding",
)
# Data alignment, not a rule threshold: a price older than this on a date counts as
# missing, so a company that stopped trading has no price there.
PRICE_TOLERANCE_DAYS = 7
FRIDAY = 4


def rebalance_dates(
    start: pd.Timestamp, end: pd.Timestamp, months: Sequence[int]
) -> list[pd.Timestamp]:
    """Return the third Friday of each of ``months`` from ``start`` to ``end``, inclusive."""
    dates: list[pd.Timestamp] = []
    for first in pd.date_range(start.replace(day=1), end, freq="MS"):
        if first.month not in months:
            continue
        day = first + pd.Timedelta(days=(FRIDAY - first.weekday()) % 7 + 14)
        if start <= day <= end:
            dates.append(day)
    return dates


@dataclass(frozen=True)
class RuleInputs:
    """What the rules see on each rebalance date.

    Parameters
    ----------
    price : pd.Series
        Close on the rebalance date, or the last trading day before it (USD,
        split-adjusted to the price fetch date); index (``symbol``, ``date``).
    annual : pd.DataFrame
        Same index; columns (field, lag): the annual value of each field for the fiscal
        year ``lag`` years before the latest one filed by the date, per-share values and
        share counts on the price's split basis. Missing where not filed or the history
        has a gap.
    dividends : pd.DataFrame
        Columns ``symbol``, ``ex_date``, ``amount`` (USD per share, split-adjusted).
    dates : tuple[pd.Timestamp, ...]
        The rebalance dates.
    listed : pd.Series
        The first date of each company's price history, indexed by ``symbol``; a
        history before the price source's start shows that start.
    """

    price: pd.Series
    annual: pd.DataFrame
    dividends: pd.DataFrame
    dates: tuple[pd.Timestamp, ...]
    listed: pd.Series

    @property
    def index(self) -> pd.MultiIndex:
        """Every (``symbol``, ``date``) a rule gives a value for."""
        return pd.MultiIndex.from_frame(self.price.index.to_frame(index=False))

    def field(self, name: str, lag: int = 0) -> pd.Series:
        """Return ``name`` at ``lag`` on every (symbol, date), missing where not filed."""
        if (name, lag) not in self.annual.columns:
            return pd.Series(float("nan"), index=self.price.index)
        return self.annual[(name, lag)]

    def history(self, name: str, years: int) -> pd.DataFrame:
        """Return ``name`` at lags 0 to ``years - 1``, one column per lag."""
        return pd.concat(
            {lag: self.field(name, lag) for lag in range(years)}, axis=1
        ).reindex(self.price.index)


@dataclass(frozen=True)
class Tables:
    """The stored tables the rule inputs are built from.

    Parameters
    ----------
    facts : pd.DataFrame
        The EDGAR ``facts`` table.
    prices : pd.DataFrame
        Columns ``symbol``, ``date``, ``close``, ``adj_close``.
    dividends : pd.DataFrame
        Columns ``symbol``, ``ex_date``, ``amount``.
    splits : pd.DataFrame
        Columns ``symbol``, ``date``, ``numerator``, ``denominator``.
    listings : pd.DataFrame
        Columns ``symbol``, ``first_date``: the start of each price history.
    universe : pd.DataFrame
        Columns ``symbol``, ``cik``, ``gics_sector`` and ``date_added`` (when the
        company joined the index): the companies scored.
    """

    facts: pd.DataFrame
    prices: pd.DataFrame
    dividends: pd.DataFrame
    splits: pd.DataFrame
    listings: pd.DataFrame
    universe: pd.DataFrame


def rule_inputs(
    tables: Tables,
    dates: Sequence[pd.Timestamp],
    *,
    depth: int,
    tolerance_days: int,
    stale_days: int,
) -> RuleInputs:
    """Build the rule inputs for ``tables.universe`` on ``dates``.

    Parameters
    ----------
    tables : Tables
        The stored tables.
    dates : Sequence[pd.Timestamp]
        Rebalance dates.
    depth : int
        Fiscal years of history kept (the longest rule window).
    tolerance_days, stale_days : int
        Passed to ``market_data.fundamentals.annual_as_of``: how far a fiscal year end
        may lie from whole years before the latest, and the oldest the latest fiscal
        year may be on a date.

    Returns
    -------
    RuleInputs
        One row per company and date with a price within ``PRICE_TOLERANCE_DAYS`` before
        the date; companies without a price on a date have no row there.

    Notes
    -----
    Filings count from their filing date, same day included (spec §3.3). Total
    liabilities missing from a filing are derived as liabilities and equity less
    stockholders' equity (``tag_map``); no other value is filled.
    """
    ciks = tables.universe[["symbol", "cik"]].astype({"cik": "int64"})
    own_facts = tables.facts[tables.facts["cik"].isin(ciks["cik"])]
    known = annual_as_of(
        own_facts,
        FIELDS,
        dates,
        depth=depth,
        tolerance_days=tolerance_days,
        stale_days=stale_days,
    )
    known = known.merge(ciks, on="cik")
    factor = split_factors(tables.splits, known, symbol="symbol", date="filed")
    per_share = known["field"].isin(PER_SHARE_FIELDS).to_numpy()
    shares = known["field"].isin(SHARE_FIELDS).to_numpy()
    value = known["value"].to_numpy(dtype=float).copy()
    value[per_share] = value[per_share] / factor.to_numpy()[per_share]
    value[shares] = value[shares] * factor.to_numpy()[shares]
    annual = known.assign(value=value).pivot_table(
        index=["symbol", "date"], columns=["field", "lag"], values="value"
    )
    price = price_on(tables.prices, ciks["symbol"], dates)
    annual = annual.reindex(price.index)
    for lag in range(depth):
        if ("liabilities_and_equity", lag) not in annual.columns:
            continue
        derived = (
            annual[("liabilities_and_equity", lag)]
            - annual[("stockholders_equity", lag)]
        )
        reported = annual.get(("total_liabilities", lag))
        annual[("total_liabilities", lag)] = (
            derived if reported is None else reported.fillna(derived)
        )
    dividends = tables.dividends
    own_dividends = dividends[dividends["symbol"].isin(ciks["symbol"])]
    return RuleInputs(
        price=price,
        annual=annual.sort_index(axis=1),
        dividends=own_dividends[["symbol", "ex_date", "amount"]],
        dates=tuple(dates),
        listed=tables.listings.set_index(tables.listings["symbol"].astype("string"))[
            "first_date"
        ],
    )


def price_on(
    prices: pd.DataFrame,
    symbols: pd.Series,
    dates: Sequence[pd.Timestamp],
    *,
    column: str = "close",
) -> pd.Series:
    """Return the last ``column`` on or before each date, per symbol.

    Indexed (``symbol``, ``date``); a symbol without a value in the
    ``PRICE_TOLERANCE_DAYS`` before a date has no row there.
    """
    grid = pd.MultiIndex.from_product(
        [sorted(symbols.astype(str)), list(dates)], names=["symbol", "date"]
    ).to_frame(index=False)
    own = prices[prices["symbol"].isin(grid["symbol"])].dropna(subset=[column])
    keys = {"symbol": "string", "date": "datetime64[ns]"}
    merged = pd.merge_asof(
        grid.astype(keys).sort_values("date"),
        own[["symbol", "date", column]].astype(keys).sort_values("date"),
        on="date",
        by="symbol",
        direction="backward",
        tolerance=pd.Timedelta(days=PRICE_TOLERANCE_DAYS),
    ).dropna(subset=[column])
    return merged.set_index(["symbol", "date"])[column].sort_index()


def window_returns(
    prices: pd.DataFrame, index: pd.MultiIndex, *, benchmark: str, months: int
) -> pd.DataFrame:
    """Return each row's total return over the next ``months``, the benchmark's and the excess.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns ``symbol``, ``date``, ``adj_close``, including ``benchmark``.
    index : pd.MultiIndex
        (``symbol``, ``date``) rows: the company and the window's start.
    benchmark : str
        Total-return benchmark symbol.
    months : int
        Length of the window.

    Returns
    -------
    pd.DataFrame
        On ``index``: ``end`` (the window's end date), ``own_return`` and
        ``benchmark_return`` (fractions, missing where a level is missing) and
        ``excess`` (their difference, also missing when the window ends after the
        benchmark's last price).
    """
    frame = index.to_frame(index=False)
    frame["end"] = frame["date"] + pd.DateOffset(months=months)
    last = prices.loc[prices["symbol"] == benchmark, "date"].max()
    days = sorted({*frame["date"], *frame["end"]})
    symbols = pd.concat([frame["symbol"], pd.Series([benchmark])]).drop_duplicates()
    levels = price_on(prices, symbols, days, column="adj_close")

    def level(symbol: pd.Series, day: pd.Series) -> pd.Series:
        keys = pd.MultiIndex.from_arrays([symbol.astype("string"), day])
        return pd.Series(levels.reindex(keys).to_numpy(dtype=float), index=frame.index)

    market = pd.Series(benchmark, index=frame.index)
    own = holding_return(
        level(frame["symbol"], frame["date"]), level(frame["symbol"], frame["end"])
    )
    index_return = holding_return(
        level(market, frame["date"]), level(market, frame["end"])
    )
    excess = (own - index_return).where(frame["end"] <= last)
    return pd.DataFrame(
        {
            "end": frame["end"].to_numpy(),
            "own_return": own.to_numpy(),
            "benchmark_return": index_return.to_numpy(),
            "excess": excess.to_numpy(),
        },
        index=index,
    )


def members_on_date(index: pd.MultiIndex, universe: pd.DataFrame) -> pd.Series:
    """Return which (``symbol``, ``date``) rows lie on or after the company joined the index.

    ``universe`` has ``symbol`` and ``date_added``. A company without a ``date_added``
    is never a member, so all its rows are False.
    """
    added = universe.set_index(universe["symbol"].astype("string"))["date_added"]
    first = added.reindex(index.get_level_values("symbol")).to_numpy()
    member = index.get_level_values("date") >= first
    return pd.Series(member, index=index)


def index_universe(
    constituents: pd.DataFrame,
    changes: pd.DataFrame,
    *,
    funds: Mapping[str, str],
    excluded: Collection[str],
) -> pd.DataFrame:
    """Return the companies scored: an index's members with CIK, sector and join date.

    Parameters
    ----------
    constituents, changes : pd.DataFrame
        The index's members and changes (``market_data.sources.constituents``).
    funds : Mapping[str, str]
        GICS sector to its fund (``[benchmarks.sectors]``).
    excluded : Collection[str]
        GICS sectors left out (``[universe] excluded_sectors``).

    Returns
    -------
    pd.DataFrame
        ``market_data.gics.gics_map`` output with ``date_added``, missing join dates
        taken from the changes (``market_data.gics.fill_date_added``).
    """
    members = fill_date_added(constituents, changes)
    added = pd.DataFrame(
        {
            "symbol": [yahoo_symbol(str(s)) for s in members["symbol"]],
            "date_added": members["date_added"].to_numpy(),
        }
    ).astype({"symbol": "string"})
    return gics_map(members, funds=funds, excluded=excluded).merge(
        added, on="symbol", how="left"
    )
