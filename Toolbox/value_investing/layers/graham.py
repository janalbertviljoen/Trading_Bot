"""
value_investing/layers/graham.py
--------------------------------
Graham's rules for the defensive and the enterprising investor, one function per rule.

Node ``metrics.graham`` (spec §2.1 to §2.3): rules G2 to G7 (The Intelligent Investor,
rev. ed., ch. 14), E1 to E6 (ch. 15) and the net-net test NN. Each function takes the
rule inputs (``value_investing.inputs``) and the shared ratios
(``value_investing.layers.valuation``), its thresholds as keyword arguments from
``config/rules.toml``, and returns the tested value and a pass flag per company and
rebalance date (``value_investing.flags``). Fundamentals are annual, first-reported and
known on the date; balance-sheet values are fiscal-year-end values (lag 0).

Owns
----
Rules G2a, G2b, G3, G4, G5, G6, G7, E1, E2, E3, E4, E5, E6 and NN: current ratio, debt
to net current assets, earnings stability, dividend record, earnings growth, price to
net tangible assets, net current asset value.

Must not
--------
Compute market cap, average EPS, P/E or P/B (``value_investing.layers.valuation``);
hold thresholds (``config/rules.toml``); score or rank (``value_investing.scoring``).

Public API
----------
g2a_current_ratio, g2b_debt_cover, g3_earnings_stability, g4_dividend_record,
g5_earnings_growth, g6_moderate_pe, g7_moderate_price_to_assets, e1_current_ratio,
e2_debt_cover, e3_earnings_stability, e4_current_dividend, e5_earnings_growth,
e6_price_to_net_tangible_assets, nn_net_net
    One rule each: ``(inputs, ratios, **thresholds) -> DataFrame[value, passed]``.

Computes
--------
current ratio : g2a_current_ratio, e1_current_ratio
    CR = current assets / current liabilities; pass when CR >= the minimum (G2a 2.0,
    E1 1.5); missing when current liabilities <= 0.
debt to net current assets, debt cover : g2b_debt_cover, e2_debt_cover
    NCA = current assets - current liabilities; pass when long-term debt <= k * NCA
    (G2b k = 1.0, E2 k = 1.10); value LTD / NCA, missing when NCA <= 0. The rule then
    fails, except at NCA = 0 with no long-term debt, which passes.
earnings stability : g3_earnings_stability, e3_earnings_stability
    Pass when diluted EPS > 0 in each of the last n fiscal years (G3 3 as configured,
    Graham's 10; E3 5); value
    the number of those years with EPS > 0; not evaluable until n consecutive years are
    filed.
dividend record : g4_dividend_record
    Count of the n calendar years before the rebalance year (G4 n = 3 as configured,
    Graham's 20) with at least one dividend ex-date; pass when it equals n.
earnings growth (Graham) : g5_earnings_growth
    g = mean EPS of lags 0 to a-1 / mean EPS of lags n-a to n-1 (a = 3, n = 6 as
    configured, Graham's 13); pass when the older mean is > 0 and g >= 1.33. Over 3
    years apart 1.33 is about 10% a year, over Graham's 10 about 2.9% a year.
current dividend : e4_current_dividend
    Trailing dividend yield over ``window_days``; pass when > 0.
earnings growth (enterprising) : e5_earnings_growth
    g = (EPS(lag 0) - EPS(lag n)) / |EPS(lag n)| (n = 4), a fraction; pass when
    EPS(lag 0) > EPS(lag n); g missing when EPS(lag n) = 0.
price to net tangible assets : e6_price_to_net_tangible_assets
    NTA = equity - goodwill - intangible assets; value market cap / NTA, missing when
    NTA <= 0; pass when market cap <= k * NTA (k = 1.20).
net current asset value, net-net : nn_net_net
    NCAV = current assets - total liabilities; value market cap / NCAV, missing when
    NCAV <= 0; pass when NCAV > 0 and market cap <= k * NCAV (k = 2/3).

Dependency rule
---------------
pandas, ``market_data.metrics`` and ``value_investing.inputs``, ``value_investing.flags``.
"""

import pandas as pd

from Toolbox.market_data.metrics import dividend_yield
from Toolbox.value_investing.flags import rule_result
from Toolbox.value_investing.inputs import RuleInputs


def _current_ratio(inputs: RuleInputs, minimum: float) -> pd.DataFrame:
    """Test current assets over current liabilities against ``minimum``."""
    liabilities = inputs.field("current_liabilities")
    ratio = (inputs.field("current_assets") / liabilities).where(liabilities > 0)
    return rule_result(ratio, ratio >= minimum, ratio.notna())


def _debt_cover(inputs: RuleInputs, multiple: float) -> pd.DataFrame:
    """Long-term debt against ``multiple`` * net current assets."""
    nca = inputs.field("current_assets") - inputs.field("current_liabilities")
    debt = inputs.field("long_term_debt")
    return rule_result(
        (debt / nca).where(nca > 0),
        debt <= multiple * nca,
        nca.notna() & debt.notna(),
    )


def _earnings_stability(inputs: RuleInputs, years: int) -> pd.DataFrame:
    """Diluted EPS positive in each of the last ``years`` fiscal years.

    ``value`` counts the positive years, comparable between companies.
    """
    eps = inputs.history("eps_diluted", years)
    positive = (eps > 0).sum(axis=1).astype(float)
    return rule_result(positive, positive >= years, eps.notna().all(axis=1))


def g2a_current_ratio(
    inputs: RuleInputs, ratios: pd.DataFrame, *, min_current_ratio: float
) -> pd.DataFrame:
    """G2a, spec §2.1: current ratio of at least ``min_current_ratio`` (2.0)."""
    return _current_ratio(inputs, min_current_ratio)


def g2b_debt_cover(
    inputs: RuleInputs, ratios: pd.DataFrame, *, max_debt_to_net_current_assets: float
) -> pd.DataFrame:
    """G2b, spec §2.1: long-term debt no more than net current assets.

    Not evaluable where long-term debt is not reported, including companies without debt
    that report no tag (a missing input stays missing, 2026-10-31).
    """
    return _debt_cover(inputs, max_debt_to_net_current_assets)


def g3_earnings_stability(
    inputs: RuleInputs, ratios: pd.DataFrame, *, years: int
) -> pd.DataFrame:
    """G3, spec §2.1: positive diluted EPS in each of the last ``years`` years.

    Configured at 3 years (Graham: 10); off by default.
    """
    return _earnings_stability(inputs, years)


def g4_dividend_record(
    inputs: RuleInputs, ratios: pd.DataFrame, *, years: int
) -> pd.DataFrame:
    """G4, spec §2.1: a dividend in each of the ``years`` calendar years before.

    Configured at 3 years (Graham: 20); off by default.

    Parameters
    ----------
    inputs : RuleInputs
        Uses ``dividends`` (Yahoo, by ex-date) and the rebalance dates.
    ratios : pd.DataFrame
        Not used.
    years : int
        Calendar years y - ``years`` to y - 1 must each hold an ex-date, y the
        rebalance year.

    Returns
    -------
    pd.DataFrame
        ``value`` the number of those years with a dividend, ``passed`` when all have
        one. Evaluable only when the company's price history (``inputs.listed``) starts
        by 1 January of year y - ``years``; a younger history gives NaN (phase-in, as
        every rule whose window is not yet full). A company with the full history that
        never paid has value 0 and fails.
    """
    index = inputs.price.index
    dividends = inputs.dividends
    paid = pd.crosstab(dividends["symbol"], dividends["ex_date"].dt.year).gt(0)
    rebalance_years = pd.DatetimeIndex(index.get_level_values("date")).year.to_numpy()
    first = int(min(min(paid.columns, default=9999), rebalance_years.min())) - years - 1
    symbols = index.get_level_values("symbol")
    # counts[s, j] = years up to first + j in which symbol s paid a dividend.
    counts = (
        paid.reindex(
            index=sorted(set(symbols)),
            columns=range(first, int(rebalance_years.max()) + 1),
            fill_value=False,
        )
        .astype(int)
        .cumsum(axis=1)
    )
    rows = counts.index.get_indexer(symbols)
    table = counts.to_numpy()
    upper = table[rows, rebalance_years - 1 - first]
    lower = table[rows, rebalance_years - years - 1 - first]
    window_start = pd.to_datetime(
        pd.Series(rebalance_years - years).astype(str) + "-01-01"
    ).to_numpy()
    listed = inputs.listed.reindex(symbols.astype("string")).to_numpy()
    evaluable = pd.Series(listed <= window_start, index=index)
    value = pd.Series((upper - lower).astype(float), index=index)
    return rule_result(value, value >= years, evaluable)


def g5_earnings_growth(
    inputs: RuleInputs,
    ratios: pd.DataFrame,
    *,
    years: int,
    average_years: int,
    min_growth: float,
) -> pd.DataFrame:
    """G5, spec §2.1: recent average EPS at least ``min_growth`` * the earlier average.

    The recent average covers lags 0 to ``average_years - 1``, the earlier one the last
    ``average_years`` of ``years`` (lags ``years - average_years`` to ``years - 1``: 3 to 5
    with the configured 6 years and 3-year averages; Graham's 13 years give lags 10 to 12).
    Fails when the earlier average is zero or negative; not evaluable until all ``years``
    years are filed.
    """
    eps = inputs.history("eps_diluted", years)
    recent = eps.iloc[:, :average_years].mean(axis=1)
    old = eps.iloc[:, years - average_years :].mean(axis=1)
    return rule_result(
        (recent / old).where(old > 0),
        (old > 0) & (recent >= min_growth * old),
        eps.notna().all(axis=1),
    )


def g6_moderate_pe(
    inputs: RuleInputs, ratios: pd.DataFrame, *, max_pe: float
) -> pd.DataFrame:
    """G6, spec §2.1: price at most ``max_pe`` (15) * average EPS.

    Fails when the average EPS is zero or negative; not evaluable without it.
    """
    pe = ratios["pe_average"]
    return rule_result(pe, pe <= max_pe, ratios["average_eps"].notna())


def g7_moderate_price_to_assets(
    inputs: RuleInputs,
    ratios: pd.DataFrame,
    *,
    max_pb: float,
    max_pe_times_pb: float,
) -> pd.DataFrame:
    """G7, spec §2.1: P/B at most ``max_pb`` (1.5), or P/E * P/B at most 22.5.

    ``value`` is P/E * P/B. With an average EPS of zero or below, P/E and the product are
    undefined, so the rule passes on P/B at most ``max_pb`` alone; with equity of zero or
    below, P/B is undefined and the rule fails.
    """
    pe, pb = ratios["pe_average"], ratios["pb"]
    product = pe * pb
    evaluable = (
        ratios["average_eps"].notna()
        & ratios["market_cap"].notna()
        & inputs.field("stockholders_equity").notna()
    )
    return rule_result(
        product, (pb <= max_pb) | (product <= max_pe_times_pb), evaluable
    )


def e1_current_ratio(
    inputs: RuleInputs, ratios: pd.DataFrame, *, min_current_ratio: float
) -> pd.DataFrame:
    """E1, spec §2.2: current ratio of at least ``min_current_ratio`` (1.5)."""
    return _current_ratio(inputs, min_current_ratio)


def e2_debt_cover(
    inputs: RuleInputs, ratios: pd.DataFrame, *, max_debt_to_net_current_assets: float
) -> pd.DataFrame:
    """E2, spec §2.2: long-term debt at most 110% of net current assets."""
    return _debt_cover(inputs, max_debt_to_net_current_assets)


def e3_earnings_stability(
    inputs: RuleInputs, ratios: pd.DataFrame, *, years: int
) -> pd.DataFrame:
    """E3, spec §2.2: positive diluted EPS in each of the last ``years`` (5) years."""
    return _earnings_stability(inputs, years)


def e4_current_dividend(
    inputs: RuleInputs, ratios: pd.DataFrame, *, window_days: int
) -> pd.DataFrame:
    """E4, spec §2.2: some current dividend: an ex-date in the last ``window_days``.

    ``value`` is the trailing dividend yield (``market_data.metrics.dividend_yield``).
    """
    prices = inputs.price.rename("close").reset_index()
    yields = dividend_yield(prices, inputs.dividends, window_days=window_days)
    value = yields.set_index(["symbol", "date"])["dividend_yield"].reindex(
        inputs.price.index
    )
    return rule_result(value, value > 0, value.notna())


def e5_earnings_growth(
    inputs: RuleInputs, ratios: pd.DataFrame, *, years: int
) -> pd.DataFrame:
    """E5, spec §2.2: latest diluted EPS above that of ``years`` (4) years before.

    ``value`` is the relative change, comparable between companies.
    """
    latest = inputs.field("eps_diluted")
    before = inputs.field("eps_diluted", years)
    change = latest - before
    growth = (change / before.abs()).where(before != 0)
    return rule_result(growth, change > 0, change.notna())


def e6_price_to_net_tangible_assets(
    inputs: RuleInputs, ratios: pd.DataFrame, *, max_price_to_net_tangible_assets: float
) -> pd.DataFrame:
    """E6, spec §2.2: market cap at most 120% of net tangible assets.

    Not evaluable where goodwill or intangible assets are not reported (a missing input
    stays missing, 2026-10-31).
    """
    tangible = (
        inputs.field("stockholders_equity")
        - inputs.field("goodwill")
        - inputs.field("intangible_assets")
    )
    cap = ratios["market_cap"]
    return rule_result(
        (cap / tangible).where(tangible > 0),
        (tangible > 0) & (cap <= max_price_to_net_tangible_assets * tangible),
        tangible.notna() & cap.notna(),
    )


def nn_net_net(
    inputs: RuleInputs, ratios: pd.DataFrame, *, max_price_to_ncav: float
) -> pd.DataFrame:
    """NN, spec §2.3: market cap at most two-thirds of net current asset value."""
    ncav = inputs.field("current_assets") - inputs.field("total_liabilities")
    cap = ratios["market_cap"]
    return rule_result(
        (cap / ncav).where(ncav > 0),
        (ncav > 0) & (cap <= max_price_to_ncav * ncav),
        ncav.notna() & cap.notna(),
    )
