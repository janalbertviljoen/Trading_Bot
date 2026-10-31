"""
value_investing/layers/valuation.py
-----------------------------------
The price-based ratios more than one rule uses, computed once per company and date.

Node ``metrics.valuation`` (spec §5): market capitalisation, average EPS, P/E on average
EPS and P/B from the rule inputs (``value_investing.inputs``). The Graham and Buffett
layers read these columns; they never recompute them.

Owns
----
Market capitalisation, average EPS, P/E on average EPS, P/B.

Must not
--------
Apply rule thresholds (``value_investing.layers.graham``,
``value_investing.layers.buffett``); align filings or adjust for splits
(``value_investing.inputs``).

Public API
----------
valuation(inputs, *, average_years)
    The shared ratios per (``symbol``, ``date``).
VALUATION_COLUMNS
    The columns it returns.

Computes
--------
market capitalisation, market cap, average EPS, multi-year average earnings, P/E on
average earnings, price-to-earnings, P/B, price-to-book : valuation
    market_cap = close * shares outstanding (lag 0), USD. average_eps = mean diluted EPS
    over lags 0 to n - 1, missing unless all n years are present. pe_average = close /
    average_eps, missing when average_eps <= 0. pb = market_cap / stockholders' equity
    (lag 0), missing when equity <= 0. Per-share values and share counts are on the
    price's split basis.

Dependency rule
---------------
pandas and ``value_investing.inputs``.
"""

import pandas as pd

from Toolbox.value_investing.inputs import RuleInputs

VALUATION_COLUMNS = ["market_cap", "average_eps", "pe_average", "pb"]


def valuation(inputs: RuleInputs, *, average_years: int) -> pd.DataFrame:
    """Return the shared valuation ratios per (``symbol``, ``date``).

    Parameters
    ----------
    inputs : RuleInputs
        Price and annual history per company and date.
    average_years : int
        Years in the EPS average (Graham: 3).

    Returns
    -------
    pd.DataFrame
        Columns ``VALUATION_COLUMNS`` on ``inputs.price``'s index: ``market_cap`` (USD),
        ``average_eps`` (USD per share), ``pe_average`` and ``pb`` (ratios).

    Notes
    -----
    A zero or negative average EPS or equity gives a missing ratio, never inf; the rules
    decide what a missing ratio means for their flag.
    """
    eps = inputs.history("eps_diluted", average_years).mean(axis=1, skipna=False)
    market_cap = inputs.price * inputs.field("shares_outstanding")
    equity = inputs.field("stockholders_equity")
    return pd.DataFrame(
        {
            "market_cap": market_cap,
            "average_eps": eps,
            "pe_average": (inputs.price / eps).where(eps > 0),
            "pb": (market_cap / equity).where(equity > 0),
        }
    )[VALUATION_COLUMNS]
