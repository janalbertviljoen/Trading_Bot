"""
value_investing/layers/buffett.py
---------------------------------
Buffett's and Munger's quality rules, one function per rule.

Node ``metrics.buffett`` (spec §2.4). The first slice holds B1, a high and consistent
return on equity without much debt; B2 to B5 and M1 follow once their thresholds are
set. Same signature and result as the Graham rules (``value_investing.flags``).

Owns
----
Rule B1: return on equity per fiscal year, debt to equity.

Must not
--------
Compute shared valuation ratios (``value_investing.layers.valuation``); hold thresholds
(``config/rules.toml``); score or rank (``value_investing.scoring``).

Public API
----------
b1_return_on_equity(inputs, ratios, *, years, min_roe, min_years_passed, max_debt_to_equity)
    B1: ROE above the minimum in most years, and debt to equity under a cap.

Computes
--------
return on equity, ROE, debt to equity : b1_return_on_equity
    ROE(k) = net income(k) / stockholders' equity(k) per fiscal year k (year-end
    equity), a year counting only when equity > 0 and ROE >= the minimum (0.15).
    D/E = long-term debt / equity (lag 0). Pass when at least ``min_years_passed`` of the
    last ``years`` count and D/E <= the cap; value the number of years that count.

Dependency rule
---------------
pandas, ``value_investing.inputs`` and ``value_investing.flags``.
"""

import pandas as pd

from Toolbox.value_investing.flags import rule_result
from Toolbox.value_investing.inputs import RuleInputs


def b1_return_on_equity(  # noqa: PLR0913  one keyword per threshold in config
    inputs: RuleInputs,
    ratios: pd.DataFrame,
    *,
    years: int,
    min_roe: float,
    min_years_passed: int,
    max_debt_to_equity: float,
) -> pd.DataFrame:
    """B1, spec §2.4: ROE >= ``min_roe`` in most of the last ``years`` years, little debt.

    Parameters
    ----------
    inputs : RuleInputs
        Net income, stockholders' equity and long-term debt per fiscal year.
    ratios : pd.DataFrame
        Not used.
    years : int
        Fiscal years examined (10).
    min_roe : float
        Minimum ROE as a fraction (0.15).
    min_years_passed : int
        Years that must reach it ("most": 7).
    max_debt_to_equity : float
        Cap on long-term debt over equity, latest year.

    Returns
    -------
    pd.DataFrame
        ``value`` the number of years with ROE >= ``min_roe``; ``passed`` when that is at
        least ``min_years_passed`` and D/E is within the cap.

    Notes
    -----
    A year with zero or negative equity does not count. Not evaluable until ``years``
    years of net income and equity are filed, or where long-term debt is not reported.
    """
    income = inputs.history("net_income", years)
    equity = inputs.history("stockholders_equity", years)
    roe = (income / equity).where(equity > 0)
    good_years = (roe >= min_roe).sum(axis=1).astype(float)
    latest_equity = inputs.field("stockholders_equity")
    debt = inputs.field("long_term_debt")
    levered = (debt / latest_equity).where(latest_equity > 0)
    evaluable = income.notna().all(axis=1) & equity.notna().all(axis=1) & debt.notna()
    passed = (good_years >= min_years_passed) & (levered <= max_debt_to_equity)
    return rule_result(good_years, passed, evaluable)
