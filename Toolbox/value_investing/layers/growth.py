"""
value_investing/layers/growth.py
--------------------------------
Growth rules that are not in Graham's or Buffett's catalogue, one function per rule.

Node ``metrics.growth``. NI1, net income growth over one year, was added by the user
(VI_202610_005, 2026-10-31) to reward companies whose profit is rising; it has no
published threshold, so it passes on any growth and is mainly graded in buckets
(``value_investing.scoring.rule_points``). Same signature and result as the Graham
rules (``value_investing.flags``).

Owns
----
Rule NI1: one-year net income growth.

Must not
--------
Compute shared valuation ratios (``value_investing.layers.valuation``); hold thresholds
(``config/rules.toml``); score or rank (``value_investing.scoring``).

Public API
----------
ni1_net_income_growth(inputs, ratios)
    NI1: net income of the latest fiscal year against the year before.

Computes
--------
net income growth, profit growth : ni1_net_income_growth
    g = (NI(lag 0) - NI(lag 1)) / |NI(lag 1)|, a fraction, annual first-reported net
    income; pass when NI(lag 0) > NI(lag 1); g missing when NI(lag 1) = 0.

Dependency rule
---------------
pandas, ``value_investing.inputs`` and ``value_investing.flags``.
"""

import pandas as pd

from Toolbox.value_investing.flags import rule_result
from Toolbox.value_investing.inputs import RuleInputs


def ni1_net_income_growth(inputs: RuleInputs, ratios: pd.DataFrame) -> pd.DataFrame:
    """NI1 (VI_202610_005): net income growth from the year before to the latest year.

    Parameters
    ----------
    inputs : RuleInputs
        Net income per fiscal year.
    ratios : pd.DataFrame
        Not used.

    Returns
    -------
    pd.DataFrame
        ``value`` the relative growth (fraction), ``passed`` when net income rose; not
        evaluable without both years. A swing from a loss to a smaller loss counts as
        growth, measured against the size of the earlier loss.
    """
    latest = inputs.field("net_income")
    before = inputs.field("net_income", 1)
    change = latest - before
    growth = (change / before.abs()).where(before != 0)
    return rule_result(growth, change > 0, change.notna())
