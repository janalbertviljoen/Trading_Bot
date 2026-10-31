"""
value_investing/flags.py
------------------------
The shape of a rule's result: the value it tests and its pass flag.

Every rule function returns ``rule_result(value, passed, evaluable)``: one row per
(``symbol``, ``date``), the tested value and a float flag that is 1 when the rule
passes, 0 when it fails and missing when it cannot be evaluated (an input missing, or
the rule's window not yet full). The score counts a missing flag as 0.

Owns
----
The result columns of a rule and the encoding of its flag.

Public API
----------
rule_result(value, passed, evaluable)
    The result table of one rule.
RESULT_COLUMNS
    ``value`` and ``passed``.

Dependency rule
---------------
pandas.
"""

import pandas as pd

RESULT_COLUMNS = ["value", "passed"]


def rule_result(
    value: pd.Series, passed: pd.Series, evaluable: pd.Series
) -> pd.DataFrame:
    """Return the rule's value and flag: 1.0 passed, 0.0 failed, NaN not evaluable.

    Parameters
    ----------
    value : pd.Series
        The quantity the rule tests (a ratio, a count of years); may be missing even when
        the rule is evaluable, e.g. a P/E on negative earnings.
    passed : pd.Series
        Boolean, on the same index.
    evaluable : pd.Series
        Boolean: every input the rule needs is present.
    """
    flag = (
        passed.fillna(False).astype(float).where(evaluable.fillna(False).astype(bool))
    )
    return pd.DataFrame({"value": value.astype(float), "passed": flag})
