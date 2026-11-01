"""
value_investing/rules.py
------------------------
The rule registry: each supported rule ID with its family and the function computing it.

The pipeline and the score select rules by ID from ``config/rules.toml`` through this
registry, without branching on IDs. Every rule function has the signature
``(inputs, ratios, **thresholds) -> DataFrame[value, passed]`` (``value_investing.flags``).
Families group the rules for the score: ``G`` Graham's defensive investor, ``E`` his
enterprising investor (with the net-net test, ch. 15), ``B`` Buffett and Munger, ``N``
net income growth (a convention of this repository, not from the sources).

Owns
----
The supported rule IDs, their family, the direction of their value and their output
column names.

Must not
--------
Compute anything (``value_investing.layers``); hold thresholds (``config/rules.toml``).

Public API
----------
Rule(rule_id, family, compute, higher_is_better)
    One registered rule.
RULES
    Every supported rule, by ID.
value_column(rule_id), pass_column(rule_id), points_column(rule_id)
    The output column names of a rule.

Dependency rule
---------------
The standard library, pandas and ``value_investing.layers``.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from Toolbox.value_investing.layers import buffett, graham, growth

RuleFunction = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class Rule:
    """One registered rule.

    Parameters
    ----------
    rule_id : str
        The spec's ID, e.g. ``"G2a"``.
    family : str
        ``"G"``, ``"E"``, ``"B"`` or ``"N"``.
    compute : RuleFunction
        ``(inputs, ratios, **thresholds) -> DataFrame[value, passed]``.
    higher_is_better : bool
        Direction of the value for the binned score: ``True`` when a higher value is
        better (current ratio), ``False`` when lower is (P/E).
    """

    rule_id: str
    family: str
    compute: RuleFunction
    higher_is_better: bool


RULES: Mapping[str, Rule] = MappingProxyType(
    {
        rule.rule_id: rule
        for rule in (
            Rule("G2a", "G", graham.g2a_current_ratio, True),
            Rule("G2b", "G", graham.g2b_debt_cover, False),
            Rule("G3", "G", graham.g3_earnings_stability, True),
            Rule("G4", "G", graham.g4_dividend_record, True),
            Rule("G5", "G", graham.g5_earnings_growth, True),
            Rule("G6", "G", graham.g6_moderate_pe, False),
            Rule("G7", "G", graham.g7_moderate_price_to_assets, False),
            Rule("E1", "E", graham.e1_current_ratio, True),
            Rule("E2", "E", graham.e2_debt_cover, False),
            Rule("E3", "E", graham.e3_earnings_stability, True),
            Rule("E4", "E", graham.e4_current_dividend, True),
            Rule("E5", "E", graham.e5_earnings_growth, True),
            Rule("E6", "E", graham.e6_price_to_net_tangible_assets, False),
            Rule("NN", "E", graham.nn_net_net, False),
            Rule("B1", "B", buffett.b1_return_on_equity, True),
            Rule("NI1", "N", growth.ni1_net_income_growth, True),
        )
    }
)


def value_column(rule_id: str) -> str:
    """Return the column of the value a rule tests, e.g. ``g2a_value``."""
    return f"{rule_id.lower()}_value"


def pass_column(rule_id: str) -> str:
    """Return the column of a rule's pass flag, e.g. ``g2a_pass``."""
    return f"{rule_id.lower()}_pass"


def points_column(rule_id: str) -> str:
    """Return the column of a rule's points in the binned score, e.g. ``g2a_points``."""
    return f"{rule_id.lower()}_points"
