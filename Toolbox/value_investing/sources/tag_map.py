"""
value_investing/sources/tag_map.py
----------------------------------
Maps XBRL tags in SEC EDGAR filings to the tool's canonical fundamental fields.

Companies report the same concept under different us-gaap tags, and change tags over
time (revenue moved to ``RevenueFromContractWithCustomerExcludingAssessedTax`` under ASC
606). Each canonical field lists its tags in priority order; the parser keeps every
listed tag with its priority, and choosing between them is left to a later stage. A
lower-priority tag can be narrower than its field (``Depreciation`` without amortisation
for ``depreciation_amortization``); its ``tag`` column shows which was used. Total
liabilities are not reported by every company; ``liabilities_and_equity`` lets a later
stage derive them. The
fields cover the data needs of the rule catalogue (spec §2). Financial companies and
REITs are excluded from the universe, so no bank, insurer or REIT tags are mapped.

Owns
----
The canonical fundamental field names, their units and their XBRL tags.

Must not
--------
Parse filings (``value_investing.sources.edgar``); choose between tags or periods
(point-in-time stage).

Public API
----------
FieldMap(field, unit, tags)
    One canonical field: its EDGAR unit and its (taxonomy, tag) pairs in priority order.
TAG_MAP
    Every mapped field.

Dependency rule
---------------
Standard library only.
"""

from dataclasses import dataclass

USD = "USD"
USD_PER_SHARE = "USD/shares"
SHARES = "shares"


@dataclass(frozen=True)
class FieldMap:
    """One canonical field and the XBRL tags that report it.

    Parameters
    ----------
    field : str
        Canonical name.
    unit : str
        EDGAR unit key the field must be reported in (``USD``, ``USD/shares``,
        ``shares``).
    tags : tuple[tuple[str, str], ...]
        ``(taxonomy, tag)`` pairs, highest priority first.
    """

    field: str
    unit: str
    tags: tuple[tuple[str, str], ...]


def _gaap(*tags: str) -> tuple[tuple[str, str], ...]:
    """Return us-gaap ``(taxonomy, tag)`` pairs."""
    return tuple(("us-gaap", tag) for tag in tags)


TAG_MAP: tuple[FieldMap, ...] = (
    # Income statement (flows)
    FieldMap(
        "revenue",
        USD,
        _gaap(
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
        ),
    ),
    FieldMap(
        "cost_of_revenue", USD, _gaap("CostOfRevenue", "CostOfGoodsAndServicesSold")
    ),
    FieldMap("gross_profit", USD, _gaap("GrossProfit")),
    FieldMap("operating_income", USD, _gaap("OperatingIncomeLoss")),
    FieldMap("net_income", USD, _gaap("NetIncomeLoss", "ProfitLoss")),
    FieldMap("eps_diluted", USD_PER_SHARE, _gaap("EarningsPerShareDiluted")),
    FieldMap("eps_basic", USD_PER_SHARE, _gaap("EarningsPerShareBasic")),
    FieldMap(
        "dividends_per_share",
        USD_PER_SHARE,
        _gaap(
            "CommonStockDividendsPerShareDeclared",
            "CommonStockDividendsPerShareCashPaid",
        ),
    ),
    FieldMap(
        "weighted_average_diluted_shares",
        SHARES,
        _gaap("WeightedAverageNumberOfDilutedSharesOutstanding"),
    ),
    # Cash flow statement (flows)
    FieldMap(
        "operating_cash_flow",
        USD,
        _gaap(
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
    ),
    FieldMap(
        "capital_expenditure", USD, _gaap("PaymentsToAcquirePropertyPlantAndEquipment")
    ),
    FieldMap(
        "depreciation_amortization",
        USD,
        _gaap(
            "DepreciationDepletionAndAmortization",
            "DepreciationAndAmortization",
            "DepreciationAmortizationAndAccretionNet",
            "Depreciation",
        ),
    ),
    FieldMap(
        "dividends_paid",
        USD,
        _gaap("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    ),
    # Balance sheet (stocks)
    FieldMap("current_assets", USD, _gaap("AssetsCurrent")),
    FieldMap("current_liabilities", USD, _gaap("LiabilitiesCurrent")),
    FieldMap("total_assets", USD, _gaap("Assets")),
    FieldMap("total_liabilities", USD, _gaap("Liabilities")),
    FieldMap("liabilities_and_equity", USD, _gaap("LiabilitiesAndStockholdersEquity")),
    FieldMap(
        "long_term_debt",
        USD,
        _gaap(
            "LongTermDebtNoncurrent",
            "LongTermDebt",
            "LongTermDebtAndCapitalLeaseObligations",
        ),
    ),
    FieldMap(
        "stockholders_equity",
        USD,
        _gaap(
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
    ),
    FieldMap("retained_earnings", USD, _gaap("RetainedEarningsAccumulatedDeficit")),
    FieldMap("cash", USD, _gaap("CashAndCashEquivalentsAtCarryingValue")),
    FieldMap("ppe_net", USD, _gaap("PropertyPlantAndEquipmentNet")),
    FieldMap("goodwill", USD, _gaap("Goodwill")),
    FieldMap("intangible_assets", USD, _gaap("IntangibleAssetsNetExcludingGoodwill")),
    FieldMap(
        "shares_outstanding",
        SHARES,
        (
            ("dei", "EntityCommonStockSharesOutstanding"),
            ("us-gaap", "CommonStockSharesOutstanding"),
        ),
    ),
)
