"""Point-in-time tests of ``market_data.fundamentals``: no value from after the date."""

import pandas as pd
import pytest

from Toolbox.market_data.fundamentals import annual_as_of

PREFERRED = "IntangibleAssetsNetExcludingGoodwill"  # tag_priority 0
NARROWER = "FiniteLivedIntangibleAssetsNet"  # tag_priority 1
DAY = pd.Timestamp("2018-06-30")


def _fact(row: tuple[str, int, float, str, str, str]) -> dict[str, object]:
    """Return one balance-sheet fact of company AAA (CIK 1) from a 10-K.

    ``row`` is (tag, tag priority, value, period end, filing date, accession).
    """
    tag, priority, value, period_end, filed, accession = row
    return {
        "cik": 1,
        "entity_name": "AAA Inc.",
        "field": "intangible_assets",
        "taxonomy": "us-gaap",
        "tag": tag,
        "tag_priority": priority,
        "unit": "USD",
        "value": value,
        "period_start": pd.NaT,
        "period_end": pd.Timestamp(period_end),
        "fiscal_year": int(period_end[:4]),
        "fiscal_period": "FY",
        "form": "10-K",
        "filed": pd.Timestamp(filed),
        "accession": accession,
        "frame": None,
    }


@pytest.fixture
def facts() -> pd.DataFrame:
    """AAA reports FY2017 under both tags, then switches to the narrower tag only.

    FY2017 10-K (filed 2018-02-15): preferred tag 100, narrower tag 80.
    FY2018 10-K (filed 2019-02-15) and FY2019 10-K (filed 2020-02-14): narrower tag only.
    On 2018-06-30 only the FY2017 10-K is known: both tags were last used on 2018-02-15,
    so the tie goes to the preferred tag and FY2017 is 100. The later filings make the
    narrower tag the most recently used, which turns FY2017 into 80 if they are seen.
    """
    rows = [
        _fact((PREFERRED, 0, 100.0, "2017-12-31", "2018-02-15", "A1")),
        _fact((NARROWER, 1, 80.0, "2017-12-31", "2018-02-15", "A1")),
        _fact((NARROWER, 1, 90.0, "2018-12-31", "2019-02-15", "A2")),
        _fact((NARROWER, 1, 95.0, "2019-12-31", "2020-02-14", "A3")),
    ]
    return pd.DataFrame(rows).astype({"frame": "string"})


def _latest(facts: pd.DataFrame) -> pd.Series:
    """Return the lag-0 row of AAA's intangible assets on ``DAY``."""
    known = annual_as_of(
        facts,
        ["intangible_assets"],
        [DAY],
        depth=3,
        tolerance_days=45,
        stale_days=550,
    )
    return known[known["lag"] == 0].iloc[0]


def test_a_filing_after_the_date_is_not_seen(facts: pd.DataFrame) -> None:
    latest = _latest(facts)
    assert latest["period_end"] == pd.Timestamp("2017-12-31")
    assert latest["filed"] <= DAY


def test_input_is_not_modified(facts: pd.DataFrame) -> None:
    before = facts.copy()
    _latest(facts)
    pd.testing.assert_frame_equal(facts, before)


def test_cutting_the_facts_at_the_date_gives_the_reference_value(
    facts: pd.DataFrame,
) -> None:
    assert _latest(facts[facts["filed"] <= DAY])["value"] == 100.0


def test_the_tag_choice_does_not_use_later_filings(facts: pd.DataFrame) -> None:
    assert _latest(facts)["value"] == 100.0
