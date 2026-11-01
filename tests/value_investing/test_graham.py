"""G5 earnings growth with hand-computed EPS histories (layers/graham.py)."""

import pandas as pd
import pytest

from Toolbox.value_investing.inputs import RuleInputs
from Toolbox.value_investing.layers.graham import g5_earnings_growth

DATE = pd.Timestamp("2020-03-20")


def _inputs(histories: dict[str, list[float]]) -> RuleInputs:
    """One row per symbol; ``histories[symbol][lag]`` is its diluted EPS at that lag."""
    index = pd.MultiIndex.from_tuples(
        [(s, DATE) for s in histories], names=["symbol", "date"]
    )
    columns = {
        ("eps_diluted", lag): [h[lag] for h in histories.values()]
        for lag in range(max(len(h) for h in histories.values()))
    }
    annual = pd.DataFrame(columns, index=index)
    annual.columns = pd.MultiIndex.from_tuples(annual.columns)
    return RuleInputs(
        price=pd.Series(10.0, index=index),
        annual=annual,
        dividends=pd.DataFrame(columns=["symbol", "ex_date", "amount"]),
        dates=(DATE,),
        listed=pd.Series(dtype="datetime64[ns]"),
    )


def _run(histories: dict[str, list[float]], **kw: float) -> pd.DataFrame:
    settings = {"years": 6, "average_years": 3, "min_growth": 1.33} | kw
    out = g5_earnings_growth(_inputs(histories), pd.DataFrame(), **settings)  # type: ignore[arg-type]
    return out.droplevel("date")


def test_recent_average_against_earlier_average_over_six_years() -> None:
    # recent (lags 0-2) mean (6 + 9 + 3) / 3 = 6; earlier (lags 3-5) mean (1 + 2 + 3) / 3 = 2
    out = _run({"AAA": [6, 9, 3, 1, 2, 3]})
    assert out.loc["AAA", "value"] == pytest.approx(3.0)
    assert out.loc["AAA", "passed"] == 1.0


def test_exact_threshold_passes_and_just_below_fails() -> None:
    # earlier mean 3; recent mean 3.99 gives 1.33 exactly, 3.9 gives 1.3
    out = _run(
        {
            "AAA": [3.99, 3.99, 3.99, 3, 3, 3],
            "BBB": [3.9, 3.9, 3.9, 3, 3, 3],
        }
    )
    assert out.loc["AAA", "value"] == pytest.approx(1.33)
    assert out.loc["AAA", "passed"] == 1.0
    assert out.loc["BBB", "value"] == pytest.approx(1.3)
    assert out.loc["BBB", "passed"] == 0.0


def test_lags_beyond_the_two_windows_are_ignored() -> None:
    # With years = 6 only lags 0 to 5 count; lag 6 is a decoy.
    out = _run({"AAA": [4, 4, 4, 2, 2, 2, 1000]})
    assert out.loc["AAA", "value"] == pytest.approx(2.0)


def test_zero_or_negative_earlier_average_fails_with_no_value() -> None:
    out = _run(
        {
            "ZERO": [5, 5, 5, 0, 0, 0],
            "NEG": [5, 5, 5, -1, -2, -3],
            "MIXED": [5, 5, 5, 3, -3, 0],  # mean 0
        }
    )
    assert (out["passed"] == 0.0).all()
    assert out["value"].isna().all()


def test_not_evaluable_until_six_years_are_filed() -> None:
    out = _run({"AAA": [6, 6, 6, 2, 2, float("nan")]})
    assert pd.isna(out.loc["AAA", "passed"])


def test_thirteen_years_compares_lags_zero_to_two_with_ten_to_twelve() -> None:
    # lags 0-2 mean 4; lags 10-12 mean 1; lags 3-9 are 100 and must not count.
    eps = [4.0] * 3 + [100.0] * 7 + [1.0] * 3
    out = _run({"AAA": eps}, years=13)
    assert out.loc["AAA", "value"] == pytest.approx(4.0)
    assert out.loc["AAA", "passed"] == 1.0


def test_thirteen_years_needs_all_thirteen() -> None:
    out = _run({"AAA": [4.0] * 3 + [100.0] * 7 + [1.0] * 2 + [float("nan")]}, years=13)
    assert pd.isna(out.loc["AAA", "passed"])


def test_rows_do_not_leak_between_companies() -> None:
    both = _run({"AAA": [6, 6, 6, 2, 2, 2], "BBB": [1, 1, 1, 2, 2, 2]})
    alone = _run({"BBB": [1, 1, 1, 2, 2, 2]})
    assert both.loc["BBB", "value"] == pytest.approx(alone.loc["BBB", "value"])
    assert both.loc["BBB", "passed"] == 0.0


def test_inputs_are_not_modified() -> None:
    inputs = _inputs({"AAA": [6, 6, 6, 2, 2, 2]})
    before = inputs.annual.copy()
    g5_earnings_growth(
        inputs, pd.DataFrame(), years=6, average_years=3, min_growth=1.33
    )
    pd.testing.assert_frame_equal(inputs.annual, before)
