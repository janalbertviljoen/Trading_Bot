"""Book weights and ranked positions of the ranked wallet (evaluation.py)."""

import math

import pandas as pd
import pytest

from Toolbox.value_investing.evaluation import Wallet, book_weights, ranked_positions


def wallet(
    names: int = 3,
    weighting: str = "score",
    rank_weights: tuple[float, ...] = (4.0, 2.0, 1.0),
) -> Wallet:
    return Wallet(
        score_column="score",
        weighting=weighting,
        names=names,
        book_usd=100.0,
        horizon_months=12,
        rank_weights=rank_weights,
    )


SCORES = pd.Series({"AAA": 1.0, "BBB": 5.0, "CCC": 3.0, "DDD": 2.0, "EEE": 4.0})


def test_buy_book_has_exactly_names_companies_in_rank_order() -> None:
    weights = book_weights(SCORES, wallet(), side="buy")
    assert list(weights.index) == ["BBB", "EEE", "CCC"]
    assert weights.sum() == pytest.approx(1.0)
    # 5 / 12, 4 / 12, 3 / 12
    assert weights["BBB"] == pytest.approx(5 / 12)
    assert weights["CCC"] == pytest.approx(3 / 12)


def test_sell_book_takes_the_lowest_with_equal_weights() -> None:
    weights = book_weights(SCORES, wallet(), side="sell")
    assert list(weights.index) == ["AAA", "DDD", "CCC"]
    assert weights.tolist() == pytest.approx([1 / 3] * 3)


def test_tie_at_the_cut_off_is_broken_alphabetically_on_buy() -> None:
    scores = pd.Series({"ZZZ": 3.0, "AAA": 3.0, "MMM": 3.0, "TOP": 9.0})
    weights = book_weights(scores, wallet(names=2), side="buy")
    assert list(weights.index) == ["TOP", "AAA"]


def test_tie_at_the_cut_off_is_broken_alphabetically_on_sell() -> None:
    scores = pd.Series({"ZZZ": 3.0, "AAA": 3.0, "MMM": 3.0, "LOW": 0.0})
    weights = book_weights(scores, wallet(names=2), side="sell")
    assert list(weights.index) == ["LOW", "AAA"]


def test_fewer_scored_than_names_gives_all_of_them() -> None:
    scores = pd.Series({"AAA": 1.0, "BBB": 3.0})
    weights = book_weights(scores, wallet(names=5), side="buy")
    assert list(weights.index) == ["BBB", "AAA"]
    assert weights.tolist() == pytest.approx([0.75, 0.25])


def test_empty_scores_give_an_empty_book() -> None:
    empty = pd.Series(dtype=float)
    assert book_weights(empty, wallet(), side="buy").empty
    assert book_weights(empty, wallet(weighting="rank"), side="sell").empty


def test_all_zero_scores_give_equal_weights() -> None:
    scores = pd.Series({"AAA": 0.0, "BBB": 0.0, "CCC": 0.0, "DDD": 0.0})
    weights = book_weights(scores, wallet(), side="buy")
    assert list(weights.index) == ["AAA", "BBB", "CCC"]
    assert weights.tolist() == pytest.approx([1 / 3] * 3)


def test_rank_weighting_uses_rank_weights_normalised() -> None:
    weights = book_weights(SCORES, wallet(weighting="rank"), side="buy")
    assert list(weights.index) == ["BBB", "EEE", "CCC"]
    assert weights.tolist() == pytest.approx([4 / 7, 2 / 7, 1 / 7])


def test_rank_weighting_shares_the_mean_weight_between_tied_companies() -> None:
    scores = pd.Series({"AAA": 5.0, "BBB": 5.0, "CCC": 2.0, "DDD": 1.0})
    weights = book_weights(scores, wallet(weighting="rank"), side="buy")
    # AAA and BBB tie on ranks 1 and 2: both get (4 + 2) / 2 = 3; CCC gets 1; sum 7.
    assert list(weights.index) == ["AAA", "BBB", "CCC"]
    assert weights.tolist() == pytest.approx([3 / 7, 3 / 7, 1 / 7])


def test_rank_weighting_with_too_few_rank_weights_raises() -> None:
    with pytest.raises(ValueError, match="rank_weights"):
        book_weights(
            SCORES,
            wallet(names=3, weighting="rank", rank_weights=(2.0, 1.0)),
            side="buy",
        )


def test_score_weighting_does_not_need_rank_weights() -> None:
    weights = book_weights(SCORES, wallet(rank_weights=()), side="buy")
    assert len(weights) == 3


def test_book_weights_does_not_change_the_input() -> None:
    scores = SCORES.copy()
    book_weights(scores, wallet(), side="buy")
    pd.testing.assert_series_equal(scores, SCORES)


def _table() -> pd.DataFrame:
    early = pd.Timestamp("2020-03-20")
    late = pd.Timestamp("2020-06-19")
    rows = [
        (early, s, v)
        for s, v in [("AAA", 1.0), ("BBB", 2.0), ("CCC", 3.0), ("DDD", 4.0)]
    ]
    rows += [
        (late, s, v)
        for s, v in [("AAA", 9.0), ("BBB", 2.0), ("CCC", 3.0), ("DDD", 1.0)]
    ]
    return pd.DataFrame(rows, columns=["date", "symbol", "score"])


def test_ranked_positions_has_both_sides_with_amounts() -> None:
    out = ranked_positions(_table(), [pd.Timestamp("2020-03-20")], wallet(names=2))
    buys = out[out["side"] == "buy"]
    sells = out[out["side"] == "sell"]
    assert list(buys["symbol"]) == ["DDD", "CCC"]
    assert list(sells["symbol"]) == ["AAA", "BBB"]
    assert buys["amount"].sum() == pytest.approx(100.0)
    assert sells["amount"].tolist() == pytest.approx([50.0, 50.0])
    assert buys["weight"].tolist() == pytest.approx([4 / 7, 3 / 7])
    assert buys["score"].tolist() == [4.0, 3.0]


def test_score_date_is_the_latest_rebalance_on_or_before_the_decision() -> None:
    decisions = [
        pd.Timestamp("2020-03-20"),  # on the date
        pd.Timestamp("2020-05-01"),  # between
        pd.Timestamp("2020-06-22"),  # after the second
    ]
    out = ranked_positions(_table(), decisions, wallet(names=2))
    by = out.groupby("decision_date")["score_date"].first()
    assert by[decisions[0]] == pd.Timestamp("2020-03-20")
    assert by[decisions[1]] == pd.Timestamp("2020-03-20")
    assert by[decisions[2]] == pd.Timestamp("2020-06-19")
    last = out[(out["decision_date"] == decisions[2]) & (out["side"] == "buy")]
    assert list(last["symbol"]) == ["AAA", "CCC"]


def test_decision_before_any_score_is_skipped() -> None:
    out = ranked_positions(
        _table(),
        [pd.Timestamp("2019-01-01"), pd.Timestamp("2020-03-20")],
        wallet(names=2),
    )
    assert set(out["decision_date"]) == {pd.Timestamp("2020-03-20")}
    assert not math.isnan(out["weight"].sum())


def test_missing_scores_are_ignored_in_positions() -> None:
    table = _table()
    table.loc[
        (table["symbol"] == "DDD") & (table["date"] == pd.Timestamp("2020-03-20")),
        "score",
    ] = float("nan")
    out = ranked_positions(table, [pd.Timestamp("2020-03-20")], wallet(names=2))
    assert list(out[out["side"] == "buy"]["symbol"]) == ["CCC", "BBB"]
