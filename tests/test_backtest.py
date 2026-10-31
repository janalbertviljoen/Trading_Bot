import backtest
import pytest


def test_split_is_chronological(prices):
    train, test = backtest.split_dataset(prices, 0.25)
    assert len(train) == 150 and len(test) == 50
    assert train["timestamp"].max() < test["timestamp"].min()
    assert test.index[0] == 0


def test_summarise_reports_buy_and_hold(prices):
    summary = backtest.summarise(backtest.simulate(prices, backtest.DEFAULT_CONFIG))
    expected = prices["price"].iloc[-1] / prices["price"].iloc[0]
    assert summary["buy_and_hold"] == pytest.approx(expected, rel=1e-5)
    assert summary["max_drawdown"] <= 0


def test_optimise_rejects_unknown_params(monkeypatch, prices):
    monkeypatch.setattr(backtest, "load_dataset", lambda _: prices)
    with pytest.raises(ValueError, match="Unknown parameters"):
        backtest.optimise("x", {"bogus": [1]}, {}, 0.3, "excess_vs_buy_and_hold", 5)


def test_optimise_returns_ranked_candidates(monkeypatch, prices):
    monkeypatch.setattr(backtest, "load_dataset", lambda _: prices)
    result = backtest.optimise(
        "x",
        {"threshold": [0.01, 0.02, 0.05]},
        {"rolling_window": 12},
        0.3,
        "excess_vs_buy_and_hold",
        2,
    )
    scores = [t["in_sample"]["excess_vs_buy_and_hold"] for t in result["top"]]
    assert result["combinations_tested"] == 3
    assert scores == sorted(scores, reverse=True)
    assert "out_of_sample" in result["top"][0]
