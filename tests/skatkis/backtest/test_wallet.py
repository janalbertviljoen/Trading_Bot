import numpy as np

from Toolbox.SandBox.btc_vwap import trade_indicators
from Toolbox.skatkis.backtest import wallet as trade_wallet


def run(prices, **kwargs):
    df = trade_indicators.detect_all(prices.copy(), threshold=0.02, rolling_window=12)
    return trade_wallet.make_trades(df, **kwargs)


def test_wallet_never_goes_negative(prices):
    df = run(prices)
    assert (df["wallet_value_available"] >= 0).all()
    assert (df["wallet_volume_available"] >= 0).all()


def test_total_value_is_cash_plus_holdings(prices):
    df = run(prices)
    expected = (
        df["wallet_value_available"] + df["wallet_volume_available"] * df["price"]
    )
    np.testing.assert_allclose(df["total_wallet_value"], expected)


def test_no_signals_matches_initial_split(prices):
    df = trade_indicators.detect_all(prices.copy(), threshold=10, rolling_window=12)
    df = trade_wallet.make_trades(
        df, initial_investment=100, initial_trade_fraction=0.5
    )
    assert (df["trade_volume"] == 0).all()
    final = 50 + 50 * df["price"].iloc[-1] / df["price"].iloc[0]
    assert np.isclose(df["total_wallet_value"].iloc[-1], final)
