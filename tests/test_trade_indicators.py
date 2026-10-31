import numpy as np

import trade_indicators


def test_rolling_vwap_equals_mean_for_constant_volume(prices):
    df = trade_indicators.rolling_vwap(prices.copy(), rolling_window=5)
    expected = prices["price"].rolling(5).mean()
    np.testing.assert_allclose(df["vwap"].iloc[4:], expected.iloc[4:])
    assert df["vwap"].iloc[:4].isna().all()


def test_detect_deviation_flags_only_beyond_threshold(prices):
    df = trade_indicators.detect_all(prices.copy(), threshold=0.02, rolling_window=12)
    assert (df.loc[df["signal"], "deviation"].abs() > 0.02).all()
    assert (df.loc[~df["signal"], "deviation"].abs().fillna(0) <= 0.02).all()
