import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ----------------------------
# 1. Fetch historical BTC data from Binance
# ----------------------------
BINANCE_URL = "https://api.binance.com/api/v3/klines"


def fetch_btc_history(interval="1h", start_date="2024-01-01", symbol="BTCUSDT"):
    """
    Fetch historical BTC price data from Binance.

    :param interval (str): Frequency of data (e.g., '1h' for 1 hour)
    :param start_date (str): Start date for fetching data
    :param symbol (str): Trading pair symbol (default is BTCUSDT)

    :return (pandas.DataFrame): DataFrame with historical price data
    """
    start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp.utcnow().timestamp() * 1000)

    all_data = []

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": 1000,
        }

        r = requests.get(BINANCE_URL, params=params)
        data = r.json()

        if not data:
            break

        all_data.extend(data)
        start_ts = data[-1][0] + 1  # move forward safely

    df = pd.DataFrame(
        all_data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_base",
            "taker_quote",
            "ignore",
        ],
    )

    df["timestamp"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["price"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)

    return df[["timestamp", "high", "low", "price", "volume"]].reset_index(drop=True)


def rolling_vwap(df, price_col="price", volume_col="volume", window=50):
    """
    Computes the volume-weighted average price (VWAP) over a rolling window.

    :param df (pandas.DataFrame): DataFrame containing price and volume data
    :param price_col (str): Column name for price data
    :param volume_col (str): Column name for volume data
    :param window (int): Rolling window size for VWAP calculation

    :return (pandas.Series): Series with rolling VWAP values
    """
    pv = df[price_col] * df[volume_col]
    vwap = (
        pv.rolling(window=window, min_periods=window).sum()
        / df[volume_col].rolling(window=window, min_periods=window).sum()
    )
    return vwap


def detect_large_moves(df, unit_price="price", metric_name="vwap", threshold=0.03):
    """
    Detects significant price deviations from metric, uses volume-weighted average price as default.

    :param df (pandas.DataFrame): DataFrame containing reference metric to compare against
    :param unit_price (str): Name of the price column to compare against (default is "price")
    :param metric_name (str): Name of the metric to compare against (default is "vwap")
    :param threshold (float): Threshold for detecting large moves (default is 0.03)

    :return (pandas.DataFrame): DataFrame with added value of 'deviation' and 'signal' if deviation exceeds threshold
    """
    df = df.copy()
    df["deviation"] = (df[unit_price] - df[metric_name]) / df[metric_name]
    df["signal"] = df["deviation"].abs() > threshold
    return df


def trade_direction(df):
    """
    Determines if a trade should be a 'Buy' or 'Sell' based on deviation sign and expected direction of market price.

    :param df (pandas.DataFrame): DataFrame containing price 'deviation' from reference metric and 'signal' columns

    :return (pandas.DataFrame): DataFrame with added 'trade' column indicating direction of trade: 1 for Buy, -1 for Sell, 0 for No Trade
    """
    df["trade"] = 0
    df.loc[(df["signal"]) & (df["deviation"] > 0), "trade"] = -1
    df.loc[(df["signal"]) & (df["deviation"] < 0), "trade"] = 1
    return df


def trade_volume(df, unit_price="price", wallet_fraction=0.01):
    """
    Assigns trade volume based on trade signal and value of wallet.

    :param df (pandas.DataFrame): DataFrame containing 'trade' column
    :param unit_price (str): Name of the price column to use for calculating trade volume (default is "price")
    :param wallet_fraction (float): Fraction of total wallet value to use for each trade (default is 0.01)

    :return (pandas.DataFrame): DataFrame with added 'trade_volume' column
    """
    df["trade_volume"] = wallet_fraction * df["signal"] * df["total_wallet_value"]
    df["trade_volume_price"] = df[unit_price] * df["trade_volume"]
    return df


def best_bot(df):
    """Copmutes the theoretical maximum profit possible given high and low prices in time interval selected.

    :param df (pandas.DataFrame): DataFrame containing 'high' and 'low' price columns

    :return (pandas.DataFrame): DataFrame with added 'max_profit' column
    """
    df["max_profit"] = df["high"] - df["low"]
    return df


def detrend_series(series, method="log_return"):
    if method == "log_return":
        return np.log(series).diff().dropna()
    elif method == "demean":
        return series - series.rolling(50).mean()
    else:
        raise ValueError("Unknown detrending method")


def rolling_fft(series, window):
    fft_results = []
    index = []

    for i in range(window, len(series)):
        segment = series.iloc[i - window : i].values
        fft_vals = np.fft.fft(segment)
        fft_mag = np.abs(fft_vals)

        fft_results.append(fft_mag)
        index.append(series.index[i])

    return np.array(fft_results), index


def fft_frequencies(window, sampling_interval=1.0):
    return np.fft.fftfreq(window, d=sampling_interval)


def dominant_cycle(series, window, sampling_interval=1.0):
    fft_mag, _ = rolling_fft(series, window)
    freqs = fft_frequencies(window, sampling_interval)

    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]

    dom_freq = pos_freqs[np.argmax(fft_mag[:, pos_mask], axis=1)]
    return 1 / dom_freq
