import requests
import pandas as pd

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
        start_ts = data[-1][0] + 1

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
    df["timestamp"] = df["timestamp"].dt.floor("s").dt.tz_localize(None)
    df["price"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)

    return df[["timestamp", "high", "low", "price", "volume"]].reset_index(drop=True)
