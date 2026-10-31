"""Price-data ingestion from the Binance public klines API."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

BINANCE_URL = "https://api.binance.com/api/v3/klines"


def fetch_btc_history(
    interval: str = "1h", start_date: str = "2024-01-01", symbol: str = "BTCUSDT"
) -> pd.DataFrame:
    """Fetch historical OHLCV bars from Binance, paging 1000 bars per request.

    Args:
        interval (str): Bar size accepted by Binance, e.g. "1m", "1h", "1d".
            Default "1h".
        start_date (str): First date to fetch, parsed as UTC (e.g. "2024-01-01").
            Default "2024-01-01".
        symbol (str): Trading pair. Default "BTCUSDT".

    Returns:
        pandas.DataFrame: One row per bar with a ``RangeIndex`` and columns:
            timestamp (datetime64[ns], bar close time in UTC, timezone-naive,
                floored to the second),
            high, low (float64, quote currency per unit),
            price (float64, bar close, quote currency per unit),
            volume (float64, base-asset units traded in the bar).

    Assumptions:
        Data runs up to the moment of the call, so results are not reproducible
        between calls; use ``mcp_server/backtest.fetch_dataset`` with a pinned
        ``end_date`` for backtests.

    Edge cases:
        The last bar may still be open (partial). No retry, rate-limit handling
        or HTTP status check; an API error payload is not detected.

    Raises:
        requests.RequestException: Network failure.
    """
    start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    all_data: list[list[Any]] = []

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": 1000,
        }

        r = requests.get(BINANCE_URL, params=params)
        data: list[list[Any]] = r.json()

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
