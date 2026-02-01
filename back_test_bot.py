import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

BINANCE_URL = "https://api.binance.com/api/v3/klines"


# TODO: Split to seperate file: "read_data.py"
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


# TODO: Split to seperate file: "indicators.py"
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


def detect_deviation(df, unit_price="price", metric_name="vwap", threshold=0.03):
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


# TODO: Split to seperate file: "trading_logic.py"
def update_wallet(df, wallet_value: float, wallet_value_fraction: float = 0.01):
    """
    Executes trading logic and updates wallet balances across time based on
    directional trade signals.

    Trading rules and conventions:
    -------------------------------
    • Trades are signal-driven using the `direction` column:
        - direction > 0 → BUY
        - direction < 0 → SELL
        - direction = 0 → NO TRADE

    • Trade sizing is asymmetric and inventory-aware:
        - BUY trades are sized as a fraction of available cash
        - SELL trades are sized as a fraction of available asset volume

    • Trade sign conventions:
        - trade_value > 0  → cash outflow (BUY)
        - trade_value < 0  → cash inflow  (SELL)
        - trade_volume > 0 → asset inflow (BUY)
        - trade_volume < 0 → asset outflow (SELL)

    • Execution constraints:
        - BUY trades are executed only if sufficient cash is available
        - SELL trades are executed only if sufficient asset volume is available
        - Short selling and leverage are not permitted

    • Wallet accounting:
        - Wallet balances are updated additively
        - Total wallet value is computed as:
              cash + (asset_volume × current_price)

    Parameters:
    -----------
    df : pandas.DataFrame
        Must contain at minimum the columns:
        ['price', 'direction']

    wallet_value : float
        Initial wallet value in base currency (e.g. USD)

    wallet_value_fraction : float, optional
        Fraction of available resources used per trade (default = 0.01)

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame augmented with:
        - trade_value
        - trade_volume
        - wallet_value_available
        - wallet_volume_available
        - total_wallet_value
    """

    df.loc[0, "wallet_value_available"] = wallet_value
    df.loc[0, "wallet_volume_available"] = 0.0
    df.loc[0, "trade_value"] = 0.0
    df.loc[0, "trade_volume"] = 0.0
    df.loc[0, "total_wallet_value"] = wallet_value

    for i in range(1, len(df)):
        price = df.loc[i, "price"]
        direction = df.loc[i, "direction"]

        prev_cash = df.loc[i - 1, "wallet_value_available"]
        prev_volume = df.loc[i - 1, "wallet_volume_available"]

        cash = prev_cash
        volume = prev_volume

        trade_val = 0.0
        trade_vol = 0.0

        # ===== BUY =====
        if direction > 0 and prev_cash > 0:
            trade_val = wallet_value_fraction * prev_cash
            trade_vol = trade_val / price

            cash -= trade_val
            volume += trade_vol

        # ===== SELL =====
        elif direction < 0 and prev_volume > 0:
            trade_vol = -wallet_value_fraction * prev_volume
            trade_val = trade_vol * price  # negative cash flow

            cash += abs(trade_val)
            volume += trade_vol

        # Persist state
        df.loc[i, "trade_value"] = trade_val
        df.loc[i, "trade_volume"] = trade_vol
        df.loc[i, "wallet_value_available"] = cash
        df.loc[i, "wallet_volume_available"] = volume
        df.loc[i, "total_wallet_value"] = cash + volume * price

    return df


def trade_value(
    direction: float, wallet_value_available: float, wallet_fraction: float
):
    """
    Assigns trade value based on trade signal and value of wallet.

    :param direction (int): The direction of the trade: 1 for Buy, -1 for Sell, 0 for No Trade
    :param wallet_value_available (float): Total value of the trading wallet available for trading
    :param wallet_fraction (float): Fraction of total wallet value to use for each trade (default is 0.01)

    :return (float): The volume of the trade in units
    """
    trade_value = direction * wallet_fraction * wallet_value_available
    return trade_value


def price_to_unit_conversion(trade_value: float, unit_price: float):
    """
    Converts transaction values to units based on specified unit price.

    :param trade_value (float): The value of the trade in base currency
    :param unit_price (float): The price per unit in base currency

    :return (float): The number of units for the given trade value and unit price
    """
    units = trade_value / unit_price
    return units


def total_wallet_value(
    wallet_value_available: float, wallet_volume_available: float, unit_price: float
):
    """
    Computes the total wallet value combining available cash and the value of held assets.

    :param wallet_value_available (float): Total value of the trading wallet available for trading
    :param wallet_volume_available (float): Total volume of assets held in the wallet
    :param unit_price (float): The price per unit in base currency

    :return (float): The total wallet value in base currency
    """
    total_value = wallet_value_available + (wallet_volume_available * unit_price)
    return total_value


def trade_direction(df):
    """
    Determines if a trade should be a 'Buy' or 'Sell' based on deviation sign and expected direction of market price.

    :param df (pandas.DataFrame): DataFrame containing price 'deviation' from reference metric and 'signal' columns

    :return (pandas.DataFrame): DataFrame with added 'trade' column indicating direction of trade: 1 for Buy, -1 for Sell, 0 for No Trade
    """
    df["direction"] = 0
    df.loc[(df["signal"]) & (df["deviation"] > 0), "direction"] = -1
    df.loc[(df["signal"]) & (df["deviation"] < 0), "direction"] = 1
    return df


def unit_to_price_conversion(trade_units: float, unit_price: float):
    """
    Converts transaction units to value based on specified unit price.

    :param trade_units (float): The number of units being traded
    :param unit_price (float): The price per unit in base currency

    :return (float): The value of the trade in base currency for the given number of units and unit price
    """
    value = trade_units * unit_price
    return value


# TODO: Split to seperate file: "performance.py"
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
