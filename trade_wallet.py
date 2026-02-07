import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def make_trades(
    df, initial_investment=100, trade_fraction_cash=0.1, trade_fraction_volume=0.1
):
    """
    Initiate all trades, determine the direction of trade and update the wallet values of both cash value and comodity volume.

    :param df (panadas.DataFrame):
    :param wallet_value: Description
    :param wallet_value_fraction: Description
    """
    df = trade_direction(df)
    df = update_wallet(
        df, initial_investment, trade_fraction_cash, trade_fraction_volume
    )
    return df


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


# TODO: Split to seperate file: "trading_logic.py"
def update_wallet(
    df,
    initial_investment: float,
    trade_fraction_cash: float,
    trade_fraction_volume: float,
):
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

    df.loc[0, "wallet_value_available"] = initial_investment
    df.loc[0, "wallet_volume_available"] = 0.0
    df.loc[0, "trade_value"] = 0.0
    df.loc[0, "trade_volume"] = 0.0
    df.loc[0, "total_wallet_value"] = initial_investment
    df.loc[0, "normalised_wallet_value"] = 1.0
    df["normalised_price"] = df["price"] / df.loc[0, "price"]

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
            trade_val = trade_value(direction, prev_cash, trade_fraction_cash)
            trade_vol = price_to_unit_conversion(trade_val, price)

            cash -= trade_val
            volume += trade_vol

        # ===== SELL =====
        elif direction < 0 and prev_volume > 0:
            trade_vol = trade_value(direction, prev_volume, trade_fraction_volume)
            trade_val = unit_to_price_conversion(trade_vol, price)

            cash += abs(trade_val)
            volume += trade_vol

        # Persist state
        df.loc[i, "trade_value"] = trade_val
        df.loc[i, "trade_volume"] = trade_vol
        df.loc[i, "wallet_value_available"] = cash
        df.loc[i, "wallet_volume_available"] = volume
        df.loc[i, "total_wallet_value"] = cash + volume * price
        df.loc[i, "normalised_wallet_value"] = (
            df.loc[i, "total_wallet_value"] / initial_investment
        )

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


def unit_to_price_conversion(trade_units: float, unit_price: float):
    """
    Converts transaction units to value based on specified unit price.

    :param trade_units (float): The number of units being traded
    :param unit_price (float): The price per unit in base currency

    :return (float): The value of the trade in base currency for the given number of units and unit price
    """
    value = trade_units * unit_price
    return value
