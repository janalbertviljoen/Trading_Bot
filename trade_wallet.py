"""Single-asset trading wallet: turns trade signals into cash / volume balances over time.

The wallet holds two balances, cash (base currency, e.g. USD) and asset volume
(units, e.g. BTC or shares), and walks a price time series row by row, buying or
selling a fraction of the available balance whenever a trade signal fires.

Input contract (one row per time step, ``RangeIndex`` starting at 0):
    price (float64): asset price in base currency per unit.
    signal (bool): True where a trade should be considered.
    deviation (float64): signed distance of price from a reference metric; its
        sign sets the trade direction (see ``trade_direction``).

Output columns added by ``make_trades`` are documented on ``update_wallet``.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def make_trades(
    df,
    initial_investment=100,
    initial_trade_fraction=0.5,
    trade_fraction_cash=0.1,
    trade_fraction_volume=0.1,
):
    """Run the full wallet simulation: derive trade directions, then update balances.

    Computes:
        direction = trade_direction(signal, deviation)
        wallet balances = update_wallet(price, direction, ...)

    Args:
        df (pandas.DataFrame): Time series with a ``RangeIndex`` starting at 0 and
            columns ``price`` (float64, base currency per unit), ``signal`` (bool)
            and ``deviation`` (float64, dimensionless).
        initial_investment (float): Starting wallet value in base currency.
            Default 100.
        initial_trade_fraction (float): Fraction of ``initial_investment`` converted
            to the asset at row 0, in [0, 1]. Default 0.5.
        trade_fraction_cash (float): Fraction of available cash spent on each BUY,
            in [0, 1]. Default 0.1.
        trade_fraction_volume (float): Fraction of held volume sold on each SELL,
            in [0, 1]. Default 0.1.

    Returns:
        pandas.DataFrame: The same object as ``df`` (mutated in place), with the
        ``direction`` column from ``trade_direction`` and the wallet columns from
        ``update_wallet``.

    Edge cases:
        Mutates ``df``; pass a copy if the original must be preserved.
    """
    df = trade_direction(df)
    df = update_wallet(
        df,
        initial_investment,
        initial_trade_fraction,
        trade_fraction_cash,
        trade_fraction_volume,
    )
    return df


def trade_direction(df):
    """Assign a BUY / SELL / no-trade direction to each row (mean-reversion rule).

    Computes:
        direction = -1  if signal and deviation > 0   (price above reference -> SELL)
        direction = +1  if signal and deviation < 0   (price below reference -> BUY)
        direction =  0  otherwise

    Args:
        df (pandas.DataFrame): Must contain ``signal`` (bool) and ``deviation``
            (float64, dimensionless; NaN allowed).

    Returns:
        pandas.DataFrame: The same object as ``df`` (mutated in place) with an added
        ``direction`` column (int64, values in {-1, 0, 1}).

    Edge cases:
        deviation NaN or exactly 0 -> direction 0.
        Mutates ``df``.

    Raises:
        KeyError: ``signal`` or ``deviation`` column is missing.
    """
    df["direction"] = 0
    df.loc[(df["signal"]) & (df["deviation"] > 0), "direction"] = -1
    df.loc[(df["signal"]) & (df["deviation"] < 0), "direction"] = 1
    return df


# TODO: Split to seperate file: "trading_logic.py"
def update_wallet(
    df,
    initial_investment: float,
    initial_trade_fraction: float,
    trade_fraction_cash: float,
    trade_fraction_volume: float,
):
    """Simulate the wallet row by row and record cash, volume and total value.

    Trading rules:
        Row 0 (initialisation):
            initial_trade = initial_trade_fraction * initial_investment
            cash_0   = initial_investment - initial_trade
            volume_0 = initial_trade / price_0
            ``direction`` at row 0 is ignored.
        Row i > 0:
            BUY  (direction > 0 and cash_{i-1} > 0):
                trade_value  = trade_fraction_cash * cash_{i-1}          (> 0)
                trade_volume = trade_value / price_i                      (> 0)
                cash_i = cash_{i-1} - trade_value
            SELL (direction < 0 and volume_{i-1} > 0):
                trade_volume = -trade_fraction_volume * volume_{i-1}     (< 0)
                trade_value  = trade_volume * price_i                     (< 0)
                cash_i = cash_{i-1} + |trade_value|
            otherwise: no trade, balances carried forward.
            volume_i = volume_{i-1} + trade_volume

    Formula:
        total_wallet_value_i      = cash_i + volume_i * price_i
        normalised_wallet_value_i = total_wallet_value_i / initial_investment
        normalised_price_i        = price_i / price_0

    Sign conventions:
        trade_value > 0 is a cash outflow (BUY), < 0 a cash inflow (SELL).
        trade_volume > 0 is an asset inflow (BUY), < 0 an asset outflow (SELL).

    Args:
        df (pandas.DataFrame): Time series with a ``RangeIndex`` starting at 0 and
            columns ``price`` (float64, base currency per unit, > 0) and
            ``direction`` (int64, values in {-1, 0, 1}).
        initial_investment (float): Starting wallet value in base currency (> 0).
        initial_trade_fraction (float): Fraction of ``initial_investment`` bought
            at row 0, in [0, 1].
        trade_fraction_cash (float): Fraction of available cash spent per BUY,
            in [0, 1].
        trade_fraction_volume (float): Fraction of held volume sold per SELL,
            in [0, 1].

    Returns:
        pandas.DataFrame: The same object as ``df`` (mutated in place) with added
        float64 columns:
            trade_value: signed trade value in base currency (0.0 if no trade).
            trade_volume: signed trade volume in units (0.0 if no trade).
            wallet_value_available: cash after the row's trade, base currency, >= 0.
            wallet_volume_available: units held after the row's trade, >= 0.
            total_wallet_value: cash + holdings at the row's price, base currency.
            normalised_wallet_value: total_wallet_value / initial_investment
                (1.0 at row 0).
            normalised_price: price / price at row 0 (the buy-and-hold benchmark).

    Assumptions:
        Trades execute at the row's ``price`` with no fees, slippage or spread.
        No leverage and no short selling; fractions in [0, 1] keep balances >= 0.

    Edge cases:
        Fractions > 1 are not rejected and can drive cash or volume negative.
        price_0 == 0 -> division by zero (inf volume / normalised price).
        Requires labels 0..n-1 (uses ``df.loc[i]``); reset the index after slicing.
        Python loop over rows: O(n) ``.loc`` calls, slow on long series.

    Raises:
        KeyError: ``price`` or ``direction`` missing, or index does not contain 0.
    """
    initial_trade = initial_trade_fraction * initial_investment
    df.loc[0, "wallet_value_available"] = initial_investment - initial_trade
    df.loc[0, "wallet_volume_available"] = price_to_unit_conversion(
        initial_trade, df.loc[0, "price"]
    )
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
    """Size a trade as a signed fraction of an available balance.

    Formula:
        trade = direction * wallet_fraction * wallet_value_available

    Args:
        direction (float): Trade direction: +1 BUY, -1 SELL, 0 no trade.
        wallet_value_available (float): Balance the trade is sized from: cash in
            base currency for a BUY, or asset volume in units for a SELL.
        wallet_fraction (float): Fraction of the balance to trade, in [0, 1].

    Returns:
        float: Signed trade size in the same unit as ``wallet_value_available``
        (base currency for a BUY, units for a SELL). Zero when direction is 0.
    """
    trade_value = direction * wallet_fraction * wallet_value_available
    return trade_value


def price_to_unit_conversion(trade_value: float, unit_price: float):
    """Convert a value in base currency to a number of asset units.

    Formula:
        units = trade_value / unit_price

    Args:
        trade_value (float): Value in base currency (sign preserved).
        unit_price (float): Price per unit in base currency (> 0).

    Returns:
        float: Number of units, same sign as ``trade_value``.

    Edge cases:
        unit_price == 0 -> ``ZeroDivisionError`` for Python floats, inf for numpy.
    """
    units = trade_value / unit_price
    return units


def total_wallet_value(
    wallet_value_available: float, wallet_volume_available: float, unit_price: float
):
    """Compute total wallet value as cash plus the market value of held units.

    Formula:
        total = wallet_value_available + wallet_volume_available * unit_price

    Args:
        wallet_value_available (float): Cash balance in base currency.
        wallet_volume_available (float): Asset units held.
        unit_price (float): Price per unit in base currency.

    Returns:
        float: Total wallet value in base currency.

    Edge cases:
        Not called by ``update_wallet``, which inlines the same formula.
    """
    total_value = wallet_value_available + (wallet_volume_available * unit_price)
    return total_value


def unit_to_price_conversion(trade_units: float, unit_price: float):
    """Convert a number of asset units to a value in base currency.

    Formula:
        value = trade_units * unit_price

    Args:
        trade_units (float): Number of units (sign preserved; negative for a SELL).
        unit_price (float): Price per unit in base currency.

    Returns:
        float: Value in base currency, same sign as ``trade_units``.
    """
    value = trade_units * unit_price
    return value
