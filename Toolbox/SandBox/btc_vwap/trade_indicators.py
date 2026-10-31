"""Price indicators and deviation signals that feed ``trade_wallet``."""

from __future__ import annotations

import pandas as pd

ALL_METRICS: list[str] = ["vwap"]


def detect_all(df: pd.DataFrame, threshold: float, rolling_window: int) -> pd.DataFrame:
    """Compute every indicator in ``ALL_METRICS`` and flag deviations from it.

    Args:
        df (pandas.DataFrame): Columns ``price`` and ``volume`` (float64).
        threshold (float): Relative deviation that triggers a signal (e.g. 0.03).
        rolling_window (int): Window length in rows for the indicators.

    Returns:
        pandas.DataFrame: New frame with ``vwap``, ``deviation`` (float64) and
        ``signal`` (bool) added.

    Edge cases:
        ``rolling_vwap`` adds ``vwap`` to the input ``df`` in place.
        With several metrics, ``deviation`` / ``signal`` hold the last one only.
    """
    df = rolling_vwap(df, rolling_window)
    for metric in ALL_METRICS:
        df = detect_deviation(df, metric_name=metric, threshold=threshold)
    return df


def rolling_vwap(
    df: pd.DataFrame,
    rolling_window: int,
    price_col: str = "price",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Compute the volume-weighted average price over a trailing rolling window.

    Formula:
        vwap_t = sum(price * volume over last w rows) / sum(volume over last w rows)

    Args:
        df (pandas.DataFrame): Contains ``price_col`` and ``volume_col`` (float64).
        rolling_window (int): Window length w in rows (> 0).
        price_col (str): Price column name. Default "price".
        volume_col (str): Volume column name. Default "volume".

    Returns:
        pandas.DataFrame: The same object as ``df`` (mutated in place) with a
        ``vwap`` column (float64, price units).

    Edge cases:
        First w - 1 rows -> NaN. Zero volume over a window -> NaN (0 / 0).
    """
    pv = df[price_col] * df[volume_col]
    vwap = (
        pv.rolling(window=rolling_window, min_periods=rolling_window).sum()
        / df[volume_col]
        .rolling(window=rolling_window, min_periods=rolling_window)
        .sum()
    )
    df["vwap"] = vwap
    return df


def detect_deviation(
    df: pd.DataFrame, metric_name: str, threshold: float, unit_price: str = "price"
) -> pd.DataFrame:
    """Flag rows where price deviates from a reference metric by more than a threshold.

    Formula:
        deviation = (price - metric) / metric
        signal    = |deviation| > threshold

    Args:
        df (pandas.DataFrame): Contains ``unit_price`` and ``metric_name`` columns
            (float64).
        metric_name (str): Reference column, e.g. "vwap".
        threshold (float): Relative deviation threshold (>= 0); strict inequality.
        unit_price (str): Price column. Default "price".

    Returns:
        pandas.DataFrame: Copy of ``df`` with ``deviation`` (float64,
        dimensionless) and ``signal`` (bool).

    Edge cases:
        NaN metric -> deviation NaN, signal False.
    """
    df = df.copy()
    df["deviation"] = (df[unit_price] - df[metric_name]) / df[metric_name]
    df["signal"] = df["deviation"].abs() > threshold
    return df
