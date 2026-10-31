"""Performance and cycle-analysis helpers for price series (not used by the wallet run)."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd


# TODO: Split to seperate file: "performance.py"
def best_bot(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the theoretical maximum profit per time step from the high-low range.

    Formula:
        max_profit = high - low

    Args:
        df (pandas.DataFrame): Must contain ``high`` and ``low`` (float64, base
            currency per unit).

    Returns:
        pandas.DataFrame: The same object as ``df`` (mutated in place) with an added
        ``max_profit`` column (float64, base currency per unit, >= 0 for valid
        bars).

    Assumptions:
        A perfect trader buys at every bar's low and sells at its high, with no
        fees.

    Raises:
        KeyError: ``high`` or ``low`` is missing.
    """
    df["max_profit"] = df["high"] - df["low"]
    return df


def detrend_series(
    series: pd.Series[float], method: str = "log_return"
) -> pd.Series[float]:
    """Remove the trend from a price series.

    Formula:
        log_return: r_t = ln(p_t) - ln(p_{t-1})
        demean:     d_t = p_t - mean(p_{t-49..t})   (50-step rolling mean)

    Args:
        series (pandas.Series): Prices (float64, > 0 for ``log_return``).
        method (str): "log_return" (default) or "demean".

    Returns:
        pandas.Series: float64. ``log_return`` drops the first row (n - 1 values);
        ``demean`` keeps the length with NaN for the first 49 rows.

    Raises:
        ValueError: Unknown ``method``.
    """
    if method == "log_return":
        # numpy ufuncs return a Series for Series input; the stubs say ndarray.
        return cast("pd.Series[float]", np.log(series)).diff().dropna()
    elif method == "demean":
        return series - series.rolling(50).mean()
    else:
        raise ValueError("Unknown detrending method")


def rolling_fft(
    series: pd.Series[float], window: int
) -> tuple[npt.NDArray[np.float64], list[Hashable]]:
    """Compute the FFT magnitude spectrum over a trailing rolling window.

    Formula:
        For i in [window, n): |FFT(series[i - window : i])|

    Args:
        series (pandas.Series): Detrended values (float64), length n.
        window (int): Number of samples per FFT (> 0).

    Returns:
        tuple[numpy.ndarray, list]: Magnitudes of shape (n - window, window),
        float64; and the index labels ``series.index[i]`` each row belongs to.

    Assumptions:
        Each window ends at i - 1, so row i uses no data at or after i.

    Edge cases:
        n <= window -> empty array of shape (0,) and an empty list.
    """
    fft_results: list[npt.NDArray[np.float64]] = []
    index: list[Hashable] = []

    for i in range(window, len(series)):
        segment = series.iloc[i - window : i].to_numpy()
        fft_vals = np.fft.fft(segment)
        fft_mag = np.abs(fft_vals)

        fft_results.append(fft_mag)
        index.append(series.index[i])

    return np.array(fft_results), index


def fft_frequencies(
    window: int, sampling_interval: float = 1.0
) -> npt.NDArray[np.floating[Any]]:
    """Return the FFT sample frequencies for a window length.

    Args:
        window (int): Number of samples per FFT.
        sampling_interval (float): Time between samples, in the time unit of the
            series (e.g. 1.0 = one bar). Default 1.0.

    Returns:
        numpy.ndarray: float64 frequencies of shape (window,), in cycles per
        ``sampling_interval`` unit (``numpy.fft.fftfreq`` order).
    """
    return np.fft.fftfreq(window, d=sampling_interval)


def dominant_cycle(
    series: pd.Series[float], window: int, sampling_interval: float = 1.0
) -> npt.NDArray[np.floating[Any]]:
    """Find the dominant cycle length in each rolling window.

    Formula:
        period_i = 1 / argmax_{f > 0} |FFT_i(f)|

    Args:
        series (pandas.Series): Detrended values (float64).
        window (int): Samples per FFT (> 1).
        sampling_interval (float): Time between samples. Default 1.0.

    Returns:
        numpy.ndarray: float64 periods of shape (n - window,), in the time unit of
        ``sampling_interval``; bounds (2, window] samples.

    Raises:
        IndexError: n <= window (empty spectrum).
    """
    fft_mag, _ = rolling_fft(series, window)
    freqs = fft_frequencies(window, sampling_interval)

    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]

    dom_freq = pos_freqs[np.argmax(fft_mag[:, pos_mask], axis=1)]
    return 1 / dom_freq
