ALL_METRICS = ["vwap"]


def detect_all(df, threshold, rolling_window):
    df = rolling_vwap(df, rolling_window)
    for metric in ALL_METRICS:
        df = detect_deviation(df, metric_name=metric, threshold=threshold)
    return df


def rolling_vwap(df, rolling_window, price_col="price", volume_col="volume"):
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
        pv.rolling(window=rolling_window, min_periods=rolling_window).sum()
        / df[volume_col]
        .rolling(window=rolling_window, min_periods=rolling_window)
        .sum()
    )
    df["vwap"] = vwap
    return df


def detect_deviation(df, metric_name: str, threshold: float, unit_price="price"):
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
