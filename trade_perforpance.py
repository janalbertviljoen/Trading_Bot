import numpy as np


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
