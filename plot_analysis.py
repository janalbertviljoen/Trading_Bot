"""Matplotlib plots of wallet runs and parameter-grid statistics."""

import pylab as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib.patches import Rectangle


def plot_trade_events(df):
    """Plot price, VWAP and deviation with BUY / SELL markers.

    Args:
        df (pandas.DataFrame): Wallet run with ``timestamp``, ``price``, ``vwap``,
            ``deviation`` and ``direction`` (1 BUY, -1 SELL).

    Returns:
        None. Draws one figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    df.plot(ax=ax, x="timestamp", y="price")
    df.plot(ax=ax, x="timestamp", y="deviation", secondary_y=True, alpha=0.2, ls="-.")
    df.plot(ax=ax, x="timestamp", y="vwap", color="k", ls=":")
    df[df["direction"] == 1].plot.scatter(
        ax=ax, x="timestamp", y="price", color="green", label="buy"
    )
    df[df["direction"] == -1].plot.scatter(
        ax=ax, x="timestamp", y="price", color="red", label="sell"
    )
    ax.legend(ncols=5)


def plot_wallet_outcomes(df):
    """Plot total wallet value against price, and held volume, over time.

    Args:
        df (pandas.DataFrame): Wallet run with ``timestamp``, ``total_wallet_value``,
            ``price`` and ``wallet_volume_available``.

    Returns:
        None. Draws one figure with two panels (axis labels assume USD / BTC).
    """
    fig, ax = plt.subplots(ncols=2, figsize=(18, 6))

    # Left panel
    df.plot(
        ax=ax[0],
        x="timestamp",
        y="total_wallet_value",
        ylabel="Wallet value (USD)",
        title="Value in USD",
    )
    ax[0].yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax2 = ax[0].twinx()
    df.plot(ax=ax2, x="timestamp", y="price", alpha=0.5, ylabel="USD/BTC (price)")
    ax2.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Right panel
    df.plot(
        ax=ax[1],
        x="timestamp",
        y="wallet_volume_available",
        ylabel="Wallet volume (BTC)",
        title="Volume of BTC",
    )
    ax[1].yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def plot_normalised_wallet(df):
    """Plot the normalised wallet value against the buy-and-hold normalised price.

    Args:
        df (pandas.DataFrame): Wallet run with ``timestamp``,
            ``normalised_wallet_value`` and ``normalised_price``.

    Returns:
        None. Draws one figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    df.plot(
        ax=ax,
        x="timestamp",
        y="normalised_wallet_value",
        ylabel="Factor of gains",
        title="Normalised value",
    )
    df.plot(ax=ax, x="timestamp", y="normalised_price", alpha=0.5)


def plot_imshow(df_stat):
    """Plot a 2-D parameter grid as a heatmap and outline its maximum cell.

    Args:
        df_stat (pandas.DataFrame): Numeric pivot table (index = one parameter,
            columns = another, values = metric).

    Returns:
        None. Draws and shows one figure.

    Raises:
        ValueError: All values are NaN (``nanargmax``).
    """
    fig, ax = plt.subplots()

    # Plot heatmap
    img = ax.imshow(df_stat.values, aspect="auto", cmap="viridis")

    # Find position of maximum value
    max_pos = np.unravel_index(np.nanargmax(df_stat.values), df_stat.shape)
    y_max, x_max = max_pos  # row, column

    # Draw red rectangle around the max cell
    rect = Rectangle(
        (x_max - 0.5, y_max - 0.5), 1, 1, linewidth=2, edgecolor="red", facecolor="none"
    )
    ax.add_patch(rect)

    # X-axis: use column labels as ranges
    ax.set_xticks(np.arange(len(df_stat.columns)))
    ax.set_xticklabels(df_stat.columns, rotation=45, ha="right")

    # Y-axis: use index values
    ax.set_yticks(np.arange(len(df_stat.index)))
    ax.set_yticklabels(df_stat.index)

    # Labels
    ax.set_xlabel(df_stat.columns.name if df_stat.columns.name else "Range")
    ax.set_ylabel(df_stat.index.name if df_stat.index.name else "Index")

    # Colorbar
    fig.colorbar(img, ax=ax)

    plt.tight_layout()
    plt.show()
