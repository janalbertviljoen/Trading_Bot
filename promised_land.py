"""Strategy runner: fetch prices, generate signals, run the wallet and grid-search parameters.

``PromisedLandTrader`` chains the pipeline
    read_data.fetch_btc_history -> trade_indicators.detect_all -> trade_wallet.make_trades
and repeats it over every combination of a parameter grid, storing the final
performance of each combination (``df_strategic_points``) and the wallet time series
of each run side by side (``df_timeseries``).
"""

from __future__ import annotations

import itertools
from typing import Any, cast

import pandas as pd

import plot_analysis
import read_data
import trade_indicators
import trade_wallet

# import trade_perforpance
EXTRACT_COL: list[str] = [
    "timestamp",
    "total_wallet_value",
    "normalised_wallet_value",
    "normalised_price",
]


class PromisedLandTrader:
    """Grid-search runner over signal-detection and wallet parameters.

    Attributes set after ``evalaute_strategy`` / ``optimise_strategy``:
        df_base (pandas.DataFrame): Raw price data from ``read_data``
            (timestamp, high, low, price, volume).
        df (pandas.DataFrame): Result of the latest run: ``df_base`` plus indicator,
            signal, direction and wallet columns (see ``trade_wallet.update_wallet``).
        df_timeseries (pandas.DataFrame): ``df_base`` + ``normalised_price``, plus
            ``<col>_<coordinate>`` for every ``EXTRACT_COL`` column of every run.
        df_strategic_points (pandas.DataFrame): One row per combination: all
            parameters, ``coordinate`` (str), ``mean_normalised_wallet`` (float64)
            and the final row of that run's ``df``.
    """

    def __init__(
        self,
        dict_read_data: dict[str, Any],
        dict_detect_trades: dict[str, Any],
        dict_trade_logic: dict[str, Any],
        dict_detect_range: dict[str, list[Any]],
        dict_trade_range: dict[str, list[Any]],
    ) -> None:
        """Store the configuration sections used by every run.

        Args:
            dict_read_data (dict): Keyword arguments for
                ``read_data.fetch_btc_history`` (``interval``, ``start_date``,
                ``symbol``).
            dict_detect_trades (dict): Keyword arguments for
                ``trade_indicators.detect_all`` (``threshold`` float,
                ``rolling_window`` int).
            dict_trade_logic (dict): Keyword arguments for
                ``trade_wallet.make_trades`` (``initial_investment``,
                ``initial_trade_fraction``, ``trade_fraction_cash``,
                ``trade_fraction_volume``).
            dict_detect_range (dict[str, list]): Candidate values per detection
                parameter, e.g. ``{"threshold": [0.01, 0.02]}``.
            dict_trade_range (dict[str, list]): Candidate values per wallet
                parameter.

        Edge cases:
            The dicts are stored by reference; ``optimise_strategy`` overwrites
            values in ``dict_detect_trades`` and ``dict_trade_logic`` in place.
        """
        self.read_config = dict_read_data
        self.detect_config = dict_detect_trades
        self.trade_config = dict_trade_logic
        self.detect_config_range = dict_detect_range
        self.trade_config_range = dict_trade_range
        self.df_base: pd.DataFrame
        self.df: pd.DataFrame
        self.df_timeseries: pd.DataFrame
        self.df_strategic_points: pd.DataFrame
        self.strategic_points: list[dict[str, Any]] = []

    def optimise_strategy(self, extract: bool = True) -> None:
        """Run the strategy for every parameter combination and collect the results.

        Computes:
            1. A base run with the current configs (fetching data if ``extract``).
            2. For each combination from ``generate_combinations``: set the values
               in the detect / trade configs, re-run on ``df_base`` and record the
               result via ``summarise_performance`` with coordinate "1", "2", ...

        Args:
            extract (bool): If True, fetch fresh data from Binance for the base run;
                if False, reuse the existing ``df_base``. Default True.

        Returns:
            None. Sets ``df_timeseries`` and ``df_strategic_points``.

        Assumptions:
            Every grid key exists in ``detect_config`` or ``trade_config``; keys in
            neither are silently ignored.

        Edge cases:
            Mutates ``detect_config`` / ``trade_config``; after the call they hold
            the last combination, not the original values.
            The full in-sample series is used; there is no holdout split.

        Raises:
            AttributeError: ``extract`` is False and ``df_base`` was never set.
        """
        self.evalaute_strategy(extract=extract)
        self.df_timeseries = self.df_base.copy()
        self.df_timeseries["normalised_price"] = self.df["normalised_price"]
        self.strategic_points = []

        i = 1
        optimise_detection = self.generate_combinations()
        for dict in optimise_detection:
            print(f"Progress: {(i / len(optimise_detection)*100):.2f} %", end="\r")
            if not dict:
                continue
            for param, p in dict.items():
                if param in self.detect_config:
                    self.detect_config[param] = p
                elif param in self.trade_config:
                    self.trade_config[param] = p
            self.evalaute_strategy(extract=False)
            self.summarise_performance(coord=f"{i}")
            i += 1

        self.df_strategic_points = pd.DataFrame(self.strategic_points)

    def evalaute_strategy(self, extract: bool = True) -> None:
        """Run one pass of signal detection and wallet simulation.

        Computes:
            df = make_trades(detect_all(df_base, **detect_config), **trade_config)

        Args:
            extract (bool): If True, fetch data with ``read_config`` into
                ``df_base`` and run on a copy; if False, run on ``df_base``.
                Default True.

        Returns:
            None. Sets ``self.df`` (and ``self.df_base`` when ``extract``).

        Edge cases:
            With ``extract=False``, ``rolling_vwap`` writes a ``vwap`` column into
            ``df_base`` itself (overwritten on each run).

        Raises:
            AttributeError: ``extract`` is False and ``df_base`` was never set.
        """
        if extract:
            self.df_base = read_data.fetch_btc_history(**self.read_config)
            df = self.df_base.copy()
        else:
            df = self.df_base
        df = trade_indicators.detect_all(df, **self.detect_config)
        df = trade_wallet.make_trades(df, **self.trade_config)
        self.df = df

    def generate_combinations(self) -> list[dict[str, Any]]:
        """Build the Cartesian product of the detection and wallet parameter grids.

        Returns:
            list[dict]: One dict per combination mapping parameter name to value,
            in ``itertools.product`` order (last key varies fastest). Length is the
            product of the candidate list lengths.

        Edge cases:
            A key present in both ranges takes the trade range's values.
            Both ranges empty -> ``[{}]`` (one empty combination, skipped later).
        """
        # Generate all combinations
        param_config = self.detect_config_range | self.trade_config_range
        keys = param_config.keys()
        values = param_config.values()

        combinations = [
            dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)
        ]
        return combinations

    def summarise_performance(self, coord: str) -> None:
        """Record the latest run's parameters, final state and wallet time series.

        Computes:
            mean_normalised_wallet = mean(df["normalised_wallet_value"])
            One record = read/detect/trade configs + coordinate +
                mean_normalised_wallet + every column of the run's last row.

        Args:
            coord (str): Label of this combination, used as the column suffix in
                ``df_timeseries`` and as ``coordinate`` in the record, e.g. "3".

        Returns:
            None. Appends to ``strategic_points`` and inner-joins the run's
            ``EXTRACT_COL`` columns (suffixed ``_<coord>``) onto ``df_timeseries``
            by ``timestamp``.

        Edge cases:
            Last-row columns override config keys with the same name.
        """
        df = self.df.copy()

        mean = df["normalised_wallet_value"].mean()
        # Column labels are str, so the record is keyed by str.
        performance = cast("dict[str, Any]", df.tail(1).to_dict(orient="records")[0])
        df = self.rename_columns(df, tag=coord)
        self.df_timeseries = self.df_timeseries.merge(df, on="timestamp")

        paramaters = self.read_config | self.detect_config | self.trade_config
        paramaters["coordinate"] = coord
        paramaters["mean_normalised_wallet"] = mean
        paramaters = paramaters | performance
        self.strategic_points.append(paramaters)

    def rename_columns(self, df: pd.DataFrame, tag: str = "0") -> pd.DataFrame:
        """Select ``EXTRACT_COL`` and suffix every column except ``timestamp``.

        Args:
            df (pandas.DataFrame): Run output containing all ``EXTRACT_COL`` columns.
            tag (str): Suffix appended as ``<col>_<tag>``. Default "0".

        Returns:
            pandas.DataFrame: New frame with ``timestamp`` (datetime64[ns]) and
            ``total_wallet_value_<tag>``, ``normalised_wallet_value_<tag>``,
            ``normalised_price_<tag>`` (float64).

        Raises:
            KeyError: A column of ``EXTRACT_COL`` is missing.
        """
        rename_col = {i: f"{i}_{tag}" for i in EXTRACT_COL[1:]}
        df_p = df[EXTRACT_COL].rename(columns=rename_col)
        return df_p

    def perform_analysis(self, sample_size: int | None = None) -> None:
        """Plot trade events, wallet outcomes and normalised wallet for the latest run.

        Args:
            sample_size (int | None): If set, plot only the last ``sample_size``
                rows of ``self.df``; otherwise plot all rows. Default None.

        Returns:
            None. Draws three matplotlib figures (see ``plot_analysis``).
        """
        df_sample = self.df.tail(sample_size) if sample_size else self.df.copy()
        plot_analysis.plot_trade_events(df_sample)
        plot_analysis.plot_wallet_outcomes(df_sample)
        plot_analysis.plot_normalised_wallet(df_sample)
