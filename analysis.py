"""Post-processing of a ``PromisedLandTrader`` grid search: parameter robustness statistics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from promised_land import PromisedLandTrader


class PromisedLandExplorer:
    """Analyse which parameter values perform well consistently across a grid search.

    Attributes set after ``pre_process``:
        df_sp (pandas.DataFrame): Strategic points sorted by the metric, merged with
            the detect / trade statistics and ``final_weighted`` (float64).
        df_stat_d (pandas.DataFrame): Metric mean / std / weighted score per
            detection-parameter combination.
        df_stat_t (pandas.DataFrame): Same per wallet-parameter combination.
        df_ts_norm (pandas.DataFrame): Each run's ``normalised_wallet_value_<c>``
            and ``normalised_price_<c>`` divided by the buy-and-hold
            ``normalised_price``, plus ``timestamp``.
        highest_coor (pandas.Series): Coordinates whose time series were extracted.
    """

    def __init__(self, trader: PromisedLandTrader) -> None:
        """Take the results of a finished grid search.

        Args:
            trader (PromisedLandTrader): Trader after ``optimise_strategy`` has run
                (``df_strategic_points`` and ``df_timeseries`` must exist).

        Raises:
            AttributeError: ``optimise_strategy`` has not been run on ``trader``.
        """
        self.trader = trader
        self.df_sp = trader.df_strategic_points
        self.df_ts = trader.df_timeseries
        self.params_detect = list(trader.detect_config_range.keys())
        self.params_trade = list(trader.trade_config_range.keys())

    def pre_process(self, metric: str = "mean_normalised_wallet") -> None:
        """Score parameter robustness and extract the top runs' time series.

        Computes:
            For detection and wallet parameters separately, grouped by parameter
            values:
                <type>_mean, <type>_std = mean, std of ``metric``
                <type>_weighted = <type>_mean / (<type>_std + 1)
            final_weighted = sqrt(detect_weighted**2 + trade_weighted**2)
            df_ts_norm = normalised wallet columns / normalised_price
                (> 1 means the run beat buy-and-hold at that time step).

        Args:
            metric (str): Column of ``df_strategic_points`` to score (float64).
                Default "mean_normalised_wallet".

        Returns:
            None. Sets ``df_sp``, ``df_stat_d``, ``df_stat_t``, ``df_ts_norm`` and
            ``highest_coor``.

        Edge cases:
            ``highest_coor`` is taken from ``self.df_sp`` before it is replaced by
            the sorted frame, so on the first call it is the first five
            coordinates in run order, not the five best by ``metric``.
            A parameter group with a single run has std NaN, so its weighted
            score is NaN.

        Raises:
            KeyError: ``metric`` or a parameter column is missing.
        """

        ## Strategic Points
        df_sp = self.df_sp.sort_values(metric, ascending=False).copy()

        df_stat_d = self.calc_stat(df_sp, self.params_detect, metric)
        df_stat_t = self.calc_stat(df_sp, self.params_trade, metric)

        df_stat_d = self.calc_weighted_metric(df_stat_d, type="detect")
        df_stat_t = self.calc_weighted_metric(df_stat_t, type="trade")

        df_sp = df_sp.merge(df_stat_d, on=self.params_detect)
        df_sp = df_sp.merge(df_stat_t, on=self.params_trade)
        df_sp["final_weighted"] = np.sqrt(
            df_sp["detect_weighted"] ** 2 + df_sp["trade_weighted"] ** 2
        )

        ## Time series
        df_ts = self.df_ts
        highest_comb = self.df_sp.head(5)["coordinate"].reset_index(drop=True)
        col_name = self.extract_ts_cols(self.df_ts, highest_comb)
        df_ts_top = df_ts[["timestamp", *col_name]]
        normalised_columns = list(
            df_ts_top.columns[df_ts_top.columns.str.contains("normalised_wallet")]
        )
        df_ts_top = df_ts_top[["timestamp", *normalised_columns]]
        df_ts_norm = df_ts[normalised_columns].div(df_ts["normalised_price"], axis=0)
        df_ts_norm = df_ts_norm.merge(
            df_ts[["timestamp"]], left_index=True, right_index=True
        )

        ## Store data tables
        self.df_sp = df_sp
        self.df_stat_d = df_stat_d
        self.df_stat_t = df_stat_t
        self.df_ts_norm = df_ts_norm
        self.highest_coor = highest_comb

    def calc_stat(
        self, df: pd.DataFrame, params: list[str], metric: str
    ) -> pd.DataFrame:
        """Compute the mean and standard deviation of a metric per parameter group.

        Args:
            df (pandas.DataFrame): Strategic points with ``metric`` and ``params``
                columns.
            params (list[str]): Parameter columns to group by.
            metric (str): Numeric column to aggregate.

        Returns:
            pandas.DataFrame: One row per unique parameter combination with the
            ``params`` columns plus ``mean`` and ``std`` (float64, sample std,
            ddof=1).

        Raises:
            ValueError: ``params`` is empty (``groupby([])``).
        """
        df_grp = (
            df[[metric, *params]]
            .groupby(params)[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        return df_grp

    def calc_weighted_metric(self, df: pd.DataFrame, type: str) -> pd.DataFrame:
        """Add a stability-weighted score and prefix the statistic columns.

        Formula:
            <type>_weighted = mean / (std + 1)

        Args:
            df (pandas.DataFrame): Output of ``calc_stat`` (``mean``, ``std``).
            type (str): Prefix, e.g. "detect" or "trade".

        Returns:
            pandas.DataFrame: ``df`` with ``mean`` / ``std`` renamed to
            ``<type>_mean`` / ``<type>_std`` and an added ``<type>_weighted``
            column (float64). Higher means a good and consistent metric.

        Edge cases:
            Adds the ``<type>_weighted`` column to the input ``df`` in place before
            renaming into a new frame.
        """
        rename = {"mean": f"{type}_mean", "std": f"{type}_std"}
        df[f"{type}_weighted"] = df["mean"] / (df["std"] + 1)
        df = df.rename(columns=rename)
        return df

    def extract_ts_cols(
        self, df: pd.DataFrame, coordinates: Iterable[str]
    ) -> list[str]:
        """List the time-series columns belonging to the given run coordinates.

        Args:
            df (pandas.DataFrame): ``df_timeseries`` with ``<col>_<coordinate>``
                columns.
            coordinates (Iterable[str]): Run coordinates, e.g. ``["3", "12"]``.

        Returns:
            list[str]: Column names ending in ``_<coordinate>``, grouped by
            coordinate in the given order.

        Edge cases:
            Coordinates are used in a regex unescaped; numeric strings are safe.
        """
        col_name: list[str] = []
        for c in coordinates:
            col_name += list(df.columns[df.columns.str.contains(f"_{c}$")])
        return col_name
