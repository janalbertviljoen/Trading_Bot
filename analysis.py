import numpy as np
from promised_land import PromisedLandTrader


class PromisedLandExplorer:

    def __init__(self, trader: PromisedLandTrader):
        self.trader = trader
        self.df_sp = trader.df_strategic_points
        self.df_ts = trader.df_timeseries
        self.params_detect = list(trader.detect_config_range.keys())
        self.params_trade = list(trader.trade_config_range.keys())

    def pre_process(self, metric="mean_normalised_wallet"):

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
        df_ts_top = df_ts[["timestamp"] + col_name]
        normalised_columns = list(
            df_ts_top.columns[df_ts_top.columns.str.contains("normalised_wallet")]
        )
        df_ts_top = df_ts_top[["timestamp"] + normalised_columns]
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

    def calc_stat(self, df, params, metric):
        df_grp = (
            df[[metric] + params]
            .groupby(params)[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        return df_grp

    def calc_weighted_metric(self, df, type):
        rename = {"mean": f"{type}_mean", "std": f"{type}_std"}
        df[f"{type}_weighted"] = df["mean"] / (df["std"] + 1)
        df = df.rename(columns=rename)
        return df

    def extract_ts_cols(self, df, coordinates: list):
        col_name = []
        for c in coordinates:
            col_name += list(df.columns[df.columns.str.contains(f"_{c}$")])
        return col_name
