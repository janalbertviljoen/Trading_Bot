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

        self.df_sp = df_sp
        self.df_stat_d = df_stat_d
        self.df_stat_t = df_stat_t

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
