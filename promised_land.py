import pandas as pd
import itertools

import read_data
import trade_indicators
import trade_wallet
import plot_analysis

# import trade_perforpance
EXTRACT_COL = [
    "timestamp",
    "total_wallet_value",
    "normalised_wallet_value",
    "normalised_price",
]


class PromisedLandTrader:
    def __init__(
        self,
        dict_read_data: dict,
        dict_detect_trades: dict,
        dict_trade_logic: dict,
        dict_detect_range: dict,
    ):
        """
        Initialise the trading bot using structured configuration sections.
        """
        self.read_config = dict_read_data
        self.detect_config = dict_detect_trades
        self.trade_config = dict_trade_logic
        self.detect_config_range = dict_detect_range

    def optimise_strategy(self):
        self.evalaute_strategy()
        self.df_timeseries = self.df_base.copy()
        self.strategic_points = []

        i = 1
        optimise_detection = self.generate_combinations()
        for dict in optimise_detection:
            for param, p in dict.items():
                self.detect_config[param] = p
                self.evalaute_strategy(extract=False)
                self.summarise_performance(coord=f"{i}")
                i += 1

        self.df_strategic_points = pd.DataFrame(self.strategic_points)

    def evalaute_strategy(self, extract=True):
        if extract:
            self.df_base = read_data.fetch_btc_history(**self.read_config)
            df = self.df_base.copy()
        else:
            df = self.df_base
        df = trade_indicators.detect_all(df, **self.detect_config)
        df = trade_wallet.make_trades(df, **self.trade_config)
        self.df = df

    def generate_combinations(self):
        # Generate all combinations
        keys = self.detect_config_range.keys()
        values = self.detect_config_range.values()

        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        return combinations

    def summarise_performance(self, coord: int):
        df = self.df.copy()
        performance = df.tail(1).to_dict(orient="records")[0]
        df = self.rename_columns(df, tag=coord)
        self.df_timeseries = self.df_timeseries.merge(df, on="timestamp")

        paramaters = self.read_config | self.detect_config | self.trade_config
        paramaters["coordinate"] = coord
        paramaters = paramaters | performance
        self.strategic_points.append(paramaters)

    def rename_columns(self, df, tag="0"):
        rename_col = {i: f"{i}_{tag}" for i in EXTRACT_COL[1:]}
        df_p = df[EXTRACT_COL].rename(columns=rename_col)
        return df_p

    def perform_analysis(self, sample_size=None):
        if sample_size:
            df_sample = self.df.tail(sample_size)
        else:
            df_sample = self.df.copy()
        plot_analysis.plot_trade_events(df_sample)
        plot_analysis.plot_wallet_outcomes(df_sample)
        plot_analysis.plot_normalised_wallet(df_sample)
