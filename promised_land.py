import read_data
import trade_indicators
import trade_wallet
import plot_analysis

# import trade_perforpance


class PromisedLandTrader:
    def __init__(
        self,
        dict_read_data: dict,
        dict_detect_trades: dict,
        dict_trade_logic: dict,
    ):
        """
        Initialise the trading bot using structured configuration sections.
        """
        self.read_config = dict_read_data
        self.detect_config = dict_detect_trades
        self.trade_config = dict_trade_logic

    def evalaute_strategy(self):
        df = read_data.fetch_btc_history(**self.read_config)
        df = trade_indicators.detect_all(df, **self.detect_config)
        df = trade_wallet.make_trades(df, **self.trade_config)
        self.df = df

    def perform_analysis(self, sample_size=None):
        if sample_size:
            df_sample = self.df.tail(sample_size)
        else:
            df_sample = self.df.copy()
        plot_analysis.plot_trade_events(df_sample)
        plot_analysis.plot_wallet_outcomes(df_sample)
        plot_analysis.plot_normalised_wallet(df_sample)
