import pandas as pd
from pandas import Series

class Source():
    def __init__(
        self, 
        open=[],
        high=[],
        low=[],
        close=[],
    ):
        self.open = pd.Series(open)
        self.high = pd.Series(high)
        self.low = pd.Series(low)
        self.close = pd.Series(close)

class Label():
    def __init__(
        self, 
        short: int, 
        long: int, 
        neutral: int,
    ):
        self.long = long
        self.short = short
        self.neutral = neutral

class Feature_Series():
    def __init__(
            self, 
            ADX_1: float,
            RSI_1: float,
            RSI_2: float,
            CCI_1: float,
            WT_1: float,
        ):
        self.ADX_1 = ADX_1
        self.RSI_1 = RSI_1
        self.RSI_2 = RSI_2
        self.CCI_1 = CCI_1
        self.WT_1 = WT_1

class Feature_Arrays():
    def __init__(
            self, 
            ADX_1: Series,
            RSI_1: Series,
            RSI_2: Series,
            CCI_1: Series,
            WT_1: Series,
        ):
        self.ADX_1 = ADX_1
        self.RSI_1 = RSI_1
        self.RSI_2 = RSI_2
        self.CCI_1 = CCI_1
        self.WT_1 = WT_1

class ML_Model():
    def __init__(
        self, 
        y_train_array: Series,
        y_test_array: list,
        distances: list,
        predictions: list,
        signals: list,
        yhat1s: list,
        yhat2s: list,
        last_distance: float,
        prediction: int | float,
        signal: int | float,
        bars_held: int,
    ):
        self.y_train_array = y_train_array
        self.y_test_array = y_test_array
        self.distances = distances
        self.predictions = predictions
        self.signals = signals
        self.yhat1s = yhat1s
        self.yhat2s = yhat2s
        self.last_distance = last_distance
        self.prediction = prediction
        self.signal = signal
        self.bars_held = bars_held

class Filter_Arrays():
    def __init__(
        self,
        is_EMA_up_trends,
        is_EMA_down_trends,
        is_SMA_up_trends,
        is_SMA_down_trends,
        is_valid_short_exits,
        is_valid_long_exits,
        start_short_trades,
        start_long_trades,
    ):
        self.is_EMA_up_trends = is_EMA_up_trends
        self.is_EMA_down_trends = is_EMA_down_trends
        self.is_SMA_up_trends = is_SMA_up_trends
        self.is_SMA_down_trends = is_SMA_down_trends
        self.is_valid_short_exits = is_valid_short_exits
        self.is_valid_long_exits = is_valid_long_exits
        self.start_short_trades = start_short_trades
        self.start_long_trades = start_long_trades            

class Filter():
    def __init__(
        self,
        volatility: bool,
        ADX: bool,
        regime: bool,
        is_EMA_up_trend: bool,
        is_EMA_down_trend: bool,
        is_SMA_up_trend: bool,
        is_SMA_down_trend: bool,
        is_held_less_n_bars: bool,
        is_held_n_bars: bool,
        is_diff_signal_type: bool,
        is_early_signal_flip: bool,
        is_short_signal: bool,
        is_last_buy_signal: bool,
        is_last_short_signal: bool,
        is_new_buy_signal: bool,
        is_new_short_signal: bool,
        was_bearish_rate: bool,
        was_bullish_rate: bool,
        is_bearish_rate: bool,
        is_bullish_rate: bool,
        is_bearish_change: bool,
        is_bullish_change: bool,
        is_bearish_cross_alert: bool,
        is_bullish_cross_alert: bool,
        is_bearish_smooth: bool,
        is_bullish_smooth: bool,
        alert_bearish: bool,
        alert_bullish: bool,
        is_bearish: bool,
        is_bullish: bool,
        start_short_trade: bool,
        start_long_trade: bool,
        last_signal_was_bearish: bool,
        last_signal_was_bullish: bool,
        bars_since_red_entry: bool,
        bars_since_red_exit: bool,
        bars_since_green_entry: bool,
        bars_since_green_exit: bool,
        is_valid_short_exit: bool,
        is_valid_long_exit: bool,
        end_short_trade_dynamic: bool,
        end_long_trade_dynamic: bool,

        end_long_trade_strict: bool,
        end_short_trade_strict: bool,
        is_dynamic_exit_valid: bool,
        end_long_trade: bool,
        end_short_trade: bool,

        is_short,
        is_long,
    ):
        # default
        self.volatility = volatility
        self.ADX = ADX
        self.regime = regime

        # SMA/EMA 
        self.is_EMA_up_trend = is_EMA_up_trend
        self.is_EMA_down_trend = is_EMA_down_trend
        self.is_SMA_up_trend = is_SMA_up_trend
        self.is_SMA_down_trend = is_SMA_down_trend
        
        # holding bars
        self.is_held_n_bars = is_held_n_bars
        self.is_held_less_n_bars = is_held_less_n_bars

        # fractal
        self.is_diff_signal_type = is_diff_signal_type
        self.is_early_signal_flip = is_early_signal_flip
        self.is_short_signal = is_short_signal
        self.is_last_buy_signal = is_last_buy_signal
        self.is_last_short_signal = is_last_short_signal
        self.is_new_buy_signal = is_new_buy_signal
        self.is_new_short_signal = is_new_short_signal

        # Kernel
        self.was_bearish_rate = was_bearish_rate
        self.was_bullish_rate = was_bullish_rate
        self.is_bearish_rate = is_bearish_rate
        self.is_bullish_rate = is_bullish_rate
        self.is_bearish_change = is_bearish_change
        self.is_bullish_change = is_bullish_change
        self.is_bearish_cross_alert = is_bearish_cross_alert
        self.is_bullish_cross_alert = is_bullish_cross_alert
        self.is_bearish_smooth = is_bearish_smooth
        self.is_bullish_smooth = is_bullish_smooth
        self.alert_bearish = alert_bearish
        self.alert_bullish = alert_bullish
        self.is_bearish = is_bearish
        self.is_bullish = is_bullish

        # Entry Conditions
        self.start_short_trade = start_short_trade
        self.start_long_trade = start_long_trade

        # Dynamic exits conditions
        self.last_signal_was_bearish = last_signal_was_bearish
        self.last_signal_was_bullish = last_signal_was_bullish
        self.bars_since_red_entry = bars_since_red_entry
        self.bars_since_red_exit = bars_since_red_exit
        self.bars_since_green_entry = bars_since_green_entry
        self.bars_since_green_exit = bars_since_green_exit
        self.is_valid_short_exit = is_valid_short_exit
        self.is_valid_long_exit = is_valid_long_exit
        self.end_short_trade_dynamic = end_short_trade_dynamic
        self.end_long_trade_dynamic = end_long_trade_dynamic

        self.end_long_trade_strict = end_long_trade_strict
        self.end_short_trade_strict = end_short_trade_strict
        self.is_dynamic_exit_valid = is_dynamic_exit_valid
        self.end_long_trade = end_long_trade
        self.end_short_trade = end_short_trade

        # <filter_all>: complex filter
        self.filter_all_default = volatility and ADX and regime
        self.is_short = is_short
        self.is_long = is_long

class Bars_Count():
    def __init__(
        self, 
        bars_used: int,
        max_period: int,
    ):
        self.bars_used = int(bars_used)
        self.max_period = int(max_period)
        self.max_bars_back = int(bars_used * 2 + max_period)
        self.bars_used_mu_2 = int(bars_used * 2)
        self.max_period_mu_2 = int(max_period * 2)

class Index_Count():
    def __init__(
        self, 
        i, 
        bars_count_struct,
        max_bars_held,
    ):
        self.i_mi_bars_used_and_max_period = i - bars_count_struct.bars_used - bars_count_struct.max_period
        self.i_mi_bars_used = i - bars_count_struct.bars_used
        self.i_mi_max_period_mu_2 = i - bars_count_struct.max_period
        self.i_mi_max_bars_held = i - max_bars_held
        self.i_a_1 = i + 1
        self.i_mi_max_period = i - bars_count_struct.max_period
        self.i_mi_max_period_mu_5 = i - bars_count_struct.max_period * 5
        self.i_mi_max_bars_held = i - max_bars_held
