from onion.l1.l1.g_indicators import *
from onion.l1.l1.g_transform import *

import ta
import pandas_ta

def g_features_series_choice(
    open,
    high, 
    low, 
    close,
):
    return {
        "ADX": lambda params: pandas_ta.\
            adx(
                high,
                low,
                close,
                **params
            )[f"ADX_{params["length"]}"].iloc[-1],
        "CCI": lambda params: ta.\
            trend.\
            CCIIndicator(high=high, low=low, close=close, **params).\
            cci()\
            .iloc[-1],
        "RSI": lambda params: ta.\
            momentum.\
            RSIIndicator(close=close, **params).\
            rsi()\
            .iloc[-1],
        "WT": lambda params: g_wt_A_iter(
            open,
            high,
            low,
            close,
            **params
        )
    }

def g_filters_values_choice(
    signal_raw,
    last_signal_raw,
    signals_held_counter,
    zeros_skip_counter,
    adx_value,
    high_max_window_filters,
    low_max_window_filters,
    close_max_window_filters,
    open_max_window_filters_MU_5,
    high_max_window_filters_MU_5,
    low_max_window_filters_MU_5,
    close_max_window_filters_MU_5,    
):
    return {
        "signals_held": lambda params: g_signals_held(
            signal_raw, 
            last_signal_raw,
            signals_held_counter,
            zeros_skip_counter,
            params["zeros_skip_held_threshold"],
        ),
        "ADX": lambda params: adx_value,
        "volatility": lambda params: g_volatility(
            high_max_window_filters,
            low_max_window_filters,
            close_max_window_filters,
            **params,
        ),
        "regime": lambda params: g_regime(
            open_max_window_filters_MU_5,
            high_max_window_filters_MU_5,
            low_max_window_filters_MU_5,
            close_max_window_filters_MU_5,
        ),
        "EMA": lambda params: g_ema(
            close_max_window_filters_MU_5,
            **params
        ).iloc[-1],
        "SMA": lambda params: g_sma(
            close_max_window_filters_MU_5,
            **params
        ).iloc[-1],
    }

def g_filters_choice(
    filters_values,
    last_price,
    signal,
    signal_last,
    short=False,
    long=False,
):
    return {
        "signals_held": lambda params: g_f_signals_held(
            filters_values["signals_held"][0], 
            params["held_threshold"],
            use_ema_f=params["use_ema_f"],
            use_sma_f=params["use_sma_f"],
            ema_value=filters_values["EMA"],
            sma_value=filters_values["SMA"],
            last_price=last_price,
            short=short,
            long=long,
        ),
        "ADX": lambda params: g_f_adx(
            filters_values["ADX"], 
            params["threshold"],
        ),
        "volatility": lambda params: g_f_volatility(
            *filters_values["volatility"], 
        ),
        "regime": lambda params: g_f_regime(
            filters_values["regime"], 
            **params,
        ),
        "EMA": lambda params: g_f_ema(
            filters_values["EMA"], 
            last_price, 
            short=short, 
            long=long,
        ),
        "SMA": lambda params: g_f_sma(
            filters_values["SMA"], 
            last_price, 
            short=short, 
            long=long,
        ),
        "is_different": lambda params: signal != signal_last,
    }