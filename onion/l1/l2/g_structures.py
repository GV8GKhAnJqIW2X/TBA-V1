from onion.l1.l1.g_settings_ import *
from onion.l1.l1.g_indicators import *
from onion.l1.l1.g_transform import *
from onion.l1.l1.g_choices import *

import numpy as np
import pandas as pd

def g_src(
    open, 
    high, 
    low, 
    close,
):
    src_ = {}

    src_["open"] = open
    src_["high"] = high
    src_["low"] = low
    src_["close"] = close

    return src_

def g_x_features_series(
    high,
    low,
    close,
    features_used: dict,
    check_params=True,
):
    """
    RETURNS:
    x_features_series: dict. feature series
    
    """
    # check params
    if check_params:
        if not isinstance(high, pd.Series):
            high = pd.Series(high)
        if not isinstance(low, pd.Series):
            low = pd.Series(low)
        if not isinstance(close, pd.Series):
            close = pd.Series(close)
        if not isinstance(features_used, dict):
            raise ValueError("Features used is not dict")
    
    # init
    choice = g_features_series_choice(
        high,
        low,
        close
    )

    # main
    return {
        key: choice[key.split("_")[0]](params)
        for key, params in features_used.items()
    }

def g_filters_values(
    high_max_window_filters,
    low_max_window_filters,
    close_max_window_filters,
    open_max_window_filters_MU_5,
    high_max_window_filters_MU_5,
    low_max_window_filters_MU_5,
    close_max_window_filters_MU_5,
    signal_raw,
    last_signal_raw,
    adx_value,
    signals_held_counter,
    zeros_skip_counter,
    filters_used: dict,
    check_params=True,
):
    # check params
    if check_params:
        if not isinstance(filters_used, dict):
            raise ValueError("Filters used is not dict")
        if not isinstance(signals_held_counter, int):
            raise ValueError("Signals held counter is not int")
        if not isinstance(zeros_skip_counter, int):
            raise ValueError("Zeros skip counter is not int")
        if not isinstance(signal_raw, int):
            raise ValueError("Signal raw is not int")
        if not isinstance(last_signal_raw, int):
            raise ValueError("Last signal raw is not int")
        if not isinstance(adx_value, (int, float)):
            raise ValueError("ADX value is not float or int")

    # init
    choice = g_filters_values_choice(
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
    )

    # main
    return {
        key: choice[key](params)
        for key, params in filters_used.items()
    }

def g_filters(
    filters_values,
    filters_used,
    last_price,
    signal_raw,
    check_params=True,
):
    # check params
    if check_params:
        if not isinstance(filters_values, dict):
            raise ValueError("Filters values is not dict")
        if not isinstance(filters_used, dict):
            raise ValueError("Filters used is not dict")
    
    # init
    choice = g_filters_choice(
        filters_values,
        last_price,
        True if signal_raw == -1 else False,
        True if signal_raw == 1 else False,
    )

    # main
    return {
        key: choice[key](params)
        for key, params in filters_used.items()
    }

def g_features_series_empty(
    features_used_keys=settings["SIGNAL_GENERATION"]["ML"]["features_used"].keys()
):
    return {
        key: np.nan
        for key in features_used_keys
    }

def g_x_train_arrays_empty(
    klines_train=settings["SIGNAL_GENERATION"]["ML"]["klines_train"],
    features_used_keys=settings["SIGNAL_GENERATION"]["ML"]["features_used"].keys(),
):
    return {
        key: np.full(klines_train, np.nan)
        for key in features_used_keys
    }

def g_plot_values(
    signals, 
    y,
    trace_close=True,
    trace_long=True,
    trace_short=True,
    trace_balance=False,
):
    plot_values = {}

    x = np.arange(len(signals))
    if trace_close:
        plot_values["x_close"] = x
        plot_values["y_close"] = y
    if trace_long:
        long = signals == 1
        plot_values["x_long"] = x[long]
        plot_values["y_long"] = y[long]
    if trace_short:
        short = signals == -1
        plot_values["x_short"] = x[short]
        plot_values["y_short"] = y[short]
    if trace_balance:
        plot_values["x_balance"] = x
        plot_values["y_balance"] = y
    
    return plot_values
