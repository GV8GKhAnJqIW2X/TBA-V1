from onion.l1.l1.g_settings_ import *
from onion.l1.l1.g_indicators import *
from onion.l1.l1.g_transform import *
from onion.l1.l1.g_choices.g_choices_signals import *

from project_exctentions.g_utils import *
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
    open,
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
        if not isinstance(open, pd.Series):
            open = pd.Series(open)
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
        open,
        high,
        low,
        close,
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
    signal_raw_last,
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
        signal_raw,
        signal_raw_last,
        True if signal_raw == -1 else False,
        True if signal_raw == 1 else False,
    )

    # main
    return {
        key: choice[key](params)
        for key, params in filters_used.items()
    }

def g_features_series_empty(
    features_used_keys=settings["ML"]["features"].keys()
):
    return {
        key: np.nan
        for key in features_used_keys
    }

def g_x_train_arrays_empty(
    klines_train=settings["ML"]["klines_train"],
    features_used_keys=settings["ML"]["features"].keys(),
):
    return {
        key: np.full(klines_train, np.nan)
        for key in features_used_keys
    }

def g_plot_close(
    y,
    signals=None, 
    trace_long=False,
    trace_short=False,
):
    plot_values = {}

    x = np.arange(len(y))
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
    
    return plot_values

def g_plot_backtest_values(
    y,
    balance,
    in_positions,
    qty,
    signals,
    check_args=False,
):
    # init
    plot_values = {}
    x = np.arange(len(y))
    inpositions_indcs = g_split_AS_bool_array_AS_indcs(in_positions)
    long = signals == 1
    short = signals == -1

    # main
    plot_values["x_close"] = x
    plot_values["y_close"] = y
    plot_values["x_dash_lines"] = []
    plot_values["y_dash_lines"] = []
    plot_values["indc_pos_close"] = []
    plot_values["indc_pos_opened"] = []
    for i, indc in enumerate(inpositions_indcs):
        print(i, len(inpositions_indcs))
        v = np.array(list(enumerate(signals != 0)))[indc]
        indc_ = v[np.where(v[:, 1] == True)][:, 0]
        indc_ = np.append(indc_, indc[-1])
        plot_values["x_dash_lines"].append(x[indc_])
        plot_values["y_dash_lines"].append(y[indc_])
        plot_values["indc_pos_close"].append(x[indc_[-1] + (1 if indc_[-1] + 1 < len(y) else 0)])
        plot_values["indc_pos_opened"].extend(x[indc_[:-1]] if signals[indc_[-1]] == 0 else x[indc_[:-2]])
        if signals[indc_[-1]] == 0:
            if indc_[-1] + 1 != len(signals):
                signal_post = -1 if signals[indc_[0]] == 1 else 1
                if signal_post == -1:
                    short[indc_[-1] + 1] = True
                else:
                    long[indc_[-1] + 1] = True

    plot_values["x_long"] = x[long]
    plot_values["y_long"] = y[long]
    plot_values["x_short"] = x[short]
    plot_values["y_short"] = y[short]
    plot_values["x_balance"] = x[plot_values["indc_pos_close"]]
    plot_values["y_balance"] = y[plot_values["indc_pos_close"]]
    plot_values["x_qty"] = x[plot_values["indc_pos_opened"]]
    plot_values["y_qty"] = y[plot_values["indc_pos_opened"]]

    plot_values["text_balance"] = "b_" + np.array(np.round(balance[plot_values["x_balance"]], 1), dtype=np.str_)
    plot_values["text_qty"] = "q_" + np.array(np.round(qty[plot_values["x_qty"]], 1), dtype=np.str_)

    return plot_values
    
    
        

