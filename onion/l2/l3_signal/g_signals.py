from onion.l1.l1.g_settings_ import *
from onion.l1.l2.g_structures import *
from onion.l2.l1_features.g_predictions_ml import *
from onion.l2.l2_features_all.g_x_y_ml import *
from project_exctentions.g_utils import g_iloc

import numpy as np
import pandas as pd

def g_signal_raw(prediction):
    signal = 0
    if prediction > 0:
        signal = 1
    elif prediction < 0:
        signal = -1
    
    return signal

def g_signal_A_distance_lorentzian_A_ANN_A_iter(
    open,
    high,
    low,
    close,
    initialized_return_value,  
    klines_train=settings["ML"]["klines_train"], 
    klines_train_held=settings["ML"]["klines_train_held"],
    max_window_features=settings["max_window_features"],
    max_window_filters=settings["max_window_filters"],
    features_used: dict = settings["ML"]["features"],
    filters_used: dict = settings["FILTERS"],
    neighbors_count=settings["ML"]["neighbors_count"],
    additional_return_x_features_series=True,
    additional_return_signal_raw=True,
    additional_return_filters_values=True,
    check_params=True,
):  
    """
    для использования этой функции нужно 
    проиницилизировать переменную вне цикла для возвращаемых значений

    src (open, high, low, close) len == (max(features windows) * 5 + 1 + klines_train + klines_test)

    RETURNS:
    signal: int. trading signal
    (необязательное) additional_returns: tuple | Any. дополнительное возвращаемое значение.
        обычно нужно для переиспользования вне функции
    initialized_return_value: tuple. возвращаемое значение

    """
    # check params
    if check_params:
        if not isinstance(open, np.ndarray):
            open = np.array(open)
        if not isinstance(high, np.ndarray):
            high = np.array(high)
        if not isinstance(low, np.ndarray):
            low = np.array(low)
        if not isinstance(close, np.ndarray):
            close = np.array(close)
        if initialized_return_value is None:
            initialized_return_value = (
                np.full(klines_train, np.nan), # y_train_array
                g_x_train_arrays_empty(klines_train, features_used.keys()), # x_train_arrays
                None, # features_series
                0, # signals_held_counter
                0, # zeros_skip_counter
                0, # last_signal_raw
            )
    
    # init
    (
        y_train_array,
        x_train_arrays,
        x_features_series,
        signals_held_counter,
        zeros_skip_counter,
        last_signal_raw,
    ) = initialized_return_value
        
    # main
    (
        x_train_arrays,
        y_train_array, 
    ) = g_x_y_arrays_partial_filling_A_comparison_A_klines_train_held(
       open,
       high,
       low,
       close,
       y_train_array_fill_target=y_train_array,
       x_train_arrays_fill_target=x_train_arrays,
       klines_train=klines_train,
       klines_train_held=klines_train_held,
       max_window_features=max_window_features,
       features_used=features_used,
       USE_shift_filling=True,
       USE_ready_made_series=True if x_features_series is not None else False,
       ready_made_series=x_features_series,
    )
    x_features_series = g_x_features_series(
        *g_iloc(
            tuple(map(pd.Series, [open, high, low, close])), 
            slice(-max_window_features * 5, None),
        ),
        features_used=features_used,
    )
    prediction = g_lorentzian_prediction(
        y_train_array=y_train_array,
        x_train_arrays=x_train_arrays,
        x_features_series=x_features_series,
        klines_train=klines_train,
        klines_train_held=klines_train_held,
        neighbors_count=neighbors_count,
    )
    signal_raw = g_signal_raw(prediction) 
    filters_values = g_filters_values(
        *g_iloc((high, low, close), slice(-max_window_filters, None)),
        *g_iloc((open, high, low, close), slice(-max_window_filters * 5, None)),
        signal_raw=signal_raw,
        last_signal_raw=last_signal_raw,
        adx_value=x_features_series["ADX_1"],
        signals_held_counter=signals_held_counter,
        zeros_skip_counter=zeros_skip_counter,
        filters_used=filters_used,
    )
    signal = 0
    if signal_raw:
        filters = g_filters(
            filters_values,
            filters_used,
            close[-1],
            signal_raw,
            last_signal_raw,
        )
        if all(filters.values()):
            signal = signal_raw
    last_signal_raw = signal_raw

    ## return values
    initialized_return_value = (
        y_train_array,
        x_train_arrays,
        x_features_series,
        *filters_values["signals_held"],
        last_signal_raw,
    )
    additional_return_values = []

    if additional_return_x_features_series:
        additional_return_values.append(x_features_series)
    if additional_return_signal_raw:
        additional_return_values.append(signal_raw)
    if additional_return_filters_values:
        additional_return_values.append(filters_values)

    len_additional_return_values = len(additional_return_values)
    if len_additional_return_values > 0:
        if len_additional_return_values == 1:
            return signal, additional_return_values[0], initialized_return_value
        return signal, additional_return_values, initialized_return_value

    return signal, initialized_return_value

# def g_signal_A_ready_made_data(
#     signal_raw,
#     signal_raw_last,
#     last_price, 
#     filters_values: dict,
#     filters_used: dict,
#     check_params=True,
# ):
#     # check params:
#     if check_params:
#         if not isinstance(signal_raw, int):
#             if isinstance(signal_raw, float):
#                 signal_raw = int(signal_raw)
#             else:
#                 raise ValueError("Signal raw is not int or float")
#         if not isinstance(last_price, (int, float)):
#             raise ValueError("Last price is not int or float")
#         if not isinstance(filters_values, dict):
#             raise ValueError("Filters values is not dict")
#         if not isinstance(filters_used, dict):
#             raise ValueError("Filters used is not dict")
    
#     # init
#     signal = 0
    
#     # main
#     if signal_raw:
#         filters = g_filters(
#             filters_values,
#             filters_used,
#             last_price,
#             signal_raw,
#             signal_raw_last,
#         )
#         if all(filters.values()):
#             signal = signal_raw
    
#     return signal

def g_signal_A_ready_made_data(
    signal_raw,
    signal_raw_last,
    signals_raw_last,
    last_price, 
    filters_values: dict,
    filters_used: dict,
    return_sighals_counter,
    return_signal_raw,
    signals_raw_l2,
    check_params=True,
):
    # check params:
    if check_params:
        if not isinstance(signal_raw, int):
            if isinstance(signal_raw, float):
                signal_raw = int(signal_raw)
            else:
                raise ValueError("Signal raw is not int or float")
        if not isinstance(last_price, (int, float)):
            raise ValueError("Last price is not int or float")
        if not isinstance(filters_values, dict):
            raise ValueError("Filters values is not dict")
        if not isinstance(filters_used, dict):
            raise ValueError("Filters used is not dict")
    
    # init
    signal = 0
    signal2 = 0 
    # signal = signal_raw
    all_ = [True,]
    all_2 = [True,]
    
    # main
    if signal_raw:
        volatility_f = g_f_volatility(*filters_values["volatility"])
        if not volatility_f:
            signal_raw = 0
        vol_signal_f = (
            signal_raw != return_signal_raw
            and signal_raw != signals_raw_l2[-1]
            and signal_raw != signals_raw_l2[-2]
            and signal_raw != signals_raw_l2[-3]
            and signal_raw != signals_raw_l2[-4]
        )
        all_.append(
            vol_signal_f
        )
        all_2.append(vol_signal_f)

        all_.append(g_f_regime(filters_values["regime"], settings["FILTERS"]["regime"]["threshold1"]))
        all_2.append(g_f_regime(filters_values["regime"], settings["FILTERS"]["regime"]["threshold2"]))

        short = True if signal_raw == -1 else False
        long = True if signal_raw == 1 else False
        all_.append(
            g_f_ema(filters_values["EMA"], last_price,short, long)
        )
        all_2.append(
            g_f_ema(filters_values["EMA"], last_price,short, long)
        )

        all_.append(g_f_sma(filters_values["SMA"], last_price,short, long))
        all_2.append(g_f_sma(filters_values["SMA"], last_price,short, long))
        
        all_.append(g_f_adx(filters_values["ADX"], settings["FILTERS"]["ADX"]["threshold1"]))
        all_2.append(g_f_adx(filters_values["ADX"], settings["FILTERS"]["ADX"]["threshold2"]))
        
        if all(all_):
            signal = signal_raw
        if all(all_2):
            signal2 = signal_raw
    
    return signal, return_sighals_counter, signal_raw, signal2,

def g_signal_raw_A_distance_lorentzian_A_ANN_A_iter(
    open,
    high,
    low,
    close,
    initialized_return_value,  
    klines_train=settings["ML"]["klines_train"], 
    klines_train_held=settings["ML"]["klines_train_held"],
    features_used: dict = settings["ML"]["features"],
    max_window_features=settings["max_window_features"],
    neighbors_count=settings["ML"]["neighbors_count"],
    tp_train=settings["ML"]["tp_train"],
    additional_return_x_features_series=True,
    check_args=True,   
):
    # check args
    if check_args:
        if initialized_return_value is None:
            initialized_return_value = (
                np.full(klines_train, np.nan), # y_train_array
                g_x_train_arrays_empty(klines_train, features_used.keys()), # x_train_arrays
                None, # features_series
            )

    # init
    (
        y_train_array,
        x_train_arrays,
        x_features_series,
    ) = initialized_return_value

    # main
    (
        x_train_arrays,
        y_train_array, 
    ) = g_x_y_arrays_partial_filling_A_comparison_A_klines_train_held(
       open,
       high,
       low,
       close,
       y_train_array_fill_target=y_train_array,
       x_train_arrays_fill_target=x_train_arrays,
       klines_train=klines_train,
       klines_train_held=klines_train_held,
       max_window_features=max_window_features,
       features_used=features_used,
       USE_shift_filling=True,
       USE_ready_made_series=True if x_features_series is not None else False,
       ready_made_series=x_features_series,
       tp_train=tp_train,
    )
    x_features_series = g_x_features_series(
        *g_iloc(
            tuple(map(pd.Series, [high, low, close])), 
            slice(-max_window_features * 2, None),
        ),
        features_used=features_used,
    )
    prediction = g_lorentzian_prediction(
        y_train_array=y_train_array,
        x_train_arrays=x_train_arrays,
        x_features_series=x_features_series,
        klines_train=klines_train,
        klines_train_held=klines_train_held,
        neighbors_count=neighbors_count,
    )
    
    if additional_return_x_features_series:
        return g_signal_raw(prediction), x_features_series, (
            y_train_array,
            x_train_arrays,
            x_features_series,
        )
    return g_signal_raw(prediction), (
        y_train_array,
        x_train_arrays,
        x_features_series,
    )