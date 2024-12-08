from onion.l1.l1.settings_ import *

import numpy as np
import pandas as pd
import ta

def g_signals_held(
    signal_raw,
    last_signal_raw,
    signals_held_counter,
    zeros_skip_counter,
    zeros_skip_held_threshold=settings["SIGNAL_GENERATION"]["filters_used"]["signals_held"]["zeros_skip_held_threshold"],
    check_params=True,
):
    # check params
    if check_params:
        if not isinstance(signal_raw, int):
            if isinstance(signal_raw, float):
                signal_raw = int(signal_raw)
            else:
                raise ValueError("signal_raw must be an integer")
        if not isinstance(last_signal_raw, int):
            if isinstance(last_signal_raw, float):
                last_signal_raw = int(last_signal_raw)
            else:
                raise ValueError("last_signal_raw must be an integer")
        if not isinstance(signals_held_counter, int):
            raise ValueError("signals_held_counter must be an integer")
        if not isinstance(zeros_skip_counter, int):
            raise ValueError("zeros_skip_counter must be an integer")
        if not isinstance(zeros_skip_held_threshold, int):
            raise ValueError("zeros_skip_held_threshold must be an integer")

    # main
    if signal_raw == 0:
        zeros_skip_counter += 1
        if zeros_skip_counter > zeros_skip_held_threshold:
            signals_held_counter = 0
    elif signal_raw:
        zeros_skip_counter = 0
        if signal_raw == last_signal_raw:
            signals_held_counter += 1
        else:
            signals_held_counter = 0

    return signals_held_counter, zeros_skip_counter

def g_ema(data, window=settings["SIGNAL_GENERATION"]["filters_used"]["EMA"]["window"]):
    """
    PARAMS:
    data - DataFrame<float> The source series.
    window - int The window for calculating EMA.

    RETURNS:
    DataFrame<float> The EMA series.

    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    return data.ewm(span=window, adjust=False).mean()

def g_sma(
    data, 
    window=settings["SIGNAL_GENERATION"]["filters_used"]["SMA"]["window"],
    check_params=True,
):
    # check params
    if check_params:
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

    return data.rolling(window=window).mean()

def g_volatility(
    high, 
    low, 
    close,
    min_length=1, 
    max_length=10, 
    check_params=True,
) -> bool:    
    # check params
    if check_params:
        if not isinstance(close, pd.Series):
            high = pd.Series(high) 
            low = pd.Series(low)
            close = pd.Series(close)

    # main        
    return (
        ta.volatility.AverageTrueRange(
            high=high, 
            low=low, 
            close=close, 
            window=min_length,
        ).average_true_range().iloc[-1],
        ta.volatility.AverageTrueRange(
            high=high, 
            low=low, 
            close=close,
            window=max_length,
        ).average_true_range().iloc[-1],
    )

def g_regime(
    open,
    high,
    low,
    close,
    check_params=True,
):
    """
    для адекватной работы этой функции необходимо в 4-5 раз больше клайнов, 
    чем указано в периоде ema.
    
    это связано с работой ema.
    
    """
    # check params
    if check_params:
        if not isinstance(open, pd.Series):
            open = pd.Series(open) 
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)
    
    # init
    ohlc4 = (open + high + low + close) / 4
    len_src = len(high)
    values1 = np.zeros(len_src)
    values2 = np.zeros(len_src)
    klmf = np.zeros(len_src)
    abs_curve_slope = np.zeros(len_src)

    # main
    for i in range(1, len_src):
        values1[i] = 0.2 * (ohlc4[i] - ohlc4[i - 1]) + 0.8 * values1[i - 1]
        values2[i] = 0.1 * (high[i] - low[i]) + 0.8 * values2[i - 1]
    values1[0] = np.mean(values1)
    values2[0] = np.mean(values2)
    omega = np.abs(values1 / values2)
    alpha = (-(omega ** 2) + np.sqrt(omega ** 4 + 16 * omega ** 2)) / 8
    for i in range(1, len_src):
        klmf[i] = alpha[i] * ohlc4[i] + (1 - alpha[i]) * klmf[i - 1]    
        abs_curve_slope[i] = np.abs(klmf[i] - klmf[i - 1])
    exponential_average_abs_curve_slope = g_ema(abs_curve_slope, 200).to_numpy()[-1]
    
    return (abs_curve_slope[-1] - exponential_average_abs_curve_slope) / exponential_average_abs_curve_slope

def g_rational_quadratic(close, lookback, relative_weight, start_at_bar):
    length_of_prices = len(close)
    result = np.zeros(length_of_prices)
    
    for index in range(length_of_prices):
        current_weight = 0.0
        cumulative_weight = 0.0
        
        # Итерация по предыдущим барам
        for i in range(start_at_bar + 1):
            if index - i < 0:
                continue  # Пропустить, если индекс выходит за пределы
            
            y = close[index - i]
            w = (1 + (i ** 2) / (lookback ** 2 * 2 * relative_weight)) ** -relative_weight
            
            current_weight += y * w
            cumulative_weight += w
        
        # Нормализация результата
        result[index] = current_weight / cumulative_weight if cumulative_weight != 0 else 0.0
    
    return result[-1]

def g_gaussian(close, lookback, start_at_bar):
    length_of_prices = len(close)
    result = np.zeros(length_of_prices)  # Массив для хранения результатов

    for index in range(length_of_prices):
        current_weight = 0.0
        cumulative_weight = 0.0
        
        # Итерация по предыдущим барам
        for i in range(start_at_bar + 1):
            if index - i < 0:
                continue  # Пропустить, если индекс выходит за пределы
            
            y = close[index - i]
            w = np.exp(-((i ** 2) / (2 * (lookback ** 2))))  # Вычисление веса
            
            current_weight += y * w
            cumulative_weight += w
        
        # Нормализация результата
        result[index] = current_weight / cumulative_weight if cumulative_weight != 0 else 0.0
    
    return result[-1]