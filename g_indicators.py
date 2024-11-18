import numpy as np
import pandas as pd

from settings_ import *
from structures import *

def g_atr(
    high, 
    low,
    close,
    period,
):
    """
    PARAMS:
    data - DataFrame<float> The source series.
    period - int The period for calculating ATR.

    RETURNS:
    DataFrame<float> The ATR series.
    
    """
    return pd.DataFrame({
        'tr1': high - low,
        'tr2': np.abs(high - close.shift()),
        'tr3': np.abs(low - close.shift())
    })\
        .max(axis=1)\
        .rolling(window=period)\
        .mean() 

def g_rsi(nums, period=14,) -> float:
    """
    PARAMS:
    nums - DataFrame<float> The source series.
    period - int The period for calculating RSI.

    RETURNS:
    float: The RSI value
    
    """
    if len(nums) < period:
        raise ValueError("Nums must be greater than period")
    
    delta = nums.diff()
    return (100 - (100 / (
        1
        +
        (delta.where(delta > 0, 0))
            .rolling(window=period)
            .mean()
        /
        (-delta.where(delta < 0, 0))\
            .rolling(window=period)\
            .mean()\
            .replace(0, delta.mean())
    ))).iloc[-1]

def g_adx(
    high,
    low,
    close,
    period=14,
) -> float:
    """
    PARAMS:
    data - DataFrame<float> The source series.
    i - Index
    period - int The period for calculating ADX.

    RETURNS:
    float: The ADX value
    
    """
    if len(close) < period:
        raise ValueError("Close must be greater than period")
    
    # Расчет истинного диапазона (TR)
    high_low = high - low
    high_prev_close = abs(high - close.shift(1))
    low_prev_close = abs(low - close.shift(1))
    
    tr = pd.Series(np.maximum(high_low, np.maximum(high_prev_close, low_prev_close)))

    # Расчет направленного движения (+DM и -DM)
    plus_dm = np.where((high.diff() > 0) & (high.diff() > low.diff()), 
                       high.diff(), 0)
    minus_dm = np.where((low.diff() > 0) & (low.diff() > high.diff()), 
                        low.diff(), 0)

    # Сглаживание через скользящую среднюю
    tr_smooth = tr.rolling(window=period).mean()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()

    # Расчет +DI и -DI
    plus_di = (plus_dm_smooth / tr_smooth) * 100
    minus_di = (minus_dm_smooth / tr_smooth) * 100

    # Расчет DX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100

    # Расчет ADX как скользящей средней DX
    adx = dx.rolling(window=period).mean()

    # Возвращаем только последнее значение ADX и другие значения
    last_adx = adx.iloc[-1]
    
    return last_adx

def g_cci(
    high,
    low,
    close,
    period=20,
) -> float:
    """
    PARAMS:
    data - DataFrame<float> The source series.
    i - Index
    period - int The period for calculating CCI.

    RETURNS:
    float: The CCI value
    
    """
    if len(close) < period * 2:
        raise ValueError("Close must be greater than period * 2")

    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    return ((typical_price - sma) / (0.015 * (typical_price - sma)\
        .abs()\
        .rolling(window=period)\
        .mean())).iloc[-1]

def g_williams_r(
    high,
    low,
    close,
    period=14,
) -> float:
    """
    PARAMS:
    data - DataFrame<float> The source series.
    i - Index
    period - int The period for calculating Williams %R.

    RETURNS:
    DataFrame<float> The Williams %R series.
    
    """
    if len(close) < period * 2:
        raise ValueError("Close must be greater than period * 2")

    highest_high = high.rolling(window=period).max()
    return (-100 * (highest_high - close) / (
        highest_high -
        low
            .rolling(window=period)
            .min()
    )).iloc[-1]

def g_tsi(close, period=14,) -> float:
    """
    PARAMS:
    data - DataFrame<float> The source series.
    i - Index
    period - int The period for calculating TSI.

    RETURNS:
    float: The TSI value
    
    """
    if len(close) < period * 2:
        raise ValueError("Close must be greater than period * 2")

    return close\
        .rolling(window=period)\
        .corr(pd.Series(np.arange(len(close))))

def g_lorentzian_distance(
    i, 
    feature_series, 
    feature_arrs
) -> float:
    """
    PARAMS:
    i - int The index of the current bar.
    feature_series - Series<float> The feature series.
    feature_arrs - List[Series] The feature series to calculate distances for.

    RETURNS:
    float The Lorentzian distance for the current bar.
    
    """
    return np.sum([
        np.log(1 + abs(feature_series[key] - array[i]))
        for key, array in enumerate(feature_arrs)
    ])

def g_lorentzian_distances(feature_arrs,bars_back,):
    """
    PARAMS:
    feature_arrs - List[Series] The feature series to calculate distances for.
    bars_back - int The number of bars back to calculate distances.

    RETURNS:
    DataFrame<float> The Lorentzian distances series.
    
    """
    return np.sum([
        feature_arr\
            .fillna(feature_arr.mean())\
            .rolling(window=bars_back)\
            .apply(lambda v: np.log(1 + np.abs(v.iloc[0] - v.iloc[-1])))
        for feature_arr in feature_arrs
    ], axis=0)

def g_series_from(
    i,
    feature_str, 
    src: Source, # close_, high_, low_, hlc3_, 
    *params, # [20, 9]
) -> Feature_Series:
    """
    PARAMS:
    feature_str - str The name of the feature to calculate.
    src - DataFrame<float> The source series.
    params - List[int] The parameters for the feature calculation.

    RETURNS:
    Feature_Series<float> The calculated feature series.
    
    """
    src = src.iloc(i - max(params) * 2 + 1, i + 1)
    return {
        "RSI": lambda: g_rsi(src.close, *params),
        "ADX": lambda: g_adx(
            src.high, 
            src.low, 
            src.close, 
            *params
        ),
        "CCI": lambda: g_cci(
            src.high, 
            src.low, 
            src.close, 
            *params
        ),
        "WT": lambda: g_williams_r( 
            src.high, 
            src.low, 
            src.close, 
            *params
        ),
        "TSI": lambda: g_tsi(src.close, *params),
    }.get(feature_str.upper(), lambda: None)()

def g_ema(data, period):
     """
     PARAMS:
     data - DataFrame<float> The source series.
     period - int The period for calculating EMA.

     RETURNS:
     DataFrame<float> The EMA series.

     """
     return data.ewm(span=period, adjust=False).mean()

def g_sma(data, period):
    """
    PARAMS:
    data - DataFrame<float> The source series.
    period - int The period for calculating SMA.

    RETURNS:
    DataFrame<float> The SMA series.
    
    """
    return data.rolling(window=period).mean()

def g_rational_quadratic(
    src, 
    lookback, 
    relative_weight, 
    start_at_bar,
):
    """
    PARAMS:
    src - Series<float> The source series.
    lookback - int The period for calculating the RQ.
    relative_weight - float The relative weight for the RQ.
    start_at_bar - int The index of the bar at which to start the calculation.

    RETURNS:
    float The RQ estimate.

    """
    current_weight = 0.0
    cumulative_weight = 0.0
    size = len(src)

    # Проходим по всем элементам начиная с start_at_bar
    for i in range(size):
        if i < start_at_bar:
            continue  # Пропускаем элементы до start_at_bar
        
        y = src[i]
        # Вычисляем вес w
        w = (1 + ((i ** 2) / (2 * (lookback ** 2) * relative_weight))) ** (-relative_weight)
        
        current_weight += y * w
        cumulative_weight += w

    # Проверка на ноль для избежания деления на ноль
    if cumulative_weight == 0:
        return np.nan  # Или любое другое значение по умолчанию
    
    yhat = current_weight / cumulative_weight
    return yhat

def g_gaussian(
    src, 
    lookback, 
    start_at_bar
):
    """
    PARAMS:
    src - Series<float> The source series.
    lookback - int The period for calculating the Gaussian.
    start_at_bar - int The index of the bar at which to start the calculation.

    RETURNS:
    float The Gaussian estimate.

    """
    current_weight = 0.0
    cumulative_weight = 0.0
    size = len(src)

    # Проходим по всем элементам начиная с start_at_bar
    for i in range(size):
        if i < start_at_bar:
            continue  # Пропускаем элементы до start_at_bar
        
        y = src[i]
        # Вычисляем вес w
        w = np.exp(-((i ** 2) / (2 * (lookback ** 2))))
        
        current_weight += y * w
        cumulative_weight += w

    # Проверка на ноль для избежания деления на ноль
    if cumulative_weight == 0:
        return np.nan  # Или любое другое значение по умолчанию
    
    yhat = current_weight / cumulative_weight
    return yhat