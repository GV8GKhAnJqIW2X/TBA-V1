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

def g_adx(high, low, close, period=14):
    atr = pd.DataFrame({
        'tr1': high - low,
        'tr2': np.abs(high - close.shift()),
        'tr3': np.abs(low - close.shift())
    })\
        .max(axis=1)\
        .rolling(window=period)\
        .mean()\
        .reset_index(drop=True)
    low_diff, high_diff = map(lambda v: v.diff(), (low, high))
    calculate_di = lambda v1, v2:\
        100 * \
        (
        pd.Series(np.where((v1 > v2) & (v1 > 0), v1, 0))
            .rolling(window=period)
            .mean()
        / atr
    )
    minus_di = calculate_di(low_diff, high_diff)
    plus_di = calculate_di(high_diff, low_diff)
    return ((100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di)))\
        .rolling(window=period)\
        .mean()).iloc[-1]

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
        np.log(1 + abs(getattr(feature_series, key) - array[i]))
        for key, array in vars(feature_arrs).items()
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
    feature_str, 
    src: Source, 
    *params,
) -> Feature_Series:
    """
    PARAMS:
    feature_str - str The name of the feature to calculate.
    src - DataFrame<float> The source series.
    params - List[int] The parameters for the feature calculation.

    RETURNS:
    Feature_Series<float> The calculated feature series.
    
    """
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
    close, 
    lookback, 
    relative_weight, 
    start_at_bar,
):
    """
    PARAMS:
    close - Series<float> The source series.
    lookback - int The period for calculating the RQ.
    relative_weight - float The relative weight for the RQ.
    start_at_bar - int The index of the bar at which to start the calculation.

    RETURNS:
    float The RQ estimate.

    """
    size = len(close)
    current_weight = 0
    cumulative_weight = 0
    
    for i in range(start_at_bar, size,):
        y = close[i]
        w = (1 + (i ** 2) / (lookback ** 2 * 2 * relative_weight)) ** -relative_weight
        print(round(w, 4))
        current_weight += y * w
        # if i == size - 1:
        #     print(current_weight[i])
    #     cumulative_weight[i] += w
    # yhat = current_weight / cumulative_weight
    # return yhat[-1]
    return current_weight

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

# def g_gaussian(
#     close, 
#     lookback, 
#     start_at_bar
# ):
#     """
#     PARAMS:
#     close - Series<float> The source series.
#     lookback - int The period for calculating the Gaussian.
#     start_at_bar - int The index of the bar at which to start the calculation.

#     RETURNS:
#     float The Gaussian estimate.

#     """
#     current_weight = 0.0
#     cumulative_weight = 0.0
#     size = len(close)

#     # Проходим по всем элементам начиная с start_at_bar
#     for i in range(size):
#         if i < start_at_bar:
#             continue  # Пропускаем элементы до start_at_bar
        
#         y = close[i]
#         # Вычисляем вес w
#         w = np.exp(-((i ** 2) / (2 * (lookback ** 2))))
        
#         current_weight += y * w
#         cumulative_weight += w

#     # Проверка на ноль для избежания деления на ноль
#     if cumulative_weight == 0:
#         return np.nan  # Или любое другое значение по умолчанию
    
#     yhat = current_weight / cumulative_weight
#     return yhat