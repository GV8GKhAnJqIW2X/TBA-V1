import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import ta 

from settings_ import *
from g_indicators import *

def g_m_imputer(data):
    """
    PARAMS:
    data - DataFrame<float> The source series.

    RETURNS:
    DataFrame<float> The imputed series.
    
    """
    return pd.DataFrame(
        SimpleImputer(strategy="mean")\
            .fit_transform(data.replace({-np.inf: np.nan, np.inf: np.nan})),
        columns=data.columns
    )

def g_f_volatility(
    src, 
    min_length=1, 
    max_length=10, 
    USE_volatility_f=settings_filter["USE_volatility_f"]
) -> bool:
    """
    PARAMS:
    src - DataFrame<float> The source series.
    min_length - int The minimum length of the series to calculate ATR.
    max_length - int The maximum length of the series to calculate ATR.
    USE_volatility_f - bool Whether to use the volatility filter.

    RETURNS:
    Boolean indicating whether or not to let the signal pass through the filter.

    """  
    if not USE_volatility_f:
        return True
    
    # Вычисляем недавний и исторический ATR
    recent_atr = g_atr(high=src.high, low=src.low, close=src.close, period=min_length).iloc[-1]  # Последнее значение ATR за minLength
    historical_atr = g_atr(high=src.high, low=src.low, close=src.close, period=max_length).iloc[-1]  # Последнее значение ATR за maxLength
    
    # Возвращаем результат фильтрации
    return recent_atr > historical_atr

def g_f_adx(
    latest_adx, 
    adx_threshold=settings_filter["ADX"], 
    USE_ADX_f=settings_filter["USE_ADX_f"],
):
    """
    PARAMS:
    latest_adx - DataFrame<float> The source series.
    adx_threshold - int The threshold for ADX.
    USE_ADX_f - bool Whether to use the ADX filter.

    RETURNS:
    DataFrame<float> The filtered series.
    
    """    
    if USE_ADX_f:
        return latest_adx > adx_threshold  # Возвращаем True или False в зависимости от порога
    
    return True  # Если фильтр не используется, возвращаем True

def g_f_regime(
    high,
    low,
    ohlc4,
    threshold=settings_filter["regime"],
    USE_regime_filter=settings_filter["USE_regime_f"],
):
    """
    для адекватной работы этой функции необходимо в 4-5 раз больше клайнов, чем указано в периоде ema.
    это связано с работой ema.
    
    """
    
    if USE_regime_filter:
        len_src = len(high)
        values1 = np.zeros(len_src)
        values2 = np.zeros(len_src)
        klmf = np.zeros(len_src)
        abs_curve_slope = np.zeros(len_src)

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
        exponential_average_abs_curve_slope = g_ema(pd.Series(abs_curve_slope), 200).to_numpy()[-1]
        
        return (abs_curve_slope[-1] - exponential_average_abs_curve_slope) / exponential_average_abs_curve_slope >= threshold
    return True

def g_f_all(src, settings_filter=settings_filter,):
    """
    PARAMS:
    src - DataFrame<float> The source series.
    settings_filter - Settings_custom The filter settings.
    USE_data_frame - bool Whether to use DataFrame for calculations.

    RETURNS:
    Boolean indicating whether or not to let the signal pass through all filters.

    """
    return all([
        g_f_volatility(src.close, *settings_filter["volatility"]),
        g_f_regime(src.close, *settings_filter["regime"]),
        g_f_adx(src.ADX, *settings_filter["ADX"]),
    ])

def g_crossover(series1, series2):
    """
    Проверяет, произошло ли пересечение series1 с series2.
    
    :param series1: Первая серия данных (Pandas Series).
    :param series2: Вторая серия данных (Pandas Series).
    :return: Серия логических значений, где True указывает на пересечение.
    """
    if len(series1) < 2:
        return False
    
    # Создаем смещения для проверки пересечения
    series1_pop = series1.pop(0)
    series2_pop = series2.pop(0)
    series1 = np.array(series1)
    series2 = np.array(series2)
    series1_pop = np.array(series1_pop)
    series2_pop = np.array(series2_pop)
    crossover_condition = (series1 > series2) & (series1_pop <= series2_pop)
    return crossover_condition[-1]

def g_crossunder(series1, series2):
    """
    Проверяет, произошло ли пересечение series1 с series2 сверху вниз.
    
    :param series1: Первая серия данных (Pandas Series).
    :param series2: Вторая серия данных (Pandas Series).
    :return: Серия логических значений, где True указывает на пересечение.
    """
    if len(series1) < 2:
        return False
    
    # Создаем смещения для проверки пересечения
    series1_pop = series1.pop(0)
    series2_pop = series2.pop(0)
    series1 = np.array(series1)
    series2 = np.array(series2)
    series1_pop = np.array(series1_pop)
    series2_pop = np.array(series2_pop)
    crossunder_condition = (series1 < series2) & (series1_pop >= series2_pop)
    return crossunder_condition[-1]

def g_bars_since(index_condition, index_now):
    return abs(index_condition - index_now)