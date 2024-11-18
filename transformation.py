import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

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
    close, 
    min_length=1, 
    max_length=10, 
    USE_volatility_f=settings_filter["USE_volatility_f"]
) -> bool:
    """
    PARAMS:
    close - DataFrame<float> The source series.
    min_length - int The minimum length of the series to calculate ATR.
    max_length - int The maximum length of the series to calculate ATR.
    USE_volatility_f - bool Whether to use the volatility filter.

    RETURNS:
    Boolean indicating whether or not to let the signal pass through the filter.

    """  
    if not USE_volatility_f:
        return True
    
    # Вычисляем недавний и исторический ATR
    recent_atr = g_atr(close, min_length).iloc[-1]  # Последнее значение ATR за minLength
    historical_atr = g_atr(close, max_length).iloc[-1]  # Последнее значение ATR за maxLength
    
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
    src, 
    threshold=settings_filter["regime"], 
    USE_regime_f=settings_filter["USE_regime_f"],
):
    """
    Функция фильтра режима.
    
    :param src: Список или массив входных данных (float).
    :param threshold: Пороговое значение (float).
    :param USE_regime_f: Флаг использования фильтра режима (bool).
    :return: Логическое значение, указывающее, пропускается ли сигнал через фильтр (bool).
    """
    # Инициализация переменных
    value1 = 0.0
    value2 = 0.0
    klmf = 0.0
    
    # Создаем DataFrame для хранения значений
    df = pd.DataFrame({'src': src})
    
    # Вычисляем value1 и value2
    df['value1'] = 0.2 * (df['src'] - df['src'].shift(1)).fillna(0) + 0.8 * value1
    df['value2'] = 0.1 * (df['src'].max() - df['src'].min()) + 0.8 * value2
    
    # Вычисляем omega
    df['omega'] = np.abs(df['value1'] / df['value2'])
    
    # Вычисляем alpha
    df['alpha'] = (-df['omega']**2 + np.sqrt(df['omega']**4 + 16 * df['omega']**2)) / 8
    
    # Вычисляем klmf
    klmf_values = []
    for alpha in df['alpha']:
        klmf = alpha * df['src'] + (1 - alpha) * klmf_values[-1] if klmf_values else 0.0
        klmf_values.append(klmf)
    
    df['klmf'] = klmf_values
    
    # Вычисляем absCurveSlope
    df['absCurveSlope'] = np.abs(df['klmf'] - df['klmf'].shift(1)).fillna(0)
    
    # Вычисляем экспоненциальную среднюю наклона
    df['exponentialAverageAbsCurveSlope'] = df['absCurveSlope'].ewm(span=200).mean()
    
    # Нормализуем наклон
    df['normalized_slope_decline'] = (df['absCurveSlope'] - df['exponentialAverageAbsCurveSlope']) / df['exponentialAverageAbsCurveSlope']
    
    # Возвращаем результат фильтрации
    return df['normalized_slope_decline'].iloc[-1] >= threshold if USE_regime_f else True

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
    # Создаем смещения для проверки пересечения
    crossover_condition = (series1 > series2) & (series1.shift(1) <= series2.shift(1))
    return crossover_condition

def g_bars_since(index_condition, index_now):
    return abs(index_condition - index_now)