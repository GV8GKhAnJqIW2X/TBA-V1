import numpy as np

from onion.l1.l1.g_settings_ import settings

def g_f_ema(
    ema_value,
    last_price,
    short=False,
    long=False,
    check_params=True,
):
    # check params
    if check_params:
        if not isinstance(ema_value, (float, int)):
            raise ValueError("ema_value must be a float or int")
        if not isinstance(last_price, (float, int)):
            raise ValueError("last_price must be a float or int")

    # main
    if long:
        return ema_value > last_price
    elif short:
        return ema_value < last_price

def g_f_sma(
    sma_value,
    last_price,
    short=False,
    long=False,
    check_params=True,
):
    # check params
    if check_params:
        if not isinstance(sma_value, (float, int)):
            raise ValueError("sma_value must be a float or int")
        if not isinstance(last_price, (float, int)):
            raise ValueError("last_price must be a float or int")

    # main
    if long:
        return sma_value > last_price
    elif short:
        return sma_value < last_price

def g_f_signals_held(
    signals_held_counter,
    held_threshold=settings["SIGNAL_GENERATION"]["filters_used"]["signals_held"]["held_threshold"],
    use_ema_f=True,
    use_sma_f=True,
    ema_value=None,
    sma_value=None,
    last_price=None,
    short=False,
    long=False,
    check_params=True,
):
    # check params
    if check_params:
        if not isinstance(signals_held_counter, int):
            raise ValueError("signals_held_counter must be an integer")
    
    # main
    add_f = [True,]
    if use_ema_f:
        add_f.append(g_f_ema(ema_value, last_price, short, long,))
    if use_sma_f:
        add_f.append(g_f_sma(sma_value, last_price, short, long,))
    return signals_held_counter == held_threshold and all(add_f)

def g_f_adx(
    latest_adx, 
    adx_threshold=settings["SIGNAL_GENERATION"]["filters_used"]["ADX"]["threshold"], 
    check_params=True,
):
    # check params
    if check_params:
        if not isinstance(latest_adx, (float, int)):
            raise ValueError("latest_adx must be a float or int")
        if not isinstance(adx_threshold, (float, int)):
            raise ValueError("adx_threshold must be a float or int")
    
    # main
    return latest_adx > adx_threshold

def g_f_volatility(
    min_length,
    max_length,
    check_params=True,
):
    # check params
    if check_params:
        if not isinstance(min_length, (int, float)):
            raise ValueError("min_length must be an integer or float")
        if not isinstance(max_length, (int, float)):
            raise ValueError("max_length must be an integer or float")
    
    # main
    return min_length > max_length

def g_f_regime(
    regime_value,
    threshold,
    check_params=True,
):
    # check_params
    if check_params:
        if not isinstance(regime_value, (float, int)):
            raise ValueError("regime_value must be a float or int")
        if not isinstance(threshold, (float, int)):
            raise ValueError("threshold must be a float or int")
    
    # main
    return regime_value > threshold

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

