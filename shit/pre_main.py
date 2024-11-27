from g_indicators import *
from transformation import *
from g_utils import g_klines_split
from g_data_api_global import g_klines
from settings_ import settings_ml, settings_filter
from struct import *

import numpy as np
import pandas as pd

async def main():
    data = await g_klines_split(await g_klines())
    f1array = []
    f2array = []
    f3array = []
    f4array = []
    f5array = []
    farrays = [f1array, f2array, f3array, f4array, f5array]

    direction: Label = {
        "long": 1,
        "short": -1,
        "neutral": 0
    }
    isEmaUptrend = data['close'].iloc[-1] > g_ema(data, settings_filter["EMA"]).iloc[-1]\
        if settings_filter["EMA"]\
        else True
    isEmaDowntrend = data['close'].iloc[-1] < g_ema(data, settings_filter["EMA"]).iloc[-1]\
        if settings_filter["EMA"]\
        else True
    isSmaUptrend = data['close'].iloc[-1] > data['sma'].iloc[-1]\
        if settings_filter["SMA"]\
        else True
    isSmaDowntrend = data['close'].iloc[-1] < data['sma'].iloc[-1]\
        if settings_filter["SMA"]\
        else True
    y_train_series = lambda data, i: direction["short"] if data["close"].iloc[i - 4] < data["close"].iloc[i]\
        else direction["long"] if data["close"].iloc[i - 4] > data["close"].iloc[i]\
        else direction["neutral"]
    y_train_array = []

    # Переменные для логики машинного обучения
    predictions = []  # Массив для хранения предсказаний
    prediction = 0.0  # Переменная для последнего предсказания
    signal = direction["neutral"]  # Начальное значение сигнала
    signals = []
    last_distance = -1.0
    distances = []  # Массив для хранения расстояний
    size = min(settings_ml["max_bars_back"] - 1, len(y_train_array) - 1)
    sizeLoop = min(settings_ml["max_bars_back"] - 1, size)
    bars_held = 0

    for i in data["close"]:
        fseries = []
        for i2, feature_str in enumerate(settings_ml["feature_used"]):
            feature_value = g_series_from(
                feature_str=feature_str,
                data=data,
                params=settings_ml["feature_used"][feature_str]
            )
            fseries.append(feature_value)
            farrays[i2].append(feature_value)
        y_train_array.append(y_train_series(data, i))
        if i >= settings_ml["max_bars_back"]:
            d = g_lorentzian_distance(i - settings_ml["max_bars_back"], fseries, farrays)
            if d >= lastDistance and i % 4 == 0:  # Проверка кратности индекса
                lastDistance = d
                distances.append(d)
                predictions.append(round(y_train_array[i]))  # Предсказание из y_train_array
                
                if len(predictions) > settings_ml["neighbors_count"]:
                    lastDistance = distances[round(settings_ml["neighbors_count"] * 3 / 4)]  # Обновление lastDistance
                    distances.pop(0)
                    predictions.pop(0)

            prediction = sum(predictions)  # Суммирование предсказаний
        
        filter_all = g_f_all(data)
        if prediction > 0 and filter_all:
            signal = direction["long"]
        elif prediction < 0 and filter_all:
            signal = direction["short"]
        signals.append(signal)
        for current_signal in signals:
            # Обновление bars_held
            if current_signal != signal:  # Если сигнал изменился
                bars_held = 0
            else:
                bars_held += 1

            # Обновление текущего сигнала
            signal = current_signal

            # Фильтры
            is_held_four_bars = (bars_held == 4)
            is_held_less_than_four_bars = (0 < bars_held < 4)
        current_signal = signals[-1]  # Текущий сигнал (последний элемент списка)
        previous_signals = signals[-5:-1]  # Последние 4 сигнала

        # Определение изменений сигнала
        isDifferentSignalType = current_signal != previous_signals[-1]
        isEarlySignalFlip = isDifferentSignalType and (previous_signals[0] != previous_signals[1] or previous_signals[1] != previous_signals[2] or previous_signals[2] != previous_signals[3])

        # Определение сигналов
        isBuySignal = current_signal == direction["long"] and isEmaUptrend[-1] and isSmaUptrend[-1]
        isSellSignal = current_signal == direction["short"] and isEmaDowntrend[-1] and isSmaDowntrend[-1]

        # Проверка предыдущих сигналов
        isLastSignalBuy = previous_signals[-4] == direction["long"] and isEmaUptrend[-4] and isSmaUptrend[-4]
        isLastSignalSell = previous_signals[-4] == direction["short"] and isEmaDowntrend[-4] and isSmaDowntrend[-4]

        # Новые сигналы
        isNewBuySignal = isBuySignal and isDifferentSignalType
        isNewSellSignal = isSellSignal and isDifferentSignalType

        h = settings_filter["KERNEl"]["lookback_windows"]
        r = settings_filter["KERNEl"]["relative_weighting"]
        x = settings_filter["KERNEl"]["regression_level"]
        lag = settings_filter["KERNEl"]["enhance_smoothing_lag"]

        # Функции для вычисления ядерной регрессии (примерные реализации)


        # Вычисляем оценки ядра
        yhat1 = rational_quadratic(data["close"], h, r, x)
        yhat2 = gaussian(data["close"], h - lag, x)

        # Ядерные изменения
        wasBearishRate = yhat1[2] > yhat1[1]
        wasBullishRate = yhat1[2] < yhat1[1]
        isBearishRate = yhat1[1] > yhat1[0]
        isBullishRate = yhat1[1] < yhat1[0]

        isBearishChange = isBearishRate and wasBullishRate
        isBullishChange = isBullishRate and wasBearishRate

        # Перекрестки ядер
        isBullishCrossAlert = np.any(yhat2 > yhat1)
        isBearishCrossAlert = np.any(yhat2 < yhat1)
        isBullishSmooth = yhat2 >= yhat1
        isBearishSmooth = yhat2 <= yhat1

        # Условия для входа и выхода из сделок (примерные условия)
        useKernelSmoothing = True  # Пример флага использования сглаживания
        startLongTrade = isBullishChange and isBullishRate  # Условие для длинной позиции
        startShortTrade = isBearishChange and isBearishRate  # Условие для короткой позиции

        # Условия входа
        start_long_trade = isNewBuySignal and isBullish and isEmaUptrend and isSmaUptrend
        start_short_trade = isNewSellSignal and isBearish and isEmaDowntrend and isSmaDowntrend

        # Динамические выходы
        bars_since_red_entry = last_signal_was_bearish  # Пример: количество баров с последнего короткого входа
        bars_since_red_exit = 3  # Пример: количество баров с последнего бычьего сигнала
        bars_since_green_entry = last_signal_was_bullish  # Пример: количество баров с последнего длинного входа
        bars_since_green_exit = 1  # Пример: количество баров с последнего медвежьего сигнала

        is_valid_short_exit = bars_since_red_exit > bars_since_red_entry
        is_valid_long_exit = bars_since_green_exit > bars_since_green_entry

        # Динамические выходы
        end_long_trade_dynamic = (isBearish and is_valid_long_exit)
        end_short_trade_dynamic = (isBullish and is_valid_short_exit)

        # Фиксированные выходы (примерные условия)
        is_held_four_bars = False  # Пример: условие удержания четырех баров
        is_last_signal_buy = True   # Пример: был ли последний сигнал на покупку
        is_last_signal_sell = False  # Пример: был ли последний сигнал на продажу

        end_long_trade_strict = ((is_held_four_bars and is_last_signal_buy) or 
                                (not is_held_four_bars and isNewSellSignal and is_last_signal_buy)) and start_long_trade 
        end_short_trade_strict = ((is_held_four_bars and is_last_signal_sell) or 
                                (not is_held_four_bars and isNewBuySignal and is_last_signal_sell)) and start_short_trade 

        # Проверка валидности динамического выхода
        is_dynamic_exit_valid = not useEmaFilter and not useSmaFilter and not useKernelSmoothing

        # Конечные условия выхода
        end_long_trade = end_long_trade_dynamic if useDynamicExits and is_dynamic_exit_valid else end_long_trade_strict 
        end_short_trade = end_short_trade_dynamic if useDynamicExits and is_dynamic_exit_valid else end_short_trade_strict 
        


async def de_de_main():
    data = await g_klines_split(await g_klines(symbol="SUIUSDT", qty=1000))
    print(g_tsi(data, 999))

if __name__ == "__main__":
    import asyncio

    asyncio.run(de_de_main())