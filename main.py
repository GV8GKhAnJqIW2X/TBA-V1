from settings_ import *
from structures import *
from g_utils import *
from g_data_api_global import *
from g_indicators import *
from transformation import *

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ta

async def main():
    # INIT CONST
    symbol = "SUIUSDT"
    bars_count = Bars_Count(
        bars_used=settings_ml["bars_used"], 
        max_period=max(g_not_iter_from_iter(list(settings_ml["feature_used"].values()) + list(settings_filter.values()))), 
    )
    src = Source(**(await g_klines_split(await g_klines(symbol=symbol, qty=bars_count.max_bars_back,))))
    feature_series = Feature_Series(*[np.nan for _ in range(5)])
    feature_arrays = Feature_Arrays(*[pd.Series(np.full(bars_count.max_bars_back, np.nan)) for _ in range(5)])
    label_direction = Label(
        short=-1, 
        long=1, 
        neutral=0,
    )
    ml = ML_Model(
        y_train_array=np.full(bars_count.bars_used, label_direction.neutral),
        y_test_array=np.full(bars_count.bars_used, np.nan),
        distances=[],
        predictions=[],
        signals=[],
        yhat1s=[],
        yhat2s=[],
        last_distance=-1.0,
        prediction=label_direction.neutral,
        signal=label_direction.neutral,
        bars_held=0,
    )
    filter_arrays = Filter_Arrays(
        is_EMA_up_trends=[],
        is_EMA_down_trends=[],
        is_SMA_up_trends=[],
        is_SMA_down_trends=[],
        is_valid_short_exits=[],
        is_valid_long_exits=[],
        start_short_trades=[],
        start_long_trades=[],
    )

    for i, price in enumerate(src.close.iloc[bars_count.max_period:], start=bars_count.max_period,):
        # INIT values
        filter = Filter(
            volatility=True,
            ADX=True,
            regime=True,
            is_EMA_up_trend=False,
            is_EMA_down_trend=False,
            is_SMA_up_trend=False,
            is_SMA_down_trend=False,
            is_held_n_bars=False,
            is_held_less_n_bars=False,
            is_diff_signal_type = False,
            is_early_signal_flip = False,
            is_short_signal = False,
            is_last_buy_signal = False,
            is_last_short_signal = False,
            is_new_buy_signal = False,
            is_new_short_signal = False,
            was_bearish_rate = False,
            was_bullish_rate = False,
            is_bearish_rate = False,
            is_bullish_rate = False,
            is_bearish_change = False,
            is_bullish_change = False,
            is_bearish_cross_alert = False,
            is_bullish_cross_alert = False,
            is_bearish_smooth = False,
            is_bullish_smooth = False,
            alert_bearish = False,
            alert_bullish = False,
            is_bearish = False,
            is_bullish = False,
            start_short_trade = False,
            start_long_trade = False,
            last_signal_was_bearish = False,
            last_signal_was_bullish = False,
            bars_since_red_entry = False,
            bars_since_red_exit = False,
            bars_since_green_entry = False,
            bars_since_green_exit = False,
            is_valid_short_exit = False,
            is_valid_long_exit = False,
            end_short_trade_dynamic = False,
            end_long_trade_dynamic = False,
            end_long_trade_strict = False,
            end_short_trade_strict = False,
            is_dynamic_exit_valid = False,
            end_long_trade = False,
            end_short_trade = False,
            is_short=False,
            is_long=False,
        )
        index_count = Index_Count(
            i=i, 
            bars_count_struct=bars_count, 
            max_bars_held=settings_ml["max_bars_held"],
        )
        
        src_mi_max_period = Source(**g_iloc(
            struct=src, 
            start=index_count.i_mi_max_period, 
            end=index_count.i_a_1,
            struct_values_set=frozenset({"open", "high", "low", "close"}),
        ))
        src_mi_max_period_mu_5 = Source(**g_iloc(
            struct=src, 
            start=index_count.i_mi_max_period_mu_5, 
            end=index_count.i_a_1,
            struct_values_set=frozenset({"open", "high", "low", "close"}),
        ))
        src_mi_max_period_indcs = Source(**g_iloc(
            struct=src, 
            start=i - max(g_not_iter_from_iter(list(settings_ml["feature_used"].values()))) * 2 + 1,  
            end=index_count.i_a_1,
            struct_values_set=frozenset({"open", "high", "low", "close"}),
        ))
        feature_series = Feature_Series(
            ADX_1=ta.trend.ADXIndicator(high=src_mi_max_period_indcs.high, low=src_mi_max_period_indcs.low, close=src_mi_max_period_indcs.close, window=settings_ml["feature_used"]["ADX"][0][0]).adx().iloc[-1],
            RSI_1=ta.momentum.RSIIndicator(close=src_mi_max_period_indcs.close, window=settings_ml["feature_used"]["RSI"][0][0]).rsi().iloc[-1],
            RSI_2=ta.momentum.RSIIndicator(close=src_mi_max_period_indcs.close, window=settings_ml["feature_used"]["RSI"][1][0]).rsi().iloc[-1],
            CCI_1=ta.trend.CCIIndicator(high=src_mi_max_period_indcs.high, low=src_mi_max_period_indcs.low, close=src_mi_max_period_indcs.close, window=settings_ml["feature_used"]["CCI"][0][0]).cci().iloc[-1],
            WT_1=ta.momentum.WilliamsRIndicator(high=src_mi_max_period_indcs.high, low=src_mi_max_period_indcs.low, close=src_mi_max_period_indcs.close,).williams_r().iloc[-1]
        )
        feature_arrays.ADX_1[index_count.i_mi_max_period] = feature_series.ADX_1
        feature_arrays.RSI_1[index_count.i_mi_max_period] = feature_series.RSI_1
        feature_arrays.RSI_2[index_count.i_mi_max_period] = feature_series.RSI_2
        feature_arrays.CCI_1[index_count.i_mi_max_period] = feature_series.CCI_1
        feature_arrays.WT_1[index_count.i_mi_max_period] = feature_series.WT_1
          
        # CLASSIFICATION
        if i < bars_count.bars_used + bars_count.max_period:
            ml.y_train_array[index_count.i_mi_max_period] = label_direction.short\
                if src.close[index_count.i_mi_max_bars_held] < price else label_direction.long\
                if src.close[index_count.i_mi_max_bars_held] > price else label_direction.neutral
        else:
            # pass
            # init values for filters
            if i % 100 == 0:
                print(i, bars_count.max_bars_back)

            sma_last_value = g_sma(src_mi_max_period_mu_5.close, settings_filter["EMA"]).iloc[-1]
            ema_last_value = g_ema(src_mi_max_period_mu_5.close, settings_filter["EMA"]).iloc[-1]
            
            yhat1 = g_rational_quadratic(
                src=src_mi_max_period.close.to_numpy(),
                lookback=settings_filter["KERNEL"]["lookback_window"], 
                relative_weight=settings_filter["KERNEL"]["relative_weight"], 
                start_at_bar=settings_filter["KERNEL"]["regression_level"],
            )
            yhat2 = g_gaussian(
                src=src_mi_max_period.close.to_numpy(),
                lookback=settings_filter["KERNEL"]["lookback_window"] - settings_filter["KERNEL"]["enhance_smoothing_lag"], 
                start_at_bar=settings_filter["KERNEL"]["regression_level"],
            )

            # FILTERS
            # signal filter l2
            # Lorentzian distance
            ml.predictions = []
            ml.distances = []
            ml.last_distance = -1.0
            ml.signal = label_direction.neutral
            for i_2 in range(settings_ml["bars_used"]):
                distance = g_lorentzian_distance(i_2, feature_series, feature_arrays)
                if distance >= ml.last_distance and i_2 % settings_ml["max_bars_held"] == 0:
                    ml.last_distance = distance
                    ml.distances.append(distance)
                    ml.predictions.append(round(ml.y_train_array[i_2]))
                    if len(ml.predictions) > settings_ml["neighbors_count"]:
                        # получаем значение ближе к концу массива (75%)
                        # Использование 75% может быть связано с тем, что последние значения в массиве расстояний могут лучше 
                        # отражать текущую тенденцию или состояние рынка.
                        ml.last_distance = ml.distances[round(settings_ml["neighbors_count"] * 3 / 4)]
                        ml.distances.pop(0)
                        ml.predictions.pop(0)

            # default filters
            filter.volatility = g_f_volatility(
                src_mi_max_period.high, 
                low=src_mi_max_period.low, 
                close=src_mi_max_period.close,
            )
            filter.ADX = g_f_adx(latest_adx=feature_series.ADX_1)
            filter.regime = g_f_regime(
                high=src_mi_max_period_mu_5.high.to_numpy(),
                low=src_mi_max_period_mu_5.low.to_numpy(),
                ohlc4=(src_mi_max_period_mu_5.close.to_numpy() + src_mi_max_period_mu_5.open.to_numpy() + src_mi_max_period_mu_5.high.to_numpy() + src_mi_max_period_mu_5.low.to_numpy()) / 4,
            )
            
            ml.prediction = sum(ml.predictions)
            
            # if filter.filter_all_default:
            ml.signal = label_direction.long\
                if ml.prediction > 0 else label_direction.short\
                if ml.prediction < 0 else label_direction.neutral
            
            # signal filter l3
            # EMA/SMA
            last_price = src.close.iloc[i]
            if settings_filter["USE_EMA_f"]:
                last_ema = g_ema(src.close[index_count.i_mi_max_period_mu_5:], settings_filter["EMA"]).iloc[-1]
                filter.is_EMA_down_trend =  last_ema < last_price
                filter.is_EMA_up_trend = last_ema > last_price
            if settings_filter["USE_SMA_f"]:
                last_sma = g_sma(src.close[index_count.i_mi_max_period_mu_5:], settings_filter["SMA"]).iloc[-1]
                filter.is_SMA_down_trend = last_sma < last_price
                filter.is_SMA_up_trend = last_sma > last_price
            
            # Bar Count filters
            if ml.signal != g_iloc(array=ml.signals, start=-1):
                ml.bars_held = 0
            else:
                ml.bars_held += 1
            filter.is_held_n_bars = ml.bars_held >= settings_ml["max_bars_held"]
            filter.is_held_less_n_bars = 0 < ml.bars_held < settings_ml["max_bars_held"]

            # Fractal Filters: Derived from relative appearances of signals in a given time series fractal/segment with a default length of 4 bars
            filter.is_different_signal_type = ml.signal != g_iloc(array=ml.signals, start=-2)
            filter.is_early_signal_flip = ml.signal != g_iloc(array=ml.signals, start=-2) and \
                g_iloc(array=ml.signals, start=-2) != g_iloc(array=ml.signals, start=-3) and \
                g_iloc(array=ml.signals, start=-3) != g_iloc(array=ml.signals, start=-4)
            filter.is_buy_signal = ml.signal == label_direction.long and \
                filter.is_EMA_up_trend and \
                filter.is_SMA_up_trend
            filter.is_sell_signal = ml.signal == label_direction.short and \
                filter.is_EMA_down_trend and \
                filter.is_SMA_down_trend
            filter.is_last_signal_buy = g_iloc(array=ml.signals, start=-4) == label_direction.long and \
                g_iloc(array=filter_arrays.is_EMA_up_trends, start=-4) and \
                g_iloc(array=filter_arrays.is_SMA_up_trends, start=-4)
            filter.is_last_signal_sell = g_iloc(array=ml.signals, start=-4) == label_direction.short and \
                g_iloc(array=filter_arrays.is_EMA_down_trends, start=-4) and \
                g_iloc(array=filter_arrays.is_SMA_down_trends, start=-4)
            filter.is_new_buy_signal = filter.is_buy_signal and filter.is_different_signal_type
            filter.is_new_sell_signal = filter.is_sell_signal and filter.is_different_signal_type

            # Kernel Rates of Change
            filter.was_bearish_rate = g_all_conditions(g_iloc(array=ml.yhat1s, start=-2) > g_iloc(array=ml.yhat2s, start=-1))
            filter.was_bullish_rate = g_all_conditions(g_iloc(array=ml.yhat1s, start=-2) < g_iloc(array=ml.yhat2s, start=-1))
            filter.is_bearish_rate = g_all_conditions(g_iloc(array=ml.yhat1s, start=-1) > g_iloc(array=ml.yhat2s, start=-2))
            filter.is_bullish_rate = g_all_conditions(g_iloc(array=ml.yhat1s, start=-1) < g_iloc(array=ml.yhat2s, start=-2))
            filter.is_bearish_change = filter.is_bearish_rate and filter.was_bullish_rate
            filter.is_bullish_change = filter.is_bullish_rate and filter.was_bearish_rate
            # Kernel Crossovers
            filter.is_bullish_cross_alert = g_crossover(ml.yhat2s, ml.yhat1s)
            filter.is_bearish_cross_alert = g_crossunder(ml.yhat1s, ml.yhat2s)
            filter.is_bullish_smooth = yhat2 >= yhat1
            filter.is_bearish_smooth = yhat2 <= yhat1
            # Alert Variables
            filter.alert_bullish = filter.is_bullish_cross_alert if settings_filter["KERNEL"]["USE_enhance_smoothing_lag"] else filter.is_bullish_change
            filter.alert_bearish = filter.is_bearish_cross_alert if settings_filter["KERNEL"]["USE_enhance_smoothing_lag"] else filter.is_bearish_change
            i_alert_bullish = i if filter.alert_bullish else np.nan
            i_alert_bearish = i if filter.alert_bearish else np.nan

            # Bullish and Bearish Filters based on Kernel
            filter.is_bullish = (filter.is_bullish_smooth if settings_filter["KERNEL"]["USE_enhance_smoothing_lag"] else filter.is_bullish_rate) if settings_filter["KERNEL"]["USE_KERNEL_f"] else True
            filter.is_bearish = (filter.is_bearish_smooth if settings_filter["KERNEL"]["USE_enhance_smoothing_lag"] else filter.is_bearish_rate) if settings_filter["KERNEL"]["USE_KERNEL_f"] else True

            # Entry Conditions: Booleans for ML Model Position Entries
            filter.start_long_trade = filter.is_new_buy_signal and filter.is_bullish and filter.is_EMA_up_trend and filter.is_SMA_up_trend
            filter.start_short_trade = filter.is_new_sell_signal and filter.is_bearish and filter.is_EMA_down_trend and filter.is_SMA_down_trend
            i_start_long_trade = i if filter.start_long_trade else np.nan
            i_start_short_trade = i if filter.start_short_trade else np.nan
            filter_arrays.start_long_trades.append(filter.start_long_trade)
            filter_arrays.start_short_trades.append(filter.start_short_trade)

            # Fixed Exit Conditions: Booleans for ML Model Position Exits based on a Bar-Count Filters
            filter.end_long_trade_strict = ((filter.is_held_less_n_bars and filter.is_last_signal_buy) or (filter.is_held_less_n_bars and filter.is_new_sell_signal and filter.is_last_signal_buy)) and g_iloc(filter_arrays.start_long_trades, index_count.i_mi_max_bars_held)
            filter.end_short_trade_strict = ((filter.is_held_less_n_bars and filter.is_last_signal_sell) or (filter.is_held_less_n_bars and filter.is_new_buy_signal and filter.is_last_signal_sell)) and g_iloc(filter_arrays.start_short_trades, index_count.i_mi_max_bars_held)
            filter.is_dynamic_exit_valid = not settings_filter["USE_EMA_f"] and not settings_filter["USE_SMA_f"] and not settings_filter["KERNEL"]["USE_enhance_smoothing_lag"]
            filter.end_long_trade = settings_ml["USE_dynamic_exits"] and filter.is_dynamic_exit_valid if filter.end_long_trade_dynamic else filter.end_long_trade_strict
            filter.end_short_trade = settings_ml["USE_dynamic_exits"] and filter.is_dynamic_exit_valid if filter.end_short_trade_dynamic else filter.end_short_trade_strict

            # Dynamic Exit Conditions: Booleans for ML Model Position Exits based on Fractal Filters and Kernel Regression Filters
            filter.last_signal_was_bullish = g_bars_since(i_start_long_trade, i) < g_bars_since(i_start_short_trade, i)
            filter.last_signal_was_bearish = g_bars_since(i_start_short_trade, i) > g_bars_since(i_start_short_trade, i)
            filter.bars_since_red_entry = g_bars_since(i_start_short_trade, i)
            filter.bars_since_red_exit = g_bars_since(i_alert_bullish, i)
            filter.bars_since_green_entry = g_bars_since(filter.start_long_trade, i)
            filter.bars_since_green_exit = g_bars_since(i_alert_bearish, i)
            filter.is_valid_short_exit = filter.bars_since_green_exit > filter.bars_since_red_entry
            filter.is_valid_long_exit = filter.bars_since_red_exit > filter.bars_since_green_entry
            filter.end_long_trade_dynamic = all((filter.is_bearish_change, g_iloc(filter_arrays.is_valid_long_exits, -2)))
            filter.end_short_trade_dynamic = all((filter.is_bullish_change, g_iloc(filter_arrays.is_valid_short_exits, -2)))

            # <appends>
            ml.signals.append(ml.signal)
            ml.yhat1s.append(yhat1)
            ml.yhat2s.append(yhat2)
            if len(ml.yhat1s) > settings_ml["max_bars_held"]:
                ml.yhat1s.pop(0)
                ml.yhat2s.pop(0)

            filter_arrays.is_EMA_up_trends.append(filter.is_EMA_up_trend)
            filter_arrays.is_EMA_down_trends.append(filter.is_EMA_down_trend)
            filter_arrays.is_SMA_up_trends.append(filter.is_SMA_up_trend)
            filter_arrays.is_SMA_down_trends.append(filter.is_SMA_down_trend)
            filter_arrays.is_valid_short_exits.append(filter.is_valid_short_exit)
            filter_arrays.is_valid_long_exits.append(filter.is_valid_long_exit)
            filter_arrays.start_short_trades.append(filter.start_short_trade)
            filter_arrays.start_long_trades.append(filter.start_long_trade)
            
            # print(filter.volatility, filter.ADX, filter.regime)
            filter.filter_all_default = all((
                filter.regime, 
                filter.ADX, 
                filter.volatility,
                filter.is_held_n_bars,
            ))
            filter.is_short = all((
                filter.filter_all_default,
                ml.signal == label_direction.short, 
                filter.is_EMA_down_trend,
                filter.is_SMA_down_trend,
            ))
            filter.is_long = all((
                filter.filter_all_default,
                ml.signal == label_direction.long, 
                filter.is_EMA_up_trend,
                filter.is_SMA_up_trend,
            ))
            # print(filter.is_short, filter.is_held_n_bars, ml.signal, ml.signal == label_direction.short and filter.is_short)
            if filter.is_short or filter.is_long:
                if filter.is_short:
                    total_signal = label_direction.short
                elif filter.is_long:
                    total_signal = label_direction.long
                else:
                    total_signal = label_direction.neutral
                print("# ----------- #")
                print(
                    f"signal: {total_signal}\n"
                    f"index: {index_count.i_mi_bars_used_and_max_period}\n"
                )
                print(
                    f"FILTERS:\n"
                    f"regime: {filter.regime}\n" 
                    f"adx: {filter.ADX}\n"
                    f"volatility: {filter.volatility}\n"
                    f"is_held_n_bars: {filter.is_held_n_bars}\n"
                )
                print("# ----------- #")
                ml.y_test_array[index_count.i_mi_bars_used_and_max_period] = total_signal
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(bars_count.bars_used),
        y=src.close[bars_count.max_bars_back - bars_count.bars_used:],
        mode="lines",
        name="close",
    ))
    target = ml.y_test_array
    index_long = target == label_direction.long
    index_short = target == label_direction.short
    # index_neutral = ml.y_test_array == label_direction.neutral
    fig.add_trace(go.Scatter(
        x=np.arange(bars_count.bars_used)[index_long],
        y=src.close[bars_count.bars_used + bars_count.max_period:].to_numpy()[index_long],
        mode="markers",
        name="long",
        marker={"color": "green"},
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(bars_count.bars_used)[index_short],
        y=src.close[bars_count.bars_used + bars_count.max_period:].to_numpy()[index_short],
        mode="markers",
        name="short",
        marker={"color": "red"},
    ))
    from datetime import datetime
    fig.write_html(f"plotly_html/{symbol}_bu{settings_ml["bars_used"]}_{datetime.now().strftime('%Y.%m.%d_%H.%M.%S')}.html")
    fig.show()

 
async def pre_main():
    from pprint import pprint
    from time import sleep
    while True:
        src = Source(**(await g_klines_split(await g_klines(symbol="SUIUSDT", qty=200*5,))))
        yhat1 = g_rational_quadratic(
            close=src.close.to_numpy(),
            lookback=settings_filter["KERNEL"]["lookback_window"], 
            relative_weight=settings_filter["KERNEL"]["relative_weight"], 
            start_at_bar=settings_filter["KERNEL"]["regression_level"],
        )
        # print(yhat1)
        yhat2 = g_gaussian(
            close=src.close.to_numpy(),
            lookback=settings_filter["KERNEL"]["lookback_window"] - settings_filter["KERNEL"]["enhance_smoothing_lag"], 
            start_at_bar=settings_filter["KERNEL"]["regression_level"],
        )
        yhat1s = []
        yhat2s = []
        was_bearish_rate = g_all_conditions(g_iloc(array=yhat1s, start=-2) > g_iloc(array=yhat2s, start=-1))
        was_bullish_rate = g_all_conditions(g_iloc(array=yhat1s, start=-2) < g_iloc(array=yhat2s, start=-1))
        is_bearish_rate = g_all_conditions(g_iloc(array=yhat1s, start=-1) > g_iloc(array=yhat2s, start=-2))
        is_bullish_rate = g_all_conditions(g_iloc(array=yhat1s, start=-1) < g_iloc(array=yhat2s, start=-2))
        is_bearish_change = is_bearish_rate and was_bullish_rate
        is_bullish_change = is_bullish_rate and was_bearish_rate
        # Kernel Crossovers
        is_bullish_cross_alert = g_crossover(yhat2s, yhat1s)
        is_bearish_cross_alert = g_crossunder(yhat1s, yhat2s)
        is_bullish_smooth = yhat2 >= yhat1
        is_bearish_smooth = yhat2 <= yhat1
        # Alert Variables
        alert_bullish = is_bullish_cross_alert if settings_filter["KERNEL"]["USE_enhance_smoothing_lag"] else is_bullish_change
        alert_bearish = is_bearish_cross_alert if settings_filter["KERNEL"]["USE_enhance_smoothing_lag"] else is_bearish_change
        # i_alert_bullish = i if alert_bullish else np.nan
        # i_alert_bearish = i if alert_bearish else np.nan

        # Bullish and Bearish Filters based on Kernel
        is_bullish = (is_bullish_smooth if settings_filter["KERNEL"]["USE_enhance_smoothing_lag"] else is_bullish_rate) if settings_filter["KERNEL"]["USE_KERNEL_f"] else True
        is_bearish = (is_bearish_smooth if settings_filter["KERNEL"]["USE_enhance_smoothing_lag"] else is_bearish_rate) if settings_filter["KERNEL"]["USE_KERNEL_f"] else True
        
        # <appends>
        yhat1s.append(yhat1)
        yhat2s.append(yhat2)
        if len(yhat1s) > settings_ml["max_bars_held"]:
            yhat1s.pop(0)
            yhat2s.pop(0)
        print(
            src.close.iloc[-1],
            # yhat1,
            # round(yhat2[-1], 4)
            is_bearish,
            is_bullish,
        )
        sleep(0.3)

import asyncio

asyncio.run(pre_main())
# asyncio.run(main())