from settings_ import *
from structures import *
from g_indicators import *
from transformation import *
from g_data_api_global import *
from g_utils import *

async def main():
    max_period = g_max_value_from_nested_heterogeneous_array(settings_ml["feature_used"] + list(settings_filter.values()))
    bars_count = Bars_Count(
        bars_used=settings_ml["bars_used"], 
        max_period=max_period, 
        max_bars_back=(settings_ml["bars_used"] + max_period) * 2, 
    )
    src = Source(**(await g_klines_split(await g_klines(qty=bars_count.max_bars_back))))
    feature_arrays = Feature_Arrays([pd.Series(np.full(bars_count.max_bars_back, np.nan)) for _ in range(5)])
    label_direction = Label(
        short=-1, 
        long=1, 
        neutral=0,
    )
    ml_model = ML_Model(
        y_train_array=pd.Series(np.full(bars_count.bars_used, label_direction.neutral)),
        distances=[],
        predictions=[],
        signals=[],
        is_EMA_up_trends=[],
        is_EMA_down_trends=[],
        is_SMA_up_trends=[],
        is_SMA_down_trends=[],
        last_distance=-1.0,
        prediction=label_direction.neutral,
        signal=label_direction.neutral,
        bars_held=0,
    )

    start_iteration = bars_count.max_bars_back - (bars_count.bars_used * 2)
    for i, price in enumerate(src.close.iloc[start_iteration:], start=start_iteration,):
        feature_series = Feature_Series(
            g_series_from(i, "ADX", src, *settings_ml["feature_used"]["ADX"][0]),
            g_series_from(i, "RSI", src, *settings_ml["feature_used"]["RSI"][0]),
            g_series_from(i, "CCI", src, *settings_ml["feature_used"]["CCI"][0]),
            g_series_from(i, "WT", src, *settings_ml["feature_used"]["WT"][0]),
            g_series_from(i, "RSI", src, *settings_ml["feature_used"]["RSI"][1]),
        )
        feature_arrays.f1[i] = feature_series.f1
        feature_arrays.f2[i] = feature_series.f2
        feature_arrays.f3[i] = feature_series.f3
        feature_arrays.f4[i] = feature_series.f4
        feature_arrays.f5[i] = feature_series.f5
        
        i_down_max_period_a1 = i - bars_count.max_period + 1
        i_a1 = i + 1
        close_down_i_a1 = src.close.iloc[:i_a1]
        
        if settings_filter["USE_EMA_f"]:
            ema_last_value = g_ema(src.close[i - settings_filter["EMA"]], settings_filter["EMA"]).iloc[-1]
            is_EMA_up_trend = src.close[i] > ema_last_value
            is_EMA_down_trend = src.close[i] < ema_last_value
        else:
            is_EMA_up_trend = True
            is_EMA_down_trend = True
        
        if settings_filter["USE_SMA_f"]:
            sma_last_value = g_sma(src.close[i - settings_filter["SMA"]], settings_filter["EMA"]).iloc[-1]
            is_SMA_up_trend = src.close[i] > sma_last_value
            is_SMA_down_trend = src.close[i] < sma_last_value

        else:
            is_SMA_up_trend = True
            is_SMA_down_trend = True
        
        # classification
        if i < bars_count.bars_used:
            ml_model.y_train_array[i] = label_direction.short\
                if src.close[i - settings_ml["y_train_bars_back"]:i] > price else label_direction.long\
                if src.close[i - settings_ml["y_train_"]] < price else label_direction.neutral
        else:
            i_down_used = i - bars_count.bars_used
            d = g_lorentzian_distance(i_down_used, feature_series, feature_arrays)
            if d >= ml_model.last_distance and i % settings_ml["y_train_bars_back"]:
                ml_model.last_distance = d
                ml_model.distances.append(d)
                ml_model.predictions.append(round(ml_model.y_train_array[i_down_used]))
                if len(ml_model.predictions) > settings_ml["neighbors_count"]:
                    ml_model.last_distance = ml_model.distances[round(settings_ml["neighbors_count"] * 3 / 4)]
                    ml_model.distances.pop(0)
                    ml_model.predictions.pop(0)
            
            ml_model.prediction = sum(ml_model.predictions)
        
        filter = Filter(
            g_f_volatility(src.close[i_down_max_period_a1:i_a1]),
            g_f_adx(feature_series.f1),
            g_f_regime(src.close[i_down_max_period_a1:i_a1]),
        )
        if filter.f1 and filter.f2 and filter.f3:
            ml_model.signal = label_direction.long\
                if ml_model.prediction > 0 else label_direction.short\
                if ml_model.prediction < 0 else label_direction.neutral
        
        # appends
        ml_model.signals.append(ml_model.signal)
        ml_model.is_EMA_up_trends(is_EMA_up_trend)
        ml_model.is_EMA_down_trends(is_EMA_down_trend)
        ml_model.is_SMA_up_trends(is_SMA_up_trend)
        ml_model.is_SMA_down_trends(is_SMA_down_trend)
        if len(ml_model.signals) > 4:
            ml_model.signals.pop(0)
            ml_model.is_EMA_up_trends.pop(0)
            ml_model.is_EMA_down_trends.pop(0)
            ml_model.is_SMA_up_trends.pop(0)
            ml_model.is_SMA_down_trends.pop(0)

        # Bar-Count Filters: Represents strict filters based on a pre-defined holding period of 4 bars
        if ml_model.signal != ml_model.signals[-2]:
            ml_model.bars_held = 0
        else:
            ml_model.bars_held += 1
        is_held_n_bars = ml_model.bars_held == settings_ml["y_train_bars_back"]
        is_held_less_than_n_bars = 0 < ml_model.bars_held < settings_ml["y_train_bars_back"]

        # Fractal Filters: Derived from relative appearances of signals in a given time series fractal/segment with a default length of 4 bars
        is_different_signal_type = ml_model.signal != ml_model.signals[-2]
        is_early_signal_flip = ml_model.signal != ml_model.signals[-2] and \
            ml_model.signals[-2] != ml_model.last_signals[-3] and \
            ml_model.signals[-3] != ml_model.signals[-4]
        is_buy_signal = ml_model.signal == label_direction.long and \
            is_EMA_up_trend and \
            is_SMA_up_trend
        is_sell_signal = ml_model.signal == label_direction.short and \
            is_EMA_down_trend and \
            is_SMA_down_trend
        is_last_signal_buy = ml_model.signals[-4] == label_direction.long and \
            ml_model.is_EMA_up_trends[-4] and \
            ml_model.is_SMA_up_trends[-4]
        is_last_signal_sell = ml_model.signals[-4] == label_direction.short and \
            ml_model.is_EMA_down_trends[-4] and \
            ml_model.is_SMA_down_trends
        is_new_buy_signal = is_buy_signal and is_different_signal_type
        is_new_sell_signal = is_sell_signal and is_different_signal_type

        # Kernel Regression Filters: Filters based on Nadaraya-Watson Kernel Regression using the Rational Quadratic Kernel
        yhat1 = g_rational_quadratic(
            src=close_down_i_a1,
            lookback=settings_filter["KERNEL"]["lookback_window"], 
            relative_weight=settings_filter["KERNEL"]["relative_weight"], 
            start_at_bar=settings_filter["KERNEL"]["regression_level"],
        )
        yhat2 = g_gaussian(
            src=close_down_i_a1,
            lookback=settings_filter["KERNEL"]["lookback_window"] - settings_filter["KERNEL"]["enhance_smoothing_lag"], 
            start_at_bar=settings_filter["KERNEL"]["regression_level"],
        )
        # Kernel Rates of Change
        was_bearish_rate = yhat1[-3] > yhat2[-2]
        was_bullish_rate = yhat1[-3] < yhat2[-2]
        is_bearish_rate = yhat1[-1] > yhat1[-2]
        is_bullish_rate = yhat1[-1] < yhat1[-2]
        is_bearish_change = is_bearish_rate and was_bullish_rate
        is_bullish_change = is_bullish_rate and was_bearish_rate
        # Kernel Crossovers
        is_bullish_cross_alert = g_crossover(yhat2, yhat1)
        is_bearish_cross_alert = g_crossover(yhat1, yhat2)
        is_bullish_smooth = yhat1[-1] >= yhat2[-1]
        is_bearish_smooth = yhat1[-1] <= yhat2[-1]
        # Alert Variables
        alert_bullish = is_bullish_cross_alert if settings_filter["KERNEL"]["USE_enhance_smoothing_lag"] else is_bullish_change
        alert_bearish = is_bearish_cross_alert if settings_filter["KERNEL"]["USE_enhance_smoothing_lag"] else is_bearish_change
        # Bullish and Bearish Filters based on Kernel
        is_bullish = (is_bullish_smooth if settings_filter["KERNEL"]["USE_enhance_smoothing_lag"] else is_bullish_rate) if settings_filter["KERNEL"]["USE_KERNEL_f"] else True
        is_bearish = (is_bearish_smooth if settings_filter["KERNEL"]["USE_enhance_smoothing_lag"] else is_bearish_rate) if settings_filter["KERNEL"]["USE_KERNEL_f"] else True

        # Entry Conditions: Booleans for ML Model Position Entries
        start_long_trade = is_new_buy_signal and is_bullish and is_EMA_up_trend and is_SMA_up_trend
        start_short_trade = is_new_sell_signal and is_bearish and is_EMA_down_trend and is_SMA_down_trend

        # Dynamic Exit Conditions: Booleans for ML Model Position Exits based on Fractal Filters and Kernel Regression Filters
        

# проверить работу функций:
#   индикаторов