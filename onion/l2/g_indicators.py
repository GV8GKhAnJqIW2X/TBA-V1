from onion.l1.s_logging import logger
from onion.l1.g_settings import settings_ml

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# @logger.catch
def g_rsi(data, period=14):
    delta = data['close'].diff()
    return 100 - (100 / (
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
    ))

# @logger.catch
def g_adx(data, period=14):
    atr = pd.DataFrame({
        'tr1': data["high"] - data["low"],
        'tr2': np.abs(data["high"] - data["close"].shift()),
        'tr3': np.abs(data["low"] - data["close"].shift())
    })\
        .max(axis=1)\
        .rolling(window=period)\
        .mean()
    low_diff, high_diff = map(lambda v: v.diff(), (data["low"], data["high"]))
    calculate_di = lambda v1, v2: 100 * (
        pd.Series(np.where((v1 > v2) & (v1 > 0), v1, 0))
            .rolling(window=period)
            .mean()
        / atr
    )
    minus_di = calculate_di(low_diff, high_diff)
    plus_di = calculate_di(high_diff, low_diff)

    return (100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di)))\
        .rolling(window=period)\
        .mean()

# @logger.catch
def g_cci(data, period=20):
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    sma = typical_price.rolling(window=period).mean()
    return (typical_price - sma) / (0.015 * (typical_price - sma)\
        .abs()\
        .rolling(window=period)\
        .mean())

# @logger.catch
def g_williams_r(data, period=14):
    highest_high = data["high"].rolling(window=period).max()
    return -100 * (highest_high - data["close"]) / (
        highest_high -
        data["low"]
            .rolling(window=period)
            .min()
    )

# @logger.catch
def g_tsi(data, period=14,):
    return data["close"]\
        .rolling(window=period)\
        .corr(pd.Series(np.arange(len(data["close"]))))

# @logger.catch
def g_lorentzian_distances(feature_arrs,bars_back,):
    return np.sum([
        feature_arr\
            .fillna(feature_arr.mean())\
            .rolling(window=bars_back)\
            .apply(lambda v: np.log(1 + np.abs(v.iloc[0] - v.iloc[-1])))
        for feature_arr in feature_arrs
    ], axis=0)

# @logger.catch
async def g_data_x(
    data,
    indcs_list,
    l1_indcs_train_sett=settings_ml["l1_indcs_train_sett"],
    l2_indcs_train_sett=settings_ml["l2_indcs_train_sett"],
):
    async def g_clean_x(x):
        return pd.DataFrame(
            SimpleImputer(strategy="mean")\
                .fit_transform(x.replace({-np.inf: np.nan, np.inf: np.nan})),
            columns=x.columns
    )

    choice_l1 = {
        "INDCS/ RSI": lambda **kwargs: g_rsi(**kwargs),
        "INDCS/ ADX": lambda **kwargs: g_adx(**kwargs),
        "INDCS/ CCI": lambda **kwargs: g_cci(**kwargs),
        "INDCS/ WT": lambda **kwargs: g_williams_r(**kwargs),
        "INDCS/ TSI": lambda **kwargs: g_tsi(**kwargs),
    }
    choice_l2 = {
        "INDCS/ LD": lambda l1_arrays, **kwargs: g_lorentzian_distances(feature_arrs=l1_arrays,**kwargs,), 
    }
    for key in l1_indcs_train_sett:
        data[key] = choice_l1[key](data=data, **l1_indcs_train_sett[key])
    indcs_serieses = [data[column] for column in l1_indcs_train_sett]
    for key in l2_indcs_train_sett:
        data[key] = choice_l2[key](indcs_serieses, **l2_indcs_train_sett[key])
    data[indcs_list] = await g_clean_x(data[indcs_list])
    return data
