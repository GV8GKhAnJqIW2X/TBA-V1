import numpy as np
import pandas as pd
from itertools import chain
from collections.abc import Iterable

async def g_rolling_apply(
    arr, 
    window, 
    func,
):
    len_arr = len(arr)
    result = np.empty(len_arr)
    result[:window] = np.nan

    for i in range(window, len_arr):
        result[i] = func(arr[i-window:i])
        if i % 1000 == 0:
            print(f"progress : {round(i / len_arr * 100, 2)}%")
    return result

async def g_klines_split(klines):
    return pd.DataFrame({
        "open": klines[:, 1],
        "high": klines[:, 2],
        "low": klines[:, 3],
        "close": klines[:, 4],
    })

def g_not_iter_from_iter(iter_, ignore_str=False):
    iter_ = iter_.values() if isinstance(iter_, dict) else iter_
    not_iter = []

    def from_iter(iter__=iter_):
        for item in iter__:
            item = item.values() if isinstance(item, dict) else item
            if isinstance(item, Iterable) and not isinstance(item, str):
                from_iter(item)
            else:
                not_iter.append(item)
    from_iter(iter_)
    return filter(lambda v: not isinstance(v, str) if not ignore_str else True, not_iter)

def g_iloc_array(
    array, 
    start=None, 
    end=None, 
    step=None,
):
    try:
        if end is None and step is None:
            return array[start] if len(array) else array
        else:
            return array[start:end:step]
    except IndexError:
        return array

def g_iloc(
    array=None,
    start=None, 
    end=None,
    step=None,
    df=None,
    struct = None,
    struct_values_set: frozenset = None,
):
    if struct is not None:
        return {
            key: g_iloc_array(arr, start, end, step)
            for key, arr, in vars(struct).items() 
            if key in struct_values_set
        }
    elif df is not None:
        return df.iloc[start:end:step]
    elif array is not None:
        return g_iloc_array(array, start, end, step)
    
def g_ohlc4(struct, df):
    if struct is not None:
        return (struct.open + struct.high + struct.low + struct.close) / 4
    else:
        return (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    
def g_all_conditions(*args):
    try:
        return all(args)
    except:
        return False