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

def g_max_value_from_nested_heterogeneous_array(lst):
    lst_not_nested = []
    lst_nested = []
    for item in lst:
        if isinstance(item, Iterable) and not isinstance(item, str):
            if isinstance(item, dict):
                lst_nested.append(list(item.values()))
            else:
                lst_nested.append(item)
        else:
            if not isinstance(item, str):
                lst_not_nested.append(item)

    lst = list(chain.from_iterable(lst_nested)) + lst_not_nested    
    return np.max(np.array(list(filter(lambda v: isinstance(v, (int, float)), lst)))\
        .flatten())