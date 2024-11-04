from onion.l1.s_logging import logger

import numpy as np
import pandas as pd

@logger.catch
async def g_rolling_apply(
    arr, 
    window, 
    func,
):
    len_arr = len(arr)
    result = np.empty(len_arr)
    result[:window] = np.nan

    for i in range(window, len_arr):
        if i % 1000 == 0:
            print(f"progress : {round(i / len_arr * 100, 2)}%")
        result[i] = func(arr[i-window:i])
    return result

@logger.catch
async def g_klines_split(klines):
    return pd.DataFrame({
        "close": klines[:, 4],
        "low": klines[:, 3],
        "high": klines[:, 2],
    })