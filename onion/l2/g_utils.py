from onion.l1.s_logging import logger

import numpy as np

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
        result[i] = func(arr[i-window:i])
    return result