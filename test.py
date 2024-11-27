import itertools
import numpy as np
from collections.abc import Iterable

dct = {
    "RSI": [[14], [9], [10], [11], [12]],
    "ADX": [[20]],
    "CCI": [[20]],
    "WT": [[15]]
}

lst = list(dct.values())
def g_max_value_from_nested_array(lst):
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

    lst = list(itertools.chain.from_iterable(lst_nested)) + lst_not_nested    
    return np.max(np.array(list(filter(lambda v: isinstance(v, (int, float)), lst)))\
        .flatten())

lst_2 = list({
    "USE_volatility_f": True,
    "USE_regime_f": True,
    "USE_ADX_f": True,
    "USE_EMA_f": True,
    "USE_SMA_f": True,
    "ADX": 20,
    "EMA": 200,
    "SMA": 200,
    "regime": -0.1,
    "KERNEL": {
        "USE_KERNEL": True,
        "USE_enhance_smoothing_lag": True,
        "lookback_window": 8,
        "relative_weighting": 8,
        "regression_level": 25,
        "enhance_smoothing_lag": 2
    }
}.values())
# print(lst_2)
print(g_max_value_from_nested_array(lst + lst_2))