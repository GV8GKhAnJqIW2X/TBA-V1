from collections.abc import Iterable

import numpy as np

async def g_klines_split(klines):
    return {
        "open": klines[:, 1],
        "high": klines[:, 2],
        "low": klines[:, 3],
        "close": klines[:, 4],
    }

def g_not_iter_from_iter(iter_, ignore_str=False):
    if len(iter_) == 0:
        return 0
    
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

def g_iloc(
    iterator, 
    slice_, 
    need=[], 
    ignore=[],
):
    """
    PARAMS:
    iterator - Iter<ordered collection>. 
    
    """
    if isinstance(iterator, dict):
        if not need:
            need = frozenset(iterator.keys())
        
        return {
            key: value[slice_]
            for key, value in iterator.items()
            if key in need and key not in ignore
        }
    elif isinstance(iterator, Iterable) and not isinstance(iterator, str):
        if not need:
            need = frozenset(range(len(iterator)))

        return [
            value[slice_] 
            for i, value in enumerate(iterator)
            if i in need and i not in ignore
        ]
    return iterator

def g_number_need_to_filled(
    array,
    empty_is_nan=True,
    empty_is_zero=False,
    check_params=True,
):
    # check params
    if check_params:
        try:
            len(array)
        except:
            raise ValueError("The array is not measured")
    
    # main
    if empty_is_nan:
        return np.isnan(array).sum()
    elif empty_is_zero:
        return np.count_nonzero(array)