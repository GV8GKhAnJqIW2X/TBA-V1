import numpy as np
import os
import pandas as pd
import pickle

def g_klines_splitting(klines):
    return pd.DataFrame({
        'close': klines[:, 4],
        'high': klines[:, 2],
        'low': klines[:, 3],
    })

def g_rolling_apply(
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

def g_df_range_create(
    data,
    columns,
    range_,
    replace,
):
    def g_df_fill(
        data,
        columns,
        value=np.nan
    ):
        for column in columns:
            data[column] = value
        return data

    data = g_df_fill(data, columns)
    for i in range(len(range_)):
        data.loc[range_[i], columns[i]] = replace[i]
    return data

def s_df_dump(
    data, 
    name, 
    dir="data_pack",
):
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(f"{dir}/{name}.pickle", "wb") as f:
        pickle.dump(data, f)

def g_df_load(
    name, 
    dir="data_pack",
):
    with open(f"{dir}/{name}.pickle", "rb") as f:
        return pickle.load(f)