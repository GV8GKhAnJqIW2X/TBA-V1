from onion.l1.s_logging import logger

import pickle
import os

# @logger.catch
async def s_data_dump(
    data, 
    name, 
    dir="data_pack/data_backtest",
):
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(f"{dir}/{name}.pickle", "wb") as f:
        pickle.dump(data, f)

# @logger.catch
async def g_data_load(name,):
    with open(f"{name}.pickle", "rb") as f:
        return pickle.load(f)

# @logger.catch
async def g_files_arr(dir="data_pack/data_backtest", suffix_del=".pickle"):
    return frozenset([file.rstrip(suffix_del) for file in os.listdir(dir)])

# @logger.catch
async def g_data_loads(dir="data_pack"):
    return [await g_data_load(name=f"{dir}/{file}") for file in await g_files_arr(dir=dir)]