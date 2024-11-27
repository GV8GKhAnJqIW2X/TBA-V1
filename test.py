from structures import *
from g_utils import *
from g_data_api_global import *
from g_indicators import *
from transformation import *

import pandas as pd
import numpy as np
import ta

async def main():
    from time import sleep
    while True:
        src = Source(**(await g_klines_split(await g_klines(symbol="SUIUSDT", qty=500,))))
        print(ta.momentum.WilliamsRIndicator(high=src.high, low=src.low, close=src.close, ).williams_r().iloc[-1])
        sleep(3)

if __name__ == "__main__":
    import asyncio
    lst = [1, 2, 3, 4, 5]
    print(np.roll(lst, 1))
    print(lst[1:], lst[:-1])