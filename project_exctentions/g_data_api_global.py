from project_exctentions.g_api_session import session_

from asyncio import (
    to_thread as aio_to_thread,
    gather as aio_gather
)
import numpy as np
from time import time

async def g_klines(
    symbol,
    qty: int,
    interval=1,
    float_=True,
):  
    start = time() * 1000
    limits = np.append(np.full(qty // 1000, 1000), qty % 1000)
    data = np.concatenate(await aio_gather(*(
        aio_to_thread(lambda i=i: session_.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limits[i],
            end=str(int(start - (i * 60_000_000)))
        )["result"]["list"])
        for i in range(len(limits))
        if limits[i] > 0
    )))[::-1]
    return np.float64(data) if float_ else data

# async def g_symbols_f(
#     klines_all_num=settings_ml["max_bars_back"],
#     volume_24=settings_bt["symbols_f_volume_24"],
#     index_volatility_24=0.9,
#     deviations=0.05,
#     klines_week_mult=1.3,
#     not_in_symbol_list=["USDC"],
# ):
#     klines_week = int(klines_all_num / 60 / 24 / 7 * klines_week_mult)
#     klines_week = klines_week if klines_week > 1 else 2
#     tasks = {
#         symbol: aio_to_thread(lambda v=symbol: session_.get_kline(
#             category="linear",
#             interval="W",
#             symbol=symbol,
#             limit=klines_week
#         )["result"]["list"])
#         for symbol in np.array([
#             el["symbol"]
#             for el in session_.get_tickers(category="linear")["result"]["list"]
#             if (
#                 (
#                     (float(el["lowPrice24h"]) + float(el["highPrice24h"])) / 2 * float(el["volume24h"]) >= volume_24 and
#                     abs(index_volatility_24 / (float(el["lowPrice24h"]) / float(el["highPrice24h"])) - 1) <= deviations
#                 ) and
#                 all(filter(lambda v: v not in el["symbol"], not_in_symbol_list))
#             )
#         ])
#     }
#     return np.array([
#         symbol
#         for symbol, v_ in zip(tasks.keys(), (await aio_gather(*tasks.values())))
#         if len(v_) >= klines_week
#     ])