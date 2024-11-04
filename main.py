from onion.l1.g_settings import settings_ml, settings_bt
from onion.l2.g_data_api_global import g_symbols_f, g_klines
from onion.l2.g_filemanager_pickle import g_files_list, s_data_dump, g_data_load, g_data_loads
from onion.l2.g_indicators import g_data_x
from onion.l2.g_model import *
from onion.l2.g_utils import g_klines_split
from onion.l3.g_data_backtest import g_data_backtest
from onion.l4.g_data_pack import g_data_mark, g_data_trace
from onion.l5.g_report_gui import g_report_plotly
from onion.l5.g_report import g_report

import asyncio
from random import randint
from pprint import pprint

async def main():
    for symbol in\
        await g_symbols_f(klines_all_num=settings_ml["klines_all_num"]):
        # ["BTCUSDT"]:
        # await g_files_list():
        print(symbol)
        
    #     data = await g_data_load(name=f"data_pack/data_backtest/{symbol}")
    #     # data = data[["close", "low", "high"]].copy()

        data = await g_klines_split(await g_klines(symbol=symbol))
        if data is not None:
            indcs_list = list(settings_ml["l1_indcs_train_sett"].keys()) + list(settings_ml["l2_indcs_train_sett"])
            print("get klines")
            data = await g_data_x(data=data, indcs_list=indcs_list)
            print("get x")
            data["train"] = await g_y_train(data=data)
            print("get train")
            data["test"] = await g_y_test(data=data, indcs_list=indcs_list)
            print("get test")
            data = await g_data_backtest(data=data)
            await asyncio.gather(
                s_data_dump(data=data, name=f"{symbol}"),
                
                # trace balance
                s_data_dump(
                    data=await g_data_trace(
                        y=data["BT/ balance"], 
                        name_trace=symbol,
                        color_rgb=tuple([randint(0, 255)] * 3)
                    ), 
                    name=f"{symbol}", 
                    dir="data_pack/trace_balance"
                ),
                
                # test labels
                s_data_dump(
                    data=await g_data_mark(
                        y=data["close"],
                        labels=data["test"],
                        classes={
                            1: {"color_rgb": (0, 255, 85), "name": "buy"},
                            -1: {"color_rgb": (255, 0, 0), "name": "sell"},
                        }
                    ), 
                    name=f"{symbol}", 
                    dir="data_pack/mark",
                ),
                
                # trace close
                s_data_dump(
                    data=await g_data_trace(
                        y=data["close"], 
                        name_trace=symbol,
                        color_rgb=tuple([randint(0, 255)] * 3)
                    ),
                    name=symbol,
                    dir="data_pack/trace"
                )
            )
    
    print(g_report(data=await g_data_loads(dir="data_pack/data_backtest")))
    g_report_plotly(traces=await g_data_loads(dir="data_pack/trace_balance"))
    # g_report_plotly(traces=await g_data_loads(dir="data_pack/trace"), markers=await g_data_loads(
    #     dir="data_pack/mark"
    # ))

if __name__ == "__main__":
    asyncio.run(main())