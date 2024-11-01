from onion.l1.g_settings import settings_ml, settings_bt
from onion.l2.g_data_api_global import g_symbols_f, g_klines
from onion.l2.g_filemanager_pickle import g_files_list, s_data_dump, g_data_load, g_data_loads
from onion.l3.g_data_test import g_data_test
from onion.l4.g_data_backtest import g_data_backtest
from onion.l5.g_data_pack import g_data_mark, g_data_trace
from onion.l6.g_report_gui import g_report_plotly
from onion.l6.g_report import g_report

import asyncio
from random import randint
from pprint import pprint

async def main():
    for symbol in\
        await g_files_list():
        # await g_symbols_f(klines_all_num=settings_ml["klines_all_num"]):
        print(symbol)
        
        data = await g_data_load(name=f"data_pack/data_backtest/{symbol}")

        # data_klines = await g_klines(symbol=symbol, qty=settings_ml["klines_all_num"])
        data_klines = data[["close", "low", "high"]].copy()
        if data_klines is not None:
            data = await g_data_test(
                data=data_klines,
                ml_sett=settings_ml,
                indcs_list=list(settings_ml["l1_indcs_train_sett"].keys()) + list(settings_ml["l2_indcs_train_sett"])
            )
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
                )
            )
    
    print(g_report(data=await g_data_loads(dir="data_pack/data_backtest")))
    # g_report_plotly(traces=await g_data_loads(dir="data_pack/trace_balance"))

if __name__ == "__main__":
    asyncio.run(main())