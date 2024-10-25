from onion.app.domain.g_settings_bt import settings_bt, settings_ml
from onion.app.domain.func_session_global import g_symbols_f, g_klines
from onion.app.infrastructure.data_processing import (
    g_klines_splitting, 
    s_df_dump, 
    g_df_load,
    g_files_list,
)
from onion.app.infrastructure.g_df_pack import g_df_pack
from onion.app.infrastructure.g_df_test import g_df_test
from onion.visual.visual import g_visualize, g_report_balance

import numpy as np
import asyncio

async def main():
    for symbol in (await g_symbols_f(settings_ml["klines_all"])):
        print(symbol)
        data = await g_df_test(
            data=await g_df_pack(
                data=g_klines_splitting(await g_klines(
                    symbol=symbol,
                    float_=True,
                    qty=settings_ml["klines_all"],
                )),
                settings_ml=settings_ml,
            ),  
            settings_bt=settings_bt,
            settings_ml=settings_ml,
        )
        if data is not None:
            s_df_dump(
                data=dict(
                    x=np.arange(settings_ml["klines_train_used"], settings_ml["klines_all"]),
                    y=data["BT/ balance"].dropna(),
                    name=symbol,
                    line=dict(color="random", color_random_defolt=[228]),
                ),
                name=symbol,
                dir="traces_pack",
            )
            s_df_dump(data=data, name=symbol)

    g_visualize(traces=[g_df_load(name=name, dir="traces_pack") for name in g_files_list(dir="traces_pack")])
    print(g_report_balance(files_list=g_files_list(), load_func=g_df_load))

if __name__ == "__main__":
    asyncio.run(main())

# покрыть все тестами и обработчиками ошибок 