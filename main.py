from onion.app.domain.g_settings_bt import settings_bt, settings_ml
from onion.app.domain.func_session_global import g_symbols_f, g_klines
from onion.app.infrastructure.data_processing import (
    g_klines_splitting, 
    s_df_dump, 
    g_df_load,
)
from onion.app.infrastructure.g_df_pack import g_df_pack
from onion.app.infrastructure.g_df_test import g_df_test
from onion.visual.visual import g_visualize

import asyncio

async def main():
    traces = []

    for symbol in await g_symbols_f(settings_ml["klines_all"]):
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
        traces.append(dict(
            x=range(settings_ml["klines_train_used"], settings_ml["klines_all"]),
            y=data["BT/ balance"],
            name=symbol,
            line=dict(
                color="random", 
                theme="dark", 
                color_random_defolt=[228],
            )
        ))
        s_df_dump(data=data, name=symbol)

    g_visualize(traces=traces)
    # data_need = g_df_load(name="10000000AIDOGEUSDT")
    # g_visualize(
    #     traces=(dict(x=data_need.index, y=data_need["close"], name="coin", line=dict(color="random", color_random_defolt=[])),),
    #     markers=(data_need["predicted_label"],),
    #     markers_settings=(
    #         (
    #             dict(
    #                 class_=-1,
    #                 color="red",
    #                 name="sell",
    #                 trace_index=0
    #             ),
    #             dict(
    #                 class_=1,
    #                 color="green",
    #                 name="buy",
    #                 trace_index=0
    #             ),
    #         ),
    #     ),
    # )

if __name__ == "__main__":
    asyncio.run(main())
