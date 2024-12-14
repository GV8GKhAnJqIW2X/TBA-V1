from onion.l1.l2.g_structures import *
from project_exctentions.g_data_api_global import *
from project_exctentions.g_utils import *
from onion.l2.l3_signal.g_signals import *
from onion.l1.l1.g_settings_ import *
from onion.l2.l4_after_signal.g_backtests import *

import pickle
import plotly.graph_objects as go
from datetime import datetime
import json
import os

async def main():
    # init
    symbol = "AAVEUSDT"
    signals = np.empty(settings["BACKTEST"]["klines_test"])
    signals2 = np.empty(settings["BACKTEST"]["klines_test"])
    datetime_now = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')
    filter_values_array = np.full(settings["BACKTEST"]["klines_test"], np.nan)
    in_position = None
    price_pos = None
    signal_pos = None
    qty = None
    balance = None
    balances_array = np.full(settings["BACKTEST"]["klines_test"], np.nan)
    signals_all = np.full(settings["BACKTEST"]["klines_test"], np.nan)
    in_positions = np.full(settings["BACKTEST"]["klines_test"], False)
    qtys = np.full(settings["BACKTEST"]["klines_test"], np.nan)
    return_backtest_values = None


    # not ready_made_data
    src = g_src(**(await g_klines_split(await g_klines(
        symbol=symbol, 
        qty=(
            settings["max_window_features"] * 5
            + settings["ML"]["klines_train"] 
            + settings["BACKTEST"]["klines_test"]
        )
    ))))
    close_backtest = src["close"][-settings["BACKTEST"]["klines_test"]:]
    signals_raw_array = np.full(settings["BACKTEST"]["klines_test"], np.nan)
    x_features_series_array = np.full(settings["BACKTEST"]["klines_test"], {})
    filters_values_array = np.full(settings["BACKTEST"]["klines_test"], {})
    signal_return_value = None

    # # ready_made_data
    # with open("backtest_values/SUIUSDT_kt200000_2024.12.11_01.29.35.pickle", "rb") as f:
    #     ready_made_data = pickle.load(f)
    # close_backtest = ready_made_data["close"]
    # signals_raw_array = ready_made_data["signals_raw_array"]
    # filters_values_array = ready_made_data["filters_values_array"]
    # return_sighals_counter = 0
    # return_signal_raw = 0
    # return_signals_raw = [0, 0, 0, 0]

    # main
    for i in range(settings["BACKTEST"]["klines_test"]):
        signal, additional_return, signal_return_value = g_signal_A_distance_lorentzian_A_ANN_A_iter(
            **g_iloc(src, slice(
                i, 
                (
                    i 
                    + settings["max_window_features"] * 5 
                    + settings["ML"]["klines_train"] 
                    + 1
                )
            )),
            initialized_return_value=signal_return_value,
            additional_return_x_features_series=True,
            additional_return_signal_raw=True,
            additional_return_filters_values=True,
        )
        (
            x_features_series, 
            signal_raw, 
            filters_values,
        ) = additional_return
        signals[i] = signal
        x_features_series_array[i] = x_features_series
        signals_raw_array[i] = signal_raw
        filters_values_array[i] = filters_values

        # ready_made_data
        # price_last = close_backtest[i]
        # signal, return_sighals_counter, return_signal_raw, signal2 = g_signal_A_ready_made_data(
        #     signals_raw_array[i],
        #     signals_raw_array[i - 1] if i else 0,
        #     signals_raw_array[i - 4:i] if i > 4 else [0, 0, 0, 0],
        #     price_last,
        #     filters_values_array[i],
        #     settings["FILTERS"],
        #     return_sighals_counter=return_sighals_counter,
        #     return_signal_raw=return_signal_raw,
        #     signals_raw_l2=return_signals_raw,
        # )
        # signals[i] = signal
        # signals2[i] = signal2
        # signals_all[i] = signal
        # if in_position:
        #     signals_all[i] = signal2
        # return_signals_raw.append(return_signal_raw)
        # return_backtest_values = g_backtest_AS_balance_A_comparison_A_single_A_iter(
        #     return_backtest_values,
        #     signal,
        #     signal2,
        #     price_last,
        # )
        # (
        #     in_position,
        #     price_pos,
        #     signal_pos,
        #     qty,
        #     balance,
        # ) = return_backtest_values
        # balances_array[i] = balance
        # in_positions[i] = in_position
        # qtys[i] = qty
        # filter_values_array[i] = filters_values_array[i]["ADX"]
        
        print(f"\r{i + 1}/{settings["BACKTEST"]["klines_test"]} klines tested", end="")
    
    # plot_values_balances = g_plot_backtest_values(
    #     close_backtest,
    #     balances_array,
    #     in_positions,
    #     qtys,
    #     signals_all
    # )
    # fig_balance = go.Figure()
    # fig_balance.add_trace(go.Scatter(
    #     x=plot_values_balances["x_close"],
    #     y=plot_values_balances["y_close"],
    #     mode="lines",
    # ))
    # fig_balance.add_trace(go.Scatter(
    #     x=plot_values_balances["x_short"],
    #     y=plot_values_balances["y_short"],
    #     mode="markers",
    #     marker={
    #         "color": "red",
    #         "symbol": 'arrow-down',
    #         "line": dict(
    #             color='white',
    #             width=1         
    #         ),
    #         "size": 10,
    #     },
    # ))
    # fig_balance.add_trace(go.Scatter(
    #     x=plot_values_balances["x_long"],
    #     y=plot_values_balances["y_long"],
    #     mode="markers",
    #     marker={
    #         "color": "green",
    #         "symbol": 'arrow-up',
    #         "line": dict(
    #             color='white',
    #             width=1         
    #         ),
    #         "size": 10,
    #     },
    # ))
    # for x_dash, y_dash in zip(
    #     plot_values_balances["x_dash_lines"], 
    #     plot_values_balances["y_dash_lines"],
    # ):
    #     fig_balance.add_trace(go.Scatter(
    #         x=x_dash,
    #         y=y_dash,
    #         mode="lines",
    #         line=dict(dash='dash', color="orange"),
    #     ))
        
    # for annotation, x, y in zip(
    #     plot_values_balances["text_balance"],
    #     plot_values_balances["x_balance"],
    #     plot_values_balances["y_balance"],
    # ):
    #     fig_balance.add_annotation(
    #         x=x,
    #         y=y + y * 0.02,
    #         text=annotation,
    #         showarrow=False,
    #         font=dict(size=9, color="black"),  
    #         bgcolor="white",              
    #     )
    # for annotation, x, y in zip(
    #     plot_values_balances["text_qty"],
    #     plot_values_balances["x_qty"],
    #     plot_values_balances["y_qty"],
    # ):
    #     fig_balance.add_annotation(
    #         x=x,
    #         y=y + y * 0.02,
    #         text=annotation,
    #         showarrow=False,
    #         font=dict(size=9, color="black"),  
    #         bgcolor="white",              
    #     )
    
    # fig_balance.show()
    # fig_balance_2 = go.Figure()
    # fig_balance_2.add_trace(go.Scatter(
    #     x=np.arange(balances_array.size),
    #     y=balances_array,
    #     mode="lines",
    # ))
    # fig_balance_2.show()

    # os.mkdir(f'plotly_html/{symbol}_kt{settings['BACKTEST']['klines_test']}_{datetime_now}')

    # fig_balance.write_html(f"plotly_html/{symbol}_kt{settings['BACKTEST']['klines_test']}_{datetime_now}/BT.html")
    # fig_balance_2.write_html(f"plotly_html/{symbol}_kt{settings['BACKTEST']['klines_test']}_{datetime_now}/balance_trace.html")
    # with open(f"plotly_html/{symbol}_kt{settings['BACKTEST']['klines_test']}_{datetime_now}/settings.json", "w", encoding="utf-8") as f:
    #     json.dump(settings, f, indent=4)


    with open(f"backtest_values/{symbol}_kt{settings["BACKTEST"]["klines_test"]}_{datetime_now}.pickle", "wb") as f:
        pickle.dump(
            {
                "signals": signals,
                "close": close_backtest,
                "x_features_series_array": x_features_series_array,
                "signals_raw_array": signals_raw_array,
                "filters_values_array": filters_values_array,
                "src_all": src,
            },
            f,
        )
    
import asyncio
asyncio.run(main())