from onion.l1.l2.g_structures import *
from project_exctentions.g_data_api_global import *
from project_exctentions.g_utils import *
from onion.l2.l3.g_signals import *

import pickle
import plotly.graph_objects as go
from datetime import datetime

async def main():
    # init
    symbol = "SUIUSDT"
    src = g_src(**(await g_klines_split(await g_klines(
        symbol=symbol, 
        qty=(
            settings["SIGNAL_GENERATION"]["max_window_features"] * 2
            + settings["SIGNAL_GENERATION"]["ML"]["klines_train"] 
            + settings["BACKTEST"]["klines_test"]
            + 1
        )
    ))))
    close_backtest = src["close"][-settings["BACKTEST"]["klines_test"]:]
    signals = np.empty(settings["BACKTEST"]["klines_test"])
    x_features_series_array = np.full((settings["BACKTEST"]["klines_test"], settings["SIGNAL_GENERATION"]["features_used_count"]), {})
    signals_raw_array = np.full(settings["BACKTEST"]["klines_test"], np.nan)
    filters_values_array = np.full(settings["BACKTEST"]["klines_test"], {})
    balances_array = np.full(settings["BACKTEST"]["klines_test"], np.nan)
    signal_return_value = None
    datetime_now = datetime.now().strftime('%Y.%m.%d_%H.%M.%S')

    # ready_made_data
    # with open("backtest_values/SUIUSDT_kt50000_2024.12.08_23.18.25.pickle", "rb") as f:
    #     ready_made_data = pickle.load(f)
    
    # close_backtest = ready_made_data["close"]
    # signals_raw_array = ready_made_data["signals_raw_array"]
    # filters_values_array = ready_made_data["filters_values_array"]

    # main
    for i in range(settings["BACKTEST"]["klines_test"]):
        signal, additional_return, signal_return_value = g_signal_A_distance_lorentzian_A_ANN_A_iter(
            **g_iloc(src, slice(
                i, 
                (
                    i 
                    + settings["SIGNAL_GENERATION"]["max_window_features"] * 2 
                    + settings["SIGNAL_GENERATION"]["ML"]["klines_train"] 
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

        # # ready_made_data
        # signals[i] = g_signal_A_ready_made_data(
        #     signals_raw_array[i],
        #     close_backtest[i],
        #     filters_values_array[i],
        #     settings["SIGNAL_GENERATION"]["filters_used"],
        # )
        
        print(f"\r{i + 1}/{settings["BACKTEST"]["klines_test"]} klines tested", end="")
    
    plot_values = g_plot_values(signals, close_backtest,)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_values["x_close"],
        y=plot_values["y_close"],
        mode="lines",
        name="close",
    ))
    fig.add_trace(go.Scatter(
        x=plot_values["x_long"],
        y=plot_values["y_long"],
        mode="markers",
        name="long",
        marker={"color": "green"},
    ))
    fig.add_trace(go.Scatter(
        x=plot_values["x_short"],
        y=plot_values["y_short"],
        mode="markers",
        name="short",
        marker={"color": "red"},
    ))
    fig.write_html(f"plotly_html/{symbol}_kt{settings["BACKTEST"]["klines_test"]}_{datetime_now}.html")
    fig.show()

    with open(f"backtest_values/{symbol}_kt{settings["BACKTEST"]["klines_test"]}_{datetime_now}.pickle", "wb") as f:
        pickle.dump(
            {
                "signals":signals,
                "close": close_backtest,
                "x_features_series_array": x_features_series_array,
                "signals_raw_array": signals_raw_array,
                "filters_values_array": filters_values_array,
            },
            f,
        )
    
import asyncio
asyncio.run(main())