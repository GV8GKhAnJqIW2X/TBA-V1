import plotly.graph_objects as go
from itertools import count
import numpy as np
from random import randint

def g_print_load(
    full_load,
    divider=1000, 
):
    counter = count(start=1)

    def dcrtr(func):
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            current_count = next(counter)
            if current_count % divider == 0:
                print(f"load {func.__name__} - {round(current_count / full_load * 100, 2)}%")
            return res
        return wrapper
    return dcrtr

def g_report(arrs_balance, arr_symbols):
    arr_balance = [arr.iloc[-1] for arr in arrs_balance]
    print(
        f"\nmean balance: {np.mean(arr_balance)}\n"
        f"min_balance: {np.min(arr_balance)}, symbol: {arr_symbols[np.argmin(arr_balance)]}\n"
        f"max_balance: {np.max(arr_balance)}, symbol: {arr_symbols[np.argmax(arr_balance)]}\n"
    )

def g_visualize(
    markers=(),
    markers_settings=(),
    traces=(),
    title="",
):
    fig = go.Figure()
    for traces_sett in traces:
        fig.add_trace(go.Scatter(
            x=traces_sett["x"],
            y=traces_sett["y"],
            mode='lines',
            name=traces_sett["name"],
            line=dict(
                color=f"rgb{
                    tuple(traces_sett["line"]["color_random_defolt"] + [randint(58, 159) for _ in range(3 - len(traces_sett["line"]["color_random_defolt"]))]) 
                    if traces_sett["line"]["color"] == "random" 
                    else traces_sett["line"]["color"]}"
            ),
        ))

    for mark, setting in zip(markers, markers_settings):
        for el in setting:
            if_key = mark == el["class_"]
            fig.add_trace(go.Scatter(
                x=mark[if_key].index,
                y=traces[el["trace_index"]]["y"][if_key],
                mode='markers',
                marker=dict(size=10, color=el["color"]),
                name=el["name"]
            ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        plot_bgcolor='rgb(26,25,24)',
        paper_bgcolor='rgb(58,52,53)',
        font=dict(color='lightgray'),
    )

    fig.update_xaxes(
        title_text='X',
        title_font=dict(color='lightgray'),
        showgrid=True,
        gridcolor="rgb(50, 50, 50)",   # Цвет сетки (темно-серый)
        gridwidth=1              # Ширина линий сетки
    )

    fig.update_yaxes(
        title_text='Y',
        title_font=dict(color='lightgray'),
        showgrid=True,
        gridcolor="rgb(50, 50, 50)",   # Цвет сетки (темно-серый)
        gridwidth=1              # Ширина линий сетки
    )
    fig.show()
