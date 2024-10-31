import plotly.graph_objects as go

def g_report_plotly(
    markers=(),
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
            line=traces_sett["line"],
        ))

    for mark in markers:
        fig.add_trace(go.Scatter(
            x=mark["x"],
            y=mark["y"],
            mode='markers',
            marker=dict(size=10, color=mark["color"], name=mark["name"]),
        ))

    # ignore:s
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
