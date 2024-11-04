import plotly.graph_objects as go

def g_report_plotly(
    markers=(),
    traces=(),
    title="",
):
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(go.Scatter(
            x=trace["x"],
            y=trace["y"],
            mode='lines',
            name=trace["name"],
            line=trace["line"],
        ))

    for mark in markers:
        for class_ in mark:
            fig.add_trace(go.Scatter(
                x=class_["x"],
                y=class_["y"],
                mode='markers',
                marker=class_["marker"],
                name=class_["name"],
            ))

    # ignore
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
