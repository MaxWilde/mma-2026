from dash import dcc, html


def chart_card(title: str, graph_id: str, class_name: str = "panel") -> html.Section:
    return html.Section(
        className=class_name,
        children=[
            html.Div(className="section-heading", children=[html.H2(title)]),
            dcc.Graph(id=graph_id, config={"displayModeBar": False}, className="chart"),
        ],
    )
