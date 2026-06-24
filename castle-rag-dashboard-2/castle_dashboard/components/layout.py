from dash import dcc, html

from castle_dashboard.components.charts import chart_card
from castle_dashboard.components.search_panel import build_search_panel


def build_layout() -> html.Div:
    return html.Div(
        className="app-shell",
        children=[
            # Hidden stores
            dcc.Store(id="filtered-result-ids", data=[]),
            dcc.Store(id="selected-result-id", data=None),
            dcc.Store(id="grounding-result", data=None),
            dcc.Store(id="transcript-evidence", data=None),
            dcc.Store(id="feedback-store", data={"liked": [], "disliked": []}),
            dcc.Store(id="query-embedding-store", data=None),
            dcc.Store(id="evaluation-state-store", data={"enabled": False, "active": False}),
            dcc.Store(id="evaluation-event-sink", data=0),
            dcc.Interval(id="evaluation-state-poll", interval=2000, n_intervals=0),
            build_search_panel(),
            html.Main(
                className="dashboard-main",
                children=[
                    # Evidence router decision banner
                    html.Div(id="router-banner"),
                    # Main content: evidence (left) + results (right)
                    html.Div(
                        className="content-grid",
                        children=[
                            html.Section(
                                className="left-column",
                                children=[
                                    html.Div(id="evidence-panel"),
                                ],
                            ),
                            html.Section(
                                className="right-column panel",
                                children=[
                                    html.Div(
                                        className="section-heading",
                                        children=[
                                            html.H2("Ranked results"),
                                            html.Span(id="search-summary", className="muted-count"),
                                        ],
                                    ),
                                    dcc.Loading(
                                        id="results-loading",
                                        type="dot",
                                        children=html.Div(id="ranked-results"),
                                    ),
                                ],
                            ),
                        ],
                    ),
                    # Charts row
                    html.Div(
                        className="chart-grid",
                        children=[
                            chart_card("Retrieval scores", "score-chart"),
                            chart_card("Timeline by viewpoint", "timeline-chart"),
                            chart_card("Modality coverage", "modality-chart"),
                        ],
                    ),
                    # Metrics footer
                    html.Footer(
                        className="hero-panel",
                        children=[html.Div(id="metrics-panel")],
                    ),
                ],
            ),
        ],
    )
