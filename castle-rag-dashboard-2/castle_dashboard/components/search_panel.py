from dash import dcc, html

from castle_dashboard.services.dashboard_service import dashboard_service

MODALITY_OPTIONS = [
    {"label": "Transcript (speech → text search)", "value": "transcript"},
    {"label": "Visual (SigLIP image similarity)", "value": "visual"},
]


def build_search_panel() -> html.Aside:
    viewpoints = [{"label": "All viewpoints", "value": "All"}] + [
        {"label": v, "value": v} for v in dashboard_service.get_available_viewpoints()
    ]
    return html.Aside(
        className="sidebar",
        children=[
            html.Div(
                className="brand-block",
                children=[
                    html.Div("CASTLE", className="brand-mark"),
                    html.Div(
                        [
                            html.H1("RAG Video Search", className="brand-title"),
                            html.P("FAISS · SigLIP2 · GroundingDINO", className="brand-subtitle"),
                        ]
                    ),
                ],
            ),
            html.Section(
                className="panel compact-panel",
                children=[
                    html.H2("Search"),
                    html.Label("Natural-language query", htmlFor="query-input"),
                    dcc.Textarea(
                        id="query-input",
                        value="",
                        placeholder="Describe what you want to find…",
                        className="query-input",
                    ),
                    html.Label("Modalities"),
                    dcc.Checklist(
                        id="modality-filter",
                        options=MODALITY_OPTIONS,
                        value=["transcript", "visual"],
                        className="checklist",
                        inputClassName="checklist-input",
                    ),
                    html.Label("Confidence range"),
                    dcc.RangeSlider(
                        id="score-filter",
                        min=0,
                        max=1,
                        step=0.01,
                        value=[0.0, 1.0],
                        marks={0: "0%", 0.25: "25%", 0.5: "50%", 0.75: "75%", 1: "100%"},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Label("Viewpoint"),
                    dcc.Dropdown(
                        id="viewpoint-filter",
                        options=viewpoints,
                        value="All",
                        clearable=False,
                    ),
                    html.Button(
                        "Search",
                        id="search-button",
                        n_clicks=0,
                        className="primary-button",
                    ),
                    dcc.Loading(
                        type="dot",
                        children=html.Div(id="search-error", className="search-error"),
                    ),
                    dcc.Loading(
                        type="dot",
                        color="#2563eb",
                        children=html.Div(
                            id="keyword-suggestions",
                            className="keyword-suggestions",
                        ),
                    ),
                    html.Div(
                        className="refine-block",
                        children=[
                            html.Button(
                                "Refine with feedback",
                                id="refine-button",
                                n_clicks=0,
                                className="secondary-button refine-button",
                                disabled=True,
                            ),
                            html.Span(
                                "",
                                id="feedback-count",
                                className="feedback-count",
                            ),
                        ],
                    ),
                ],
            ),
            html.Section(
                id="evaluation-panel",
                className="panel compact-panel evaluation-panel",
                style={"display": "none"},
                children=[
                    html.Div(
                        className="evaluation-heading",
                        children=[
                            html.H2("Expert evaluation"),
                            html.Span("Optional logger", className="evaluation-badge"),
                        ],
                    ),
                    html.Label("Task prompt", htmlFor="evaluation-task-prompt"),
                    dcc.Textarea(
                        id="evaluation-task-prompt",
                        value="",
                        placeholder="Paste one evaluation task prompt…",
                        className="query-input evaluation-task-input",
                    ),
                    html.Button(
                        "Start evaluation task",
                        id="evaluation-toggle-button",
                        n_clicks=0,
                        className="evaluation-start-button",
                    ),
                    html.Div(
                        "Evaluation mode is ready.",
                        id="evaluation-status",
                        className="evaluation-status",
                    ),
                ],
            ),
        ],
    )
