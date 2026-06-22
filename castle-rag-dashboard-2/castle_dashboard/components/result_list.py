from dash import html

from castle_dashboard.models.schemas import RetrievalResult
from castle_dashboard.utils.formatting import format_percent


def _combined_ts(result: RetrievalResult) -> str:
    ts = result.timestamp
    mm_ss = ts[3:] if len(ts) == 8 and ts[2] == ":" else ts
    cam = str(result.camera).strip()
    return f"{cam}:{mm_ss}" if cam else ts


def build_result_list(
    results: list[RetrievalResult],
    active_id: str | None,
    feedback: dict | None = None,
) -> html.Div:
    if not results:
        return html.Div("No results match the current filters.", className="empty-state")
    liked = set((feedback or {}).get("liked", []))
    disliked = set((feedback or {}).get("disliked", []))
    return html.Div(
        [_build_result_card(r, active_id, liked, disliked) for r in results],
        className="result-list",
    )


def _build_result_card(
    result: RetrievalResult,
    active_id: str | None,
    liked: set[str],
    disliked: set[str],
) -> html.Div:
    card_cls = "result-card active" if result.id == active_id else "result-card"
    if result.is_routed_choice:
        card_cls += " routed-choice"

    up_cls = "thumb-btn thumb-up thumb-active" if result.id in liked else "thumb-btn thumb-up"
    dn_cls = "thumb-btn thumb-down thumb-active" if result.id in disliked else "thumb-btn thumb-down"

    return html.Div(
        className="result-card-wrap",
        children=[
            html.Button(
                id={"type": "result-card", "index": result.id},
                n_clicks=0,
                className=card_cls,
                children=[
                    html.Div(
                        className="result-card-header",
                        children=[
                            html.Span(f"#{result.rank}", className="rank-pill"),
                            html.Span("Router pick", className="router-choice-pill") if result.is_routed_choice else None,
                            html.Span(format_percent(result.score), className="score-pill"),
                        ],
                    ),
                    html.H3(result.title),
                    html.Div(
                        className="author-row",
                        children=[
                            html.Span("Author: ", className="author-label"),
                            html.Span(result.viewpoint, className="author-name"),
                        ],
                    ),
                    html.P(result.caption),
                    html.Div(
                        className="meta-row",
                        children=[
                            html.Span(_combined_ts(result)),
                            html.Span(result.modality.title()),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="thumb-row",
                children=[
                    html.Button(
                        "▲",
                        id={"type": "thumb-up", "index": result.id},
                        n_clicks=0,
                        className=up_cls,
                        title="Mark as relevant",
                    ),
                    html.Button(
                        "▼",
                        id={"type": "thumb-down", "index": result.id},
                        n_clicks=0,
                        className=dn_cls,
                        title="Mark as not relevant",
                    ),
                ],
            ),
        ],
    )
