from __future__ import annotations

from typing import Any

from dash import html


def build_router_banner(router_debug: dict[str, Any] | None) -> html.Div | None:
    """Shows the evidence_router's decision for the current query: which
    modality it would pick as the single best answer, and why — distinct
    from the ranked-results list below, which shows all candidates."""
    if not router_debug:
        return None

    route = router_debug.get("chosen_route", "")
    reason = router_debug.get("reason", "")
    second_choice = router_debug.get("router_second_choice", "")
    confidence_percent = router_debug.get("router_confidence_percent")
    visual_score = router_debug.get("top_visual_score")
    transcript_score = router_debug.get("top_transcript_score")

    badge_class = "router-pill router-pill-visual" if route == "visual" else "router-pill router-pill-transcript"
    confidence_label = (
        f"{confidence_percent:.0f}% confidence over {second_choice}"
        if confidence_percent is not None and second_choice
        else ""
    )

    return html.Div(
        className="router-banner",
        children=[
            html.Span(f"Router picked: {route or 'n/a'}", className=badge_class),
            html.Span(confidence_label, className="router-confidence") if confidence_label else None,
            html.Span(
                f"raw visual={_fmt(visual_score)} · raw transcript={_fmt(transcript_score)}",
                className="router-scores",
            ),
        ],
    )


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"
