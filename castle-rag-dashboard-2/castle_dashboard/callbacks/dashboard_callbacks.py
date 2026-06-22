from __future__ import annotations

import traceback
from typing import Any

from dash import ALL, Dash, Input, Output, State, callback_context, html, no_update

from castle_dashboard.components.evidence_viewer import build_evidence_viewer
from castle_dashboard.components.metrics import build_metrics
from castle_dashboard.components.result_list import build_result_list
from castle_dashboard.components.router_banner import build_router_banner
from castle_dashboard.models.schemas import SearchFilters
from castle_dashboard.services.dashboard_service import dashboard_service
from castle_dashboard.utils.figures import (
    make_modality_figure,
    make_score_figure,
    make_timeline_figure,
)


def register_callbacks(app: Dash) -> None:

    # ── 1. Run search on button click ─────────────────────────────────────────
    @app.callback(
        Output("filtered-result-ids", "data"),
        Output("query-embedding-store", "data"),
        Output("feedback-store", "data"),
        Output("search-error", "children"),
        Input("search-button", "n_clicks"),
        State("query-input", "value"),
        State("modality-filter", "value"),
        State("viewpoint-filter", "value"),
        State("score-filter", "value"),
        prevent_initial_call=True,
    )
    def run_search(n_clicks, query, modalities, viewpoint, score_range):
        print("[dashboard] search callback fired", n_clicks, query, flush=True)
        try:
            from castle_dashboard import pipeline as _pl
            filters = SearchFilters(
                query=query or "",
                modalities=tuple(modalities or ["transcript", "visual"]),
                viewpoint=viewpoint or "All",
                min_score=float(score_range[0]) if score_range else 0.0,
                max_score=float(score_range[1]) if score_range else 1.0,
                top_k=20,
            )
            results = dashboard_service.search(filters)
            embedding = _pl.get_query_embedding(query or "") if (query or "").strip() else None
            print(f"[dashboard] search callback returning {len(results)} results", flush=True)
            return [r.id for r in results], embedding, {"liked": [], "disliked": []}, ""
        except Exception as exc:
            print("[dashboard] search callback failed", flush=True)
            traceback.print_exc()
            message = f"Search failed: {type(exc).__name__}: {exc}"
            return [], None, {"liked": [], "disliked": []}, message

    # ── 1a. Generate keyword chips after results render ──────────────────────
    @app.callback(
        Output("keyword-suggestions", "children"),
        Input("filtered-result-ids", "data"),
        State("query-input", "value"),
        prevent_initial_call=True,
    )
    def update_keyword_suggestions(result_ids, query):
        if not (query or "").strip():
            return html.Div("Enter a query to get recommended keywords.", className="keyword-empty")
        if not result_ids:
            return html.Div("No keyword suggestions available.", className="keyword-empty")
        try:
            payload = dashboard_service.compute_keyword_suggestions()
            return build_keyword_suggestions(payload)
        except Exception:
            print("[dashboard] keyword suggestion callback failed", flush=True)
            traceback.print_exc()
            return html.Div("Keyword suggestions unavailable; see server log.", className="keyword-empty")

    # ── 1b. Add a recommended keyword to the query box ───────────────────────
    @app.callback(
        Output("query-input", "value"),
        Input({"type": "keyword-chip", "term": ALL}, "n_clicks"),
        State("query-input", "value"),
        prevent_initial_call=True,
    )
    def append_keyword_to_query(_clicks, current_query):
        trigger = callback_context.triggered_id
        if not isinstance(trigger, dict) or trigger.get("type") != "keyword-chip":
            return no_update
        term = str(trigger.get("term", "")).strip()
        if not term:
            return no_update
        current = (current_query or "").strip()
        if term.lower() in current.lower():
            return current_query
        return f"{current} {term}".strip()

    # ── 2. Select result from card click or chart click ───────────────────────
    @app.callback(
        Output("selected-result-id", "data"),
        Input({"type": "result-card", "index": ALL}, "n_clicks"),
        Input("score-chart", "clickData"),
        Input("timeline-chart", "clickData"),
        State("selected-result-id", "data"),
        prevent_initial_call=True,
    )
    def select_result(_card_clicks, score_click, timeline_click, current_id):
        trigger = callback_context.triggered_id
        if isinstance(trigger, dict) and trigger.get("type") == "result-card":
            return trigger.get("index")
        click_data = {
            "score-chart": score_click,
            "timeline-chart": timeline_click,
        }.get(trigger)
        return _extract_result_id(click_data) or current_id or no_update

    # ── 3. Toggle feedback (thumbs up / down per result card) ─────────────────
    @app.callback(
        Output("feedback-store", "data", allow_duplicate=True),
        Input({"type": "thumb-up", "index": ALL}, "n_clicks"),
        Input({"type": "thumb-down", "index": ALL}, "n_clicks"),
        State("feedback-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_feedback(up_clicks, down_clicks, feedback):
        # Guard against spurious fires when result cards are re-created (n_clicks=0)
        all_clicks = list(up_clicks or []) + list(down_clicks or [])
        if not any(c for c in all_clicks if c):
            return no_update

        trigger = callback_context.triggered_id
        if not trigger or not isinstance(trigger, dict):
            return no_update
        if trigger.get("type") not in ("thumb-up", "thumb-down"):
            return no_update

        result_id = trigger["index"]
        direction = trigger["type"]
        feedback = dict(feedback or {"liked": [], "disliked": []})
        liked = list(feedback.get("liked", []))
        disliked = list(feedback.get("disliked", []))

        if direction == "thumb-up":
            if result_id in liked:
                liked.remove(result_id)      # toggle off
            else:
                liked.append(result_id)
                if result_id in disliked:
                    disliked.remove(result_id)
        else:
            if result_id in disliked:
                disliked.remove(result_id)   # toggle off
            else:
                disliked.append(result_id)
                if result_id in liked:
                    liked.remove(result_id)

        return {"liked": liked, "disliked": disliked}

    # ── 4. Enable / disable Refine button + update feedback count label ────────
    @app.callback(
        Output("refine-button", "disabled"),
        Output("feedback-count", "children"),
        Input("feedback-store", "data"),
        Input("query-embedding-store", "data"),
    )
    def update_refine_controls(feedback, embedding):
        liked = (feedback or {}).get("liked", [])
        disliked = (feedback or {}).get("disliked", [])
        n_liked, n_disliked = len(liked), len(disliked)
        has_feedback = (n_liked + n_disliked) > 0
        disabled = not (has_feedback and embedding)
        if not has_feedback:
            label = ""
        else:
            parts = []
            if n_liked:
                parts.append(f"▲ {n_liked}")
            if n_disliked:
                parts.append(f"▼ {n_disliked}")
            label = "  ".join(parts)
        return disabled, label

    # ── 5. Refine search via Rocchio ───────────────────────────────────────────
    @app.callback(
        Output("filtered-result-ids", "data", allow_duplicate=True),
        Output("feedback-store", "data", allow_duplicate=True),
        Input("refine-button", "n_clicks"),
        State("query-embedding-store", "data"),
        State("feedback-store", "data"),
        State("modality-filter", "value"),
        State("viewpoint-filter", "value"),
        State("score-filter", "value"),
        State("query-input", "value"),
        prevent_initial_call=True,
    )
    def refine_search(n_clicks, embedding, feedback, modalities, viewpoint, score_range, query):
        if not n_clicks or not embedding:
            return no_update, no_update

        from castle_dashboard import pipeline as _pl

        feedback = feedback or {"liked": [], "disliked": []}
        liked_ids = feedback.get("liked", [])
        disliked_ids = feedback.get("disliked", [])

        # Translate dashboard result IDs → FAISS row indices
        liked_rows = [
            r.faiss_row_id for rid in liked_ids
            if (r := dashboard_service.get_result(rid)) and r.faiss_row_id is not None
        ]
        disliked_rows = [
            r.faiss_row_id for rid in disliked_ids
            if (r := dashboard_service.get_result(rid)) and r.faiss_row_id is not None
        ]

        refined_embedding = _pl.rocchio_refine(embedding, liked_rows, disliked_rows)

        filters = SearchFilters(
            query=query or "",
            modalities=tuple(modalities or ["transcript", "visual"]),
            viewpoint=viewpoint or "All",
            min_score=float(score_range[0]) if score_range else 0.0,
            max_score=float(score_range[1]) if score_range else 1.0,
            top_k=20,
        )
        refined_visual = _pl.retrieve_visual_from_embedding(refined_embedding, top_k=filters.top_k)
        results = dashboard_service.refine_visual(filters, refined_visual)

        print(
            f"[feedback] Rocchio refine — liked={len(liked_rows)} disliked={len(disliked_rows)}"
            f" → {len(results)} results"
        )
        return [r.id for r in results], {"liked": [], "disliked": []}

    # ── 6. Run GroundingDINO on demand ────────────────────────────────────────
    @app.callback(
        Output("grounding-result", "data"),
        Input("localize-button", "n_clicks"),
        State("selected-result-id", "data"),
        State("query-input", "value"),
        prevent_initial_call=True,
    )
    def run_grounding(n_clicks, selected_id, query):
        if not n_clicks or not selected_id or not query:
            return no_update
        result = dashboard_service.get_result(selected_id)
        if result is None or not result.keyframe_path:
            return no_update
        chunk = {
            "keyframe_path": result.keyframe_path,
            "visual_caption": result.caption,
            "source_name": result.source_name,
            "video_id": result.video_id,
            "start_sec": result.start_sec,
            "end_sec": result.end_sec,
            "score": result.score,
        }
        from castle_dashboard import pipeline as _pl
        return _pl.run_grounding(query, chunk)

    # ── 6b. Run transcript heatmap + extractive QA span on demand ─────────────
    @app.callback(
        Output("transcript-evidence", "data"),
        Input("highlight-answer-button", "n_clicks"),
        State("selected-result-id", "data"),
        prevent_initial_call=True,
    )
    def run_transcript_evidence(n_clicks, selected_id):
        if not n_clicks or not selected_id:
            return no_update
        enriched = dashboard_service.compute_transcript_evidence(selected_id)
        if enriched is None:
            return no_update
        enriched["_result_id"] = selected_id
        return enriched

    # ── 7a. Result list — feedback-only update (thumb clicks) ─────────────────
    # Separate from update_dashboard so thumb clicks don't re-render charts /
    # evidence panel / metrics — only the result list.
    @app.callback(
        Output("ranked-results", "children", allow_duplicate=True),
        Input("feedback-store", "data"),
        State("filtered-result-ids", "data"),
        State("selected-result-id", "data"),
        prevent_initial_call=True,
    )
    def update_result_feedback(feedback, result_ids, selected_id):
        results = dashboard_service.get_results_by_ids(result_ids or [])
        ids_set = {r.id for r in results}
        effective_id = selected_id if selected_id in ids_set else _first_id(results)
        return build_result_list(results, effective_id, feedback)

    # ── 7b. Update all panels (search / selection / grounding changes) ────────
    @app.callback(
        Output("ranked-results", "children"),
        Output("evidence-panel", "children"),
        Output("metrics-panel", "children"),
        Output("search-summary", "children"),
        Output("router-banner", "children"),
        Output("score-chart", "figure"),
        Output("timeline-chart", "figure"),
        Output("modality-chart", "figure"),
        Input("filtered-result-ids", "data"),
        Input("selected-result-id", "data"),
        Input("grounding-result", "data"),
        Input("transcript-evidence", "data"),
        State("feedback-store", "data"),
    )
    def update_dashboard(result_ids, selected_id, grounding_result, transcript_evidence, feedback):
        results = dashboard_service.get_results_by_ids(result_ids or [])

        ids_set = {r.id for r in results}
        effective_id = selected_id if selected_id in ids_set else _first_id(results)
        selected = dashboard_service.get_result(effective_id)

        active_grounding: dict[str, Any] | None = None
        if grounding_result and selected and selected.keyframe_path:
            if grounding_result.get("keyframe_path") == selected.keyframe_path:
                active_grounding = grounding_result

        active_transcript_evidence: dict[str, Any] | None = None
        if transcript_evidence and effective_id and transcript_evidence.get("_result_id") == effective_id:
            active_transcript_evidence = transcript_evidence

        transcript_ctx: list[dict[str, Any]] = []
        if selected and selected.source_name and selected.video_id:
            transcript_ctx = dashboard_service.get_transcript_context(
                selected.source_name,
                selected.video_id,
                selected.start_sec,
            )

        stats = dashboard_service.get_stats()
        summary = f"{len(results)} result{'s' if len(results) != 1 else ''}"

        return (
            build_result_list(results, effective_id, feedback),
            build_evidence_viewer(selected, active_grounding, transcript_ctx, active_transcript_evidence),
            build_metrics(stats, len(results)),
            summary,
            build_router_banner(dashboard_service.get_router_debug() if results else None),
            make_score_figure(results, effective_id),
            make_timeline_figure(results, effective_id),
            make_modality_figure(results),
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _first_id(results):
    return results[0].id if results else None


def _extract_result_id(click_data):
    if not click_data or not click_data.get("points"):
        return None
    return click_data["points"][0].get("customdata")


def build_keyword_suggestions(payload: dict[str, Any] | None):
    terms = (payload or {}).get("suggested_terms") or []
    if not terms:
        return html.Div("No recommended keywords yet.", className="keyword-empty")

    chips = []
    for index, item in enumerate(terms[:10], start=1):
        term = str(item.get("term", "")).strip()
        if not term:
            continue
        source = str(item.get("source", payload.get("source", "")))
        confidence = item.get("confidence")
        title_parts = []
        if item.get("reason"):
            title_parts.append(str(item["reason"]))
        if confidence is not None:
            try:
                title_parts.append(f"confidence {float(confidence) * 100:.0f}%")
            except (TypeError, ValueError):
                pass
        chips.append(
            html.Button(
                term,
                id={"type": "keyword-chip", "term": term},
                n_clicks=0,
                className="keyword-chip",
                title=" · ".join(title_parts) or source,
            )
        )

    if not chips:
        return html.Div("No recommended keywords yet.", className="keyword-empty")

    return html.Div(
        className="keyword-panel",
        children=[
            html.Div("Recommended keywords", className="keyword-title"),
            html.Div(chips, className="keyword-chip-row"),
        ],
    )
