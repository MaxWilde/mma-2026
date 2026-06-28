from __future__ import annotations

import traceback
import re
import time
from typing import Any

from dash import ALL, Dash, Input, Output, State, callback_context, html, no_update

from castle_dashboard.components.evidence_viewer import build_evidence_viewer
from castle_dashboard.components.metrics import build_metrics
from castle_dashboard.components.result_list import build_result_list
from castle_dashboard.components.router_banner import build_router_banner
from castle_dashboard.models.schemas import DashboardStats, SearchFilters
from castle_dashboard.services.dashboard_service import dashboard_service
from castle_dashboard.services.evaluation_logger import evaluation_logger
from castle_dashboard.services.startup_manager import startup_manager
from castle_dashboard.utils.figures import (
    make_modality_figure,
    make_score_figure,
    make_timeline_figure,
)


def register_callbacks(app: Dash) -> None:

    # Bring newly selected or newly enriched evidence into view. This is
    # client-side so scrolling is immediate and does not add a server request.
    app.clientside_callback(
        """
        function(scrollRequest) {
            if (!scrollRequest) {
                return window.dash_clientside.no_update;
            }
            window.setTimeout(function() {
                const panel = document.getElementById("evidence-panel");
                if (panel) {
                    panel.scrollIntoView({behavior: "smooth", block: "start"});
                }
            }, 120);
            return Date.now();
        }
        """,
        Output("evidence-scroll-sink", "data"),
        Input("evidence-scroll-request", "data"),
        prevent_initial_call=True,
    )

    # ── Startup progress and readiness ───────────────────────────────────────
    @app.callback(
        Output("startup-stage", "children"),
        Output("startup-percent", "children"),
        Output("startup-progress-fill", "style"),
        Output("startup-detail", "children"),
        Output("startup-status-panel", "className"),
        Output("search-button", "disabled"),
        Output("viewpoint-filter", "options"),
        Input("startup-state-poll", "n_intervals"),
    )
    def update_startup_progress(_n_intervals):
        state = startup_manager.snapshot()
        percent = int(state.get("percent", 0))
        status = state.get("status", "not_started")
        detail = state.get("detail") or ""
        elapsed = float(state.get("elapsed_seconds", 0.0))
        if status == "loading":
            detail = f"{detail} · elapsed {elapsed:.1f}s"
        elif status == "ready":
            grounding = state.get("grounding_status", "loading")
            detail = f"{detail} · GroundingDINO: {grounding}"
        panel_class = f"startup-status-panel startup-{status}"
        viewpoints = [
            {"label": viewpoint if viewpoint != "All" else "All viewpoints", "value": viewpoint}
            for viewpoint in state.get("viewpoints", ["All"])
        ]
        return (
            state.get("stage", "Starting dashboard"),
            f"{percent}%",
            {"width": f"{percent}%"},
            detail,
            panel_class,
            not bool(state.get("ready")),
            viewpoints,
        )

    # ── 0. Optional externally-enabled expert evaluation controls ────────────
    @app.callback(
        Output("evaluation-panel", "style"),
        Output("evaluation-toggle-button", "children"),
        Output("evaluation-toggle-button", "className"),
        Output("evaluation-task-prompt", "disabled"),
        Output("evaluation-status", "children"),
        Output("evaluation-state-store", "data"),
        Input("evaluation-state-poll", "n_intervals"),
    )
    def refresh_evaluation_controls(_n_intervals):
        state = evaluation_logger.state()
        enabled = bool(state.get("enabled"))
        active = state.get("active_task") or None
        if not enabled:
            return (
                {"display": "none"},
                "Start evaluation task",
                "evaluation-start-button",
                False,
                "Evaluation mode is disabled.",
                {"enabled": False, "active": False},
            )
        if active:
            return (
                {"display": "block"},
                "Stop evaluation task",
                "evaluation-stop-button",
                True,
                f"Recording {active.get('task_id', 'task')} · {active.get('prompt', '')}",
                {
                    "enabled": True,
                    "active": True,
                    "task_id": active.get("task_id"),
                    "run_id": active.get("run_id"),
                },
            )
        return (
            {"display": "block"},
            "Start evaluation task",
            "evaluation-start-button",
            False,
            f"Evaluation session {state.get('session_id', '')} is ready.",
            {"enabled": True, "active": False},
        )

    @app.callback(
        Output("query-input", "value", allow_duplicate=True),
        Output("filtered-result-ids", "data", allow_duplicate=True),
        Output("selected-result-id", "data", allow_duplicate=True),
        Output("grounding-result", "data", allow_duplicate=True),
        Output("transcript-evidence", "data", allow_duplicate=True),
        Output("feedback-store", "data", allow_duplicate=True),
        Output("modality-filter", "value"),
        Output("score-filter", "value"),
        Output("viewpoint-filter", "value"),
        Output("evaluation-status", "children", allow_duplicate=True),
        Input("evaluation-toggle-button", "n_clicks"),
        State("evaluation-task-prompt", "value"),
        State("query-input", "value"),
        State("modality-filter", "value"),
        State("viewpoint-filter", "value"),
        State("score-filter", "value"),
        State("selected-result-id", "data"),
        State("filtered-result-ids", "data"),
        prevent_initial_call=True,
    )
    def toggle_evaluation_task(
        n_clicks,
        task_prompt,
        current_query,
        modalities,
        viewpoint,
        score_range,
        selected_id,
        result_ids,
    ):
        if not n_clicks:
            return (no_update,) * 10
        try:
            state = evaluation_logger.state()
            if state.get("active_task"):
                selected = dashboard_service.get_result(selected_id)
                path = evaluation_logger.stop_task(
                    {
                        "query_text": current_query or "",
                        "modalities": modalities or [],
                        "viewpoint": viewpoint or "All",
                        "score_range": score_range or [0.0, 1.0],
                        "selected_result": _result_snapshot(selected),
                        "visible_result_ids": result_ids or [],
                    }
                )
                return (
                    "",
                    [],
                    None,
                    None,
                    None,
                    {"liked": [], "disliked": []},
                    ["transcript", "visual"],
                    [0.0, 1.0],
                    "All",
                    f"Task saved to {path}. Complete its expert_annotation section.",
                )

            active = evaluation_logger.start_task(task_prompt or "")
            return (
                task_prompt or "",
                [],
                None,
                None,
                None,
                {"liked": [], "disliked": []},
                no_update,
                no_update,
                no_update,
                f"Recording {active['task_id']}. Submit the prompt when ready.",
            )
        except Exception as exc:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                f"Evaluation control failed: {type(exc).__name__}: {exc}",
            )

    # ── 1. Run search on button click ─────────────────────────────────────────
    @app.callback(
        Output("filtered-result-ids", "data"),
        Output("query-embedding-store", "data"),
        Output("feedback-store", "data"),
        Output("search-error", "children"),
        Output("selected-result-id", "data", allow_duplicate=True),
        Output("grounding-result", "data", allow_duplicate=True),
        Output("transcript-evidence", "data", allow_duplicate=True),
        Output("selected-keywords-store", "data"),
        Input("search-button", "n_clicks"),
        State("query-input", "value"),
        State("modality-filter", "value"),
        State("viewpoint-filter", "value"),
        State("score-filter", "value"),
        prevent_initial_call=True,
    )
    def run_search(n_clicks, query, modalities, viewpoint, score_range):
        print("[dashboard] search callback fired", n_clicks, query, flush=True)
        evaluation_logger.log_event(
            "search_submitted",
            {
                "query": query or "",
                "modalities": modalities or [],
                "viewpoint": viewpoint or "All",
                "score_range": score_range or [0.0, 1.0],
            },
        )
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
            evaluation_logger.log_event(
                "search_completed",
                {
                    "query": query or "",
                    "latency_ms": dashboard_service.get_stats().avg_latency_ms,
                    "router": dashboard_service.get_router_debug(),
                    "result_count": len(results),
                    "results": [_result_snapshot(result) for result in results],
                },
            )
            print(f"[dashboard] search callback returning {len(results)} results", flush=True)
            result_ids = [r.id for r in results]
            return (
                result_ids,
                embedding,
                {"liked": [], "disliked": []},
                "",
                result_ids[0] if result_ids else None,
                None,
                None,
                {"base_query": query or "", "terms": []},
            )
        except Exception as exc:
            evaluation_logger.log_event(
                "search_failed",
                {"query": query or "", "error": f"{type(exc).__name__}: {exc}"},
            )
            print("[dashboard] search callback failed", flush=True)
            traceback.print_exc()
            message = f"Search failed: {type(exc).__name__}: {exc}"
            return (
                [],
                None,
                {"liked": [], "disliked": []},
                message,
                None,
                None,
                None,
                {"base_query": query or "", "terms": []},
            )

    # ── 1a. Clear stale keyword chips the instant Search is clicked ──────────
    # Fires immediately on button click (before run_search returns), so the
    # old chips disappear rather than lingering through retrieval + keyword ranking.
    @app.callback(
        Output("keyword-suggestions", "children", allow_duplicate=True),
        Input("search-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_keyword_suggestions_on_search(_n_clicks):
        return html.Div("Computing keywords…", className="keyword-empty")

    # ── 1b. Generate keyword chips after results render ──────────────────────
    @app.callback(
        Output("keyword-suggestions", "children"),
        Input("filtered-result-ids", "data"),
        Input("selected-keywords-store", "data"),
        State("query-input", "value"),
        prevent_initial_call=True,
    )
    def update_keyword_suggestions(result_ids, selected_keywords, query):
        if not (query or "").strip():
            return html.Div("Enter a query to get recommended keywords.", className="keyword-empty")
        if not result_ids:
            return html.Div("No keyword suggestions available.", className="keyword-empty")
        try:
            payload = dashboard_service.compute_keyword_suggestions()
            return build_keyword_suggestions(payload, selected_keywords, query)
        except Exception:
            print("[dashboard] keyword suggestion callback failed", flush=True)
            traceback.print_exc()
            return html.Div("Keyword suggestions unavailable; see server log.", className="keyword-empty")

    # ── 1c. Add a recommended keyword to the query box ───────────────────────
    @app.callback(
        Output("query-input", "value"),
        Output("selected-keywords-store", "data", allow_duplicate=True),
        Input({"type": "keyword-chip", "term": ALL}, "n_clicks"),
        State("query-input", "value"),
        State("selected-keywords-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_keyword_in_query(_clicks, current_query, selected_keywords):
        # Guard against spurious fires when chip components are created/re-created.
        # Dash fires ALL-pattern callbacks on component creation even though n_clicks=0;
        # the triggered list value tells us the actual click count on the firing component.
        triggered = callback_context.triggered
        if not triggered or not triggered[0].get("value"):
            return no_update, no_update
        trigger = callback_context.triggered_id
        if not isinstance(trigger, dict) or trigger.get("type") != "keyword-chip":
            return no_update, no_update
        term = str(trigger.get("term", "")).strip()
        if not term:
            return no_update, no_update
        current = (current_query or "").strip()
        keyword_state = _normalise_keyword_state(selected_keywords, current)
        selected = list(keyword_state["terms"])
        base_query = keyword_state["base_query"]
        is_selected = term in selected
        if is_selected:
            selected = [item for item in selected if item != term]
            updated_query = _compose_keyword_query(base_query, selected)
            action = "removed"
        else:
            # Suggestions already present in the user's own query should not be
            # selectable. This guard also protects an older/stale chip.
            if _contains_phrase(current, term):
                return no_update, no_update
            selected.append(term)
            updated_query = _compose_keyword_query(base_query, selected)
            action = "added"
        evaluation_logger.log_event(
            "keyword_clicked",
            {
                "term": term,
                "action": action,
                "query_before": current,
                "query_after": updated_query,
            },
        )
        return updated_query, {"base_query": base_query, "terms": selected}

    # ── 2. Select result from card click or chart click ───────────────────────
    @app.callback(
        Output("selected-result-id", "data"),
        Output("evidence-scroll-request", "data", allow_duplicate=True),
        Input({"type": "result-card", "index": ALL}, "n_clicks"),
        Input("score-chart", "clickData"),
        Input("timeline-chart", "clickData"),
        State("selected-result-id", "data"),
        prevent_initial_call=True,
    )
    def select_result(_card_clicks, score_click, timeline_click, current_id):
        trigger = callback_context.triggered_id
        if isinstance(trigger, dict) and trigger.get("type") == "result-card":
            triggered = callback_context.triggered
            if not triggered or not triggered[0].get("value"):
                return current_id or no_update, no_update
            result_id = trigger.get("index")
            evaluation_logger.log_event(
                "result_selected",
                {
                    "source": "result_card",
                    "result_id": result_id,
                    "result": _result_snapshot(dashboard_service.get_result(result_id)),
                },
            )
            return result_id, {
                "reason": "result_selected",
                "result_id": result_id,
                "nonce": time.time_ns(),
            }
        click_data = {
            "score-chart": score_click,
            "timeline-chart": timeline_click,
        }.get(trigger)
        result_id = _extract_result_id(click_data)
        if result_id:
            evaluation_logger.log_event(
                "result_selected",
                {
                    "source": trigger,
                    "result_id": result_id,
                    "result": _result_snapshot(dashboard_service.get_result(result_id)),
                },
            )
        if result_id:
            return result_id, {
                "reason": "chart_result_selected",
                "result_id": result_id,
                "nonce": time.time_ns(),
            }
        return current_id or no_update, no_update

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

        updated = {"liked": liked, "disliked": disliked}
        evaluation_logger.log_event(
            "feedback_changed",
            {
                "result_id": result_id,
                "direction": direction,
                "feedback": updated,
            },
        )
        return updated

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
        modality_feedback = _modality_feedback_summary(liked_ids, disliked_ids)

        refined_embedding = _pl.rocchio_refine(embedding, liked_rows, disliked_rows)
        evaluation_logger.log_event(
            "rocchio_refinement_requested",
            {
                "query": query or "",
                "liked_result_ids": liked_ids,
                "disliked_result_ids": disliked_ids,
                "liked_visual_vectors": len(liked_rows),
                "disliked_visual_vectors": len(disliked_rows),
                "modality_feedback": modality_feedback,
            },
        )

        filters = SearchFilters(
            query=query or "",
            modalities=tuple(modalities or ["transcript", "visual"]),
            viewpoint=viewpoint or "All",
            min_score=float(score_range[0]) if score_range else 0.0,
            max_score=float(score_range[1]) if score_range else 1.0,
            top_k=20,
        )
        refined_visual = _pl.retrieve_visual_from_embedding(refined_embedding, top_k=filters.top_k)
        results = dashboard_service.refine_visual(
            filters,
            refined_visual,
            modality_feedback=modality_feedback,
        )

        disliked_set = set(disliked_ids)
        results = [r for r in results if r.id not in disliked_set]

        print(
            f"[feedback] Rocchio refine — liked={len(liked_rows)} disliked={len(disliked_rows)}"
            f" → {len(results)} results (after hiding disliked)"
        )
        evaluation_logger.log_event(
            "rocchio_refinement_completed",
            {
                "result_count": len(results),
                "results": [_result_snapshot(result) for result in results],
            },
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
        evaluation_logger.log_event(
            "grounding_requested",
            {"query": query, "result_id": selected_id, "result": _result_snapshot(result)},
        )
        grounded = _pl.run_grounding(query, chunk)
        evaluation_logger.log_event(
            "grounding_completed",
            {"result_id": selected_id, "grounding": grounded},
        )
        return grounded

    # ── 6b. Run transcript heatmap + extractive QA span on demand ─────────────
    @app.callback(
        Output("transcript-evidence", "data"),
        Output("evidence-scroll-request", "data", allow_duplicate=True),
        Input("highlight-answer-button", "n_clicks"),
        State("selected-result-id", "data"),
        prevent_initial_call=True,
    )
    def run_transcript_evidence(n_clicks, selected_id):
        if not n_clicks or not selected_id:
            return no_update, no_update
        evaluation_logger.log_event(
            "transcript_highlight_requested",
            {
                "result_id": selected_id,
                "result": _result_snapshot(dashboard_service.get_result(selected_id)),
            },
        )
        enriched = dashboard_service.compute_transcript_evidence(selected_id)
        if enriched is None:
            return no_update, no_update
        enriched["_result_id"] = selected_id
        evaluation_logger.log_event(
            "transcript_highlight_completed",
            {
                "result_id": selected_id,
                "answer_span": enriched.get("answer_span"),
                "answer_candidates": enriched.get("answer_candidates"),
            },
        )
        return enriched, {
            "reason": "transcript_highlight_completed",
            "result_id": selected_id,
            "nonce": time.time_ns(),
        }

    # ── 6c. Passive evaluation-only interaction observations ─────────────────
    @app.callback(
        Output("evaluation-event-sink", "data", allow_duplicate=True),
        Input("modality-filter", "value"),
        Input("score-filter", "value"),
        Input("viewpoint-filter", "value"),
        State("evaluation-event-sink", "data"),
        prevent_initial_call=True,
    )
    def log_filter_changes(modalities, score_range, viewpoint, sink):
        evaluation_logger.log_event(
            "filters_changed",
            {
                "modalities": modalities or [],
                "score_range": score_range or [0.0, 1.0],
                "viewpoint": viewpoint or "All",
            },
            dedupe_seconds=0.5,
        )
        return int(sink or 0) + 1

    @app.callback(
        Output("evaluation-event-sink", "data", allow_duplicate=True),
        Input("score-chart", "hoverData"),
        Input("timeline-chart", "hoverData"),
        State("evaluation-event-sink", "data"),
        prevent_initial_call=True,
    )
    def log_chart_hover(score_hover, timeline_hover, sink):
        trigger = callback_context.triggered_id
        hover_data = score_hover if trigger == "score-chart" else timeline_hover
        result_id = _extract_result_id(hover_data)
        if result_id:
            evaluation_logger.log_event(
                "chart_hovered",
                {"chart": trigger, "result_id": result_id},
                dedupe_seconds=1.0,
            )
        return int(sink or 0) + 1

    @app.callback(
        Output("evaluation-event-sink", "data", allow_duplicate=True),
        Input("youtube-evidence-link", "n_clicks"),
        State("selected-result-id", "data"),
        State("evaluation-event-sink", "data"),
        prevent_initial_call=True,
    )
    def log_youtube_open(n_clicks, selected_id, sink):
        if n_clicks:
            evaluation_logger.log_event(
                "youtube_opened",
                {
                    "result_id": selected_id,
                    "result": _result_snapshot(dashboard_service.get_result(selected_id)),
                },
            )
        return int(sink or 0) + 1

    @app.callback(
        Output("evaluation-event-sink", "data", allow_duplicate=True),
        Input("original-frame-link", "n_clicks"),
        State("selected-result-id", "data"),
        State("evaluation-event-sink", "data"),
        prevent_initial_call=True,
    )
    def log_original_frame_open(n_clicks, selected_id, sink):
        if n_clicks:
            evaluation_logger.log_event(
                "original_frame_opened",
                {"result_id": selected_id},
            )
        return int(sink or 0) + 1

    @app.callback(
        Output("evaluation-event-sink", "data", allow_duplicate=True),
        Input("visual-transcript-details", "open"),
        State("selected-result-id", "data"),
        State("evaluation-event-sink", "data"),
        prevent_initial_call=True,
    )
    def log_transcript_details(opened, selected_id, sink):
        evaluation_logger.log_event(
            "visual_transcript_toggled",
            {"result_id": selected_id, "open": bool(opened)},
        )
        return int(sink or 0) + 1

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

        if startup_manager.snapshot().get("ready"):
            stats = dashboard_service.get_stats()
        else:
            # Do not let the initial page-render callback duplicate model/index
            # loading while the background warmup is still in progress.
            stats = DashboardStats(
                total_videos=0,
                total_keyframes=0,
                transcript_chunks=0,
                indexed_items=0,
                avg_latency_ms=0,
                retrieval_quality=0.0,
                grounding_quality=0.0,
            )
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


def _result_snapshot(result):
    if result is None:
        return None
    return {
        "id": result.id,
        "rank": result.rank,
        "title": result.title,
        "modality": result.modality,
        "score": result.score,
        "raw_score": result.raw_score,
        "viewpoint": result.viewpoint,
        "video_id": result.video_id,
        "timestamp": result.timestamp,
        "start_sec": result.start_sec,
        "end_sec": result.end_sec,
        "youtube_url": result.youtube_url,
        "keyframe_path": result.keyframe_path,
        "is_routed_choice": result.is_routed_choice,
    }


def _modality_feedback_summary(
    liked_ids: list[str],
    disliked_ids: list[str],
) -> dict[str, int]:
    summary = {
        "liked_visual": 0,
        "disliked_visual": 0,
        "liked_transcript": 0,
        "disliked_transcript": 0,
    }
    for direction, result_ids in (("liked", liked_ids), ("disliked", disliked_ids)):
        for result_id in result_ids:
            result = dashboard_service.get_result(result_id)
            if result is None or result.modality not in {"visual", "transcript"}:
                continue
            summary[f"{direction}_{result.modality}"] += 1
    return summary


def build_keyword_suggestions(
    payload: dict[str, Any] | None,
    selected_keywords: dict[str, Any] | list[str] | None = None,
    current_query: str = "",
):
    terms = (payload or {}).get("suggested_terms") or []
    if not terms:
        return html.Div("No recommended keywords yet.", className="keyword-empty")

    keyword_state = _normalise_keyword_state(selected_keywords, current_query)
    selected = set(keyword_state["terms"])
    base_query = keyword_state["base_query"]
    chips = []
    for index, item in enumerate(terms[:20], start=1):
        term = str(item.get("term", "")).strip()
        if not term:
            continue
        # Do not recommend words or phrases already written by the user. Keep
        # currently selected chips visible so they can still be toggled off.
        if term not in selected and _contains_phrase(base_query, term):
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
                className=(
                    "keyword-chip keyword-chip-selected"
                    if term in selected
                    else "keyword-chip"
                ),
                **{"aria-pressed": "true" if term in selected else "false"},
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


def _contains_phrase(query: str, phrase: str) -> bool:
    return bool(
        re.search(
            rf"(?<!\w){re.escape(phrase)}(?!\w)",
            query,
            flags=re.IGNORECASE,
        )
    )


def _normalise_keyword_state(
    value: dict[str, Any] | list[str] | None,
    current_query: str,
) -> dict[str, Any]:
    if isinstance(value, dict):
        base_query = str(value.get("base_query", "")).strip()
        terms = [
            str(term).strip()
            for term in value.get("terms", [])
            if str(term).strip()
        ]
        return {
            "base_query": base_query or (current_query or "").strip(),
            "terms": terms,
        }

    # Compatibility with logs/pages created before the store became structured.
    terms = [str(term).strip() for term in (value or []) if str(term).strip()]
    return {"base_query": (current_query or "").strip(), "terms": terms}


def _compose_keyword_query(base_query: str, selected_terms: list[str]) -> str:
    """Append only chip-managed terms; never edit words inside the base query."""
    pieces = [base_query.strip(), *(term.strip() for term in selected_terms)]
    return " ".join(piece for piece in pieces if piece)
