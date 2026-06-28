from __future__ import annotations

import time
from typing import Any

from castle_dashboard import pipeline as _pipeline
from castle_dashboard.models.schemas import (
    DashboardStats,
    ProjectionPoint,
    RetrievalResult,
    SearchFilters,
    TranscriptSegment,
)


def _fmt_sec(sec: float) -> str:
    h = int(sec) // 3600
    m = (int(sec) % 3600) // 60
    s = int(sec) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _evidence_key(item: dict[str, Any]) -> str:
    """Mirrors scripts/query_evidence.py::evidence_key — identifies the same
    underlying evidence item across the router's `chosen` result and the
    mixed-evidence list, so we can flag which row is the router's pick."""
    if item.get("evidence_type") == "visual" or item.get("keyframe_path"):
        return str(item.get("keyframe_path") or item.get("source_id") or id(item))
    return str(item.get("source_id") or f"{item.get('transcript_path')}:{item.get('start_sec')}:{item.get('end_sec')}")


def _common_fields(item: dict[str, Any], rank: int, is_routed_choice: bool) -> dict[str, Any]:
    components = item.get("score_components") or {}
    return dict(
        rank=rank,
        evidence_type=str(item.get("evidence_type", "")),
        confidence_percent=float(item.get("confidence_percent", 0.0)),
        score=float(item.get("confidence", 0.0)),
        raw_score=float(components.get("raw_score", item.get("score", 0.0))),
        is_routed_choice=is_routed_choice,
    )


def _visual_to_result(item: dict[str, Any], rank: int, is_routed_choice: bool = False) -> RetrievalResult:
    start = float(item.get("start_sec", item.get("keyframe_time_sec", 0)))
    source = item.get("source_name", "")
    vid = item.get("video_id", item.get("hour_id", ""))
    title = f"{source} / {vid} @ {_fmt_sec(start)}"
    caption = (
        item.get("visual_caption")
        or item.get("caption")
        or item.get("grounding_target")
        or "Visual keyframe evidence"
    )
    return RetrievalResult(
        id=str(item.get("source_id") or item.get("keyframe_path") or f"v-{rank}"),
        title=title,
        answer="",
        timestamp=_fmt_sec(start),
        timestamp_seconds=start,
        viewpoint=source,
        camera=vid,
        modality="visual",
        grounding="none",
        keyframe_path=item.get("keyframe_path", ""),
        video_path=item.get("video_path", ""),
        caption=str(caption),
        transcript=(),
        mask=None,
        tags=(source, vid, item.get("day", "")),
        youtube_url=item.get("youtube_timestamp_url") or item.get("youtube_url", ""),
        start_sec=start,
        end_sec=float(item.get("end_sec", start)),
        source_name=source,
        day=item.get("day", ""),
        video_id=vid,
        faiss_row_id=item.get("faiss_row_id"),
        **_common_fields(item, rank, is_routed_choice),
    )


def _transcript_to_result(item: dict[str, Any], rank: int, is_routed_choice: bool = False) -> RetrievalResult:
    start = float(item.get("start_sec", 0))
    end = float(item.get("end_sec", start))
    source = item.get("source_name", "")
    vid = item.get("video_id", item.get("hour_id", ""))
    text = str(item.get("transcript_snippet") or item.get("text", "")).strip()
    title = (text[:70] + "…") if len(text) > 70 else text
    kf = item.get("closest_keyframe_path") or ""
    return RetrievalResult(
        id=str(item.get("source_id") or _evidence_key(item) or f"t-{rank}"),
        title=title or f"{source} / {vid} @ {_fmt_sec(start)}",
        answer="",
        timestamp=_fmt_sec(start),
        timestamp_seconds=start,
        viewpoint=source,
        camera=vid,
        modality="transcript",
        grounding="none",
        keyframe_path=kf,
        video_path=item.get("video_path", ""),
        caption=text,
        transcript=(TranscriptSegment(_fmt_sec(start), _fmt_sec(end), source, text),),
        mask=None,
        tags=(source, vid, item.get("day", "")),
        youtube_url=item.get("youtube_timestamp_url") or item.get("youtube_url", ""),
        start_sec=start,
        end_sec=end,
        source_name=source,
        day=item.get("day", ""),
        video_id=vid,
        **_common_fields(item, rank, is_routed_choice),
    )


class DashboardService:
    def __init__(self) -> None:
        self._cache: dict[str, RetrievalResult] = {}
        self._raw_cache: dict[str, dict[str, Any]] = {}
        self._last_query_ms: int = 0
        self._last_router_debug: dict[str, Any] = {}
        self._last_query: str = ""
        self._last_transcript_results: list[dict[str, Any]] = []
        self._last_visual_results: list[dict[str, Any]] = []
        self._last_keyword_suggestions: dict[str, Any] = {
            "source": "not_run",
            "suggested_terms": [],
        }

    def search(self, filters: SearchFilters) -> list[RetrievalResult]:
        if not filters.query.strip():
            self._last_router_debug = {}
            self._last_keyword_suggestions = {"source": "empty_query", "suggested_terms": []}
            return []

        modalities = set(filters.modalities)
        top_k = max(1, min(filters.top_k, 100))
        t0 = time.perf_counter()

        # Always retrieve both channels — the evidence router needs both top
        # scores to decide (and weight) which modality this question favors,
        # even if the analyst has filtered the displayed list to one of them.
        print(f"[dashboard] visual retrieval started: {filters.query!r}", flush=True)
        visual_results = _pipeline.retrieve_visual(filters.query, top_k=top_k)
        print(f"[dashboard] transcript retrieval started: {filters.query!r}", flush=True)
        transcript_results = _pipeline.retrieve_transcript(filters.query, top_k=top_k)
        self._last_visual_results = visual_results
        self._last_transcript_results = transcript_results
        self._last_keyword_suggestions = {
            "source": "pending",
            "suggested_terms": [],
        }
        print("[dashboard] routing evidence", flush=True)
        chosen = _pipeline.route(filters.query, visual_results, transcript_results)
        router_debug = chosen.get("router_debug", {})

        print("[dashboard] building mixed evidence", flush=True)
        mixed = _pipeline.get_mixed_evidence(
            filters.query, visual_results, transcript_results, router_debug, top_k=top_k
        )

        self._last_query_ms = int((time.perf_counter() - t0) * 1000)
        self._last_router_debug = router_debug
        self._last_query = filters.query

        # Apply modality filter
        filtered = [x for x in mixed if x.get("evidence_type") in modalities]

        # Apply confidence range filter (0..1)
        filtered = [
            x for x in filtered
            if filters.min_score <= float(x.get("confidence", 0)) <= filters.max_score
        ]

        # Apply viewpoint filter
        if filters.viewpoint and filters.viewpoint != "All":
            filtered = [x for x in filtered if x.get("source_name") == filters.viewpoint]

        chosen_key = _evidence_key(chosen)

        # Convert to RetrievalResult and cache (raw dicts kept too, for
        # on-demand transcript heatmap / QA span enrichment on selection).
        results: list[RetrievalResult] = []
        raw_cache: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(filtered, start=1):
            is_routed_choice = _evidence_key(item) == chosen_key
            if item.get("evidence_type") == "visual":
                result = _visual_to_result(item, rank, is_routed_choice)
            else:
                result = _transcript_to_result(item, rank, is_routed_choice)
            results.append(result)
            raw_cache[result.id] = item

        self._cache = {r.id: r for r in results}
        self._raw_cache = raw_cache
        return results

    def compute_keyword_suggestions(self) -> dict[str, Any]:
        if not self._last_query.strip() or not self._last_transcript_results:
            self._last_keyword_suggestions = {
                "source": "unavailable_no_transcript_results",
                "suggested_terms": [],
            }
            return self._last_keyword_suggestions

        print("[dashboard] keyword recommendation started", flush=True)
        self._last_keyword_suggestions = _pipeline.recommend_keywords_semantic(
            self._last_query,
            self._last_transcript_results,
            top_n=min(20, max(1, len(self._last_transcript_results))),
            max_keywords=20,
        )
        print("[dashboard] keyword recommendation finished", flush=True)
        return self._last_keyword_suggestions

    def get_results_by_ids(self, result_ids: list[str]) -> list[RetrievalResult]:
        return [self._cache[rid] for rid in result_ids if rid in self._cache]

    def get_result(self, result_id: str | None) -> RetrievalResult | None:
        if result_id is None:
            return None
        return self._cache.get(result_id)

    def get_raw_item(self, result_id: str | None) -> dict[str, Any] | None:
        if result_id is None:
            return None
        return self._raw_cache.get(result_id)

    def get_router_debug(self) -> dict[str, Any]:
        return self._last_router_debug

    def get_keyword_suggestions(self) -> dict[str, Any]:
        return self._last_keyword_suggestions

    def compute_transcript_evidence(self, result_id: str) -> dict[str, Any] | None:
        """On-demand QA answer span + transcript heatmap for the given result.
        Mirrors --include-transcript-heatmap in scripts/query_evidence.py."""
        item = self.get_raw_item(result_id)
        if item is None or item.get("evidence_type") != "transcript" or not self._last_query:
            return None
        return _pipeline.add_transcript_heatmap(self._last_query, item)

    def refine_visual(
        self,
        filters: "SearchFilters",
        refined_visual: list[dict[str, Any]],
        modality_feedback: dict[str, int] | None = None,
    ) -> list["RetrievalResult"]:
        """Re-run routing + mixing with Rocchio-refined visual results, reusing
        the cached transcript results so we don't re-embed the query."""
        top_k = max(1, min(filters.top_k, 100))
        t0 = time.perf_counter()

        chosen = _pipeline.route(
            self._last_query,
            refined_visual,
            self._last_transcript_results,
            feedback=modality_feedback,
        )
        router_debug = chosen.get("router_debug", {})
        mixed = _pipeline.get_mixed_evidence(
            self._last_query, refined_visual, self._last_transcript_results,
            router_debug, top_k=top_k,
        )

        self._last_query_ms = int((time.perf_counter() - t0) * 1000)
        self._last_router_debug = router_debug

        modalities = set(filters.modalities)
        filtered = [x for x in mixed if x.get("evidence_type") in modalities]
        filtered = [
            x for x in filtered
            if filters.min_score <= float(x.get("confidence", 0)) <= filters.max_score
        ]
        if filters.viewpoint and filters.viewpoint != "All":
            filtered = [x for x in filtered if x.get("source_name") == filters.viewpoint]

        chosen_key = _evidence_key(chosen)
        results: list[RetrievalResult] = []
        raw_cache: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(filtered, start=1):
            is_routed_choice = _evidence_key(item) == chosen_key
            if item.get("evidence_type") == "visual":
                result = _visual_to_result(item, rank, is_routed_choice)
            else:
                result = _transcript_to_result(item, rank, is_routed_choice)
            results.append(result)
            raw_cache[result.id] = item

        self._cache = {r.id: r for r in results}
        self._raw_cache = raw_cache
        return results

    def get_stats(self) -> DashboardStats:
        raw = _pipeline.get_stats()
        return DashboardStats(
            total_videos=0,
            total_keyframes=raw["total_keyframes"],
            transcript_chunks=raw["transcript_chunks"],
            indexed_items=raw["total_keyframes"] + raw["transcript_chunks"],
            avg_latency_ms=self._last_query_ms,
            retrieval_quality=0.0,
            grounding_quality=0.0,
        )

    def get_projection_points(self) -> list[ProjectionPoint]:
        return []

    def get_available_viewpoints(self) -> list[str]:
        return _pipeline.get_available_viewpoints()

    def get_transcript_context(
        self, source_name: str, video_id: str, center_sec: float
    ) -> list[dict[str, Any]]:
        return _pipeline.get_transcript_context(source_name, video_id, center_sec)

    def get_nearby_keyframes(
        self, source_name: str, video_id: str, center_sec: float
    ) -> list[dict[str, Any]]:
        return _pipeline.get_nearby_keyframes(source_name, video_id, center_sec)


dashboard_service = DashboardService()
