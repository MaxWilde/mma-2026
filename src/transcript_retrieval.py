from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.evidence_links import youtube_timestamp_url
from src.retriever import load_embedding_model, load_index
from src.vqa import format_timestamp


DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ROOT = Path(__file__).resolve().parents[1]
REFINEMENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "did",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "with",
}


def retrieve_transcript_evidence(
    question: str,
    index_dir: str,
    top_k: int = 5,
    dense_k: int = 100,
    lexical_k: int = 100,
    rerank_k: int = 50,
    use_cross_encoder: bool = True,
    refine_timestamps: bool = False,
    align_across_povs: bool = False,
    align_playback: bool = False,
) -> list[dict[str, Any]]:
    debug = retrieve_transcript_evidence_debug(
        question,
        index_dir,
        top_k=top_k,
        dense_k=dense_k,
        lexical_k=lexical_k,
        rerank_k=rerank_k,
        use_cross_encoder=use_cross_encoder,
        refine_timestamps=refine_timestamps,
        align_across_povs=align_across_povs,
        align_playback=align_playback,
    )
    return debug["reranked"][:top_k]


def retrieve_transcript_evidence_debug(
    question: str,
    index_dir: str,
    top_k: int = 5,
    dense_k: int = 100,
    lexical_k: int = 100,
    rerank_k: int = 50,
    use_cross_encoder: bool = True,
    cross_encoder_name: str = DEFAULT_CROSS_ENCODER,
    refine_timestamps: bool = False,
    align_across_povs: bool = False,
    align_playback: bool = False,
) -> dict[str, Any]:
    timings: dict[str, float] = {}

    start = time.perf_counter()
    index, metadata, model_name, embedding_model = _load_dense_resources(str(index_dir))
    timings["load_dense_resources_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    dense = dense_candidates(question, index, metadata, embedding_model, dense_k)
    timings["dense_retrieval_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    lexical = lexical_candidates(question, str(index_dir), lexical_k)
    timings["lexical_retrieval_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    fused = rrf_fuse(dense, lexical)
    fused = [_finalize_result(item) for item in fused]
    timings["rrf_fusion_sec"] = time.perf_counter() - start

    cross_encoder_status = "disabled"
    reranked = fused[:rerank_k]
    if use_cross_encoder and reranked:
        start = time.perf_counter()
        cross_encoder = _load_cross_encoder(cross_encoder_name)
        timings["load_cross_encoder_sec"] = time.perf_counter() - start
        if cross_encoder is None:
            cross_encoder_status = "unavailable in local cache; used MiniLM passage fallback"
            start = time.perf_counter()
            reranked = minilm_passage_rerank(question, reranked, embedding_model)
            timings["minilm_passage_rerank_sec"] = time.perf_counter() - start
        else:
            cross_encoder_status = "used"
            start = time.perf_counter()
            reranked = cross_encoder_rerank(question, reranked, cross_encoder)
            timings["cross_encoder_rerank_sec"] = time.perf_counter() - start
    reranked = [_finalize_result(item) for item in reranked]

    start = time.perf_counter()
    reranked = apply_source_name_boost(question, reranked, metadata)
    timings["source_name_boost_sec"] = time.perf_counter() - start

    if refine_timestamps:
        start = time.perf_counter()
        reranked = [refine_transcript_timestamp(question, item) for item in reranked]
        timings["timestamp_refinement_sec"] = time.perf_counter() - start

    if align_across_povs:
        start = time.perf_counter()
        reranked = [align_transcript_across_povs(question, item) for item in reranked]
        timings["cross_pov_alignment_sec"] = time.perf_counter() - start

    if align_playback:
        start = time.perf_counter()
        reranked = [align_playback_start_across_povs(question, item) for item in reranked]
        timings["playback_alignment_sec"] = time.perf_counter() - start

    return {
        "question": question,
        "index_dir": str(index_dir),
        "embedding_model": model_name,
        "cross_encoder": cross_encoder_name,
        "cross_encoder_status": cross_encoder_status,
        "dense": dense,
        "lexical": lexical,
        "fused": fused,
        "reranked": reranked,
        "timings": timings,
    }


def dense_candidates(
    question: str,
    index: Any,
    metadata: list[dict[str, Any]],
    model: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    query_embedding = model.encode(
        [question],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.asarray(query_embedding, dtype="float32")
    scores, ids = index.search(query_embedding, min(top_k, len(metadata)))

    results: list[dict[str, Any]] = []
    for rank, (score, row_id) in enumerate(zip(scores[0], ids[0]), start=1):
        if row_id < 0:
            continue
        item = dict(metadata[int(row_id)])
        item["dense_rank"] = rank
        item["dense_score"] = float(score)
        item["score"] = float(score)
        results.append(_finalize_result(item))
    return results


def lexical_candidates(question: str, index_dir: str, top_k: int) -> list[dict[str, Any]]:
    metadata, windows, idf, average_window_length = _load_lexical_resources(index_dir)
    query_tokens = tokenize(question)
    if not query_tokens:
        return []

    best_by_chunk: dict[int, tuple[float, str]] = {}
    for window in windows:
        score = bm25_score(query_tokens, window, idf, average_window_length)
        if score <= 0:
            continue
        existing = best_by_chunk.get(window["metadata_index"])
        if existing is None or score > existing[0]:
            best_by_chunk[window["metadata_index"]] = (score, window["text"])

    ranked = sorted(best_by_chunk.items(), key=lambda item: item[1][0], reverse=True)[:top_k]

    results: list[dict[str, Any]] = []
    for rank, (row_id, (score, matched_window)) in enumerate(ranked, start=1):
        item = dict(metadata[int(row_id)])
        item["lexical_rank"] = rank
        item["lexical_score"] = score
        item["lexical_matched_window"] = matched_window
        item["score"] = score
        results.append(_finalize_result(item))
    return results


def rrf_fuse(
    dense: list[dict[str, Any]],
    lexical: list[dict[str, Any]],
    *,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for source_name, candidates in (("dense", dense), ("lexical", lexical)):
        for rank, candidate in enumerate(candidates, start=1):
            key = candidate_key(candidate)
            item = by_id.setdefault(key, dict(candidate))
            item["rrf_score"] = float(item.get("rrf_score", 0.0)) + 1.0 / (rrf_k + rank)
            if source_name == "dense":
                item["dense_rank"] = min(int(item.get("dense_rank", rank)), rank)
                item["dense_score"] = max(float(item.get("dense_score", 0.0)), float(candidate.get("dense_score", 0.0)))
            else:
                item["lexical_rank"] = min(int(item.get("lexical_rank", rank)), rank)
                item["lexical_score"] = max(
                    float(item.get("lexical_score", 0.0)),
                    float(candidate.get("lexical_score", 0.0)),
                )
    fused = sorted(by_id.values(), key=lambda item: float(item.get("rrf_score", 0.0)), reverse=True)
    for rank, item in enumerate(fused, start=1):
        item["fused_rank"] = rank
        item["score"] = float(item.get("rrf_score", 0.0))
    return fused


def cross_encoder_rerank(question: str, candidates: list[dict[str, Any]], cross_encoder: Any) -> list[dict[str, Any]]:
    pairs = [(question, str(item.get("text", ""))) for item in candidates]
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    ranked: list[dict[str, Any]] = []
    for item, score in zip(candidates, scores):
        updated = dict(item)
        updated["cross_encoder_score"] = float(score)
        updated["score"] = float(score)
        ranked.append(updated)
    ranked.sort(key=lambda item: float(item.get("cross_encoder_score", item.get("score", 0.0))), reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rerank_rank"] = rank
    return ranked


def minilm_passage_rerank(question: str, candidates: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    passages = [
        str(item.get("lexical_matched_window") or item.get("transcript_snippet") or item.get("text", ""))
        for item in candidates
    ]
    embeddings = model.encode(
        [question] + passages,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.asarray(embeddings[0], dtype="float32")
    passage_embeddings = np.asarray(embeddings[1:], dtype="float32")
    scores = passage_embeddings @ query_embedding

    ranked: list[dict[str, Any]] = []
    for item, score in zip(candidates, scores):
        updated = dict(item)
        updated["minilm_passage_score"] = float(score)
        updated["score"] = float(score)
        ranked.append(updated)
    ranked.sort(key=lambda item: float(item.get("minilm_passage_score", item.get("score", 0.0))), reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["rerank_rank"] = rank
    return ranked


def apply_source_name_boost(
    question: str,
    candidates: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    *,
    boost: float = 0.25,
) -> list[dict[str, Any]]:
    mentioned = mentioned_source_names(question, metadata)
    if not mentioned:
        return candidates

    boosted: list[dict[str, Any]] = []
    for item in candidates:
        updated = dict(item)
        source_name = str(updated.get("source_name", "")).lower()
        if source_name in mentioned:
            updated["source_name_boost"] = boost
            updated["score"] = float(updated.get("score", 0.0)) + boost
        else:
            updated["source_name_boost"] = 0.0
        boosted.append(updated)
    boosted.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    for rank, item in enumerate(boosted, start=1):
        item["rerank_rank"] = rank
    return boosted


def mentioned_source_names(question: str, metadata: list[dict[str, Any]]) -> set[str]:
    question_tokens = set(tokenize(question))
    mentioned: set[str] = set()
    for item in metadata:
        source = str(item.get("source_name", "")).strip().lower()
        if source and source in question_tokens:
            mentioned.add(source)
    return mentioned


def candidate_key(item: dict[str, Any]) -> str:
    return str(
        item.get("source_id")
        or f"{item.get('transcript_path')}:{item.get('start_sec')}:{item.get('end_sec')}"
    )


def preview_text(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_score(
    query_tokens: list[str],
    window: dict[str, Any],
    idf: dict[str, float],
    average_window_length: float,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    score = 0.0
    counts: Counter[str] = window["counts"]
    length = max(1, int(window["length"]))
    for token in query_tokens:
        token_idf = idf.get(token)
        if token_idf is None:
            continue
        term_frequency = counts.get(token, 0)
        if term_frequency == 0:
            continue
        denominator = term_frequency + k1 * (1 - b + b * length / average_window_length)
        score += token_idf * (term_frequency * (k1 + 1)) / denominator
    return score


def refine_transcript_timestamp(question: str, result: dict[str, Any]) -> dict[str, Any]:
    transcript_path = resolve_transcript_path(result.get("transcript_path"))
    if transcript_path is None:
        return _with_refinement_failure(result, "missing transcript_path")

    try:
        transcript_entries = load_transcript_entries(str(transcript_path))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _with_refinement_failure(result, f"could not load transcript JSON: {exc}")

    try:
        broad_start = float(result["start_sec"])
        broad_end = float(result["end_sec"])
    except (KeyError, TypeError, ValueError):
        return _with_refinement_failure(result, "missing broad start_sec/end_sec")

    fully_inside_entries = [
        entry
        for entry in transcript_entries
        if entry["start_sec"] >= broad_start - 0.001 and entry["end_sec"] <= broad_end + 0.001
    ]
    overlapping_entries = overlapping_transcript_entries(transcript_entries, broad_start, broad_end)
    extraction_debug = transcript_extraction_debug(
        transcript_path=transcript_path,
        broad_start=broad_start,
        broad_end=broad_end,
        all_entries=transcript_entries,
        overlapping_entries=overlapping_entries,
        fully_inside_entries=fully_inside_entries,
    )
    if not overlapping_entries:
        failed = _with_refinement_failure(result, "no transcript entries overlapping broad chunk")
        failed["timestamp_refinement_extraction_debug"] = extraction_debug
        return failed

    refinement_entries = overlapping_entries
    search_window_expanded = False
    windows = build_refinement_windows(refinement_entries, broad_start_sec=broad_start, broad_end_sec=broad_end)
    if not windows:
        expanded_start = broad_start - 60.0
        expanded_end = broad_end + 60.0
        expanded_entries = overlapping_transcript_entries(transcript_entries, expanded_start, expanded_end)
        search_window_expanded = True
        extraction_debug["expanded_start_sec"] = expanded_start
        extraction_debug["expanded_end_sec"] = expanded_end
        extraction_debug["expanded_overlapping_entries"] = len(expanded_entries)
        extraction_debug["expanded_short_entries_duration_le_30s"] = count_short_entries(expanded_entries)
        extraction_debug["first_20_expanded_overlapping_entries"] = summarize_entries(expanded_entries[:20])
        refinement_entries = expanded_entries
        windows = build_refinement_windows(refinement_entries, broad_start_sec=broad_start, broad_end_sec=broad_end)

    broad_evidence_text = str(
        result.get("lexical_matched_window")
        or result.get("minilm_matched_window")
        or result.get("transcript_snippet")
        or result.get("text", "")
    )
    ranked_candidates = ranked_refinement_windows(question, windows, broad_evidence_text)
    best = ranked_candidates[0] if ranked_candidates else None
    if best is None:
        failed = _with_refinement_failure(result, "no usable short transcript windows inside broad chunk")
        failed["timestamp_refinement_extraction_debug"] = extraction_debug
        failed["timestamp_refinement_candidates"] = []
        return failed

    refined = dict(result)
    refined["broad_start_sec"] = broad_start
    refined["broad_end_sec"] = broad_end
    refined["broad_timestamp"] = format_timestamp(broad_start, broad_end)
    refined["broad_transcript_snippet"] = result.get("transcript_snippet", preview_text(str(result.get("text", ""))))
    refined["broad_youtube_timestamp_url"] = result.get("youtube_timestamp_url") or youtube_timestamp_url(result)
    refined["refined_start_sec"] = best["start_sec"]
    refined["refined_end_sec"] = best["end_sec"]
    refined["refined_timestamp"] = format_timestamp(float(best["start_sec"]), float(best["end_sec"]))
    refined["refined_transcript_snippet"] = preview_text(str(best["text"]), limit=360)
    refined["refined_youtube_timestamp_url"] = youtube_timestamp_url(refined, timestamp_sec=float(best["start_sec"]))
    refined["timestamp_refinement_method"] = "transcript_json_query_plus_broad_bm25_window"
    refined["timestamp_refinement_score"] = float(best["final_score"])
    refined["timestamp_refinement_search_window_expanded"] = search_window_expanded
    refined["timestamp_refinement_extraction_debug"] = extraction_debug
    refined["timestamp_refinement_candidates"] = [
        {
            "start_sec": candidate["start_sec"],
            "end_sec": candidate["end_sec"],
            "timestamp": format_timestamp(float(candidate["start_sec"]), float(candidate["end_sec"])),
            "duration_sec": float(candidate["end_sec"]) - float(candidate["start_sec"]),
            "query_score": float(candidate["query_score"]),
            "broad_score": float(candidate["broad_score"]),
            "density_score": float(candidate["density_score"]),
            "final_score": float(candidate["final_score"]),
            "text_preview": preview_text(str(candidate["text"]), limit=220),
        }
        for candidate in ranked_candidates[:5]
    ]
    return refined


def align_transcript_across_povs(
    question: str,
    result: dict[str, Any],
    *,
    search_radius_sec: float = 120.0,
) -> dict[str, Any]:
    try:
        start_sec = float(result["start_sec"])
        end_sec = float(result["end_sec"])
    except (KeyError, TypeError, ValueError):
        return result

    day = str(result.get("day") or "")
    hour_id = str(result.get("hour_id", result.get("video_id", "")))
    if not day or not hour_id:
        return result

    duration = end_sec - start_sec
    target_time = start_sec if duration <= 30.0 else (start_sec + end_sec) / 2.0
    transcript_files = same_day_hour_transcript_files(day, hour_id)
    broad_evidence_text = str(
        result.get("lexical_matched_window")
        or result.get("refined_transcript_snippet")
        or result.get("transcript_snippet")
        or result.get("text", "")
    )

    windows: list[dict[str, Any]] = []
    for transcript_file in transcript_files:
        try:
            entries = load_transcript_entries(str(transcript_file))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        nearby_entries = overlapping_transcript_entries(
            entries,
            target_time - search_radius_sec,
            target_time + search_radius_sec,
        )
        for window in build_alignment_windows(nearby_entries, transcript_file, day, hour_id):
            windows.append(window)

    ranked = ranked_alignment_windows(question, windows, broad_evidence_text, target_time)
    updated = dict(result)
    updated["original_source_name"] = result.get("source_name")
    updated["original_transcript_path"] = result.get("transcript_path")
    updated["original_start_sec"] = result.get("start_sec")
    updated["original_end_sec"] = result.get("end_sec")
    updated["original_timestamp"] = result.get("timestamp")
    updated["original_transcript_snippet"] = result.get("transcript_snippet", result.get("text", ""))
    updated["original_youtube_timestamp_url"] = result.get("youtube_timestamp_url")
    updated["alignment_target_time_sec"] = target_time
    updated["alignment_search_radius_sec"] = search_radius_sec
    updated["alignment_transcript_files_considered"] = len(transcript_files)
    updated["alignment_candidates"] = [alignment_candidate_summary(item) for item in ranked[:10]]
    original_text = str(
        result.get("refined_transcript_snippet")
        or result.get("lexical_matched_window")
        or result.get("transcript_snippet")
        or result.get("text", "")
    )
    original_alignment_scores = score_text_for_alignment(question, original_text, broad_evidence_text)
    updated["original_query_score"] = original_alignment_scores["query_score"]
    updated["original_query_overlap_count"] = original_alignment_scores["query_overlap_count"]
    updated["original_alignment_score"] = original_alignment_scores["final_score"]
    updated["original_duration_sec"] = duration

    if not ranked:
        updated["alignment_method"] = "unavailable"
        updated["alignment_error"] = "no same-day/hour short transcript windows near target time"
        updated["alignment_override_decision"] = "keep_original"
        updated["alignment_override_reason"] = "no alignment candidates"
        apply_final_transcript_fields(updated, aligned=False)
        return updated

    eligible = eligible_alignment_candidates(ranked, updated)
    updated["eligible_alignment_candidates"] = [alignment_candidate_summary(item) for item in eligible[:10]]
    highest_scoring = ranked[0]
    selected = earliest_alignment_candidate(eligible)
    updated["highest_scoring_alignment_candidate"] = alignment_candidate_summary(highest_scoring)
    if selected is not None:
        updated["earliest_eligible_alignment_candidate"] = alignment_candidate_summary(selected)
    else:
        updated["earliest_eligible_alignment_candidate"] = None

    if selected is None:
        updated["alignment_method"] = "cross_pov_temporal_query_broad_bm25"
        updated["alignment_override_decision"] = "keep_original"
        updated["alignment_override_reason"] = "no eligible alignment candidate with direct query support"
        apply_final_transcript_fields(updated, aligned=False)
        return updated

    best = selected
    youtube_url = youtube_url_for(day, str(best["source_name"]), hour_id) or str(result.get("youtube_url") or "")
    updated["alignment_method"] = "cross_pov_temporal_query_broad_bm25"
    updated["aligned_source_name"] = best["source_name"]
    updated["aligned_transcript_path"] = str(best["transcript_path"])
    updated["aligned_start_sec"] = float(best["start_sec"])
    updated["aligned_end_sec"] = float(best["end_sec"])
    updated["aligned_timestamp"] = format_timestamp(float(best["start_sec"]), float(best["end_sec"]))
    updated["aligned_snippet"] = preview_text(str(best["text"]), limit=420)
    updated["aligned_score"] = float(best["final_score"])
    updated["aligned_query_score"] = float(best["query_score"])
    updated["aligned_broad_score"] = float(best["broad_score"])
    updated["aligned_temporal_score"] = float(best["temporal_score"])
    updated["aligned_query_overlap_count"] = int(best.get("query_overlap_count", 0))
    updated["aligned_youtube_url"] = youtube_url
    updated["aligned_youtube_timestamp_url"] = youtube_timestamp_url(
        {"youtube_url": youtube_url},
        timestamp_sec=float(best["start_sec"]),
    )
    should_override, reason = should_override_with_alignment(updated)
    updated["alignment_override_decision"] = "use_aligned" if should_override else "keep_original"
    updated["alignment_override_reason"] = reason
    apply_final_transcript_fields(updated, aligned=should_override)
    if should_override:
        update_main_transcript_fields_from_alignment(updated)
    return updated


def align_playback_start_across_povs(
    question: str,
    result: dict[str, Any],
    *,
    search_radius_sec: float = 120.0,
    similarity_threshold: float = 1.0,
    min_overlap_count: int = 3,
    min_candidate_match_ratio: float = 0.35,
) -> dict[str, Any]:
    """Find a same-conversation playback start without replacing retrieved evidence."""
    updated = dict(result)
    original_youtube_url = result.get("youtube_timestamp_url") or youtube_timestamp_url(result)
    updated["original_youtube_timestamp_url"] = original_youtube_url

    try:
        start_sec = float(result["start_sec"])
        end_sec = float(result["end_sec"])
    except (KeyError, TypeError, ValueError):
        updated["playback_alignment_debug"] = {"decision": "keep_original", "reason": "missing start_sec/end_sec"}
        return _set_original_playback_fields(updated)

    day = str(result.get("day") or "")
    hour_id = str(result.get("hour_id", result.get("video_id", "")))
    if not day or not hour_id:
        updated["playback_alignment_debug"] = {"decision": "keep_original", "reason": "missing day/hour_id"}
        return _set_original_playback_fields(updated)

    evidence_text = str(result.get("text") or result.get("transcript_snippet") or "").strip()
    anchor = extract_local_anchor_passage(question, evidence_text)
    anchor_text = anchor["text"]
    anchor_tokens = dedupe_tokens(refinement_tokens(anchor_text))
    if not anchor_tokens:
        updated["playback_alignment_debug"] = {"decision": "keep_original", "reason": "empty selected evidence text"}
        return _set_original_playback_fields(updated)

    duration = end_sec - start_sec
    target_time = start_sec if duration <= 30.0 else (start_sec + end_sec) / 2.0
    transcript_files = same_day_hour_transcript_files(day, hour_id)

    windows: list[dict[str, Any]] = []
    for transcript_file in transcript_files:
        try:
            entries = load_transcript_entries(str(transcript_file))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        nearby_entries = overlapping_transcript_entries(
            entries,
            target_time - search_radius_sec,
            target_time + search_radius_sec,
        )
        windows.extend(build_alignment_windows(nearby_entries, transcript_file, day, hour_id))

    ranked = ranked_playback_alignment_windows(anchor_tokens, windows, target_time)
    candidates = [
        candidate
        for candidate in ranked
        if is_playback_alignment_eligible(
            candidate,
            original_start_sec=start_sec,
            original_end_sec=end_sec,
            similarity_threshold=similarity_threshold,
            min_overlap_count=min_overlap_count,
            min_candidate_match_ratio=min_candidate_match_ratio,
        )
    ]
    selected = earliest_playback_candidate(candidates)

    updated["playback_alignment_debug"] = {
        "decision": "use_aligned_playback" if selected else "keep_original",
        "reason": "selected earliest sufficiently similar same-day/hour transcript window"
        if selected
        else "no sufficiently similar same-day/hour transcript window",
        "original_source_name": result.get("source_name"),
        "original_timestamp": result.get("timestamp"),
        "original_youtube_timestamp_url": original_youtube_url,
        "target_time_sec": target_time,
        "search_radius_sec": search_radius_sec,
        "transcript_files_considered": len(transcript_files),
        "similarity_threshold": similarity_threshold,
        "min_overlap_count": min_overlap_count,
        "min_candidate_match_ratio": min_candidate_match_ratio,
        "anchor_passage": anchor_text,
        "anchor_score": anchor["score"],
        "anchor_overlap_count": anchor["overlap_count"],
        "top_similar_chunks": [playback_alignment_summary(candidate) for candidate in ranked[:10]],
        "eligible_chunks": [playback_alignment_summary(candidate) for candidate in candidates[:10]],
    }

    if selected is None:
        return _set_original_playback_fields(updated)

    playback_youtube_url = youtube_url_for(day, str(selected["source_name"]), hour_id) or str(result.get("youtube_url") or "")
    updated["playback_source_name"] = selected["source_name"]
    updated["playback_transcript_path"] = str(selected["transcript_path"])
    updated["playback_start_sec"] = float(selected["start_sec"])
    updated["playback_end_sec"] = float(selected["end_sec"])
    updated["playback_timestamp"] = format_timestamp(float(selected["start_sec"]), float(selected["end_sec"]))
    updated["playback_snippet"] = preview_text(str(selected["text"]), limit=360)
    updated["playback_similarity_score"] = float(selected["similarity_score"])
    updated["playback_overlap_count"] = int(selected["anchor_overlap_count"])
    updated["playback_candidate_match_ratio"] = float(selected["candidate_match_ratio"])
    updated["playback_youtube_url"] = playback_youtube_url
    updated["playback_youtube_timestamp_url"] = youtube_timestamp_url(
        {"youtube_url": playback_youtube_url},
        timestamp_sec=float(selected["start_sec"]),
    )
    updated["youtube_timestamp_url"] = updated["playback_youtube_timestamp_url"]
    return updated


def extract_local_anchor_passage(
    question: str,
    evidence_text: str,
    *,
    window_size: int = 56,
    stride: int = 18,
) -> dict[str, Any]:
    words = evidence_text.split()
    if not words:
        return {"text": "", "score": 0.0, "overlap_count": 0}

    question_tokens = set(dedupe_tokens(refinement_tokens(question)))
    if not question_tokens or len(words) <= window_size:
        text = " ".join(words)
        overlap = query_overlap_count(refinement_tokens(text), list(question_tokens))
        return {"text": text, "score": float(overlap), "overlap_count": overlap}

    best: dict[str, Any] | None = None
    for start_index in range(0, len(words), stride):
        window_words = words[start_index : start_index + window_size]
        if not window_words:
            continue
        text = " ".join(window_words)
        tokens = refinement_tokens(text)
        token_set = set(tokens)
        overlap_count = len(token_set & question_tokens)
        overlap_ratio = overlap_count / max(1, len(question_tokens))
        density = overlap_count / max(1, len(token_set))
        score = overlap_count + overlap_ratio + 0.5 * density
        candidate = {
            "text": text,
            "score": score,
            "overlap_count": overlap_count,
            "start_word_index": start_index,
            "end_word_index": start_index + len(window_words),
        }
        if best is None or (
            candidate["score"],
            -candidate["start_word_index"],
        ) > (
            best["score"],
            -best["start_word_index"],
        ):
            best = candidate
        if start_index + window_size >= len(words):
            break

    if best is None or float(best["score"]) <= 0.0:
        text = " ".join(words[:window_size])
        return {"text": text, "score": 0.0, "overlap_count": 0, "start_word_index": 0, "end_word_index": len(text.split())}
    return best


def ranked_playback_alignment_windows(
    anchor_tokens: list[str],
    windows: list[dict[str, Any]],
    target_time_sec: float,
) -> list[dict[str, Any]]:
    if not anchor_tokens or not windows:
        return []
    idf = refinement_idf(windows)
    average_length = sum(int(window["length"]) for window in windows) / max(1, len(windows))
    anchor_terms = set(anchor_tokens)
    ranked: list[dict[str, Any]] = []
    for window in windows:
        overlap_count = query_overlap_count(window["tokens"], anchor_tokens)
        if overlap_count <= 0:
            continue
        similarity_score = bm25_score(anchor_tokens, window, idf, average_length)
        if similarity_score <= 0:
            continue
        midpoint = (float(window["start_sec"]) + float(window["end_sec"])) / 2.0
        candidate_terms = set(window["tokens"])
        candidate_match_ratio = overlap_count / max(1, len(candidate_terms))
        anchor_coverage_ratio = overlap_count / max(1, len(anchor_terms))
        candidate = dict(window)
        candidate["distance_sec"] = abs(midpoint - target_time_sec)
        candidate["similarity_score"] = similarity_score
        candidate["anchor_overlap_count"] = overlap_count
        candidate["candidate_match_ratio"] = candidate_match_ratio
        candidate["anchor_coverage_ratio"] = anchor_coverage_ratio
        ranked.append(candidate)
    ranked.sort(
        key=lambda item: (
            -float(item.get("similarity_score", 0.0)),
            float(item.get("start_sec", 0.0)),
        )
    )
    return ranked


def is_playback_alignment_eligible(
    candidate: dict[str, Any],
    *,
    original_start_sec: float,
    original_end_sec: float,
    similarity_threshold: float,
    min_overlap_count: int,
    min_candidate_match_ratio: float,
) -> bool:
    if float(candidate.get("start_sec", 0.0)) >= original_end_sec - 0.001:
        return False
    if float(candidate.get("end_sec", 0.0)) < original_start_sec - 120.0:
        return False
    if float(candidate.get("similarity_score", 0.0)) < similarity_threshold:
        return False
    if int(candidate.get("anchor_overlap_count", 0)) < min_overlap_count:
        return False
    if float(candidate.get("candidate_match_ratio", 0.0)) < min_candidate_match_ratio:
        return False
    return True


def earliest_playback_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            float(item.get("start_sec", 0.0)),
            -float(item.get("similarity_score", 0.0)),
        ),
    )[0]


def playback_alignment_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_name": item.get("source_name"),
        "transcript_path": item.get("transcript_path"),
        "start_sec": float(item["start_sec"]),
        "end_sec": float(item["end_sec"]),
        "timestamp": format_timestamp(float(item["start_sec"]), float(item["end_sec"])),
        "similarity_score": float(item.get("similarity_score", 0.0)),
        "anchor_overlap_count": int(item.get("anchor_overlap_count", 0)),
        "candidate_match_ratio": float(item.get("candidate_match_ratio", 0.0)),
        "anchor_coverage_ratio": float(item.get("anchor_coverage_ratio", 0.0)),
        "distance_sec": float(item.get("distance_sec", 0.0)),
        "text_preview": preview_text(str(item.get("text", "")), limit=240),
    }


def _set_original_playback_fields(item: dict[str, Any]) -> dict[str, Any]:
    item["playback_source_name"] = item.get("source_name")
    item["playback_transcript_path"] = item.get("transcript_path")
    item["playback_start_sec"] = item.get("start_sec")
    item["playback_end_sec"] = item.get("end_sec")
    item["playback_timestamp"] = item.get("timestamp")
    item["playback_snippet"] = item.get("transcript_snippet", item.get("text", ""))
    item["playback_youtube_url"] = item.get("youtube_url")
    item["playback_youtube_timestamp_url"] = item.get("youtube_timestamp_url")
    return item


def same_day_hour_transcript_files(day: str, hour_id: str) -> list[Path]:
    transcripts_dir = ROOT / "all_transcripts"
    return sorted(transcripts_dir.glob(f"{day}_*_{hour_id}.json"))


def build_alignment_windows(
    entries: list[dict[str, Any]],
    transcript_path: Path,
    day: str,
    hour_id: str,
    *,
    max_window_entries: int = 5,
    max_window_duration_sec: float = 30.0,
) -> list[dict[str, Any]]:
    source_name = source_name_from_transcript_path(transcript_path, day, hour_id)
    windows: list[dict[str, Any]] = []
    for start_index in range(len(entries)):
        for size in range(1, max_window_entries + 1):
            selected = entries[start_index : start_index + size]
            if len(selected) != size:
                continue
            start_sec = float(selected[0]["start_sec"])
            end_sec = float(selected[-1]["end_sec"])
            duration = end_sec - start_sec
            if duration <= 0 or duration > max_window_duration_sec:
                continue
            text = " ".join(str(entry["text"]).strip() for entry in selected if str(entry["text"]).strip())
            tokens = refinement_tokens(text)
            if not tokens:
                continue
            windows.append(
                {
                    "source_name": source_name,
                    "day": day,
                    "hour_id": hour_id,
                    "transcript_path": str(transcript_path),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "text": text,
                    "tokens": tokens,
                    "counts": Counter(tokens),
                    "length": len(tokens),
                }
            )
    return windows


def ranked_alignment_windows(
    question: str,
    windows: list[dict[str, Any]],
    broad_evidence_text: str,
    target_time_sec: float,
) -> list[dict[str, Any]]:
    query_tokens = refinement_tokens(question)
    broad_tokens = dedupe_tokens(refinement_tokens(broad_evidence_text))
    if not windows or (not query_tokens and not broad_tokens):
        return []
    idf = refinement_idf(windows)
    average_length = sum(int(window["length"]) for window in windows) / max(1, len(windows))
    broad_terms = set(broad_tokens)
    ranked = []
    for window in windows:
        midpoint = (float(window["start_sec"]) + float(window["end_sec"])) / 2.0
        distance_sec = abs(midpoint - target_time_sec)
        temporal_score = 1.0 / (1.0 + distance_sec / 30.0)
        query_score = bm25_score(query_tokens, window, idf, average_length) if query_tokens else 0.0
        broad_score = bm25_score(broad_tokens, window, idf, average_length) if broad_tokens else 0.0
        density_score = content_density_score(window["tokens"], broad_terms)
        final_score = query_score + 0.7 * broad_score + 0.1 * density_score + 0.5 * temporal_score
        if final_score <= 0:
            continue
        candidate = dict(window)
        candidate["distance_sec"] = distance_sec
        candidate["temporal_score"] = temporal_score
        candidate["query_score"] = query_score
        candidate["query_overlap_count"] = query_overlap_count(window["tokens"], query_tokens)
        candidate["broad_score"] = broad_score
        candidate["density_score"] = density_score
        candidate["final_score"] = final_score
        ranked.append(candidate)
    ranked.sort(key=lambda item: float(item["final_score"]), reverse=True)
    return ranked


def alignment_candidate_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_name": item.get("source_name"),
        "transcript_path": item.get("transcript_path"),
        "start_sec": float(item["start_sec"]),
        "end_sec": float(item["end_sec"]),
        "timestamp": format_timestamp(float(item["start_sec"]), float(item["end_sec"])),
        "duration_sec": float(item["end_sec"]) - float(item["start_sec"]),
        "distance_sec": float(item.get("distance_sec", 0.0)),
        "query_score": float(item.get("query_score", 0.0)),
        "query_overlap_count": int(item.get("query_overlap_count", 0)),
        "broad_score": float(item.get("broad_score", 0.0)),
        "temporal_score": float(item.get("temporal_score", 0.0)),
        "density_score": float(item.get("density_score", 0.0)),
        "final_score": float(item.get("final_score", 0.0)),
        "text_preview": preview_text(str(item.get("text", "")), limit=240),
    }


def source_name_from_transcript_path(path: Path, day: str, hour_id: str) -> str:
    stem = path.stem
    prefix = f"{day}_"
    suffix = f"_{hour_id}"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix) : -len(suffix)]
    return stem


def youtube_url_for(day: str, source_name: str, hour_id: str) -> str | None:
    mapping = load_youtube_video_map()
    return mapping.get(f"{day}/{source_name}/{hour_id}")


@lru_cache(maxsize=1)
def load_youtube_video_map() -> dict[str, str]:
    path = ROOT / "artifacts" / "youtube_video_map.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def score_text_for_alignment(question: str, text: str, broad_evidence_text: str) -> dict[str, Any]:
    tokens = refinement_tokens(text)
    if not tokens:
        return {"query_score": 0.0, "query_overlap_count": 0, "broad_score": 0.0, "final_score": 0.0}
    window = {"counts": Counter(tokens), "length": len(tokens), "tokens": tokens}
    query_tokens = refinement_tokens(question)
    broad_tokens = dedupe_tokens(refinement_tokens(broad_evidence_text))
    windows = [{"tokens": tokens, "counts": Counter(tokens), "length": len(tokens)}]
    idf = refinement_idf(windows)
    average_length = len(tokens)
    query_score = bm25_score(query_tokens, window, idf, average_length) if query_tokens else 0.0
    broad_score = bm25_score(broad_tokens, window, idf, average_length) if broad_tokens else 0.0
    final_score = query_score + 0.7 * broad_score
    return {
        "query_score": query_score,
        "query_overlap_count": query_overlap_count(tokens, query_tokens),
        "broad_score": broad_score,
        "final_score": final_score,
    }


def query_overlap_count(candidate_tokens: list[str], query_tokens: list[str]) -> int:
    return len(set(candidate_tokens) & set(query_tokens))


def eligible_alignment_candidates(candidates: list[dict[str, Any]], original: dict[str, Any]) -> list[dict[str, Any]]:
    eligible = []
    original_strong = original_has_strong_query_support(original)
    original_end = float(original.get("original_end_sec", original.get("end_sec", 0.0)) or 0.0)
    search_radius = float(original.get("alignment_search_radius_sec", 120.0) or 120.0)
    for candidate in candidates:
        query_score = float(candidate.get("query_score", 0.0))
        overlap = int(candidate.get("query_overlap_count", 0))
        if overlap < 2 and query_score < 1.0:
            continue
        if float(candidate.get("distance_sec", search_radius + 1.0)) > search_radius:
            continue
        if original_strong and float(candidate.get("start_sec", 0.0)) >= original_end - 0.001:
            continue
        eligible.append(candidate)
    return eligible


def earliest_alignment_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            float(item.get("start_sec", 0.0)),
            -float(item.get("query_score", 0.0)),
            -float(item.get("final_score", 0.0)),
        ),
    )[0]


def should_override_with_alignment(item: dict[str, Any]) -> tuple[bool, str]:
    aligned_query_score = float(item.get("aligned_query_score", 0.0))
    aligned_overlap = int(item.get("aligned_query_overlap_count", 0))
    aligned_final_score = float(item.get("aligned_score", 0.0))
    original_query_score = float(item.get("original_query_score", 0.0))
    original_overlap = int(item.get("original_query_overlap_count", 0))
    original_baseline = float(item.get("original_alignment_score", original_query_score))
    original_duration = float(item.get("original_duration_sec", 0.0) or 0.0)
    aligned_duration = float(item.get("aligned_end_sec", 0.0) or 0.0) - float(item.get("aligned_start_sec", 0.0) or 0.0)

    if aligned_query_score <= 0.0 and aligned_overlap < 2:
        return False, "aligned candidate lacks direct query support"
    if original_has_strong_query_support(item):
        if float(item.get("aligned_start_sec", 0.0) or 0.0) >= float(item.get("original_end_sec", 0.0) or 0.0) - 0.001:
            return False, "original evidence is strong and aligned candidate is not earlier than the original broad end"
        comparable_query_support = (
            aligned_query_score >= 0.5 * original_query_score
            or aligned_overlap >= min(max(original_overlap, 2), 4)
        )
        if not comparable_query_support:
            return False, "original evidence is strong and aligned candidate has weaker query support"
        more_precise = original_duration <= 0.0 or aligned_duration <= 0.5 * original_duration
        if not more_precise:
            return False, "original evidence is strong and aligned candidate is not more precise"
        return True, "aligned candidate is more precise and has comparable query support"
    if aligned_final_score <= original_baseline * 1.10:
        return False, "aligned candidate does not improve over original/local evidence by 10 percent"
    return True, "aligned candidate has query support and improves over original/local evidence"


def original_has_strong_query_support(item: dict[str, Any]) -> bool:
    return float(item.get("original_query_score", 0.0)) >= 1.0 or int(item.get("original_query_overlap_count", 0)) >= 2


def update_main_transcript_fields_from_alignment(item: dict[str, Any]) -> None:
    item["source_name"] = item.get("aligned_source_name")
    item["transcript_path"] = item.get("aligned_transcript_path")
    item["start_sec"] = item.get("aligned_start_sec")
    item["end_sec"] = item.get("aligned_end_sec")
    item["timestamp"] = item.get("aligned_timestamp")
    item["transcript_snippet"] = item.get("aligned_snippet")
    item["youtube_url"] = item.get("aligned_youtube_url")
    item["youtube_timestamp_url"] = item.get("aligned_youtube_timestamp_url")


def apply_final_transcript_fields(item: dict[str, Any], *, aligned: bool) -> None:
    if aligned and item.get("aligned_start_sec") is not None:
        item["final_source_name"] = item.get("aligned_source_name")
        item["final_transcript_path"] = item.get("aligned_transcript_path")
        item["final_start_sec"] = item.get("aligned_start_sec")
        item["final_end_sec"] = item.get("aligned_end_sec")
        item["final_timestamp"] = item.get("aligned_timestamp")
        item["final_transcript_snippet"] = item.get("aligned_snippet")
        item["final_youtube_timestamp_url"] = item.get("aligned_youtube_timestamp_url")
        item["final_selection_method"] = "cross_pov_alignment"
        return

    if item.get("refined_start_sec") is not None:
        item["final_source_name"] = item.get("source_name")
        item["final_transcript_path"] = item.get("transcript_path")
        item["final_start_sec"] = item.get("refined_start_sec")
        item["final_end_sec"] = item.get("refined_end_sec")
        item["final_timestamp"] = item.get("refined_timestamp")
        item["final_transcript_snippet"] = item.get("refined_transcript_snippet")
        item["final_youtube_timestamp_url"] = item.get("refined_youtube_timestamp_url")
        item["final_selection_method"] = "local_timestamp_refinement"
        return

    item["final_source_name"] = item.get("source_name")
    item["final_transcript_path"] = item.get("transcript_path")
    item["final_start_sec"] = item.get("start_sec")
    item["final_end_sec"] = item.get("end_sec")
    item["final_timestamp"] = item.get("timestamp")
    item["final_transcript_snippet"] = item.get("transcript_snippet", item.get("text"))
    item["final_youtube_timestamp_url"] = item.get("youtube_timestamp_url")
    item["final_selection_method"] = "original_retrieved_chunk"


def overlapping_transcript_entries(
    entries: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> list[dict[str, Any]]:
    return [
        entry
        for entry in entries
        if float(entry["end_sec"]) >= start_sec - 0.001 and float(entry["start_sec"]) <= end_sec + 0.001
    ]


def transcript_extraction_debug(
    *,
    transcript_path: Path,
    broad_start: float,
    broad_end: float,
    all_entries: tuple[dict[str, Any], ...],
    overlapping_entries: list[dict[str, Any]],
    fully_inside_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "transcript_path": str(transcript_path),
        "broad_start_sec": broad_start,
        "broad_end_sec": broad_end,
        "loaded_entries": len(all_entries),
        "overlapping_entries": len(overlapping_entries),
        "fully_inside_entries": len(fully_inside_entries),
        "short_entries_duration_le_30s": count_short_entries(overlapping_entries),
        "first_20_overlapping_entries": summarize_entries(overlapping_entries[:20]),
    }


def count_short_entries(entries: list[dict[str, Any]]) -> int:
    return sum(1 for entry in entries if 0 < float(entry["end_sec"]) - float(entry["start_sec"]) <= 30.0)


def summarize_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start_sec": float(entry["start_sec"]),
            "end_sec": float(entry["end_sec"]),
            "timestamp": format_timestamp(float(entry["start_sec"]), float(entry["end_sec"])),
            "duration_sec": float(entry["end_sec"]) - float(entry["start_sec"]),
            "text_preview": preview_text(str(entry.get("text", "")), limit=180),
        }
        for entry in entries
    ]


def resolve_transcript_path(path_value: Any) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    return None


@lru_cache(maxsize=256)
def load_transcript_entries(transcript_path: str) -> tuple[dict[str, Any], ...]:
    with Path(transcript_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("transcript JSON has no chunks list")

    entries = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        timestamp = chunk.get("timestamp")
        if not isinstance(timestamp, list | tuple) or len(timestamp) != 2:
            continue
        try:
            start_sec = float(timestamp[0])
            end_sec = float(timestamp[1])
        except (TypeError, ValueError):
            continue
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        entries.append({"start_sec": start_sec, "end_sec": end_sec, "text": text})
    return tuple(entries)


def build_refinement_windows(
    entries: list[dict[str, Any]],
    *,
    broad_start_sec: float,
    broad_end_sec: float,
    max_window_entries: int = 5,
    max_window_duration_sec: float = 30.0,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    broad_duration = broad_end_sec - broad_start_sec
    for start_index in range(len(entries)):
        for size in range(1, max_window_entries + 1):
            selected = entries[start_index : start_index + size]
            if len(selected) != size:
                continue
            start_sec = float(selected[0]["start_sec"])
            end_sec = float(selected[-1]["end_sec"])
            duration = end_sec - start_sec
            if duration <= 0 or duration > max_window_duration_sec:
                continue
            if abs(start_sec - broad_start_sec) < 0.001 and abs(end_sec - broad_end_sec) < 0.001:
                continue
            if broad_duration > 0 and duration >= 0.8 * broad_duration:
                continue
            text = " ".join(str(entry["text"]).strip() for entry in selected if str(entry["text"]).strip())
            tokens = refinement_tokens(text)
            if not tokens:
                continue
            windows.append(
                {
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "text": text,
                    "tokens": tokens,
                    "counts": Counter(tokens),
                    "length": len(tokens),
                }
            )
    return windows


def best_refinement_window(
    question: str,
    windows: list[dict[str, Any]],
    broad_evidence_text: str = "",
) -> dict[str, Any] | None:
    ranked = ranked_refinement_windows(question, windows, broad_evidence_text)
    return ranked[0] if ranked else None


def ranked_refinement_windows(
    question: str,
    windows: list[dict[str, Any]],
    broad_evidence_text: str = "",
) -> list[dict[str, Any]]:
    query_tokens = refinement_tokens(question)
    broad_tokens = dedupe_tokens(refinement_tokens(broad_evidence_text))
    if not windows or (not query_tokens and not broad_tokens):
        return []
    idf = refinement_idf(windows)
    average_length = sum(int(window["length"]) for window in windows) / max(1, len(windows))
    broad_terms = set(broad_tokens)
    ranked = []
    for window in windows:
        query_score = bm25_score(query_tokens, window, idf, average_length) if query_tokens else 0.0
        broad_score = bm25_score(broad_tokens, window, idf, average_length) if broad_tokens else 0.0
        density_score = content_density_score(window["tokens"], broad_terms)
        final_score = query_score + 0.7 * broad_score + 0.1 * density_score
        if final_score > 0:
            candidate = dict(window)
            candidate["query_score"] = query_score
            candidate["broad_score"] = broad_score
            candidate["density_score"] = density_score
            candidate["final_score"] = final_score
            candidate["score"] = final_score
            ranked.append(candidate)
    if not ranked:
        return []
    ranked.sort(key=lambda item: float(item["final_score"]), reverse=True)
    return ranked


def content_density_score(window_tokens: list[str], broad_terms: set[str]) -> float:
    if not window_tokens or not broad_terms:
        return 0.0
    overlap = sum(1 for token in window_tokens if token in broad_terms)
    return overlap / len(window_tokens)


def dedupe_tokens(tokens: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def refinement_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in REFINEMENT_STOPWORDS and len(token) > 1]


def refinement_idf(windows: list[dict[str, Any]]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for window in windows:
        document_frequency.update(set(window["tokens"]))
    document_count = max(1, len(windows))
    return {
        token: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
        for token, frequency in document_frequency.items()
    }


def _with_refinement_failure(result: dict[str, Any], reason: str) -> dict[str, Any]:
    updated = dict(result)
    try:
        updated["broad_start_sec"] = float(updated["start_sec"])
        updated["broad_end_sec"] = float(updated["end_sec"])
        updated["broad_timestamp"] = format_timestamp(float(updated["start_sec"]), float(updated["end_sec"]))
    except (KeyError, TypeError, ValueError):
        pass
    updated["broad_transcript_snippet"] = updated.get("transcript_snippet", preview_text(str(updated.get("text", ""))))
    updated["broad_youtube_timestamp_url"] = updated.get("youtube_timestamp_url") or youtube_timestamp_url(updated)
    updated["timestamp_refinement_method"] = "unavailable"
    updated["timestamp_refinement_error"] = reason
    return updated


def _finalize_result(item: dict[str, Any]) -> dict[str, Any]:
    finalized = dict(item)
    finalized["evidence_type"] = "transcript"
    finalized["hour_id"] = finalized.get("hour_id", finalized.get("video_id"))
    finalized["timestamp"] = format_timestamp(float(finalized["start_sec"]), float(finalized["end_sec"]))
    finalized["youtube_timestamp_url"] = youtube_timestamp_url(finalized)
    finalized["transcript_snippet"] = preview_text(str(finalized.get("text", "")))
    return finalized


@lru_cache(maxsize=2)
def _load_dense_resources(index_dir: str):
    index, metadata, model_name = load_index(index_dir)
    model = load_embedding_model(model_name)
    return index, metadata, model_name, model


@lru_cache(maxsize=2)
def _load_lexical_resources(index_dir: str):
    _index, metadata, _model_name = load_index(index_dir)
    windows = build_passage_windows(metadata)
    document_frequency: Counter[str] = Counter()
    total_length = 0
    for window in windows:
        total_length += int(window["length"])
        document_frequency.update(window["counts"].keys())
    window_count = max(1, len(windows))
    average_window_length = total_length / window_count
    idf = {
        token: math.log(1.0 + (window_count - frequency + 0.5) / (frequency + 0.5))
        for token, frequency in document_frequency.items()
    }
    return metadata, windows, idf, average_window_length


def build_passage_windows(
    metadata: list[dict[str, Any]],
    *,
    window_size: int = 80,
    stride: int = 40,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for metadata_index, item in enumerate(metadata):
        tokens = tokenize(str(item.get("text", "")))
        if not tokens:
            continue
        starts = [0] if len(tokens) <= window_size else list(range(0, len(tokens), stride))
        for start in starts:
            window_tokens = tokens[start : start + window_size]
            if not window_tokens:
                continue
            windows.append(
                {
                    "metadata_index": metadata_index,
                    "tokens": window_tokens,
                    "counts": Counter(window_tokens),
                    "length": len(window_tokens),
                    "text": " ".join(window_tokens),
                }
            )
            if start + window_size >= len(tokens):
                break
    return windows


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None

    candidates = []
    env_path = os.environ.get("TRANSCRIPT_CROSS_ENCODER_MODEL")
    if env_path:
        candidates.append(Path(env_path))
    explicit_path = Path(model_name)
    if explicit_path.is_dir():
        candidates.append(explicit_path)
    candidates.append(Path("/scratch-shared/group_h/models") / model_name.split("/")[-1])

    for path in candidates:
        if not path.is_dir():
            continue
        try:
            return CrossEncoder(
                str(path),
                model_kwargs={"local_files_only": True},
                processor_kwargs={"local_files_only": True},
            )
        except Exception:
            continue
    return None
