#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clip_retrieval import embed_texts_clip_profile, load_clip_model, load_clip_text_model  # noqa: E402
from src.answer_span_highlight import find_answer_span  # noqa: E402
from src.evidence_links import youtube_timestamp_url  # noqa: E402
from src.evidence_router import route_evidence  # noqa: E402
from src.mixed_evidence_ranker import build_mixed_evidence_list  # noqa: E402
from src.retriever import require_faiss  # noqa: E402
from src.retriever import load_embedding_model, load_index, query_index  # noqa: E402
from src.transcript_keyword_recommender import recommend_transcript_keywords  # noqa: E402
from src.transcript_heatmap import build_transcript_heatmap  # noqa: E402
from src.transcript_answer_reranker import rerank_transcript_answers  # noqa: E402
from src.transcript_reasoning_answer import reason_over_transcript_candidates  # noqa: E402
from src.transcript_retrieval import retrieve_transcript_evidence  # noqa: E402
from src.vqa import format_timestamp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Route a question to one visual or transcript evidence result.")
    parser.add_argument("question")
    parser.add_argument("--visual-index-dir", default="artifacts/siglip_index_day1")
    parser.add_argument("--transcript-index-dir", default="artifacts/transcript_index_day1")
    parser.add_argument("--text-model-name", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--visual-query-variants", action="store_true", default=True)
    parser.add_argument("--no-visual-query-variants", dest="visual_query_variants", action="store_false")
    parser.add_argument("--synonyms-file", default=None)
    parser.add_argument("--diversity-window-sec", type=float, default=30.0)
    parser.add_argument("--candidate-multiplier", type=int, default=5)
    parser.add_argument("--grounding", choices=("none", "dino", "dino-siglip-rerank"), default="none")
    parser.add_argument("--grounding-threshold", default="0.25,0.15,0.10,0.05")
    parser.add_argument("--grounding-prompts", default="")
    parser.add_argument("--grounding-max-boxes", type=int, default=5)
    parser.add_argument("--grounding-alpha", type=float, default=1.0)
    parser.add_argument("--grounding-beta", type=float, default=1.0)
    parser.add_argument("--grounding-gamma", type=float, default=1.0)
    parser.add_argument("--debug-router", action="store_true")
    parser.add_argument("--include-transcript-heatmap", action="store_true")
    parser.add_argument("--transcript-answer-rerank-top-n", type=int, default=0)
    parser.add_argument("--transcript-reasoning-top-n", type=int, default=30)
    parser.add_argument("--reasoning-mode", choices=("global", "per_candidate"), default="per_candidate")
    parser.add_argument("--keyword-extraction-top-n", type=int, default=20)
    parser.add_argument("--suggest-steering-terms", action="store_true")
    parser.add_argument("--steering-terms", default="")
    parser.add_argument("--mixed-top-k", type=int, default=0)
    parser.add_argument("--mixed-calibration", choices=("max", "percentile"), default="max")
    parser.add_argument("--debug-diversity", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    retrieval_top_k = max(
        args.top_k,
        args.mixed_top_k,
        args.transcript_answer_rerank_top_n,
        args.transcript_reasoning_top_n,
        args.keyword_extraction_top_n,
    )
    args.retrieval_top_k = retrieval_top_k
    visual_results, visual_runtime = retrieve_visual(args)
    steering_terms = parse_steering_terms(args.steering_terms)
    transcript_results = retrieve_transcript_with_steering(
        args.question,
        args.transcript_index_dir,
        retrieval_top_k,
        steering_terms,
    )
    add_relative_retrieval_confidence(visual_results, "visual")
    add_relative_retrieval_confidence(transcript_results, "transcript")
    transcript_reasoning = run_transcript_reasoning(
        args.question,
        transcript_results,
        args.transcript_reasoning_top_n,
        args.reasoning_mode,
    )
    reasoning_ordered_transcript_results = apply_reasoning_order_to_transcript_results(
        transcript_results,
        transcript_reasoning,
    )
    keyword_candidates, keyword_candidates_source = select_keyword_candidates(
        transcript_results,
        reasoning_ordered_transcript_results,
        transcript_reasoning,
        limit=20,
    )
    steering_suggestions = recommend_transcript_keywords(
        args.question,
        keyword_candidates,
        top_n=len(keyword_candidates),
    )
    annotate_keyword_candidate_source(
        steering_suggestions,
        transcript_reasoning,
        keyword_candidates,
        keyword_candidates_source,
    )
    chosen = route_evidence(args.question, visual_results, transcript_results)
    chosen["steering_suggestions"] = steering_suggestions

    if chosen["evidence_type"] == "visual" and args.grounding != "none":
        chosen = add_grounding(args, chosen, visual_results)
    if chosen["evidence_type"] == "transcript" and args.transcript_answer_rerank_top_n > 0:
        chosen = add_transcript_answer_rerank(args.question, chosen, transcript_results, args.transcript_answer_rerank_top_n)
    if chosen["evidence_type"] == "transcript" and args.transcript_reasoning_top_n > 0:
        chosen = add_transcript_reasoning_answer(
            args.question,
            chosen,
            transcript_results,
            args.transcript_reasoning_top_n,
            reasoning=transcript_reasoning,
            reasoning_mode=args.reasoning_mode,
        )
    chosen.setdefault("steering_suggestions", steering_suggestions)
    if chosen["evidence_type"] == "transcript" and (args.include_transcript_heatmap or args.debug_router):
        chosen = add_transcript_heatmap(args.question, chosen)
    chosen = compute_evidence_confidence(chosen)
    mixed_evidence = []
    if args.mixed_top_k:
        visual_for_mixed = merge_chosen_into_candidates(chosen, visual_results, "visual")
        transcript_candidates_for_mixed = reasoning_ordered_transcript_results[:10]
        transcript_for_mixed = update_existing_chosen_candidate(chosen, transcript_candidates_for_mixed, "transcript")
        mixed_result = build_mixed_evidence_list(
            args.question,
            visual_for_mixed,
            transcript_for_mixed,
            chosen.get("router_debug", {}),
            top_k=args.mixed_top_k,
            calibration_mode=args.mixed_calibration,
            return_debug=args.debug_diversity,
        )
        if args.debug_diversity:
            mixed_evidence = mixed_result["items"]
            mixed_diversity_debug = mixed_result["diversity_debug"]
        else:
            mixed_evidence = mixed_result

    print(f"Question: {args.question}")
    print(f"Evidence type: {chosen['evidence_type']}")
    print_steering_suggestions(chosen.get("steering_suggestions") or steering_suggestions)
    if steering_terms:
        print_steering_terms_used(args.question, steering_terms, transcript_results)
    if args.debug_router:
        debug = chosen.get("router_debug", {})
        print("\nROUTER DEBUG")
        print(f"heuristic visual score: {debug.get('heuristic_visual_score')}")
        print(f"heuristic transcript score: {debug.get('heuristic_transcript_score')}")
        print(f"top visual score: {debug.get('top_visual_score')}")
        print(f"top transcript score: {debug.get('top_transcript_score')}")
        print(f"combined visual score: {debug.get('combined_visual_score')}")
        print(f"combined transcript score: {debug.get('combined_transcript_score')}")
        print(f"router margin: {debug.get('router_margin')}")
        print(f"router confidence: {debug.get('router_confidence')}")
        print(f"router confidence percent: {debug.get('router_confidence_percent')}")
        print(f"router second choice: {debug.get('router_second_choice')}")
        print(f"chosen route: {debug.get('chosen_route')}")
        print(f"reason: {debug.get('reason')}")
        print(f"visual retrieval runtime: {visual_runtime:.3f}s")
        keyword_debug = (chosen.get("steering_suggestions") or {}).get("keyword_recommender_debug")
        if keyword_debug is not None:
            print("Keyword recommender debug JSON:")
            print(json.dumps(keyword_debug, ensure_ascii=False))

    if chosen["evidence_type"] == "transcript":
        print_transcript_evidence(chosen)
    else:
        print_visual_evidence(chosen)
    if mixed_evidence:
        print_mixed_evidence(mixed_evidence)
        if args.debug_diversity:
            print_diversity_debug(mixed_diversity_debug)


def retrieve_visual(args: argparse.Namespace) -> tuple[list[dict[str, Any]], float]:
    import time

    start = time.perf_counter()
    metadata, model_name = load_metadata(args.visual_index_dir)
    index = load_faiss_index(args.visual_index_dir)
    query_model_name = resolve_query_model_name(model_name, args.text_model_name, "none")
    model, processor, torch = load_clip_text_model(query_model_name, local_files_only=not args.allow_download)

    variants = (
        query_variants(args.question, load_synonym_map(args.synonyms_file))
        if args.visual_query_variants
        else [args.question]
    )
    result_top_k = int(getattr(args, "retrieval_top_k", args.top_k))
    search_k = min(max(result_top_k, result_top_k * max(1, args.candidate_multiplier)), len(metadata))
    embeddings, _profile = embed_texts_clip_profile(variants, model=model, processor=processor, torch=torch)
    scores, ids = index.search(embeddings, search_k)
    if args.visual_query_variants:
        variant_results = collect_variant_results(scores, ids, metadata, variants, result_top_k, args.diversity_window_sec)
        results = merge_variant_results(variant_results, result_top_k, args.diversity_window_sec)
    else:
        results = collect_results(scores, ids, metadata)[:result_top_k]
    for result in results:
        result["evidence_type"] = "visual"
        result["timestamp"] = format_timestamp(float(result["start_sec"]), float(result["end_sec"]))
        result["youtube_timestamp_url"] = youtube_timestamp_url(result)
    return results, time.perf_counter() - start


def retrieve_transcript(question: str, index_dir: str, top_k: int) -> list[dict[str, Any]]:
    return retrieve_transcript_evidence(question, index_dir, top_k=top_k, align_playback=True)


def retrieve_transcript_with_steering(
    question: str,
    index_dir: str,
    top_k: int,
    steering_terms: list[str],
) -> list[dict[str, Any]]:
    original = retrieve_transcript(question, index_dir, top_k)
    for item in original:
        item["retrieval_query_source"] = "original"
        item["matched_steering_term"] = None
        item["steering_queries_used"] = [question]
    if not steering_terms:
        return original

    merged: dict[str, dict[str, Any]] = {evidence_key(item): dict(item) for item in original}
    query_contributions = [{"query": question, "term": None, "count": len(original), "source": "original"}]
    for term in steering_terms:
        queries = steering_queries(question, term)
        for query in queries:
            results = retrieve_transcript(query, index_dir, top_k)
            query_contributions.append({"query": query, "term": term, "count": len(results), "source": "steering"})
            for item in results:
                key = evidence_key(item)
                existing = merged.get(key)
                if existing is None or safe_float(item.get("score"), 0.0) > safe_float(existing.get("score"), 0.0):
                    updated = dict(item)
                    updated["retrieval_query_source"] = "steering"
                    updated["matched_steering_term"] = term
                    updated["steering_queries_used"] = [query]
                    merged[key] = updated
                else:
                    existing.setdefault("steering_queries_used", [])
                    if query not in existing["steering_queries_used"]:
                        existing["steering_queries_used"].append(query)
                    if existing.get("retrieval_query_source") != "original":
                        existing["matched_steering_term"] = existing.get("matched_steering_term") or term
    ranked = sorted(merged.values(), key=lambda item: safe_float(item.get("score"), 0.0), reverse=True)
    for item in ranked:
        item["steering_terms_used"] = steering_terms
        item["steering_query_contributions"] = query_contributions
    return ranked[:top_k]


def steering_queries(question: str, term: str) -> list[str]:
    queries = [f"{question} {term}".strip()]
    if " " in term.strip():
        queries.append(term.strip())
    return dedupe_strings(queries)


def parse_steering_terms(value: str) -> list[str]:
    return dedupe_strings([term.strip() for term in value.split(",") if term.strip()])


def run_transcript_reasoning(
    question: str,
    transcript_results: list[dict[str, Any]],
    top_n: int,
    reasoning_mode: str,
) -> dict[str, Any]:
    if top_n <= 0:
        return {
            "reasoning_ordered_transcript_candidates": [],
            "reasoning_order_used": False,
            "reason": "transcript reasoning disabled",
        }
    return call_reason_over_transcript_candidates(question, transcript_results, top_n, reasoning_mode)


def call_reason_over_transcript_candidates(
    question: str,
    transcript_results: list[dict[str, Any]],
    top_n: int,
    reasoning_mode: str,
) -> dict[str, Any]:
    try:
        return reason_over_transcript_candidates(
            question,
            transcript_results,
            top_n=top_n,
            reasoning_mode=reasoning_mode,
        )
    except TypeError as exc:
        if "reasoning_mode" not in str(exc):
            raise
        return reason_over_transcript_candidates(question, transcript_results, top_n=top_n)


def apply_reasoning_order_to_transcript_results(
    transcript_results: list[dict[str, Any]],
    reasoning: dict[str, Any],
) -> list[dict[str, Any]]:
    ordered_specs = reasoning.get("reasoning_ordered_transcript_candidates") or []
    if not reasoning.get("reasoning_order_used") or not ordered_specs:
        return annotate_raw_transcript_order(transcript_results, reasoning_order_used=False)

    by_key = {evidence_key(item): item for item in transcript_results}
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fallback_rank, spec in enumerate(ordered_specs, start=1):
        key = str(spec.get("candidate_id") or f"{spec.get('transcript_path')}:{spec.get('start_sec')}:{spec.get('end_sec')}")
        item = by_key.get(key)
        if item is None or key in seen:
            continue
        updated = dict(item)
        updated["raw_retrieval_rank"] = int(spec.get("raw_retrieval_rank") or spec.get("candidate_index") or fallback_rank)
        updated["reasoning_rank"] = int(spec.get("reasoning_rank") or fallback_rank)
        updated["reasoning_score"] = spec.get("reasoning_score")
        updated["reasoning_reason"] = spec.get("reasoning_reason") or spec.get("reason")
        updated["reasoning_order_used"] = True
        ordered.append(updated)
        seen.add(key)

    for raw_rank, item in enumerate(transcript_results, start=1):
        key = evidence_key(item)
        if key in seen:
            continue
        updated = dict(item)
        updated.setdefault("raw_retrieval_rank", raw_rank)
        updated["reasoning_rank"] = None
        updated["reasoning_score"] = None
        updated["reasoning_order_used"] = False
        ordered.append(updated)
    return ordered


def annotate_raw_transcript_order(transcript_results: list[dict[str, Any]], *, reasoning_order_used: bool) -> list[dict[str, Any]]:
    annotated = []
    for raw_rank, item in enumerate(transcript_results, start=1):
        updated = dict(item)
        updated["raw_retrieval_rank"] = raw_rank
        updated["reasoning_rank"] = raw_rank
        updated["reasoning_score"] = None
        updated["reasoning_order_used"] = reasoning_order_used
        annotated.append(updated)
    return annotated


def select_keyword_candidates(
    transcript_results: list[dict[str, Any]],
    reasoning_ordered_transcript_results: list[dict[str, Any]],
    reasoning: dict[str, Any],
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], str]:
    support_indices = parse_int_list(reasoning.get("supporting_candidate_indices"))
    if support_indices:
        by_raw_rank = {index: item for index, item in enumerate(transcript_results, start=1)}
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for reasoning_rank, raw_rank in enumerate(support_indices, start=1):
            item = by_raw_rank.get(raw_rank)
            if item is None:
                continue
            updated = dict(item)
            updated["raw_retrieval_rank"] = raw_rank
            updated["reasoning_rank"] = reasoning_rank
            updated["reasoning_order_used"] = True
            updated["keyword_supporting_evidence"] = True
            selected.append(updated)
            seen.add(evidence_key(updated))
        for item in reasoning_ordered_transcript_results:
            key = evidence_key(item)
            if key in seen:
                continue
            selected.append(item)
            seen.add(key)
            if len(selected) >= limit:
                break
        if selected:
            return selected[:limit], "supporting_evidence"

    if reasoning.get("reasoning_order_used"):
        return reasoning_ordered_transcript_results[:limit], "reasoning_ordered"
    return annotate_raw_transcript_order(transcript_results, reasoning_order_used=False)[:limit], "raw_retrieval_fallback"


def parse_int_list(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    parsed = []
    seen = set()
    for value in values:
        try:
            integer = int(value)
        except (TypeError, ValueError):
            continue
        if integer in seen:
            continue
        seen.add(integer)
        parsed.append(integer)
    return parsed


def annotate_keyword_candidate_source(
    steering_suggestions: dict[str, Any],
    reasoning: dict[str, Any],
    keyword_candidates: list[dict[str, Any]],
    keyword_candidates_source: str,
) -> None:
    debug = steering_suggestions.setdefault("keyword_recommender_debug", {})
    reasoning_used = bool(reasoning.get("reasoning_order_used"))
    debug["keyword_candidates_source"] = keyword_candidates_source
    debug["keyword_candidate_count"] = len(keyword_candidates)
    debug["reasoning_order_used"] = reasoning_used
    debug["selected_candidate_index"] = reasoning.get("selected_candidate_index")
    debug["answer_hypothesis"] = reasoning.get("answer_hypothesis", "")
    debug["supporting_candidate_indices"] = reasoning.get("supporting_candidate_indices", [])
    debug["reasoning_selected_candidate_id"] = reasoning.get("selected_candidate_id")
    debug["reasoning_ordered_candidate_indices"] = reasoning.get("reasoning_ordered_candidate_indices", [])
    debug["keyword_candidates_rank_debug"] = [
        {
            "raw_retrieval_rank": item.get("raw_retrieval_rank"),
            "reasoning_rank": item.get("reasoning_rank"),
            "reasoning_score": item.get("reasoning_score"),
            "source_name": item.get("source_name"),
            "timestamp": item.get("timestamp"),
            "transcript_path": item.get("transcript_path"),
            "text_snippet": str(item.get("transcript_snippet") or item.get("text") or "")[:300],
        }
        for item in keyword_candidates
    ]


def add_transcript_answer_rerank(
    question: str,
    chosen: dict[str, Any],
    transcript_results: list[dict[str, Any]],
    top_n: int,
    *,
    margin: float = 0.05,
) -> dict[str, Any]:
    rerank_result = rerank_transcript_answers(question, transcript_results, top_n=top_n)
    candidates = rerank_result.get("answer_rerank_candidates") or []
    best_transcript = rerank_result.get("best_transcript") or {}
    best_answer = rerank_result.get("best_answer") or {}
    original_key = evidence_key(chosen)
    original_candidate = next((item for item in candidates if evidence_key_from_candidate(item) == original_key), None)
    if original_candidate is None and candidates:
        original_candidate = candidates[0]
    best_candidate = candidates[0] if candidates else None
    original_score = safe_float((original_candidate or {}).get("final_answer_score"), 0.0)
    best_score = safe_float((best_candidate or {}).get("final_answer_score"), 0.0)
    replacement = bool(best_transcript and best_candidate and best_score > original_score + margin)

    updated = dict(best_transcript if replacement else chosen)
    updated["evidence_type"] = "transcript"
    updated["router_debug"] = chosen.get("router_debug", {})
    for key in (
        "router_confidence",
        "router_confidence_percent",
        "router_margin",
        "router_second_choice",
        "router_confidence_note",
    ):
        if key in chosen:
            updated[key] = chosen[key]
    if replacement:
        updated["answer_span"] = answer_span_from_rerank(best_answer)
        updated["answer_candidates"] = best_answer.get("answer_candidates", [])
    else:
        for key in ("answer_span", "answer_candidates"):
            if key in chosen:
                updated[key] = chosen[key]
    updated["transcript_answer_rerank"] = {
        "enabled": True,
        "top_n": top_n,
        "replacement": replacement,
        "margin": margin,
        "original_transcript": transcript_debug_summary(chosen, original_score),
        "reranked_best_transcript": transcript_debug_summary(best_transcript, best_score),
        "reason": (
            "replaced because reranked best final_answer_score exceeded original by margin"
            if replacement
            else "kept original because reranked best did not exceed original by margin"
        ),
        "answer_rerank_candidates": candidates,
    }
    return updated


def add_transcript_reasoning_answer(
    question: str,
    chosen: dict[str, Any],
    transcript_results: list[dict[str, Any]],
    top_n: int,
    *,
    confidence_threshold: float = 0.5,
    reasoning: dict[str, Any] | None = None,
    reasoning_mode: str = "global",
) -> dict[str, Any]:
    if reasoning is None:
        reasoning = call_reason_over_transcript_candidates(question, transcript_results, top_n, reasoning_mode)
    selected_id = reasoning.get("selected_candidate_id")
    confidence = safe_float(reasoning.get("confidence"), 0.0)
    selected = next((item for item in transcript_results if evidence_key(item) == selected_id), None)
    reasoning_succeeded = selected is not None and confidence > 0.0
    replacement = bool(selected is not None and confidence >= confidence_threshold)
    updated = dict(selected if replacement else chosen)
    updated["evidence_type"] = "transcript"
    updated["router_debug"] = chosen.get("router_debug", {})
    for key in (
        "router_confidence",
        "router_confidence_percent",
        "router_margin",
        "router_second_choice",
        "router_confidence_note",
    ):
        if key in chosen:
            updated[key] = chosen[key]
    reasoning_debug = {
        "enabled": True,
        "top_n": top_n,
        "reasoning_mode": reasoning.get("reasoning_mode", reasoning_mode),
        "selected_candidate_index": reasoning.get("selected_candidate_index"),
        "selected_candidate_id": selected_id,
        "reasoning_selected_retrieval_rank": reasoning.get("reasoning_selected_retrieval_rank")
        or reasoning.get("selected_candidate_index"),
        "selected_candidate_overridden": reasoning.get("selected_candidate_overridden", False),
        "original_selected_candidate_index": reasoning.get("original_selected_candidate_index"),
        "final_selected_candidate_index": reasoning.get("final_selected_candidate_index"),
        "selected_candidate_override_reason": reasoning.get("selected_candidate_override_reason", ""),
        "answer_hypothesis": reasoning.get("answer_hypothesis", ""),
        "supporting_candidate_indices": reasoning.get("supporting_candidate_indices", []),
        "num_supporting_candidate_indices_raw": reasoning.get("num_supporting_candidate_indices_raw"),
        "num_supporting_candidate_indices_final": reasoning.get("num_supporting_candidate_indices_final"),
        "num_reasoning_ordered_indices_raw": reasoning.get("num_reasoning_ordered_indices_raw"),
        "num_reasoning_ordered_indices_final": reasoning.get("num_reasoning_ordered_indices_final"),
        "raw_model_output_length": reasoning.get("raw_model_output_length"),
        "max_new_tokens": reasoning.get("max_new_tokens"),
        "generation_token_count": reasoning.get("generation_token_count"),
        "generation_hit_limit": reasoning.get("generation_hit_limit"),
        "raw_output_char_count": reasoning.get("raw_output_char_count"),
        "json_parse_failed": reasoning.get("json_parse_failed"),
        "output_truncated_suspected": reasoning.get("output_truncated_suspected"),
        "answer": reasoning.get("answer", ""),
        "evidence_sentence": reasoning.get("evidence_sentence", ""),
        "confidence": confidence,
        "reasoning_succeeded": reasoning_succeeded,
        "replacement": replacement,
        "reason": reasoning.get("reason", ""),
        "steering_keywords": reasoning.get("steering_keywords", []),
        "steering_paraphrases": [],
        "raw_model_output": reasoning.get("raw_model_output", ""),
        "raw_batch_scoring_output": reasoning.get("raw_batch_scoring_output", ""),
        "raw_batch_scoring_output_length": reasoning.get("raw_batch_scoring_output_length"),
        "batch_scoring_prompt_length_chars": reasoning.get("batch_scoring_prompt_length_chars"),
        "batch_scoring_output_length_chars": reasoning.get("batch_scoring_output_length_chars"),
        "batch_scoring_generation_hit_limit": reasoning.get("batch_scoring_generation_hit_limit"),
        "batch_scoring_num_candidates_requested": reasoning.get("batch_scoring_num_candidates_requested"),
        "batch_scoring_num_candidates_completed": reasoning.get("batch_scoring_num_candidates_completed"),
        "batch_scoring_json_parse_failed": reasoning.get("batch_scoring_json_parse_failed"),
        "batch_scoring_exception": reasoning.get("batch_scoring_exception"),
        "batch_scoring_candidate_count_returned": reasoning.get("batch_scoring_candidate_count_returned"),
        "batch_scoring_num_batches": reasoning.get("batch_scoring_num_batches"),
        "batch_scoring_batch_size": reasoning.get("batch_scoring_batch_size"),
        "batch_scoring_completed_batches": reasoning.get("batch_scoring_completed_batches"),
        "batch_scoring_completed_candidates": reasoning.get("batch_scoring_completed_candidates"),
        "model_name": reasoning.get("model_name", ""),
        "reasoning_candidates": reasoning.get("reasoning_candidates", []),
        "candidate_scores": reasoning.get("candidate_scores", []),
        "reasoning_order_used": reasoning.get("reasoning_order_used", False),
        "reasoning_ordered_transcript_candidates": reasoning.get("reasoning_ordered_transcript_candidates", []),
        "reasoning_ordered_candidate_indices": reasoning.get("reasoning_ordered_candidate_indices", []),
    }
    if selected is None:
        reasoning_debug["replacement_reason"] = "kept original because no valid selected candidate was returned"
    elif confidence < confidence_threshold:
        reasoning_debug["replacement_reason"] = "kept original because reasoning confidence was below threshold"
    else:
        reasoning_debug["replacement_reason"] = "replaced with reasoning-selected transcript candidate"
    updated["transcript_reasoning_answer"] = reasoning_debug
    updated["reasoning_answer"] = reasoning_debug["answer"]
    updated["reasoning_evidence_sentence"] = reasoning_debug["evidence_sentence"]
    updated["reasoning_confidence"] = confidence
    updated["reasoning_confidence_percent"] = confidence * 100.0
    if "steering_suggestions" in chosen:
        updated["steering_suggestions"] = chosen["steering_suggestions"]
    if replacement:
        updated["answer_span"] = answer_span_from_reasoning(updated, reasoning_debug)
        updated["answer_candidates"] = []
    else:
        for key in ("answer_span", "answer_candidates"):
            if key in chosen:
                updated[key] = chosen[key]
    return updated


def answer_span_from_reasoning(item: dict[str, Any], reasoning: dict[str, Any]) -> dict[str, Any]:
    transcript_text = str(item.get("text") or item.get("transcript_snippet") or "")
    evidence = str(reasoning.get("evidence_sentence") or reasoning.get("answer") or "")
    char_start, char_end = find_text_span(transcript_text, evidence)
    if char_start < 0 and reasoning.get("answer"):
        char_start, char_end = find_text_span(transcript_text, str(reasoning.get("answer")))
    return {
        "text": evidence or str(reasoning.get("answer") or ""),
        "char_start": char_start,
        "char_end": char_end,
        "score": reasoning.get("confidence", 0.0),
        "raw_score": None,
        "method": "transcript_reasoning_model",
        "answer_confidence": reasoning.get("confidence", 0.0),
        "answer_confidence_percent": safe_float(reasoning.get("confidence"), 0.0) * 100.0,
        "answer_confidence_note": "Reasoning-model answer confidence; not calibrated probability.",
    }


def find_text_span(text: str, needle: str) -> tuple[int, int]:
    if not text or not needle:
        return -1, -1
    index = text.find(needle)
    if index >= 0:
        return index, index + len(needle)
    normalized_text = re.sub(r"\s+", " ", text)
    normalized_needle = re.sub(r"\s+", " ", needle).strip()
    index = normalized_text.lower().find(normalized_needle.lower())
    if index < 0:
        return -1, -1
    original_index = text.lower().find(normalized_needle.lower())
    if original_index >= 0:
        return original_index, original_index + len(normalized_needle)
    return -1, -1


def evidence_key_from_candidate(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_key") or f"{candidate.get('transcript_path')}:{candidate.get('timestamp')}")


def transcript_debug_summary(item: dict[str, Any], final_answer_score: float) -> dict[str, Any]:
    return {
        "source_name": item.get("source_name"),
        "timestamp": item.get("timestamp"),
        "transcript_path": item.get("transcript_path"),
        "retrieval_score": item.get("score"),
        "final_answer_score": final_answer_score,
    }


def answer_span_from_rerank(best_answer: dict[str, Any]) -> dict[str, Any]:
    confidence = best_answer.get("confidence")
    return {
        "text": best_answer.get("text", ""),
        "char_start": best_answer.get("char_start", -1),
        "char_end": best_answer.get("char_end", -1),
        "score": confidence if confidence is not None else 0.0,
        "raw_score": best_answer.get("raw_score"),
        "method": best_answer.get("method", "unavailable"),
        "qa_context_method": best_answer.get("qa_context_method"),
        "qa_context_text": best_answer.get("qa_context_text"),
        "answer_confidence": confidence,
        "answer_confidence_percent": confidence * 100.0 if confidence is not None else None,
        "answer_confidence_note": "Extractive QA span confidence if available; not calibrated probability.",
    }


def add_relative_retrieval_confidence(results: list[dict[str, Any]], evidence_type: str) -> None:
    top_score = max((safe_float(item.get("score"), 0.0) for item in results), default=0.0)
    confidence_key = f"{evidence_type}_retrieval_confidence"
    percent_key = f"{evidence_type}_retrieval_confidence_percent"
    note_key = f"{evidence_type}_retrieval_confidence_note"
    note = f"Relative {evidence_type} retrieval confidence within {evidence_type} candidates; not calibrated probability."
    for item in results:
        score = safe_float(item.get("score"), 0.0)
        confidence = score / top_score if top_score > 0 else 0.0
        confidence = max(0.0, min(1.0, confidence))
        item[confidence_key] = confidence
        item[percent_key] = confidence * 100.0
        item[note_key] = note


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def answer_confidence_value(answer_span: dict[str, Any]) -> float | None:
    value = answer_span.get("score")
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= confidence <= 1.0:
        return confidence
    return None


def compute_evidence_confidence(item: dict[str, Any]) -> dict[str, Any]:
    updated = dict(item)
    if updated.get("evidence_type") == "visual":
        components = [
            updated.get("router_confidence"),
            updated.get("visual_retrieval_confidence"),
            updated.get("grounding_confidence"),
        ]
    else:
        answer_span = updated.get("answer_span") or {}
        components = [
            updated.get("router_confidence"),
            updated.get("transcript_retrieval_confidence"),
            answer_span.get("answer_confidence"),
            updated.get("reasoning_confidence"),
        ]
    confidence = geometric_mean(components)
    updated["evidence_confidence"] = confidence
    updated["evidence_confidence_percent"] = confidence * 100.0 if confidence is not None else None
    updated["evidence_confidence_note"] = (
        "Relative evidence-chain confidence combining router, retrieval, and answer/grounding confidence "
        "when available; not calibrated probability."
    )
    return updated


def geometric_mean(values: list[Any]) -> float | None:
    usable = []
    for value in values:
        parsed = safe_float(value, None)
        if parsed is not None and 0.0 <= parsed <= 1.0:
            usable.append(parsed)
    if not usable:
        return None
    product = 1.0
    for value in usable:
        product *= max(value, 1e-6)
    return product ** (1.0 / len(usable))


def add_transcript_heatmap(question: str, chosen: dict[str, Any]) -> dict[str, Any]:
    updated = dict(chosen)
    transcript_text = str(updated.get("text") or updated.get("transcript_snippet") or "")
    anchor_text = str((updated.get("playback_alignment_debug") or {}).get("anchor_passage") or "")
    if updated.get("answer_span"):
        answer_span = normalize_existing_answer_span(updated["answer_span"])
    else:
        raw_answer_span = find_answer_span(question, transcript_text, anchor_text=anchor_text or None)
        answer_span = {
            "text": raw_answer_span.get("answer_span_text", ""),
            "char_start": raw_answer_span.get("char_start", -1),
            "char_end": raw_answer_span.get("char_end", -1),
            "score": raw_answer_span.get("score", 0.0),
            "raw_score": raw_answer_span.get("raw_score"),
            "method": raw_answer_span.get("method", "unavailable"),
            "qa_context_method": raw_answer_span.get("qa_context_method"),
            "qa_context_text": raw_answer_span.get("qa_context_text"),
            "qa_context_original_char_start": raw_answer_span.get("qa_context_original_char_start"),
            "qa_context_original_char_end": raw_answer_span.get("qa_context_original_char_end"),
        }
        answer_confidence = answer_confidence_value(raw_answer_span)
        answer_span["answer_confidence"] = answer_confidence
        answer_span["answer_confidence_percent"] = answer_confidence * 100.0 if answer_confidence is not None else None
        answer_span["answer_confidence_note"] = "Extractive QA span confidence if available; not calibrated probability."
        updated["answer_candidates"] = raw_answer_span.get("answer_candidates", [])
    updated["answer_span"] = answer_span
    updated["transcript_heatmap"] = build_transcript_heatmap(
        question,
        transcript_text,
        anchor_text=anchor_text or None,
        answer_span={
            "answer_span_text": answer_span.get("text", ""),
            "char_start": answer_span.get("char_start", -1),
            "char_end": answer_span.get("char_end", -1),
        },
    )
    return updated


def normalize_existing_answer_span(answer_span: dict[str, Any]) -> dict[str, Any]:
    updated = dict(answer_span)
    confidence = updated.get("answer_confidence")
    if confidence is None:
        confidence = answer_confidence_value(updated)
        updated["answer_confidence"] = confidence
    updated["answer_confidence_percent"] = confidence * 100.0 if confidence is not None else None
    updated.setdefault("answer_confidence_note", "Extractive QA span confidence if available; not calibrated probability.")
    return updated


def merge_chosen_into_candidates(chosen: dict[str, Any], candidates: list[dict[str, Any]], evidence_type: str) -> list[dict[str, Any]]:
    if chosen.get("evidence_type") != evidence_type:
        return candidates
    key = evidence_key(chosen)
    merged = []
    inserted = False
    for item in candidates:
        if evidence_key(item) == key:
            updated = dict(item)
            updated.update(chosen)
            merged.append(updated)
            inserted = True
        else:
            merged.append(item)
    if not inserted:
        merged.insert(0, chosen)
    return merged


def update_existing_chosen_candidate(chosen: dict[str, Any], candidates: list[dict[str, Any]], evidence_type: str) -> list[dict[str, Any]]:
    if chosen.get("evidence_type") != evidence_type:
        return candidates
    key = evidence_key(chosen)
    updated_candidates = []
    for item in candidates:
        if evidence_key(item) == key:
            updated = dict(item)
            updated.update(chosen)
            updated_candidates.append(updated)
        else:
            updated_candidates.append(item)
    return updated_candidates


def evidence_key(item: dict[str, Any]) -> str:
    if item.get("evidence_type") == "visual" or item.get("keyframe_path"):
        return str(item.get("keyframe_path") or item.get("source_id") or id(item))
    return str(item.get("source_id") or f"{item.get('transcript_path')}:{item.get('start_sec')}:{item.get('end_sec')}")


def lexical_transcript_candidates(
    question: str,
    metadata: list[dict[str, Any]],
    top_k: int,
    mentioned_sources: set[str] | None = None,
) -> list[dict[str, Any]]:
    scored = []
    for item in metadata:
        lexical_score = transcript_lexical_score(question, item, mentioned_sources or set())
        if lexical_score > 0:
            candidate = dict(item)
            candidate["score"] = float(candidate.get("score", 0.0))
            candidate["lexical_score"] = lexical_score
            scored.append(candidate)
    scored.sort(key=lambda item: float(item["lexical_score"]), reverse=True)
    return scored[:top_k]


def merge_transcript_candidates(*candidate_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for candidates in candidate_lists:
        for item in candidates:
            key = str(item.get("source_id") or f"{item.get('transcript_path')}:{item.get('start_sec')}:{item.get('end_sec')}")
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(item)
            else:
                existing["score"] = max(float(existing.get("score", 0.0)), float(item.get("score", 0.0)))
                existing["lexical_score"] = max(float(existing.get("lexical_score", 0.0)), float(item.get("lexical_score", 0.0)))
    return list(merged.values())


def transcript_rerank_score(question: str, result: dict[str, Any], mentioned_sources: set[str] | None = None) -> float:
    score = float(result.get("score", 0.0))
    score += float(result.get("lexical_score", 0.0))
    question_tokens = content_tokens(question)
    text_tokens = content_tokens(str(result.get("text", "")))
    source_tokens = content_tokens(str(result.get("source_name", "")))
    if question_tokens & source_tokens:
        score += 0.5
    elif mentioned_sources:
        score -= 1.0
    if question_tokens:
        score += 0.3 * (len(question_tokens & text_tokens) / len(question_tokens))
    return score


def transcript_lexical_score(question: str, result: dict[str, Any], mentioned_sources: set[str] | None = None) -> float:
    question_tokens = content_tokens(question)
    if not question_tokens:
        return 0.0
    text_tokens = content_tokens(str(result.get("text", "")))
    source_tokens = content_tokens(str(result.get("source_name", "")))
    overlap = question_tokens & text_tokens
    if len(overlap) < 2 and not (question_tokens & source_tokens):
        return 0.0
    score = len(overlap) / len(question_tokens)
    if question_tokens & source_tokens:
        score += 0.5
    elif mentioned_sources:
        score -= 1.0
    if "how long" in question.lower():
        if text_tokens & {"hour", "hours"}:
            score += 0.5
        elif text_tokens & {"minute", "minutes", "second", "seconds"}:
            score += 0.25
    return score


def mentioned_source_tokens(question: str, metadata: list[dict[str, Any]]) -> set[str]:
    question_tokens = content_tokens(question)
    sources = set()
    for item in metadata:
        source_tokens = content_tokens(str(item.get("source_name", "")))
        if question_tokens & source_tokens:
            sources |= source_tokens
    return sources


def content_tokens(text: str) -> set[str]:
    return {stem_token(token) for token in re_tokens(text) if token not in STOPWORDS}


def stem_token(token: str) -> str:
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def add_grounding(args: argparse.Namespace, chosen: dict[str, Any], visual_results: list[dict[str, Any]]) -> dict[str, Any]:
    from src.visual_grounding import GroundingConfig, ground_visual_evidence, ground_visual_evidence_with_rerank

    model = processor = torch = None
    config = GroundingConfig(
        thresholds=parse_thresholds(args.grounding_threshold),
        prompts=parse_prompts(args.grounding_prompts),
        rerank="siglip-crop" if args.grounding == "dino-siglip-rerank" else "none",
        max_boxes_per_frame=args.grounding_max_boxes,
        alpha=args.grounding_alpha,
        beta=args.grounding_beta,
        gamma=args.grounding_gamma,
        debug=args.debug_router,
    )
    if args.grounding == "dino-siglip-rerank":
        _metadata, model_name = load_metadata(args.visual_index_dir)
        model, processor, torch = load_clip_model(model_name, local_files_only=not args.allow_download)
        grounding = ground_visual_evidence_with_rerank(
            args.question,
            visual_results,
            siglip_model=model,
            siglip_processor=processor,
            torch=torch,
            config=config,
        )
    else:
        grounding = ground_visual_evidence(args.question, chosen, config=config)
    updated = dict(chosen)
    updated["grounding"] = grounding
    if grounding.confidence is not None:
        updated["grounding_confidence"] = grounding.confidence
        updated["grounding_confidence_percent"] = grounding.confidence * 100.0
        updated["grounding_confidence_note"] = "GroundingDINO selected-box confidence; not calibrated probability."
    return updated


def print_transcript_evidence(item: dict[str, Any]) -> None:
    print(f"Source: {item.get('source_name')} {item.get('day')} hour {item.get('hour_id', item.get('video_id'))}")
    print(f"Timestamp: {item.get('timestamp')}")
    print(f"YouTube: {item.get('youtube_timestamp_url') or ''}")
    if item.get("evidence_confidence") is not None:
        print(f"Evidence confidence: {float(item.get('evidence_confidence', 0.0)):.3f}")
        print(f"Evidence confidence percent: {float(item.get('evidence_confidence_percent', 0.0)):.1f}")
    if item.get("transcript_retrieval_confidence") is not None:
        print(f"Transcript retrieval confidence: {float(item.get('transcript_retrieval_confidence', 0.0)):.3f}")
        print(f"Transcript retrieval confidence percent: {float(item.get('transcript_retrieval_confidence_percent', 0.0)):.1f}")
    print_playback_alignment(item)
    if item.get("transcript_path"):
        print(f"Transcript path: {item.get('transcript_path')}")
    if item.get("transcript_answer_rerank"):
        print_transcript_answer_rerank(item["transcript_answer_rerank"])
    if item.get("transcript_reasoning_answer"):
        print_transcript_reasoning_answer(item["transcript_reasoning_answer"])
    print("Transcript:")
    print(f"\"{item.get('transcript_snippet', item.get('text', ''))}\"")
    if item.get("answer_span"):
        answer_confidence = item["answer_span"].get("answer_confidence")
        if answer_confidence is not None:
            print(f"Answer confidence: {float(answer_confidence):.3f}")
            print(f"Answer confidence percent: {float(item['answer_span'].get('answer_confidence_percent', 0.0)):.1f}")
        print("Answer span JSON:")
        print(json.dumps(item["answer_span"], ensure_ascii=False))
    if item.get("answer_candidates"):
        print("Answer candidates JSON:")
        print(json.dumps(item["answer_candidates"], ensure_ascii=False))
    if item.get("transcript_heatmap"):
        print("Transcript heatmap JSON:")
        print(json.dumps(item["transcript_heatmap"], ensure_ascii=False))


def print_transcript_answer_rerank(debug: dict[str, Any]) -> None:
    print("Transcript answer reranking:")
    print(f"  enabled top_n={debug.get('top_n')}")
    original = debug.get("original_transcript") or {}
    best = debug.get("reranked_best_transcript") or {}
    print(
        "  original top transcript: "
        f"{original.get('source_name')} {original.get('timestamp')} "
        f"retrieval_score={safe_float(original.get('retrieval_score'), 0.0):.4f} "
        f"final_answer_score={safe_float(original.get('final_answer_score'), 0.0):.4f}"
    )
    print(
        "  reranked best transcript: "
        f"{best.get('source_name')} {best.get('timestamp')} "
        f"final_answer_score={safe_float(best.get('final_answer_score'), 0.0):.4f}"
    )
    print(f"  replacement: {'yes' if debug.get('replacement') else 'no'}")
    print(f"  reason: {debug.get('reason')}")
    print("Answer rerank candidates JSON:")
    print(json.dumps(debug.get("answer_rerank_candidates") or [], ensure_ascii=False))


def print_steering_suggestions(payload: dict[str, Any]) -> None:
    print("Recommended steering keywords:")
    for item in payload.get("suggested_terms", []):
        print(
            f"- {item.get('term')} ({item.get('type')}) "
            f"source={item.get('source', payload.get('source', 'unknown'))} "
            f"confidence={safe_float(item.get('confidence'), 0.0):.2f}: {item.get('reason')}"
        )
    print("Steering suggestions JSON:")
    print(json.dumps(payload, ensure_ascii=False))


def print_steering_terms_used(question: str, steering_terms: list[str], transcript_results: list[dict[str, Any]]) -> None:
    contributions = []
    if transcript_results:
        contributions = transcript_results[0].get("steering_query_contributions") or []
    print("Steering terms used:")
    print(", ".join(steering_terms))
    print("Expanded transcript retrieval queries:")
    for item in contributions:
        print(f"- {item.get('query')} source={item.get('source')} count={item.get('count')} term={item.get('term')}")
    print("Steering retrieval JSON:")
    print(
        json.dumps(
            {
                "question": question,
                "steering_terms": steering_terms,
                "query_contributions": contributions,
            },
            ensure_ascii=False,
        )
    )


def print_transcript_reasoning_answer(debug: dict[str, Any]) -> None:
    print("Transcript reasoning answer:")
    print(f"  enabled top_n={debug.get('top_n')}")
    print(f"  selected candidate: {debug.get('selected_candidate_index')} {debug.get('selected_candidate_id')}")
    print(f"  answer: {debug.get('answer')}")
    print(f"  evidence sentence: {debug.get('evidence_sentence')}")
    print(f"  confidence: {safe_float(debug.get('confidence'), 0.0):.3f}")
    print(f"  replacement: {'yes' if debug.get('replacement') else 'no'}")
    print(f"  reason: {debug.get('reason')}")
    print(f"  replacement reason: {debug.get('replacement_reason')}")
    print(f"  model name: {debug.get('model_name')}")
    print(f"  max_new_tokens: {debug.get('max_new_tokens')}")
    print(f"  generation token count: {debug.get('generation_token_count')}")
    print(f"  generation hit limit: {debug.get('generation_hit_limit')}")
    print(f"  raw output char count: {debug.get('raw_output_char_count')}")
    if debug.get("raw_batch_scoring_output") is not None:
        raw_batch_output = str(debug.get("raw_batch_scoring_output", ""))
        print(f"  batch scoring prompt length chars: {debug.get('batch_scoring_prompt_length_chars')}")
        print(f"  batch scoring output length chars: {debug.get('batch_scoring_output_length_chars')}")
        print(f"  batch scoring generation hit limit: {debug.get('batch_scoring_generation_hit_limit')}")
        print(f"  batch scoring num candidates requested: {debug.get('batch_scoring_num_candidates_requested')}")
        print(f"  batch scoring num candidates completed: {debug.get('batch_scoring_num_candidates_completed')}")
        print(f"  batch scoring num batches: {debug.get('batch_scoring_num_batches')}")
        print(f"  batch scoring batch size: {debug.get('batch_scoring_batch_size')}")
        print(f"  batch scoring completed batches: {debug.get('batch_scoring_completed_batches')}")
        print(f"  batch scoring completed candidates: {debug.get('batch_scoring_completed_candidates')}")
        print(f"  raw batch scoring output length: {debug.get('raw_batch_scoring_output_length')}")
        print(f"  batch scoring JSON parse failed: {debug.get('batch_scoring_json_parse_failed')}")
        print(f"  batch scoring exception: {debug.get('batch_scoring_exception')}")
        print(f"  batch scoring candidate count returned: {debug.get('batch_scoring_candidate_count_returned')}")
        print("  Last 500 chars of raw batch scoring output:")
        print(raw_batch_output[-500:])
        print("  Raw batch scoring output:")
        print(raw_batch_output)
    candidate_scores = debug.get("candidate_scores") or []
    if candidate_scores:
        print("  Candidate evidence scores:")
        for item in candidate_scores:
            print(
                f"  - candidate_index={item.get('candidate_index')} "
                f"raw_rank={item.get('raw_retrieval_rank')} "
                f"retrieval_score={safe_float(item.get('retrieval_score'), 0.0):.4f} "
                f"evidence_score={safe_float(item.get('evidence_score'), 0.0):.3f} "
                f"answer_likelihood={safe_float(item.get('answer_likelihood'), 0.0):.3f} "
                f"directness={item.get('directness')} "
                f"supports_answer={item.get('supports_answer')} "
                f"possible_answer_terms={item.get('possible_answer_terms')} "
                f"reason={item.get('reason')}"
            )
            print(f"    snippet: {item.get('snippet', '')}")
    print("Transcript reasoning answer JSON:")
    print(json.dumps(debug, ensure_ascii=False))


def print_playback_alignment(item: dict[str, Any]) -> None:
    debug = item.get("playback_alignment_debug") or {}
    if not debug:
        return
    original_url = item.get("original_youtube_timestamp_url") or item.get("youtube_timestamp_url") or ""
    playback_url = item.get("playback_youtube_timestamp_url") or item.get("youtube_timestamp_url") or ""
    print("Playback alignment:")
    print(f"  original evidence source: {item.get('source_name')}")
    print(f"  original evidence timestamp: {item.get('timestamp')}")
    print(f"  original YouTube: {original_url}")
    print(f"  playback source: {item.get('playback_source_name')}")
    print(f"  playback timestamp: {item.get('playback_timestamp')}")
    print(f"  playback YouTube: {playback_url}")
    print(f"  decision: {debug.get('decision')}")
    print(f"  reason: {debug.get('reason')}")
    if debug.get("anchor_passage"):
        print("  local anchor passage:")
        print(f"     {debug.get('anchor_passage')}")
    candidates = debug.get("top_similar_chunks") or []
    if candidates:
        print("  Top similar aligned chunks:")
        for rank, candidate in enumerate(candidates[:10], start=1):
            print(
                f"  {rank}. {candidate.get('source_name')} {candidate.get('timestamp')} "
                f"similarity={float(candidate.get('similarity_score', 0.0)):.4f} "
                f"overlap={int(candidate.get('anchor_overlap_count', 0))} "
                f"match_ratio={float(candidate.get('candidate_match_ratio', 0.0)):.3f}"
            )
            print(f"     {candidate.get('text_preview', '')}")


def print_timestamp_refinement_extraction_debug(item: dict[str, Any]) -> None:
    debug = item.get("timestamp_refinement_extraction_debug") or {}
    if not debug:
        return
    print("Timestamp refinement extraction debug:")
    print(f"  transcript_path: {debug.get('transcript_path')}")
    print(f"  broad_start_sec: {debug.get('broad_start_sec')}")
    print(f"  broad_end_sec: {debug.get('broad_end_sec')}")
    print(f"  loaded entries: {debug.get('loaded_entries')}")
    print(f"  overlapping entries: {debug.get('overlapping_entries')}")
    print(f"  fully inside entries: {debug.get('fully_inside_entries')}")
    print(f"  short entries duration <=30s: {debug.get('short_entries_duration_le_30s')}")
    if debug.get("expanded_start_sec") is not None:
        print(f"  expanded_start_sec: {debug.get('expanded_start_sec')}")
        print(f"  expanded_end_sec: {debug.get('expanded_end_sec')}")
        print(f"  expanded overlapping entries: {debug.get('expanded_overlapping_entries')}")
        print(f"  expanded short entries duration <=30s: {debug.get('expanded_short_entries_duration_le_30s')}")
    entries = debug.get("first_20_overlapping_entries") or []
    if entries:
        print("First 20 overlapping transcript entries:")
        for rank, entry in enumerate(entries[:20], start=1):
            print(
                f"{rank}. {entry.get('timestamp')} "
                f"duration={float(entry.get('duration_sec', 0.0)):.2f}s"
            )
            print(f"   {entry.get('text_preview', '')}")
    expanded = debug.get("first_20_expanded_overlapping_entries") or []
    if expanded:
        print("First 20 expanded-window transcript entries:")
        for rank, entry in enumerate(expanded[:20], start=1):
            print(
                f"{rank}. {entry.get('timestamp')} "
                f"duration={float(entry.get('duration_sec', 0.0)):.2f}s"
            )
            print(f"   {entry.get('text_preview', '')}")


def print_alignment_debug(item: dict[str, Any]) -> None:
    if item.get("alignment_target_time_sec") is None:
        return
    print("Cross-POV transcript alignment:")
    print(f"  target_time_sec: {float(item.get('alignment_target_time_sec')):.2f}")
    print(f"  search_radius_sec: {item.get('alignment_search_radius_sec')}")
    print(f"  transcript files considered: {item.get('alignment_transcript_files_considered')}")
    print(f"  original_query_score: {float(item.get('original_query_score', 0.0)):.4f}")
    print(f"  original_query_overlap_count: {int(item.get('original_query_overlap_count', 0))}")
    print(f"  original_alignment_score: {float(item.get('original_alignment_score', 0.0)):.4f}")
    print(f"  aligned_query_score: {float(item.get('aligned_query_score', 0.0)):.4f}")
    print(f"  aligned_query_overlap_count: {int(item.get('aligned_query_overlap_count', 0))}")
    print(f"  alignment_override_decision: {item.get('alignment_override_decision')}")
    print(f"  alignment_override_reason: {item.get('alignment_override_reason')}")
    if item.get("alignment_error"):
        print(f"  alignment unavailable: {item.get('alignment_error')}")
    highest = item.get("highest_scoring_alignment_candidate")
    if highest:
        print("Highest-scoring alignment candidate:")
        print(
            f"  {highest.get('source_name')} {highest.get('timestamp')} "
            f"query_score={float(highest.get('query_score', 0.0)):.4f} "
            f"query_overlap={int(highest.get('query_overlap_count', 0))} "
            f"final_score={float(highest.get('final_score', 0.0)):.4f}"
        )
        print(f"  {highest.get('text_preview', '')}")
    earliest = item.get("earliest_eligible_alignment_candidate")
    if earliest:
        print("Earliest eligible alignment candidate:")
        print(
            f"  {earliest.get('source_name')} {earliest.get('timestamp')} "
            f"query_score={float(earliest.get('query_score', 0.0)):.4f} "
            f"query_overlap={int(earliest.get('query_overlap_count', 0))} "
            f"final_score={float(earliest.get('final_score', 0.0)):.4f}"
        )
        print(f"  {earliest.get('text_preview', '')}")
    if item.get("aligned_timestamp"):
        print("Aligned transcript evidence:")
        print(f"  source: {item.get('aligned_source_name')}")
        print(f"  transcript_path: {item.get('aligned_transcript_path')}")
        print(f"  timestamp: {item.get('aligned_timestamp')}")
        print(f"  youtube: {item.get('aligned_youtube_timestamp_url') or ''}")
        print(f"  score: {float(item.get('aligned_score', 0.0)):.4f}")
        print(f"  snippet: {item.get('aligned_snippet') or ''}")
    candidates = item.get("alignment_candidates") or []
    if candidates:
        print("Top 10 alignment candidates:")
        for rank, candidate in enumerate(candidates[:10], start=1):
            print(
                f"{rank}. {candidate.get('source_name')} {candidate.get('timestamp')} "
                f"duration={float(candidate.get('duration_sec', 0.0)):.2f}s "
                f"distance={float(candidate.get('distance_sec', 0.0)):.2f}s "
                f"query_score={float(candidate.get('query_score', 0.0)):.4f} "
                f"query_overlap={int(candidate.get('query_overlap_count', 0))} "
                f"broad_score={float(candidate.get('broad_score', 0.0)):.4f} "
                f"temporal_score={float(candidate.get('temporal_score', 0.0)):.4f} "
                f"final_score={float(candidate.get('final_score', 0.0)):.4f}"
            )
            print(f"   {candidate.get('text_preview', '')}")
    eligible = item.get("eligible_alignment_candidates") or []
    if eligible:
        print("Eligible alignment candidates:")
        for rank, candidate in enumerate(eligible[:10], start=1):
            print(
                f"{rank}. {candidate.get('source_name')} {candidate.get('timestamp')} "
                f"duration={float(candidate.get('duration_sec', 0.0)):.2f}s "
                f"query_score={float(candidate.get('query_score', 0.0)):.4f} "
                f"query_overlap={int(candidate.get('query_overlap_count', 0))} "
                f"final_score={float(candidate.get('final_score', 0.0)):.4f}"
            )
            print(f"   {candidate.get('text_preview', '')}")

def print_visual_evidence(item: dict[str, Any]) -> None:
    print(f"Keyframe: {item.get('keyframe_path')}")
    print(f"Timestamp: {item.get('timestamp')}")
    print(f"YouTube: {item.get('youtube_timestamp_url') or ''}")
    if item.get("evidence_confidence") is not None:
        print(f"Evidence confidence: {float(item.get('evidence_confidence', 0.0)):.3f}")
        print(f"Evidence confidence percent: {float(item.get('evidence_confidence_percent', 0.0)):.1f}")
    if item.get("visual_retrieval_confidence") is not None:
        print(f"Visual retrieval confidence: {float(item.get('visual_retrieval_confidence', 0.0)):.3f}")
        print(f"Visual retrieval confidence percent: {float(item.get('visual_retrieval_confidence_percent', 0.0)):.1f}")
    grounding = item.get("grounding")
    if grounding:
        print(f"Bounding box image: {grounding.output_image_path or ''}")
        print(f"Candidate boxes image: {grounding.candidates_image_path or ''}")
        if grounding.box_xyxy:
            print("Bounding box: [" + ", ".join(f"{value:.1f}" for value in grounding.box_xyxy) + "]")
        if grounding.confidence is not None:
            print(f"GroundingDINO confidence: {grounding.confidence:.3f}")
            print(f"Grounding confidence: {grounding.confidence:.3f}")
            print(f"Grounding confidence percent: {grounding.confidence * 100.0:.1f}")


def print_mixed_evidence(items: list[dict[str, Any]]) -> None:
    print("\nMixed top-" + str(len(items)) + " evidence:")
    for item in items:
        rank = item.get("rank")
        evidence_type = item.get("evidence_type")
        confidence = float(item.get("mixed_rank_confidence_percent", item.get("confidence_percent", 0.0)))
        components = item.get("score_components") or {}
        router_weight = float(components.get("router_channel_weight", 0.0))
        timestamp = item.get("timestamp") or ""
        youtube = item.get("youtube_timestamp_url") or ""
        evidence_confidence = item.get("evidence_confidence")
        evidence_confidence_text = (
            f"{float(evidence_confidence):.3f}" if evidence_confidence is not None else "n/a"
        )
        if evidence_type == "visual":
            retrieval_confidence = float(item.get("visual_retrieval_confidence", 0.0))
            print(
                f"{rank}. [visual] evidence_confidence={evidence_confidence_text} mixed_confidence={confidence:.1f}% "
                f"retrieval_confidence={retrieval_confidence:.3f} router_weight={router_weight:.3f} "
                f"keyframe={item.get('keyframe_path') or ''} timestamp={timestamp} youtube={youtube}"
            )
        else:
            source = f"{item.get('source_name')} {item.get('day')} hour {item.get('hour_id', item.get('video_id'))}"
            snippet = item.get("transcript_snippet") or item.get("text") or ""
            retrieval_confidence = float(item.get("transcript_retrieval_confidence", 0.0))
            print(
                f"{rank}. [transcript] evidence_confidence={evidence_confidence_text} mixed_confidence={confidence:.1f}% "
                f"retrieval_confidence={retrieval_confidence:.3f} router_weight={router_weight:.3f} "
                f"source={source} timestamp={timestamp} youtube={youtube} snippet={preview_inline(str(snippet))}"
            )
    print("Mixed evidence JSON:")
    print(json.dumps(to_jsonable(items), ensure_ascii=False))


def print_diversity_debug(debug: dict[str, Any]) -> None:
    print("Mixed diversity debug JSON:")
    print(json.dumps(to_jsonable(debug), ensure_ascii=False))


def preview_inline(text: str, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def extract_snippet(question: str, text: str, window: int = 28) -> str:
    tokens = [token for token in re_tokens(question) if token not in STOPWORDS]
    words = text.split()
    if not words:
        return text
    best_index = 0
    best_score = -1
    for idx, word in enumerate(words):
        normalized = normalize_token(word)
        score = 1 if normalized in tokens else 0
        if score > best_score:
            best_score = score
            best_index = idx
    start = max(0, best_index - window // 2)
    end = min(len(words), start + window)
    return " ".join(words[start:end]).strip()


def normalize_token(value: str) -> str:
    tokens = re_tokens(value)
    return tokens[0] if tokens else ""


def re_tokens(value: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", value.lower())


STOPWORDS = {
    "a",
    "an",
    "are",
    "did",
    "for",
    "how",
    "is",
    "it",
    "long",
    "not",
    "of",
    "the",
    "there",
    "they",
    "to",
    "what",
    "where",
    "why",
    "would",
}


def parse_thresholds(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    return parsed or (0.25,)


def parse_prompts(value: str) -> tuple[str, ...] | None:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed or None


def load_metadata(index_dir: str | Path) -> tuple[list[dict[str, Any]], str]:
    metadata_path = Path(index_dir) / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["items"], data["model_name"]


def load_faiss_index(index_dir: str | Path):
    faiss = require_faiss()
    return faiss.read_index(str(Path(index_dir) / "transcript.faiss"))


def resolve_query_model_name(index_model_name: str, text_model_name: str | None, grounding_mode: str) -> str:
    if grounding_mode == "dino-siglip-rerank":
        return index_model_name
    if text_model_name:
        return text_model_name
    import os

    env_text_model = os.environ.get("SIGLIP_TEXT_MODEL_NAME")
    if env_text_model:
        return env_text_model
    path = Path(index_model_name)
    if path.exists():
        sibling = path.with_name(path.name + "-text")
        if sibling.is_dir():
            return str(sibling)
    return index_model_name


def collect_results(scores: Any, ids: Any, metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for score, row_id in zip(scores[0], ids[0]):
        if row_id < 0:
            continue
        item = dict(metadata[int(row_id)])
        item["score"] = float(score)
        results.append(item)
    return results


def collect_variant_results(
    scores: Any,
    ids: Any,
    metadata: list[dict[str, Any]],
    variants: list[str],
    top_k: int,
    diversity_window_sec: float,
) -> list[dict[str, Any]]:
    grouped = []
    for variant_index, variant in enumerate(variants):
        results = collect_results(scores[variant_index : variant_index + 1], ids[variant_index : variant_index + 1], metadata)
        for result in results:
            result["query_variant"] = variant
        grouped.append({"variant": variant, "results": diversify_results(results, top_k, diversity_window_sec)})
    return grouped


def merge_variant_results(variant_results: list[dict[str, Any]], top_k: int, diversity_window_sec: float) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for group in variant_results:
        for result in group["results"]:
            key = str(result.get("keyframe_path"))
            existing = by_path.get(key)
            if existing is None or float(result["score"]) > float(existing["score"]):
                by_path[key] = dict(result)
    ranked = sorted(by_path.values(), key=lambda item: float(item["score"]), reverse=True)
    return diversify_results(ranked, top_k, diversity_window_sec)


def diversify_results(results: list[dict[str, Any]], top_k: int, window_sec: float) -> list[dict[str, Any]]:
    if window_sec <= 0:
        return results[:top_k]
    selected: list[dict[str, Any]] = []
    for result in results:
        if not is_near_duplicate(result, selected, window_sec):
            selected.append(result)
        if len(selected) >= top_k:
            break
    return selected


def is_near_duplicate(candidate: dict[str, Any], selected: list[dict[str, Any]], window_sec: float) -> bool:
    try:
        candidate_time = float(candidate.get("keyframe_time_sec", candidate.get("start_sec")))
    except (TypeError, ValueError):
        return False
    candidate_source = (candidate.get("day"), candidate.get("source_name"), candidate.get("video_id"))
    for item in selected:
        try:
            item_time = float(item.get("keyframe_time_sec", item.get("start_sec")))
        except (TypeError, ValueError):
            continue
        item_source = (item.get("day"), item.get("source_name"), item.get("video_id"))
        if item_source == candidate_source and abs(candidate_time - item_time) <= window_sec:
            return True
    return False


def query_variants(question: str, synonym_map: dict[str, list[str]] | None = None) -> list[str]:
    variants = [question.strip()]
    phrases = extract_noun_phrases(question)
    variants.extend(phrases[:4])
    for phrase in phrases[:4]:
        variants.extend(visual_prompt_templates(phrase))
        for value in (synonym_map or {}).get(phrase, []):
            variants.append(value)
    return dedupe_strings(variants)


def extract_noun_phrases(text: str) -> list[str]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS and len(token) > 1]
    phrases = []
    for size in (3, 2, 1):
        for start in range(0, max(0, len(tokens) - size + 1)):
            phrases.append(" ".join(tokens[start : start + size]))
    return dedupe_strings(phrases)


def visual_prompt_templates(phrase: str) -> list[str]:
    return [f"a photo of {phrase}", f"an image of {phrase}", f"{phrase} in the scene"] if phrase else []


def load_synonym_map(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected JSON object synonym map: {path}")
    return {str(key).lower(): [str(value).lower() for value in values] for key, values in data.items() if isinstance(values, list)}


def dedupe_strings(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", value.strip().lower())
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


if __name__ == "__main__":
    main()
