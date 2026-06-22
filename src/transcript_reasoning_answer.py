from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REASONING_MODEL = "/scratch-shared/group_h/models/qwen2.5-vl-3b"
DEFAULT_REASONING_MAX_NEW_TOKENS = 512
DEFAULT_BATCH_SCORING_SIZE = 5
ModelFunction = Callable[[str], str]


@dataclass(frozen=True)
class ReasoningCandidate:
    candidate_index: int
    candidate_id: str
    source_name: str
    timestamp: str
    transcript_path: str
    retrieval_score: float
    snippet: str


def reason_over_transcript_candidates(
    question: str,
    transcript_candidates: list[dict],
    top_n: int = 10,
    *,
    model_function: ModelFunction | None = None,
    reasoning_mode: str = "global",
) -> dict[str, Any]:
    candidates = build_reasoning_candidates(transcript_candidates[: max(0, top_n)])
    if not candidates:
        return unavailable_result("no_transcript_candidates", candidates)
    if reasoning_mode == "per_candidate":
        return reason_over_transcript_candidates_per_candidate(
            question,
            candidates,
            model_function=model_function,
        )
    if reasoning_mode != "global":
        return unavailable_result(f"unsupported_reasoning_mode: {reasoning_mode}", candidates)

    prompt = build_prompt(question, candidates)
    model_name = resolve_reasoning_model_name()
    raw_output = ""
    generation_debug = empty_generation_debug(raw_output)
    try:
        if model_function:
            raw_output = model_function(prompt)
            generation_debug = empty_generation_debug(raw_output)
        else:
            raw_output, generation_debug = run_local_reasoning_model_with_debug(
                prompt,
                model_name,
                max_new_tokens=DEFAULT_REASONING_MAX_NEW_TOKENS,
            )
        parsed = parse_strict_json(raw_output)
    except Exception as exc:
        result = unavailable_result(f"model_or_json_failed: {type(exc).__name__}: {exc}", candidates)
        result["raw_model_output"] = raw_output
        result["raw_model_output_length"] = len(raw_output)
        result.update(generation_debug)
        result["json_parse_failed"] = True
        result["output_truncated_suspected"] = output_truncated_suspected(raw_output)
        result["model_name"] = model_name
        return result

    raw_supporting_candidate_indices = parse_candidate_indices(parsed.get("supporting_candidate_indices"))
    raw_reasoning_ordered_candidate_indices = parse_candidate_indices(parsed.get("reasoning_ordered_candidate_indices"))
    supporting_candidate_indices = raw_supporting_candidate_indices[:10]
    reasoning_input_indices = raw_reasoning_ordered_candidate_indices[:20] or supporting_candidate_indices
    original_selected_candidate_index = parse_candidate_index(parsed.get("selected_candidate_index"))
    selection = validate_selected_candidate(
        candidates,
        original_selected_candidate_index,
        supporting_candidate_indices,
        str(parsed.get("evidence_sentence") or ""),
    )
    selected_index = selection["selected_candidate_index"]
    selected_candidate = selection["selected_candidate"]
    confidence = bounded_confidence(parsed.get("confidence"))
    if selected_candidate is None:
        selected_index = None
        selected_candidate_id = None
        selected_retrieval_rank = None
    else:
        selected_candidate_id = selected_candidate.candidate_id
        selected_retrieval_rank = selected_candidate.candidate_index

    evidence_sentence = str(selection["evidence_sentence"])
    reasoning_ordered_candidates = build_reasoning_ordered_candidates(
        reasoning_input_indices,
        candidates,
        supporting_candidate_indices,
    )
    reasoning_ordered_candidate_indices = [
        int(item["raw_retrieval_rank"]) for item in reasoning_ordered_candidates
    ]
    answer_hypothesis = str(parsed.get("answer_hypothesis") or parsed.get("answer") or "")
    return {
        "selected_candidate_index": selected_index,
        "selected_candidate_id": selected_candidate_id,
        "reasoning_selected_retrieval_rank": selected_retrieval_rank,
        "selected_candidate_overridden": selection["selected_candidate_overridden"],
        "original_selected_candidate_index": original_selected_candidate_index,
        "final_selected_candidate_index": selected_index,
        "selected_candidate_override_reason": selection["selected_candidate_override_reason"],
        "answer_hypothesis": answer_hypothesis,
        "supporting_candidate_indices": supporting_candidate_indices,
        "num_supporting_candidate_indices_raw": len(raw_supporting_candidate_indices),
        "num_supporting_candidate_indices_final": len(supporting_candidate_indices),
        "num_reasoning_ordered_indices_raw": len(raw_reasoning_ordered_candidate_indices),
        "num_reasoning_ordered_indices_final": len(reasoning_ordered_candidate_indices),
        "raw_model_output_length": len(raw_output),
        **generation_debug,
        "json_parse_failed": False,
        "output_truncated_suspected": False,
        "answer": str(parsed.get("answer") or answer_hypothesis),
        "evidence_sentence": evidence_sentence,
        "confidence": confidence if confidence is not None else 0.0,
        "reason": str(parsed.get("reason") or ""),
        "steering_keywords": normalize_reasoning_terms(
            parsed.get("steering_keywords"),
            question=question,
            evidence_sentence=evidence_sentence,
            require_evidence_terms=True,
        ),
        "steering_paraphrases": [],
        "raw_model_output": raw_output,
        "model_name": model_name,
        "reasoning_candidates": [candidate_to_dict(candidate) for candidate in candidates],
        "reasoning_ordered_transcript_candidates": reasoning_ordered_candidates,
        "reasoning_ordered_candidate_indices": reasoning_ordered_candidate_indices,
        "reasoning_order_used": bool(reasoning_ordered_candidate_indices),
    }


def reason_over_transcript_candidates_per_candidate(
    question: str,
    candidates: list[ReasoningCandidate],
    *,
    model_function: ModelFunction | None = None,
) -> dict[str, Any]:
    model_name = resolve_reasoning_model_name()
    scoring_raw_output = ""
    scoring_prompt = ""
    scoring_generation_debug = empty_generation_debug(scoring_raw_output)
    scoring_json_parse_failed = False
    scoring_exception = ""
    batch_scoring_candidate_count_returned = 0
    scoring_result = run_batched_candidate_scoring(
        question,
        candidates,
        model_name=model_name,
        model_function=model_function,
        batch_size=DEFAULT_BATCH_SCORING_SIZE,
    )
    candidate_scores = scoring_result["candidate_scores"]
    scoring_raw_output = scoring_result["raw_output"]
    scoring_prompt = scoring_result["prompt_text"]
    scoring_generation_debug = scoring_result["generation_debug"]
    scoring_json_parse_failed = scoring_result["json_parse_failed"]
    scoring_exception = scoring_result["exception"]
    batch_scoring_candidate_count_returned = scoring_result["candidate_count_returned"]

    candidate_scores.sort(key=candidate_score_sort_key, reverse=True)
    supporting_candidate_indices = [
        int(item["candidate_index"])
        for item in candidate_scores
        if item.get("supports_answer") or safe_float(item.get("evidence_score"), 0.0) >= 0.6
    ][:10]
    reasoning_ordered_candidate_indices = [int(item["candidate_index"]) for item in candidate_scores[:20]]

    answer_result = generate_answer_hypothesis_from_scores(
        question,
        candidates,
        candidate_scores,
        supporting_candidate_indices,
        model_name=model_name,
        model_function=model_function,
    )
    parsed = {
        "answer_hypothesis": answer_result.get("answer_hypothesis", ""),
        "supporting_candidate_indices": supporting_candidate_indices,
        "selected_candidate_index": answer_result.get("selected_candidate_index"),
        "evidence_sentence": answer_result.get("evidence_sentence", ""),
        "confidence": answer_result.get("confidence", 0.0),
        "reason": answer_result.get("reason", ""),
        "steering_keywords": answer_result.get("steering_keywords", []),
        "reasoning_ordered_candidate_indices": reasoning_ordered_candidate_indices,
    }
    raw_supporting_candidate_indices = parse_candidate_indices(parsed.get("supporting_candidate_indices"))
    raw_reasoning_ordered_candidate_indices = parse_candidate_indices(parsed.get("reasoning_ordered_candidate_indices"))
    supporting_candidate_indices = raw_supporting_candidate_indices[:10]
    reasoning_input_indices = raw_reasoning_ordered_candidate_indices[:20] or supporting_candidate_indices
    original_selected_candidate_index = parse_candidate_index(parsed.get("selected_candidate_index"))
    selection = validate_selected_candidate(
        candidates,
        original_selected_candidate_index,
        supporting_candidate_indices,
        str(parsed.get("evidence_sentence") or ""),
    )
    selected_index = selection["selected_candidate_index"]
    selected_candidate = selection["selected_candidate"]
    if selected_candidate is None:
        selected_candidate_id = None
        selected_retrieval_rank = None
    else:
        selected_candidate_id = selected_candidate.candidate_id
        selected_retrieval_rank = selected_candidate.candidate_index
    evidence_sentence = str(selection["evidence_sentence"])
    reasoning_ordered_candidates = build_reasoning_ordered_candidates(
        reasoning_input_indices,
        candidates,
        supporting_candidate_indices,
    )
    final_reasoning_ordered_indices = [int(item["raw_retrieval_rank"]) for item in reasoning_ordered_candidates]
    answer_raw = str(answer_result.get("raw_model_output", ""))
    answer_generation_debug = answer_result.get("generation_debug") or empty_generation_debug(answer_raw)
    return {
        "reasoning_mode": "per_candidate",
        "candidate_scores": candidate_scores,
        "selected_candidate_index": selected_index,
        "selected_candidate_id": selected_candidate_id,
        "reasoning_selected_retrieval_rank": selected_retrieval_rank,
        "selected_candidate_overridden": selection["selected_candidate_overridden"],
        "original_selected_candidate_index": original_selected_candidate_index,
        "final_selected_candidate_index": selected_index,
        "selected_candidate_override_reason": selection["selected_candidate_override_reason"],
        "answer_hypothesis": str(parsed.get("answer_hypothesis") or ""),
        "supporting_candidate_indices": supporting_candidate_indices,
        "num_supporting_candidate_indices_raw": len(raw_supporting_candidate_indices),
        "num_supporting_candidate_indices_final": len(supporting_candidate_indices),
        "num_reasoning_ordered_indices_raw": len(raw_reasoning_ordered_candidate_indices),
        "num_reasoning_ordered_indices_final": len(final_reasoning_ordered_indices),
        "raw_model_output_length": len(scoring_raw_output) + len(answer_raw),
        **answer_generation_debug,
        "json_parse_failed": bool(answer_result.get("json_parse_failed", False)) or scoring_json_parse_failed,
        "output_truncated_suspected": output_truncated_suspected(answer_raw),
        "answer": str(parsed.get("answer_hypothesis") or ""),
        "evidence_sentence": evidence_sentence,
        "confidence": safe_float(parsed.get("confidence"), 0.0),
        "reason": str(parsed.get("reason") or ""),
        "steering_keywords": normalize_reasoning_terms(
            parsed.get("steering_keywords"),
            question=question,
            evidence_sentence=evidence_sentence,
            require_evidence_terms=True,
        ),
        "steering_paraphrases": [],
        "raw_model_output": answer_raw,
        "raw_batch_scoring_output": scoring_raw_output,
        "raw_batch_scoring_output_length": len(scoring_raw_output),
        "batch_scoring_prompt_length_chars": len(scoring_prompt),
        "batch_scoring_output_length_chars": len(scoring_raw_output),
        "batch_scoring_generation_hit_limit": scoring_generation_debug.get("generation_hit_limit"),
        "batch_scoring_num_candidates_requested": len(candidates),
        "batch_scoring_num_candidates_completed": batch_scoring_candidate_count_returned,
        "batch_scoring_json_parse_failed": scoring_json_parse_failed,
        "batch_scoring_exception": scoring_exception,
        "batch_scoring_candidate_count_returned": batch_scoring_candidate_count_returned,
        "batch_scoring_num_batches": scoring_result["num_batches"],
        "batch_scoring_batch_size": scoring_result["batch_size"],
        "batch_scoring_completed_batches": scoring_result["completed_batches"],
        "batch_scoring_completed_candidates": scoring_result["completed_candidates"],
        "candidate_score_raw_output": scoring_raw_output,
        "candidate_score_generation_debug": scoring_generation_debug,
        "model_name": model_name,
        "reasoning_candidates": [candidate_to_dict(candidate) for candidate in candidates],
        "reasoning_ordered_transcript_candidates": reasoning_ordered_candidates,
        "reasoning_ordered_candidate_indices": final_reasoning_ordered_indices,
        "reasoning_order_used": bool(final_reasoning_ordered_indices),
    }


def run_batched_candidate_scoring(
    question: str,
    candidates: list[ReasoningCandidate],
    *,
    model_name: str,
    model_function: ModelFunction | None,
    batch_size: int,
) -> dict[str, Any]:
    all_raw_scores: list[dict[str, Any]] = []
    raw_outputs = []
    prompt_texts = []
    generation_debug_items = []
    exceptions = []
    completed_batches = 0
    batch_size = max(1, int(batch_size))
    batches = [candidates[index : index + batch_size] for index in range(0, len(candidates), batch_size)]
    for batch in batches:
        prompt = build_batch_candidate_scoring_prompt(question, batch)
        raw_output = ""
        generation_debug = empty_generation_debug(raw_output)
        prompt_texts.append(prompt)
        try:
            if model_function:
                raw_output = model_function(prompt)
                generation_debug = empty_generation_debug(raw_output)
            else:
                raw_output, generation_debug = run_local_reasoning_model_with_debug(
                    prompt,
                    model_name,
                    max_new_tokens=DEFAULT_REASONING_MAX_NEW_TOKENS,
                )
            parsed_scores = parse_strict_json(raw_output)
            raw_candidate_scores = parsed_scores.get("candidate_scores")
            if isinstance(raw_candidate_scores, list):
                all_raw_scores.extend(item for item in raw_candidate_scores if isinstance(item, dict))
                completed_batches += 1
        except Exception as exc:
            exceptions.append(f"{type(exc).__name__}: {exc}")
        raw_outputs.append(raw_output)
        generation_debug_items.append(generation_debug)

    candidate_scores = normalize_candidate_scores(all_raw_scores, candidates)
    merged_generation_debug = merge_generation_debug(generation_debug_items)
    return {
        "candidate_scores": candidate_scores,
        "raw_output": "\n\n---BATCH---\n\n".join(raw_outputs),
        "prompt_text": "\n\n---BATCH PROMPT---\n\n".join(prompt_texts),
        "generation_debug": merged_generation_debug,
        "json_parse_failed": bool(exceptions),
        "exception": " | ".join(exceptions),
        "candidate_count_returned": len(all_raw_scores),
        "num_batches": len(batches),
        "batch_size": batch_size,
        "completed_batches": completed_batches,
        "completed_candidates": len({parse_candidate_index(item.get("candidate_index")) for item in all_raw_scores}),
    }


def merge_generation_debug(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return empty_generation_debug("")
    token_counts = [safe_float(item.get("generation_token_count"), 0.0) or 0.0 for item in items]
    char_counts = [safe_float(item.get("raw_output_char_count"), 0.0) or 0.0 for item in items]
    return {
        "max_new_tokens": max((safe_float(item.get("max_new_tokens"), 0.0) or 0.0 for item in items), default=0.0),
        "generation_token_count": int(sum(token_counts)),
        "generation_hit_limit": any(bool(item.get("generation_hit_limit")) for item in items),
        "raw_output_char_count": int(sum(char_counts)),
    }


def build_batch_candidate_scoring_prompt(question: str, candidates: list[ReasoningCandidate]) -> str:
    candidate_blocks = []
    for candidate in candidates:
        candidate_blocks.append(
            "\n".join(
                [
                    f"Candidate {candidate.candidate_index}",
                    f"source: {candidate.source_name}",
                    f"timestamp: {candidate.timestamp}",
                    f"retrieval_score: {candidate.retrieval_score:.6f}",
                    "text:",
                    candidate.snippet,
                ]
            )
        )
    return (
        "You are batch-scoring transcript candidates for a multimedia evidence retrieval system.\n"
        "Use only the user question and transcript candidates below. Do not use outside knowledge.\n"
        "Score every provided candidate independently. If 30 candidates are provided, candidate_scores should "
        "contain 30 entries. Do not stop after candidate 1 or after finding one plausible answer.\n"
        "High evidence_score only if that candidate itself contains answer-bearing evidence. Low score if it "
        "only shares generic words from the question. Penalize wrong-sense matches: if the query asks about a "
        "named entity or title, candidates using the same words as common nouns should score low unless they "
        "clearly discuss that entity/title or a direct answer.\n"
        "Prefer candidates containing concrete answer evidence, names, entities, answer phrases, direct answer "
        "statements, or nearby quiz question/answer content. Return compact JSON only.\n"
        "Do not include explanations, summaries, possible answer terms, or candidate text in the output.\n\n"
        f"Question: {question}\n\n"
        "Transcript candidates:\n\n"
        + "\n\n---\n\n".join(candidate_blocks)
        + "\n\nJSON schema:\n"
        + json.dumps(
            {
                "candidate_scores": [
                    {
                        "candidate_index": 1,
                        "evidence_score": 0.0,
                        "answer_likelihood": 0.0,
                        "directness": "none",
                        "supports_answer": False,
                    }
                ]
            }
        )
    )


def normalize_candidate_scores(values: Any, candidates: list[ReasoningCandidate]) -> list[dict[str, Any]]:
    parsed_by_index: dict[int, dict[str, Any]] = {}
    if isinstance(values, list):
        for item in values:
            if not isinstance(item, dict):
                continue
            index = parse_candidate_index(item.get("candidate_index"))
            if index is None or index in parsed_by_index:
                continue
            parsed_by_index[index] = item
    scores = []
    for candidate in candidates:
        parsed = parsed_by_index.get(candidate.candidate_index)
        if parsed is None:
            scores.append(fallback_candidate_score(candidate, "missing from batch candidate_scores"))
        else:
            score = normalize_candidate_score(parsed, candidate)
            score["json_parse_failed"] = False
            scores.append(score)
    return scores


def build_candidate_scoring_prompt(question: str, candidate: ReasoningCandidate) -> str:
    return (
        "You are scoring one transcript candidate independently for a multimedia evidence retrieval system.\n"
        "Use only this candidate text and the user question. Do not use outside knowledge.\n"
        "Score whether this candidate itself contains answer-bearing evidence.\n"
        "Do not compare only to candidate 1. Do not stop because another candidate may be plausible.\n"
        "High score only if this candidate contains direct answer evidence, concrete entities, names, objects, "
        "answer phrases, direct answer statements, or nearby quiz question/answer content.\n"
        "Low score if it only shares generic words from the question.\n"
        "Penalize wrong-sense matches. For example, if the query asks about a named entity or title such as "
        "'The Police' as a band/title, mentions like 'police officer', 'cops', or 'police were going' should "
        "receive low evidence_score unless the candidate clearly discusses the band or quiz answer.\n"
        "Return compact JSON only.\n\n"
        f"Question: {question}\n\n"
        f"Candidate {candidate.candidate_index}\n"
        f"source: {candidate.source_name}\n"
        f"timestamp: {candidate.timestamp}\n"
        f"retrieval_score: {candidate.retrieval_score:.6f}\n"
        "text:\n"
        f"{candidate.snippet}\n\n"
        "JSON schema:\n"
        + json.dumps(
            {
                "candidate_index": candidate.candidate_index,
                "evidence_score": 0.0,
                "answer_likelihood": 0.0,
                "directness": "none",
                "supports_answer": False,
                "candidate_summary": "short summary",
                "possible_answer_terms": ["term"],
                "reason": "one sentence",
            }
        )
    )


def normalize_candidate_score(parsed: dict[str, Any], candidate: ReasoningCandidate) -> dict[str, Any]:
    directness = str(parsed.get("directness") or "none").lower()
    if directness not in {"none", "context", "partial", "direct"}:
        directness = "none"
    terms = parsed.get("possible_answer_terms")
    if not isinstance(terms, list):
        terms = []
    return {
        "candidate_index": candidate.candidate_index,
        "raw_retrieval_rank": candidate.candidate_index,
        "retrieval_score": candidate.retrieval_score,
        "source_name": candidate.source_name,
        "timestamp": candidate.timestamp,
        "transcript_path": candidate.transcript_path,
        "evidence_score": bounded_confidence(parsed.get("evidence_score")) or 0.0,
        "answer_likelihood": bounded_confidence(parsed.get("answer_likelihood")) or 0.0,
        "directness": directness,
        "supports_answer": bool(parsed.get("supports_answer")),
        "candidate_summary": str(parsed.get("candidate_summary") or "")[:240],
        "possible_answer_terms": [str(term)[:80] for term in terms[:8]],
        "reason": str(parsed.get("reason") or "")[:300],
        "snippet": candidate.snippet,
    }


def fallback_candidate_score(candidate: ReasoningCandidate, reason: str) -> dict[str, Any]:
    return {
        "candidate_index": candidate.candidate_index,
        "raw_retrieval_rank": candidate.candidate_index,
        "retrieval_score": candidate.retrieval_score,
        "source_name": candidate.source_name,
        "timestamp": candidate.timestamp,
        "transcript_path": candidate.transcript_path,
        "evidence_score": 0.0,
        "answer_likelihood": 0.0,
        "directness": "none",
        "supports_answer": False,
        "candidate_summary": "",
        "possible_answer_terms": [],
        "reason": reason,
        "snippet": candidate.snippet,
    }


def candidate_score_sort_key(item: dict[str, Any]) -> tuple[float, float, int, float]:
    directness_rank = {"none": 0, "context": 1, "partial": 2, "direct": 3}.get(str(item.get("directness")), 0)
    return (
        safe_float(item.get("evidence_score"), 0.0),
        safe_float(item.get("answer_likelihood"), 0.0),
        directness_rank,
        safe_float(item.get("retrieval_score"), 0.0),
    )


def generate_answer_hypothesis_from_scores(
    question: str,
    candidates: list[ReasoningCandidate],
    candidate_scores: list[dict[str, Any]],
    supporting_candidate_indices: list[int],
    *,
    model_name: str,
    model_function: ModelFunction | None,
) -> dict[str, Any]:
    candidate_map = {candidate.candidate_index: candidate for candidate in candidates}
    evidence_indices = supporting_candidate_indices or [int(item["candidate_index"]) for item in candidate_scores[:3]]
    blocks = []
    for index in evidence_indices[:10]:
        candidate = candidate_map.get(index)
        if candidate is None:
            continue
        blocks.append(
            "\n".join(
                [
                    f"Candidate {candidate.candidate_index}",
                    f"source: {candidate.source_name}",
                    f"timestamp: {candidate.timestamp}",
                    "text:",
                    candidate.snippet,
                ]
            )
        )
    prompt = (
        "Use only these scored supporting transcript candidates to produce a concise answer hypothesis and "
        "select the best single supporting candidate. Return compact JSON only.\n"
        "selected_candidate_index must be one of the provided candidates. evidence_sentence must be copied "
        "or closely extracted from selected_candidate_index only. If evidence is weak, keep confidence low.\n\n"
        f"Question: {question}\n\n"
        "Supporting transcript candidates:\n\n"
        + "\n\n---\n\n".join(blocks)
        + "\n\nJSON schema:\n"
        + json.dumps(
            {
                "answer_hypothesis": "one sentence",
                "selected_candidate_index": evidence_indices[0] if evidence_indices else None,
                "evidence_sentence": "copied sentence",
                "confidence": 0.0,
                "reason": "one sentence",
                "steering_keywords": ["term"],
            }
        )
    )
    raw_output = ""
    generation_debug = empty_generation_debug(raw_output)
    try:
        if model_function:
            raw_output = model_function(prompt)
            generation_debug = empty_generation_debug(raw_output)
        else:
            raw_output, generation_debug = run_local_reasoning_model_with_debug(
                prompt,
                model_name,
                max_new_tokens=256,
            )
        parsed = parse_strict_json(raw_output)
        parsed["raw_model_output"] = raw_output
        parsed["generation_debug"] = generation_debug
        parsed["json_parse_failed"] = False
        return parsed
    except Exception as exc:
        selected = candidate_map.get(evidence_indices[0]) if evidence_indices else None
        return {
            "answer_hypothesis": "",
            "selected_candidate_index": selected.candidate_index if selected else None,
            "evidence_sentence": selected.snippet if selected else "",
            "confidence": 0.0,
            "reason": f"answer_hypothesis_generation_failed: {type(exc).__name__}: {exc}",
            "steering_keywords": [],
            "raw_model_output": raw_output,
            "generation_debug": generation_debug,
            "json_parse_failed": True,
        }


def validate_selected_candidate(
    candidates: list[ReasoningCandidate],
    original_selected_candidate_index: int | None,
    supporting_candidate_indices: list[int],
    evidence_sentence: str,
) -> dict[str, Any]:
    final_index = original_selected_candidate_index
    overridden = False
    override_reason = ""

    if supporting_candidate_indices and final_index not in supporting_candidate_indices:
        final_index = supporting_candidate_indices[0]
        overridden = True
        override_reason = "selected_candidate_index was not in supporting_candidate_indices"

    selected_candidate = candidate_by_index(candidates, final_index)
    if selected_candidate is None:
        final_index = None
        selected_text = ""
    else:
        selected_text = selected_candidate.snippet

    final_evidence_sentence = evidence_sentence.strip()
    if selected_candidate is not None and not evidence_sentence_supported_by_candidate(final_evidence_sentence, selected_text):
        final_evidence_sentence = selected_text
        if not overridden:
            override_reason = "evidence_sentence was not found in selected candidate text"

    return {
        "selected_candidate_index": final_index,
        "selected_candidate": selected_candidate,
        "selected_candidate_overridden": overridden,
        "selected_candidate_override_reason": override_reason,
        "evidence_sentence": final_evidence_sentence,
    }


def evidence_sentence_supported_by_candidate(evidence_sentence: str, candidate_text: str) -> bool:
    if not evidence_sentence:
        return False
    normalized_evidence = normalize_text(evidence_sentence)
    normalized_candidate = normalize_text(candidate_text)
    return bool(normalized_evidence and normalized_evidence in normalized_candidate)


def output_truncated_suspected(raw_output: str) -> bool:
    text = raw_output.strip()
    if not text:
        return False
    return not text.endswith("}")


def build_reasoning_candidates(transcript_candidates: list[dict]) -> list[ReasoningCandidate]:
    candidates = []
    for index, item in enumerate(transcript_candidates, start=1):
        candidates.append(
            ReasoningCandidate(
                candidate_index=index,
                candidate_id=candidate_id(item),
                source_name=str(item.get("source_name") or ""),
                timestamp=str(item.get("timestamp") or ""),
                transcript_path=str(item.get("transcript_path") or ""),
                retrieval_score=safe_float(item.get("score"), 0.0),
                snippet=snippet_text(item),
            )
        )
    return candidates


def build_prompt(question: str, candidates: list[ReasoningCandidate]) -> str:
    candidate_blocks = []
    for candidate in candidates:
        candidate_blocks.append(
            "\n".join(
                [
                    f"Candidate {candidate.candidate_index}",
                    f"id: {candidate.candidate_id}",
                    f"source: {candidate.source_name}",
                    f"timestamp: {candidate.timestamp}",
                    f"retrieval_score: {candidate.retrieval_score:.6f}",
                    "text:",
                    candidate.snippet,
                ]
            )
        )
    return (
        "You are a transcript reranker and evidence selector for a multimedia evidence retrieval system.\n\n"
        "Use only the transcript candidates provided below.\n"
        "Do not use outside knowledge.\n\n"
        "Your task is to form an answer hypothesis from the transcript candidates, identify direct supporting "
        "evidence, and rank the transcript candidates by usefulness.\n\n"
        "The answer_hypothesis must be concise, maximum one sentence, and inferred from the provided "
        "transcript candidates only. It may combine direct evidence across multiple candidates.\n\n"
        "Before selecting one evidence chunk, evaluate ALL transcript candidates.\n\n"
        "Do not stop after finding a plausible candidate.\n\n"
        "Your supporting evidence and ranking should represent your judgment of the entire candidate set.\n\n"
        "A response containing only 1-3 ranked candidates is usually incomplete when many candidates are "
        "available.\n\n"
        "Supporting evidence objective:\n\n"
        "Return supporting_candidate_indices containing candidates that directly support the answer_hypothesis. "
        "Do not return more than 10 supporting_candidate_indices. Include only the strongest direct evidence "
        "candidates. "
        "Prefer candidates containing concrete entities, names, objects, quantities, lists, direct statements, "
        "or presence/absence statements. Do not include candidates that only imply the answer or merely repeat "
        "question wording. Generic statements like \"I'm gathering X\" should not support a concrete answer "
        "unless they contain the concrete answer terms.\n\n"
        "selected_candidate_index must be one of supporting_candidate_indices and must contain direct evidence "
        "for at least part of answer_hypothesis. evidence_sentence must be copied or closely extracted from "
        "selected_candidate_index only.\n\n"
        "If no direct supporting evidence exists, supporting_candidate_indices may be empty and confidence "
        "should be low.\n\n"
        "Ranking objective:\n\n"
        "Rank candidates by expected usefulness for answering the question.\n\n"
        "Higher-ranked candidates should contain:\n\n"
        "* explicit factual information\n"
        "* direct evidence\n"
        "* concrete entities\n"
        "* proper names\n"
        "* specific objects\n"
        "* locations\n"
        "* quantities\n"
        "* dates or times\n"
        "* statements of presence or absence\n"
        "* statements of possession or non-possession\n"
        "* lists of specific items\n"
        "* direct observations\n"
        "* direct answers\n\n"
        "Medium-ranked candidates may contain:\n\n"
        "* partial evidence\n"
        "* supporting evidence\n"
        "* contextual evidence\n\n"
        "Lower-ranked candidates typically contain:\n\n"
        "* plans\n"
        "* intentions\n"
        "* speculation\n"
        "* generic discussion\n"
        "* conversational filler\n"
        "* meta commentary\n"
        "* topic mentions without evidence\n"
        "* statements that merely repeat words from the question\n\n"
        "Do not rank candidates highly simply because they contain words from the question.\n\n"
        "Prefer evidence-bearing statements over topic-related statements.\n\n"
        "After evaluating all candidates:\n\n"
        "1. Produce answer_hypothesis.\n"
        "2. Produce supporting_candidate_indices with direct evidence for answer_hypothesis.\n"
        "3. Produce a relevance ordering of the candidate indices.\n"
        "4. reasoning_ordered_candidate_indices should start with supporting_candidate_indices.\n"
        "5. Then append other useful candidates.\n"
        "6. Do not put generic implication-only candidates before direct supporting evidence.\n"
        "7. Do not return more than 20 reasoning_ordered_candidate_indices.\n"
        "8. Do not attempt to rank all candidates.\n\n"
        "Then:\n\n"
        "1. Set selected_candidate_index to the best single supporting candidate.\n"
        "2. selected_candidate_index must be one of supporting_candidate_indices.\n"
        "3. Generate evidence_sentence using only selected_candidate_index.\n"
        "4. The evidence_sentence must be copied or closely extracted from the selected candidate text.\n"
        "5. If no candidate provides meaningful direct evidence, return selected_candidate_index = null and "
        "confidence <= 0.3.\n\n"
        "For steering_keywords:\n\n"
        "* Extract only from the selected evidence_sentence.\n"
        "* Prefer exact words or short phrases.\n"
        "* Do not invent new terms.\n"
        "* Do not generate synonyms.\n"
        "* Do not generate category labels.\n"
        "* Do not paraphrase the question.\n"
        "* Do not include terms that do not occur in the evidence_sentence.\n\n"
        "Output constraints:\n\n"
        "* Return compact JSON only.\n"
        "* Keep answer_hypothesis to one sentence.\n"
        "* Keep reason to one sentence.\n"
        "* Do not include candidate text in the output.\n"
        "* Do not return more than 10 supporting_candidate_indices.\n"
        "* Do not return more than 20 reasoning_ordered_candidate_indices.\n"
        "* Do not return more than 5 steering_keywords.\n\n"
        "Return strict JSON only with keys:\n\n"
        "{\n"
        "\"answer_hypothesis\": string,\n"
        "\"supporting_candidate_indices\": [int],\n"
        "\"selected_candidate_index\": int | null,\n"
        "\"evidence_sentence\": string,\n"
        "\"confidence\": float,\n"
        "\"reason\": string,\n"
        "\"steering_keywords\": [string],\n"
        "\"reasoning_ordered_candidate_indices\": [int]\n"
        "}\n\n"
        f"Question: {question}\n\n"
        "Transcript candidates:\n\n"
        + "\n\n---\n\n".join(candidate_blocks)
        + "\n\nJSON schema:\n"
        + json.dumps(
            {
                "answer_hypothesis": "concise answer inferred only from transcript candidates",
                "supporting_candidate_indices": [1, 3],
                "selected_candidate_index": 1,
                "evidence_sentence": "copied evidence sentence from the selected candidate",
                "confidence": 0.8,
                "reason": "why this candidate answers the question",
                "steering_keywords": ["exact phrase from evidence"],
                "reasoning_ordered_candidate_indices": [1, 2, 3],
            }
        )
        + "\n\nJSON:"
    )


def build_reasoning_ordered_candidates(
    reasoning_input_indices: list[int],
    candidates: list[ReasoningCandidate],
    supporting_candidate_indices: list[int],
) -> list[dict[str, Any]]:
    ordered_indices = supporting_candidate_indices + reasoning_input_indices
    if not ordered_indices:
        return []
    candidate_map = {candidate.candidate_index: candidate for candidate in candidates}
    ordered: list[dict[str, Any]] = []
    seen: set[int] = set()

    for reasoning_rank, candidate_index in enumerate(ordered_indices, start=1):
        candidate = candidate_map.get(candidate_index or -1)
        if candidate is None or candidate.candidate_index in seen:
            continue
        seen.add(candidate.candidate_index)
        item = candidate_to_dict(candidate)
        item["raw_retrieval_rank"] = candidate.candidate_index
        item["reasoning_rank"] = reasoning_rank
        item["reasoning_score"] = None
        item["reasoning_reason"] = ""
        ordered.append(item)

    return ordered


def parse_candidate_index(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_candidate_indices(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    indices = []
    seen = set()
    for value in values:
        index = parse_candidate_index(value)
        if index is None or index in seen:
            continue
        seen.add(index)
        indices.append(index)
    return indices


def run_local_reasoning_model(prompt: str, model_name: str, *, max_new_tokens: int = 256) -> str:
    text, debug = run_local_reasoning_model_with_debug(prompt, model_name, max_new_tokens=max_new_tokens)
    del debug
    return text


def run_local_reasoning_model_with_debug(
    prompt: str,
    model_name: str,
    *,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    if not model_name:
        raise RuntimeError("no local reasoning model configured")
    model = load_reasoning_model(model_name)
    if model is None:
        raise RuntimeError(f"local reasoning model unavailable: {model_name}")
    tokenizer, loaded_model, torch, device, model_kind = model
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    if model_kind == "processor":
        inputs = tokenizer(text=[text], return_tensors="pt").to(device)
    else:
        inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = loaded_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    text_output = tokenizer.decode(generated, skip_special_tokens=True).strip()
    generation_token_count = int(generated.shape[-1])
    return text_output, {
        "max_new_tokens": int(max_new_tokens),
        "generation_token_count": generation_token_count,
        "generation_hit_limit": generation_token_count >= int(max_new_tokens),
        "raw_output_char_count": len(text_output),
    }


def empty_generation_debug(raw_output: str) -> dict[str, Any]:
    return {
        "max_new_tokens": None,
        "generation_token_count": None,
        "generation_hit_limit": None,
        "raw_output_char_count": len(raw_output),
    }


@lru_cache(maxsize=1)
def load_reasoning_model(model_name: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    path = Path(model_name)
    if not path.is_dir():
        return None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    except Exception:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model_kind = "tokenizer"
    except Exception:
        try:
            from transformers import AutoModelForImageTextToText

            tokenizer = AutoProcessor.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(
                str(path),
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            model_kind = "processor"
        except Exception:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration

                tokenizer = AutoProcessor.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(path),
                    local_files_only=True,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                model_kind = "processor"
            except Exception:
                return None
    device = next(model.parameters()).device
    model.eval()
    return tokenizer, model, torch, device, model_kind


def parse_strict_json(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("model output JSON is not an object")
    return parsed


def unavailable_result(reason: str, candidates: list[ReasoningCandidate]) -> dict[str, Any]:
    return {
        "selected_candidate_index": None,
        "selected_candidate_id": None,
        "reasoning_selected_retrieval_rank": None,
        "answer": "",
        "answer_hypothesis": "",
        "supporting_candidate_indices": [],
        "num_supporting_candidate_indices_raw": 0,
        "num_supporting_candidate_indices_final": 0,
        "num_reasoning_ordered_indices_raw": 0,
        "num_reasoning_ordered_indices_final": 0,
        "evidence_sentence": "",
        "confidence": 0.0,
        "reason": reason,
        "steering_keywords": [],
        "steering_paraphrases": [],
        "raw_model_output": "",
        "raw_model_output_length": 0,
        "max_new_tokens": DEFAULT_REASONING_MAX_NEW_TOKENS,
        "generation_token_count": None,
        "generation_hit_limit": None,
        "raw_output_char_count": 0,
        "json_parse_failed": False,
        "output_truncated_suspected": False,
        "model_name": resolve_reasoning_model_name(),
        "reasoning_candidates": [candidate_to_dict(candidate) for candidate in candidates],
        "reasoning_ordered_transcript_candidates": [],
        "reasoning_ordered_candidate_indices": [],
        "reasoning_order_used": False,
    }


def resolve_reasoning_model_name() -> str:
    env_value = os.environ.get("TRANSCRIPT_REASONING_MODEL")
    if env_value:
        return env_value
    if Path(DEFAULT_REASONING_MODEL).is_dir():
        return DEFAULT_REASONING_MODEL
    shared = ROOT.parent / "models"
    for name in ("qwen2.5-vl-3b", "Qwen2.5-VL-3B-Instruct", "qwen2.5-3b-instruct"):
        candidate = shared / name
        if candidate.is_dir():
            return str(candidate)
    return DEFAULT_REASONING_MODEL


def candidate_by_index(candidates: list[ReasoningCandidate], selected_index: Any) -> ReasoningCandidate | None:
    try:
        index = int(selected_index)
    except (TypeError, ValueError):
        return None
    for candidate in candidates:
        if candidate.candidate_index == index:
            return candidate
    return None


def candidate_to_dict(candidate: ReasoningCandidate) -> dict[str, Any]:
    return {
        "candidate_index": candidate.candidate_index,
        "candidate_id": candidate.candidate_id,
        "source_name": candidate.source_name,
        "timestamp": candidate.timestamp,
        "transcript_path": candidate.transcript_path,
        "retrieval_score": candidate.retrieval_score,
        "snippet": candidate.snippet,
    }


def candidate_id(item: dict[str, Any]) -> str:
    return str(item.get("source_id") or f"{item.get('transcript_path')}:{item.get('start_sec')}:{item.get('end_sec')}")


def snippet_text(item: dict[str, Any], limit: int = 1600) -> str:
    text = str(item.get("transcript_snippet") or item.get("text") or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def normalize_reasoning_terms(
    values: Any,
    *,
    question: str,
    evidence_sentence: str,
    require_evidence_terms: bool,
    reject_unsupported_single_token: bool = False,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    out = []
    seen = set()
    evidence_normalized = normalize_text(evidence_sentence)
    evidence_tokens = set(tokenize(evidence_sentence))
    question_tokens = set(content_tokens(question))
    for item in values:
        if isinstance(item, str):
            raw_term = item
            raw_type = ""
            raw_reason = "Suggested by transcript reasoning model."
            raw_confidence = None
        elif isinstance(item, dict):
            raw_term = str(item.get("term") or "")
            raw_type = str(item.get("type") or "")
            raw_reason = str(item.get("reason") or "Suggested by transcript reasoning model.")
            raw_confidence = item.get("confidence")
        else:
            continue
        term = normalize_text(raw_term)
        if not valid_reasoning_term(term):
            continue
        if require_evidence_terms and not term_is_supported_by_evidence(term, evidence_normalized, evidence_tokens):
            continue
        if reject_unsupported_single_token and len(content_tokens(term)) == 1:
            if not term_is_supported_by_evidence(term, evidence_normalized, evidence_tokens):
                continue
        if mostly_question_not_evidence(term, question_tokens, evidence_tokens):
            continue
        if term in seen:
            continue
        seen.add(term)
        term_type = raw_type
        if term_type not in {"keyword", "phrase"}:
            term_type = "phrase" if " " in term else "keyword"
        confidence = bounded_confidence(raw_confidence)
        out.append(
            {
                "term": term,
                "type": term_type,
                "source": "reasoning_lm",
                "reason": raw_reason,
                "confidence": confidence if confidence is not None else 0.5,
            }
        )
        if len(out) >= limit:
            break
    return out


def valid_reasoning_term(term: str) -> bool:
    if not term or len(term) > 40:
        return False
    if re.search(r"(.)\1{5,}", term):
        return False
    if not re.search(r"[a-z0-9]", term):
        return False
    if not content_tokens(term):
        return False
    return True


def term_is_supported_by_evidence(term: str, evidence_normalized: str, evidence_tokens: set[str]) -> bool:
    del evidence_tokens
    return bool(term) and term in evidence_normalized


def mostly_question_not_evidence(term: str, question_tokens: set[str], evidence_tokens: set[str]) -> bool:
    content = content_tokens(term)
    if not content:
        return True
    question_only = [token for token in content if token in question_tokens and token not in evidence_tokens]
    evidence_overlap = [token for token in content if token in evidence_tokens]
    return len(question_only) >= 2 and len(question_only) >= len(evidence_overlap)


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in TERM_STOPWORDS]


def normalize_text(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9' ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bounded_confidence(value: Any) -> float | None:
    parsed = safe_float(value, None)
    if parsed is None:
        return None
    return max(0.0, min(1.0, parsed))


def safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


TERM_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
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
    "with",
}
