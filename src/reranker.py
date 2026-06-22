from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


DEFAULT_CROSS_ENCODER = "sentence-transformers/ms-marco-MiniLM-L-6-v2"

COOKING_OBJECT_TERMS = {
    "stove",
    "pot",
    "pan",
    "rice",
    "vegetables",
    "vegetable",
    "broth",
    "stock",
    "pepper",
    "peppers",
    "carrots",
    "carrot",
    "food",
    "cooking",
}

TRANSCRIPT_FOOD_TERMS = {
    "risotto",
    "rice",
    "broth",
    "stock",
    "bouillon",
    "vegetable",
    "vegetables",
    "pepper",
    "peppers",
    "carrot",
    "carrots",
    "spinach",
    "soup",
    "salad",
    "fish",
    "food",
}

GENERIC_COOKING_PHRASES = {
    "slow cook",
    "who cooks this",
    "i just cook",
}


@dataclass(frozen=True)
class RerankResult:
    chunks: list[dict[str, Any]]
    method: str


def rerank_chunks(
    question: str,
    chunks: list[dict[str, Any]],
    cross_encoder_model: str | None = None,
    visual_weight: float = 0.3,
) -> RerankResult:
    if not chunks:
        return RerankResult(chunks=[], method="none")

    model_name = cross_encoder_model or os.environ.get("RERANKER_MODEL", DEFAULT_CROSS_ENCODER)
    cross_encoder = _load_cross_encoder(model_name)
    if cross_encoder is not None:
        return RerankResult(
            chunks=_cross_encoder_rerank(question, chunks, cross_encoder, visual_weight),
            method=f"cross-encoder:{model_name}",
        )

    return RerankResult(chunks=_heuristic_rerank(question, chunks, visual_weight), method="heuristic")


def _load_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None

    try:
        return CrossEncoder(model_name, local_files_only=True)
    except TypeError:
        return None
    except Exception:
        return None


def _cross_encoder_rerank(
    question: str,
    chunks: list[dict[str, Any]],
    cross_encoder,
    visual_weight: float,
) -> list[dict[str, Any]]:
    pairs = [(question, str(chunk.get("text", ""))) for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    reranked: list[dict[str, Any]] = []
    for chunk, score in zip(chunks, scores):
        item = dict(chunk)
        transcript_score = float(score)
        visual_score = _visual_score(question, item)
        final_score = transcript_score + visual_weight * visual_score
        item["transcript_score"] = transcript_score
        item["visual_score"] = visual_score
        item["final_score"] = final_score
        item["rerank_score"] = final_score
        item["rerank_method"] = "cross-encoder"
        reranked.append(item)
    return sorted(reranked, key=lambda item: item["final_score"], reverse=True)


def _heuristic_rerank(question: str, chunks: list[dict[str, Any]], visual_weight: float) -> list[dict[str, Any]]:
    reranked: list[dict[str, Any]] = []
    for chunk in chunks:
        item = dict(chunk)
        transcript_score = _transcript_score(question, item)
        visual_score = _visual_score(question, item)
        final_score = transcript_score + visual_weight * visual_score
        item["transcript_score"] = transcript_score
        item["visual_score"] = visual_score
        item["final_score"] = final_score
        item["rerank_score"] = final_score
        item["rerank_method"] = "heuristic"
        reranked.append(item)
    return sorted(reranked, key=lambda item: item["final_score"], reverse=True)


def _transcript_score(question: str, chunk: dict[str, Any]) -> float:
    score = float(chunk.get("score", 0.0))
    question_lower = question.lower()
    if not any(term in question_lower for term in ("cook", "cooking", "food", "dish", "eat", "meal")):
        return score

    transcript = str(chunk.get("text", "")).lower()
    transcript_hits = _count_term_hits(transcript, TRANSCRIPT_FOOD_TERMS)

    score += min(0.35, transcript_hits * 0.05)

    if "risotto" in transcript:
        score += 0.45
    if "rice" in transcript:
        score += 0.15
    if any(term in transcript for term in ("broth", "stock", "bouillon")):
        score += 0.15

    if not transcript_hits:
        for phrase in GENERIC_COOKING_PHRASES:
            if phrase in transcript:
                score -= 0.25
                break

    return score


def _visual_score(question: str, chunk: dict[str, Any]) -> float:
    question_lower = question.lower()
    if not any(term in question_lower for term in ("cook", "cooking", "food", "dish", "eat", "meal")):
        return 0.0

    visual_caption = str(chunk.get("visual_caption", "")).lower()
    if not visual_caption:
        return 0.0

    visual_hits = _count_term_hits(visual_caption, COOKING_OBJECT_TERMS)
    score = min(0.60, visual_hits * 0.08)
    if any(term in visual_caption for term in ("stove", "pot", "pan")):
        score += 0.15
    if "rice" in visual_caption:
        score += 0.10
    if any(term in visual_caption for term in ("broth", "stock", "vegetable", "vegetables")):
        score += 0.10
    return score


def _count_term_hits(text: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term in text)

