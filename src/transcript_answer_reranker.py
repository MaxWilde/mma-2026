from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.answer_span_highlight import find_answer_span, load_minilm_model


DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QAFunction = Callable[[str, str], dict[str, Any]]
RelevanceFunction = Callable[[str, list[str]], list[float]]


def rerank_transcript_answers(
    question: str,
    transcript_candidates: list[dict],
    top_n: int = 20,
    *,
    qa_function: QAFunction | None = None,
    relevance_function: RelevanceFunction | None = None,
) -> dict[str, Any]:
    candidates = [dict(item) for item in transcript_candidates[: max(0, top_n)]]
    if not candidates:
        return {
            "best_transcript": {},
            "best_answer": unavailable_best_answer("no_transcript_candidates"),
            "answer_rerank_candidates": [],
        }

    qa_function = qa_function or run_local_qa
    qa_results = []
    for index, candidate in enumerate(candidates, start=1):
        transcript_text = str(candidate.get("text") or candidate.get("transcript_snippet") or "")
        answer = qa_function(question, transcript_text)
        answer_text = str(answer.get("answer_span_text") or answer.get("text") or "")
        qa_context = str(answer.get("qa_context_text") or "")
        qa_results.append(
            {
                "candidate": candidate,
                "retrieval_rank": index,
                "retrieval_score": safe_float(candidate.get("score"), 0.0),
                "answer": answer,
                "answer_text": answer_text,
                "qa_context_text": qa_context,
                "semantic_text": build_semantic_text(answer_text, qa_context, transcript_text),
            }
        )

    semantic_scores = score_semantic_relevance(
        question,
        [item["semantic_text"] for item in qa_results],
        relevance_function=relevance_function,
    )
    retrieval_norm = normalize_values([item["retrieval_score"] for item in qa_results])
    semantic_norm = normalize_values(semantic_scores)

    ranked = []
    for index, item in enumerate(qa_results):
        answer_confidence = bounded_confidence(item["answer"].get("score"))
        final_answer_score = (
            0.35 * retrieval_norm[index]
            + 0.30 * (answer_confidence or 0.0)
            + 0.35 * semantic_norm[index]
        )
        candidate = item["candidate"]
        ranked.append(
            {
                "rank": item["retrieval_rank"],
                "source_name": candidate.get("source_name"),
                "timestamp": candidate.get("timestamp"),
                "transcript_path": candidate.get("transcript_path"),
                "retrieval_score": item["retrieval_score"],
                "retrieval_norm": retrieval_norm[index],
                "answer_text": item["answer_text"],
                "answer_confidence": answer_confidence,
                "answer_relevance_score": semantic_norm[index],
                "answer_relevance_raw_score": semantic_scores[index],
                "final_answer_score": final_answer_score,
                "snippet": preview_text(str(candidate.get("transcript_snippet") or candidate.get("text") or "")),
                "qa_context_text": item["qa_context_text"],
                "candidate": candidate,
                "answer": item["answer"],
            }
        )

    ranked.sort(key=lambda item: float(item["final_answer_score"]), reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["answer_rerank_rank"] = rank

    best = ranked[0]
    best_answer = format_best_answer(best["answer"])
    public_candidates = [public_candidate(item) for item in ranked]
    return {
        "best_transcript": dict(best["candidate"]),
        "best_answer": best_answer,
        "answer_rerank_candidates": public_candidates,
    }


def run_local_qa(question: str, transcript_text: str) -> dict[str, Any]:
    return find_answer_span(question, transcript_text)


def score_semantic_relevance(
    question: str,
    texts: list[str],
    *,
    relevance_function: RelevanceFunction | None = None,
) -> list[float]:
    if not texts:
        return []
    if relevance_function is not None:
        return [safe_float(value, 0.0) for value in relevance_function(question, texts)]

    cross_encoder = load_local_cross_encoder(DEFAULT_CROSS_ENCODER)
    if cross_encoder is not None:
        try:
            scores = cross_encoder.predict([(question, text) for text in texts], show_progress_bar=False)
            return [safe_float(score, 0.0) for score in scores]
        except Exception:
            pass

    model = load_minilm_model()
    if model is None:
        return lexical_relevance_scores(question, texts)
    try:
        embeddings = model.encode(
            [question] + texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = np.asarray(embeddings[0], dtype="float32")
        text_embeddings = np.asarray(embeddings[1:], dtype="float32")
        return [float(score) for score in (text_embeddings @ query_embedding)]
    except Exception:
        return lexical_relevance_scores(question, texts)


@lru_cache(maxsize=2)
def load_local_cross_encoder(model_name: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return None

    candidates: list[Path] = []
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


def build_semantic_text(answer_text: str, qa_context: str, transcript_text: str) -> str:
    context = qa_context.strip() or transcript_text.strip()
    if answer_text.strip():
        return f"{answer_text.strip()}\n{context}"
    return context


def normalize_values(values: list[float]) -> list[float]:
    if not values:
        return []
    values = [safe_float(value, 0.0) for value in values]
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [1.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def lexical_relevance_scores(question: str, texts: list[str]) -> list[float]:
    query_tokens = set(tokenize(question))
    scores = []
    for text in texts:
        text_tokens = set(tokenize(text))
        if not query_tokens or not text_tokens:
            scores.append(0.0)
        else:
            scores.append(len(query_tokens & text_tokens) / len(query_tokens | text_tokens))
    return scores


def tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def bounded_confidence(value: Any) -> float | None:
    if value is None:
        return None
    parsed = safe_float(value, None)
    if parsed is None:
        return None
    if 0.0 <= parsed <= 1.0:
        return parsed
    return None


def format_best_answer(answer: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": answer.get("answer_span_text") or answer.get("text") or "",
        "char_start": answer.get("char_start"),
        "char_end": answer.get("char_end"),
        "confidence": bounded_confidence(answer.get("score")),
        "raw_score": answer.get("raw_score"),
        "method": answer.get("method", "unavailable"),
        "qa_context_text": answer.get("qa_context_text"),
        "qa_context_method": answer.get("qa_context_method"),
        "answer_candidates": answer.get("answer_candidates", []),
    }


def unavailable_best_answer(method: str) -> dict[str, Any]:
    return {
        "text": "",
        "char_start": None,
        "char_end": None,
        "confidence": None,
        "method": method,
    }


def public_candidate(item: dict[str, Any]) -> dict[str, Any]:
    candidate = item["candidate"]
    return {
        "rank": item["rank"],
        "answer_rerank_rank": item["answer_rerank_rank"],
        "source_name": item.get("source_name"),
        "timestamp": item.get("timestamp"),
        "transcript_path": item.get("transcript_path"),
        "candidate_key": candidate_key(candidate),
        "retrieval_score": item["retrieval_score"],
        "retrieval_norm": item["retrieval_norm"],
        "answer_text": item["answer_text"],
        "answer_confidence": item["answer_confidence"],
        "answer_relevance_score": item["answer_relevance_score"],
        "answer_relevance_raw_score": item["answer_relevance_raw_score"],
        "final_answer_score": item["final_answer_score"],
        "snippet": item["snippet"],
    }


def candidate_key(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("source_id")
        or f"{candidate.get('transcript_path')}:{candidate.get('start_sec')}:{candidate.get('end_sec')}"
    )


def preview_text(text: str, limit: int = 260) -> str:
    import re

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
