from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_QA_MODEL_CANDIDATES = (
    "deepset/roberta-base-squad2",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-uncased-distilled-squad",
)
ROOT = Path(__file__).resolve().parents[1]
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class LocalQAModel:
    model_name: str
    tokenizer: Any
    model: Any
    torch: Any
    device: Any


@dataclass(frozen=True)
class TextSegment:
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ReducedContext:
    text: str
    mappings: tuple[tuple[int, int, int, int], ...]
    method: str
    original_char_start: int
    original_char_end: int


def find_answer_span(question: str, transcript_text: str, anchor_text: str | None = None) -> dict[str, Any]:
    if not question.strip() or not transcript_text.strip():
        return unavailable_answer_span("empty_input")

    qa = load_local_qa_model()
    if qa is None:
        return unavailable_answer_span("unavailable_no_qa_model")

    try:
        context = select_qa_context(question, transcript_text, anchor_text=anchor_text)
        return run_manual_qa(qa, question, transcript_text, context)
    except Exception:
        return unavailable_answer_span("unavailable_qa_inference_failed")


def run_manual_qa(
    qa: LocalQAModel,
    question: str,
    original_transcript_text: str,
    context: ReducedContext,
    *,
    max_answer_tokens: int = 20,
) -> dict[str, Any]:
    encoded = qa.tokenizer(
        question,
        context.text,
        return_tensors="pt",
        truncation="only_second",
        max_length=512,
        return_offsets_mapping=True,
    )
    offset_mapping = encoded.pop("offset_mapping")[0].tolist()
    sequence_ids = encoded.sequence_ids(0)
    inputs = {key: value.to(qa.device) for key, value in encoded.items()}

    qa.model.eval()
    with qa.torch.no_grad():
        outputs = qa.model(**inputs)
    start_logits = outputs.start_logits[0].detach().cpu()
    end_logits = outputs.end_logits[0].detach().cpu()

    candidates: list[dict[str, Any]] = []
    for start_index, sequence_id in enumerate(sequence_ids):
        if sequence_id != 1:
            continue
        start_offset = offset_mapping[start_index]
        if not valid_offset(start_offset):
            continue
        max_end_index = min(len(sequence_ids) - 1, start_index + max_answer_tokens - 1)
        for end_index in range(start_index, max_end_index + 1):
            if sequence_ids[end_index] != 1:
                continue
            end_offset = offset_mapping[end_index]
            if not valid_offset(end_offset):
                continue
            char_start = int(start_offset[0])
            char_end = int(end_offset[1])
            if char_end <= char_start:
                continue
            score = float(start_logits[start_index] + end_logits[end_index])
            candidates.append(
                {
                    "char_start": char_start,
                    "char_end": char_end,
                    "raw_score": score,
                    "start_index": start_index,
                    "end_index": end_index,
                }
            )

    answer_candidates = top_answer_candidates(
        candidates,
        original_transcript_text=original_transcript_text,
        context=context,
        limit=10,
    )
    if not answer_candidates:
        return unavailable_answer_span("unavailable_qa_empty_span")

    best = answer_candidates[0]

    return {
        "answer_span_text": best["text"],
        "char_start": best["char_start"],
        "char_end": best["char_end"],
        "score": best["score"],
        "raw_score": best["raw_score"],
        "method": f"extractive_qa_manual:{context.method}:{qa.model_name}",
        "answer_candidates": answer_candidates,
        "qa_context_method": context.method,
        "qa_context_text": context.text,
        "qa_context_original_char_start": context.original_char_start,
        "qa_context_original_char_end": context.original_char_end,
    }


def top_answer_candidates(
    candidates: list[dict[str, Any]],
    *,
    original_transcript_text: str,
    context: ReducedContext,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=lambda item: float(item["raw_score"]), reverse=True)
    deduped: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for candidate in ranked:
        original_start, original_end = map_context_offsets_to_original(
            context,
            int(candidate["char_start"]),
            int(candidate["char_end"]),
        )
        original_start, original_end = trim_original_offsets(original_transcript_text, original_start, original_end)
        text = original_transcript_text[original_start:original_end]
        normalized = normalize_candidate_text(text)
        if not normalized or normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        deduped.append(
            {
                "text": text,
                "char_start": original_start,
                "char_end": original_end,
                "score": normalized_span_confidence(float(candidate["raw_score"])),
                "raw_score": float(candidate["raw_score"]),
            }
        )
        if len(deduped) >= limit:
            break
    return deduped


def normalize_candidate_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def select_qa_context(
    question: str,
    transcript_text: str,
    *,
    anchor_text: str | None = None,
    window_words: int = 60,
    stride_words: int = 30,
) -> ReducedContext:
    segments = split_overlapping_word_windows(
        transcript_text,
        window_words=window_words,
        stride_words=stride_words,
    )
    if len(segments) <= 1:
        return build_reduced_context(segments, "full_context")

    model = load_minilm_model()
    if model is None:
        return build_reduced_context([TextSegment(transcript_text, 0, len(transcript_text))], "full_context_no_minilm")

    queries = [question]
    if anchor_text and anchor_text.strip():
        queries.append(anchor_text)
    try:
        embeddings = model.encode(
            queries + [segment.text for segment in segments],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception:
        return build_reduced_context([TextSegment(transcript_text, 0, len(transcript_text))], "full_context_minilm_failed")

    query_embedding = np.asarray(embeddings[0], dtype="float32")
    segment_embeddings = np.asarray(embeddings[len(queries) :], dtype="float32")
    scores = segment_embeddings @ query_embedding
    if len(queries) > 1:
        anchor_embedding = np.asarray(embeddings[1], dtype="float32")
        anchor_scores = segment_embeddings @ anchor_embedding
        scores = 0.65 * scores + 0.35 * anchor_scores
    best_index = int(np.argmax(scores))
    selected = merge_nearby_windows(transcript_text, segments, best_index)
    method = "minilm_window_context_with_anchor" if len(queries) > 1 else "minilm_window_context"
    return build_reduced_context([selected], method)


def split_overlapping_word_windows(
    text: str,
    *,
    window_words: int = 60,
    stride_words: int = 30,
) -> list[TextSegment]:
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    segments: list[TextSegment] = []
    if len(words) <= window_words:
        return [TextSegment(text.strip(), len(text) - len(text.lstrip()), len(text.rstrip()))]
    for start_word in range(0, len(words), stride_words):
        selected = words[start_word : start_word + window_words]
        if not selected:
            continue
        char_start = selected[0].start()
        char_end = selected[-1].end()
        segments.append(TextSegment(text[char_start:char_end], char_start, char_end))
        if start_word + window_words >= len(words):
            break
    return segments


def merge_nearby_windows(text: str, segments: list[TextSegment], best_index: int) -> TextSegment:
    selected_indices = [best_index]
    if best_index > 0:
        selected_indices.insert(0, best_index - 1)
    if best_index + 1 < len(segments):
        selected_indices.append(best_index + 1)
    char_start = min(segments[index].char_start for index in selected_indices)
    char_end = max(segments[index].char_end for index in selected_indices)
    return TextSegment(text[char_start:char_end], char_start, char_end)


def build_reduced_context(segments: list[TextSegment], method: str) -> ReducedContext:
    parts: list[str] = []
    mappings: list[tuple[int, int, int, int]] = []
    cursor = 0
    for segment in segments:
        if parts:
            parts.append("\n")
            cursor += 1
        reduced_start = cursor
        parts.append(segment.text)
        cursor += len(segment.text)
        mappings.append((reduced_start, cursor, segment.char_start, segment.char_end))
    if not mappings:
        return ReducedContext("", tuple(), method, -1, -1)
    return ReducedContext(
        "".join(parts),
        tuple(mappings),
        method,
        min(mapping[2] for mapping in mappings),
        max(mapping[3] for mapping in mappings),
    )


def map_context_offsets_to_original(context: ReducedContext, char_start: int, char_end: int) -> tuple[int, int]:
    if not context.mappings:
        return 0, 0
    for reduced_start, reduced_end, original_start, original_end in context.mappings:
        if char_start >= reduced_start and char_end <= reduced_end:
            return (
                original_start + (char_start - reduced_start),
                original_start + (char_end - reduced_start),
            )
    return context.mappings[0][2], context.mappings[-1][3]


def trim_original_offsets(text: str, char_start: int, char_end: int) -> tuple[int, int]:
    while char_start < char_end and text[char_start].isspace():
        char_start += 1
    while char_end > char_start and text[char_end - 1].isspace():
        char_end -= 1
    return char_start, char_end


@lru_cache(maxsize=1)
def load_minilm_model() -> Any | None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from sentence_transformers import SentenceTransformer

        model_name = os.environ.get("MINILM_MODEL_DIR", MINILM_MODEL_NAME)
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_local_qa_model() -> LocalQAModel | None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    except Exception:
        return None

    for model_name in qa_model_candidates():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name, local_files_only=True)
        except Exception:
            continue
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return LocalQAModel(
            model_name=model_name,
            tokenizer=tokenizer,
            model=model,
            torch=torch,
            device=device,
        )
    return None


def valid_offset(offset: list[int] | tuple[int, int]) -> bool:
    return len(offset) == 2 and int(offset[1]) > int(offset[0])


def normalized_span_confidence(raw_score: float) -> float:
    # Logit sums are uncalibrated; sigmoid gives a bounded confidence-like value.
    try:
        import math

        return max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-raw_score))))
    except OverflowError:
        return 0.0 if raw_score < 0 else 1.0


def qa_model_candidates() -> list[str]:
    values: list[str] = []
    env_value = os.environ.get("ANSWER_SPAN_QA_MODEL")
    if env_value:
        values.append(env_value)
    shared_models = ROOT.parent / "models"
    for name in (
        "roberta-base-squad2",
        "distilbert-base-cased-distilled-squad",
        "distilbert-base-uncased-distilled-squad",
    ):
        candidate = shared_models / name
        if candidate.is_dir():
            values.append(str(candidate))
    values.extend(DEFAULT_QA_MODEL_CANDIDATES)
    return dedupe_strings(values)


def dedupe_strings(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def unavailable_answer_span(method: str) -> dict[str, Any]:
    return {
        "answer_span_text": "",
        "char_start": -1,
        "char_end": -1,
        "score": 0.0,
        "method": method,
    }
