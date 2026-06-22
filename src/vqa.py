from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from src.retriever import DEFAULT_MODEL_NAME
from src.retriever import load_embedding_model, load_index, query_index


EXTRACTIVE_SEMANTIC_THRESHOLD = 0.25
EXTRACTIVE_TOKEN_THRESHOLD = 0.08


@dataclass(frozen=True)
class EvidenceSentence:
    text: str
    source: str
    chunk_rank: int
    chunk: dict
    score: float = 0.0


@dataclass(frozen=True)
class ExtractiveAnswer:
    text: str
    best_sentence: str | None
    best_chunk: dict | None
    best_source: str | None

    def __str__(self) -> str:
        return self.text


def format_timestamp(start_sec: float, end_sec: float) -> str:
    return f"{_format_seconds(start_sec)}-{_format_seconds(end_sec)}"


def answer_from_chunks(question: str, chunks: list[dict], max_chunks: int = 5) -> str:
    if not chunks:
        return "I could not find transcript or visual-caption evidence relevant to the question."

    lines = [
        "Retrieved evidence:",
        f"Question: {question}",
        "",
    ]
    for idx, chunk in enumerate(chunks[:max_chunks], start=1):
        timestamp = format_timestamp(float(chunk["start_sec"]), float(chunk["end_sec"]))
        source = f"{chunk['source_name']} {chunk['day']} {chunk['video_id']}"
        transcript = str(chunk.get("text", "")).strip()
        visual_caption = str(chunk.get("visual_caption", "")).strip()
        if transcript:
            lines.append(f"{idx}. [{source}, {timestamp}] Transcript: {transcript}")
        if visual_caption:
            lines.append(f"{idx}. [{source}, {timestamp}] Visual caption: {visual_caption}")
    return "\n".join(lines)


def synthesize_answer_from_chunks(question: str, chunks: list[dict], max_chunks: int = 5) -> str:
    if not chunks:
        return "I could not find transcript or visual-caption evidence relevant to the question."

    evidence_chunks = chunks[:max_chunks]
    evidence_summary = _generic_answer_summary(evidence_chunks)
    lines = [
        evidence_summary,
        "",
        "Evidence:",
    ]
    for idx, chunk in enumerate(evidence_chunks, start=1):
        timestamp = format_timestamp(float(chunk["start_sec"]), float(chunk["end_sec"]))
        source = f"{chunk['source_name']} {chunk['day']} {chunk['video_id']}"
        transcript = _shorten(str(chunk.get("text", "")).strip(), 280)
        lines.append(f"{idx}. [{source}, {timestamp}] Transcript: {transcript}")
        visual_caption = str(chunk.get("visual_caption", "")).strip()
        if visual_caption:
            lines.append(f"   Visual caption: {_shorten(visual_caption, 220)}")
    return "\n".join(lines)


def synthesize_answer_extractive(
    question: str,
    chunks: list[dict],
    max_chunks: int = 5,
    embedding_model=None,
) -> ExtractiveAnswer:
    sentences = _collect_evidence_sentences(chunks[:max_chunks])
    if not sentences:
        return ExtractiveAnswer("The retrieved evidence is inconclusive.", None, None, None)

    ranked_sentences, threshold = _rank_evidence_sentences(question, sentences, embedding_model=embedding_model)
    selected = [sentence for sentence in ranked_sentences if sentence.score >= threshold][:3]
    if not selected:
        return ExtractiveAnswer("The retrieved evidence is inconclusive.", None, None, None)

    best_text = _trim_evidence_span(question, selected[0].text)
    lines = [f"Best evidence: {best_text}"]
    if len(selected) > 1:
        lines.append("Supporting evidence:")
        for sentence in selected[1:]:
            lines.append(f"- {_trim_evidence_span(question, sentence.text)}")
    return ExtractiveAnswer("\n".join(lines), best_text, selected[0].chunk, selected[0].source)


def match_transcript_evidence(
    question: str,
    chunks: list[dict],
    max_chunks: int = 5,
    embedding_model=None,
) -> ExtractiveAnswer:
    return synthesize_answer_extractive(
        question,
        chunks,
        max_chunks=max_chunks,
        embedding_model=embedding_model,
    )


def ask_question(index_dir: str, question: str, top_k: int = 5) -> tuple[str, list[dict]]:
    index, metadata, model_name = load_index(index_dir)
    model = load_embedding_model(model_name)
    chunks = query_index(question, index, metadata, model, top_k=top_k)
    return answer_from_chunks(question, chunks), chunks


def _collect_evidence_sentences(chunks: list[dict]) -> list[EvidenceSentence]:
    sentences: list[EvidenceSentence] = []
    for chunk_rank, chunk in enumerate(chunks, start=1):
        for field_name, source_label in (("text", "transcript"), ("visual_caption", "visual caption")):
            text = str(chunk.get(field_name, "")).strip()
            if not text:
                continue
            for sentence in _split_sentences(text):
                if sentence:
                    sentence_chunk = _chunk_with_keyframe_alias(chunk, field_name)
                    sentences.append(
                        EvidenceSentence(
                            text=sentence,
                            source=source_label,
                            chunk_rank=chunk_rank,
                            chunk=sentence_chunk,
                        )
                    )
    return sentences


def _rank_evidence_sentences(
    question: str,
    sentences: list[EvidenceSentence],
    embedding_model=None,
) -> tuple[list[EvidenceSentence], float]:
    semantic_scores = _semantic_sentence_scores(question, [sentence.text for sentence in sentences], embedding_model)
    if semantic_scores is not None:
        scored = [
            EvidenceSentence(sentence.text, sentence.source, sentence.chunk_rank, sentence.chunk, float(score))
            for sentence, score in zip(sentences, semantic_scores)
        ]
        return sorted(scored, key=_sentence_sort_key, reverse=True), EXTRACTIVE_SEMANTIC_THRESHOLD

    scored = [
        EvidenceSentence(
            sentence.text,
            sentence.source,
            sentence.chunk_rank,
            sentence.chunk,
            _token_overlap_score(question, sentence.text),
        )
        for sentence in sentences
    ]
    return sorted(scored, key=_sentence_sort_key, reverse=True), EXTRACTIVE_TOKEN_THRESHOLD


def _chunk_with_keyframe_alias(chunk: dict, field_name: str) -> dict:
    item = dict(chunk)
    if field_name == "visual_caption":
        keyframe_path = item.get("visual_caption_keyframe_path") or item.get("closest_keyframe_path")
    else:
        keyframe_path = item.get("closest_keyframe_path") or item.get("visual_caption_keyframe_path")
    item["keyframe_path"] = keyframe_path
    return item


def _semantic_sentence_scores(question: str, sentence_texts: list[str], embedding_model=None) -> list[float] | None:
    try:
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer

            embedding_model = SentenceTransformer(DEFAULT_MODEL_NAME, local_files_only=True)
        embeddings = embedding_model.encode(
            [question, *sentence_texts],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception:
        return None

    vectors = np.asarray(embeddings, dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != len(sentence_texts) + 1:
        return None
    question_vector = vectors[0]
    sentence_vectors = vectors[1:]
    return [float(score) for score in sentence_vectors @ question_vector]


def _token_overlap_score(question: str, sentence: str) -> float:
    question_tokens = _content_tokens(question)
    sentence_tokens = _content_tokens(sentence)
    if not question_tokens or not sentence_tokens:
        return 0.0
    overlap = question_tokens & sentence_tokens
    return len(overlap) / max(len(question_tokens), 1)


def _content_tokens(text: str) -> set[str]:
    stopwords = _content_stopwords()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if not normalized:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|(?:\s+-\s+)|(?:\s+\d+\.\s+)", normalized)
    return [_shorten(piece.strip(" -;:"), 320) for piece in pieces if len(piece.strip(" -;:")) >= 4]


def _sentence_sort_key(sentence: EvidenceSentence) -> tuple[float, int]:
    return sentence.score, -sentence.chunk_rank


def _trim_evidence_span(question: str, evidence: str, min_words: int = 15, max_words: int = 30) -> str:
    words = re.findall(r"\S+", evidence)
    if len(words) <= max_words:
        return evidence

    question_tokens = _content_tokens(question)
    if not question_tokens:
        return " ".join(words[:max_words])

    normalized_words = [_normalize_token(word) for word in words]
    overlap_positions = [
        index for index, token in enumerate(normalized_words) if token in question_tokens
    ]
    if overlap_positions:
        start, end = _best_overlap_context_window(normalized_words, overlap_positions, question_tokens)
        if end - start > max_words:
            end = min(len(words), start + max_words)
        start, end = _trim_filler_edges(words, normalized_words, start, end, min_words=1)
        return " ".join(words[start:end]).strip(" ,;:-")

    best_start = 0
    best_end = min(len(words), max_words)
    best_score = -1.0

    target_words = min(max_words, max(min_words, 22))
    for window_size in range(min_words, max_words + 1):
        for start in range(0, len(words) - window_size + 1):
            end = start + window_size
            window_tokens = {token for token in normalized_words[start:end] if token}
            overlap = question_tokens & window_tokens
            if not overlap:
                continue
            density = len(overlap) / window_size
            size_bonus = 1.0 - abs(window_size - target_words) / max_words
            score = len(overlap) + density + (0.15 * size_bonus)
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    start, end = _trim_filler_edges(words, normalized_words, best_start, best_end, min_words)
    span = " ".join(words[start:end]).strip()
    return span.strip(" ,;:-")


def _trim_filler_edges(
    words: list[str],
    normalized_words: list[str],
    start: int,
    end: int,
    min_words: int,
) -> tuple[int, int]:
    while end - start > min_words and _is_low_value_edge_token(normalized_words[start]):
        start += 1
    while end - start > min_words and _is_low_value_edge_token(normalized_words[end - 1]):
        end -= 1
    return start, end


def _best_overlap_context_window(
    normalized_words: list[str],
    overlap_positions: list[int],
    question_tokens: set[str],
    pre_context: int = 10,
    post_context: int = 8,
) -> tuple[int, int]:
    best_start = 0
    best_end = min(len(normalized_words), pre_context + post_context + 1)
    best_score = -1.0
    for position in overlap_positions:
        start = max(0, position - pre_context)
        end = min(len(normalized_words), position + post_context)
        window_tokens = {token for token in normalized_words[start:end] if token}
        overlap = question_tokens & window_tokens
        density = len(overlap) / max(end - start, 1)
        score = len(overlap) + density
        if score > best_score:
            best_score = score
            best_start = start
            best_end = end
    return best_start, best_end


def _is_low_value_edge_token(token: str) -> bool:
    if not token:
        return False
    return token in _content_stopwords() or len(token) <= 2


def _content_stopwords() -> set[str]:
    return {
        "a",
        "an",
        "and",
        "are",
        "at",
        "do",
        "does",
        "for",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "one",
        "the",
        "there",
        "they",
        "to",
        "what",
        "where",
        "who",
        "with",
        "this",
        "that",
        "yeah",
        "yes",
        "like",
        "probably",
        "maybe",
        "just",
        "really",
        "very",
        "uh",
        "um",
        "i",
        "don",
        "know",
    }


def _normalize_token(text: str) -> str:
    match = re.search(r"[a-z0-9]+", text.lower())
    return match.group(0) if match else ""


def _generic_answer_summary(chunks: list[dict]) -> str:
    best = chunks[0] if chunks else {}
    visual_caption = str(best.get("visual_caption", "")).strip()
    transcript = str(best.get("text", "")).strip()
    if visual_caption and transcript:
        return _shorten(
            f"Top retrieved evidence combines visual caption evidence ({visual_caption}) "
            f"with transcript evidence ({transcript}).",
            300,
        )
    if visual_caption:
        return _shorten(f"Top retrieved evidence is visual: {visual_caption}", 300)
    if transcript:
        return _shorten(f"Top retrieved evidence is transcript: {transcript}", 300)
    return "The retrieved evidence does not contain enough text for an evidence summary."


def _shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _format_seconds(total_seconds: float) -> str:
    total = max(0, int(round(total_seconds)))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"
