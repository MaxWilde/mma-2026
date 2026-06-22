from __future__ import annotations

import re
from collections import Counter


STOPWORDS = {
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
    "how",
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
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
}


def build_transcript_heatmap(
    question: str,
    transcript_text: str,
    anchor_text: str | None = None,
    answer_span: dict | None = None,
) -> list[dict[str, float | str]]:
    question_terms = weighted_terms(question)
    anchor_terms = weighted_terms(anchor_text or "")
    spans = split_preserving_whitespace(transcript_text)
    raw_scores: list[float] = []
    cursor = 0
    answer_start = int(answer_span.get("char_start", -1)) if answer_span else -1
    answer_end = int(answer_span.get("char_end", -1)) if answer_span else -1

    for span in spans:
        token = normalize_token(span)
        span_start = cursor
        span_end = cursor + len(span)
        cursor = span_end
        if not token:
            raw_scores.append(0.0)
            continue
        stemmed = stem_token(token)
        score = 0.0
        if stemmed in question_terms:
            score += question_terms[stemmed]
        if stemmed in anchor_terms:
            score += 0.7 * anchor_terms[stemmed]
        if answer_start >= 0 and answer_end > answer_start:
            if span_start < answer_end and span_end > answer_start:
                score = max(score, 3.0)
            elif min(abs(span_start - answer_end), abs(span_end - answer_start)) <= 80:
                score = max(score, 1.2)
        raw_scores.append(score)

    max_score = max(raw_scores, default=0.0)
    if max_score <= 0.0:
        normalized_scores = [0.0 for _ in raw_scores]
    else:
        normalized_scores = [min(1.0, score / max_score) for score in raw_scores]

    return [
        {"text": span, "score": float(score)}
        for span, score in zip(spans, normalized_scores)
    ]


def weighted_terms(text: str) -> dict[str, float]:
    tokens = [stem_token(token) for token in tokenize(text) if token not in STOPWORDS]
    counts = Counter(tokens)
    return {token: 1.0 + min(1.0, 0.25 * (count - 1)) for token, count in counts.items()}


def split_preserving_whitespace(text: str) -> list[str]:
    return [span for span in re.findall(r"\s+|[^\s]+", text) if span]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize_token(text: str) -> str:
    tokens = tokenize(text)
    return tokens[0] if tokens else ""


def stem_token(token: str) -> str:
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token
