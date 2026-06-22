from __future__ import annotations

import re
from typing import Any


# Deprecated: retained as a lightweight fallback/experiment. The production steering
# path uses src.transcript_keyword_recommender.recommend_transcript_keywords.


def extract_keywords_from_transcript_candidates(
    question: str,
    transcript_candidates: list[dict],
    top_n: int = 10,
    max_keywords: int = 12,
) -> dict[str, Any]:
    question_tokens = set(content_tokens(question))
    question_topic_phrases = topic_phrases(question)
    candidates: dict[str, dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(transcript_candidates[: max(0, top_n)], start=1):
        text = str(candidate.get("transcript_snippet") or candidate.get("text") or "")
        candidate_id = candidate_key(candidate)
        for term, reason, base_score in extract_terms_from_text(text):
            term = normalize_term(term)
            protected = base_score >= 0.83 or reason.startswith("Quoted phrase")
            if not valid_term(term, question_tokens, question_topic_phrases, protected=protected):
                continue
            item = candidates.get(term)
            rank_boost = 1.0 / max(candidate_index, 1)
            confidence = min(1.0, base_score + 0.12 * rank_boost)
            if item is None:
                candidates[term] = {
                    "term": term,
                    "type": "phrase" if " " in term else "keyword",
                    "source": "candidate_keyword_extractor",
                    "reason": reason,
                    "confidence": confidence,
                    "candidate_indices": [candidate_index],
                    "candidate_ids": [candidate_id],
                }
            else:
                item["confidence"] = max(float(item["confidence"]), confidence)
                if candidate_index not in item["candidate_indices"]:
                    item["candidate_indices"].append(candidate_index)
                if candidate_id not in item["candidate_ids"]:
                    item["candidate_ids"].append(candidate_id)

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            float(item["confidence"]),
            len(item["candidate_ids"]),
            1.0 / max(min(item["candidate_indices"]), 1),
        ),
        reverse=True,
    )
    return {
        "source": "multi_candidate_transcript_keywords",
        "top_n": top_n,
        "suggested_terms": ranked[:max_keywords],
    }


def extract_terms_from_text(text: str) -> list[tuple[str, str, float]]:
    terms: list[tuple[str, str, float]] = []
    for clause in split_clauses(text):
        normalized = normalize_term(clause)
        if not normalized:
            continue
        terms.extend(absence_terms(normalized))
        terms.extend(answer_cue_terms(normalized))
        terms.extend(spoken_answer_terms(normalized))
        terms.extend(quoted_terms(clause))
    return terms


def absence_terms(clause: str) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    if re.search(r"\bnothing\b", clause):
        out.append(("nothing", "Explicit absence term from retrieved transcript evidence.", 0.86))

    patterns = [
        (r"\bwe\s+have\s+no\s+(.+)$", "we have no"),
        (r"\bthere\s+(?:is|are|was|were)\s+no\s+(.+)$", "there is no"),
        (r"\b(?:do\s+not|don't|dont)\s+have\s+(.+)$", "don't have"),
        (r"\b(?:not\s+available|missing|without|empty)\s+(.+)$", None),
        (r"\bno\s+(.+)$", None),
    ]
    for pattern, cue in patterns:
        match = re.search(pattern, clause)
        if not match:
            continue
        if cue:
            out.append((cue, "Exact absence cue from retrieved transcript evidence.", 0.88))
        remainder = clean_remainder(match.group(1))
        for phrase in nounish_phrases(remainder):
            out.append((phrase, "Concrete phrase from an explicit absence statement.", 0.82))
    return out


def spoken_answer_terms(clause: str) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    patterns = [
        r"\banswer\s+(?:was|is)\s+(.+)$",
        r"\b(?:answer(?:ed)?|said|called|named)\s+(.+)$",
        r"\b(?:it's|it\s+is|that\s+is|that's)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, clause)
        if not match:
            continue
        remainder = clean_remainder(match.group(1))
        for phrase in nounish_phrases(remainder, max_phrases=2):
            out.append((phrase, "Phrase near a spoken answer cue in retrieved transcript evidence.", 0.74))
    return out


def answer_cue_terms(clause: str) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    for cue in ("answer", "said", "called", "named"):
        if re.search(rf"\b{cue}\b", clause):
            out.append((cue, "Answer-bearing cue from retrieved transcript evidence.", 0.83))
    tokens = content_tokens(clause)
    if len(tokens) <= 3 and tokens:
        for token in tokens:
            out.append((token, "Short spoken answer token from retrieved transcript evidence.", 0.78))
    return out


def quoted_terms(clause: str) -> list[tuple[str, str, float]]:
    out = []
    for match in re.finditer(r"(?<![A-Za-z0-9])['\"]([A-Za-z0-9][^'\"]{0,38}[A-Za-z0-9])['\"](?![A-Za-z0-9])", clause):
        phrase = normalize_term(match.group(1))
        if valid_quoted_phrase(phrase):
            out.append((phrase, "Quoted phrase from retrieved transcript evidence.", 0.72))
    return out


def nounish_phrases(text: str, max_phrases: int = 4) -> list[str]:
    tokens = [token for token in tokenize(text) if token not in STOPWORDS]
    phrases = []
    if not tokens:
        return phrases
    if len(tokens) <= 3:
        phrases.append(" ".join(tokens))
    else:
        for size in (3, 2, 1):
            for start in range(0, len(tokens) - size + 1):
                phrases.append(" ".join(tokens[start : start + size]))
                if len(phrases) >= max_phrases:
                    return dedupe_strings(phrases)
    return dedupe_strings(phrases[:max_phrases])


def split_clauses(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,.;!?]\s+|[,.;!?]$", text) if part.strip()]


def clean_remainder(text: str) -> str:
    text = re.split(r"\b(?:and then|but|so|because|while|when)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return normalize_term(text)


def valid_term(term: str, question_tokens: set[str], question_topic_phrases: set[str], protected: bool = False) -> bool:
    if not term or len(term) > 40:
        return False
    if re.search(r"(.)\1{5,}", term):
        return False
    tokens = content_tokens(term)
    if not tokens:
        return False
    if len(tokens) == 1 and len(tokens[0]) < 3:
        return False
    if protected or term in ANSWER_CUE_TERMS:
        return True
    overlap = sum(1 for token in tokens if token in question_tokens)
    if overlap and overlap / len(tokens) >= 0.67:
        return False
    if is_question_topic_term(term, tokens, question_tokens, question_topic_phrases):
        return False
    return True


def valid_quoted_phrase(phrase: str) -> bool:
    tokens = tokenize(phrase)
    if not tokens:
        return False
    if not content_tokens(phrase):
        return False
    if tokens[0] in ORPHAN_FRAGMENTS or tokens[-1] in ORPHAN_FRAGMENTS:
        return False
    if len(tokens) > 8:
        return False
    return True


def is_question_topic_term(term: str, tokens: list[str], question_tokens: set[str], question_topic_phrases: set[str]) -> bool:
    if term in question_topic_phrases:
        return True
    if len(tokens) == 1 and tokens[0] in question_tokens:
        return True
    question_overlap = sum(1 for token in tokens if token in question_tokens)
    return len(tokens) > 1 and question_overlap >= max(1, len(tokens) - 1)


def topic_phrases(question: str) -> set[str]:
    tokens = content_tokens(question)
    phrases = set(tokens)
    for size in (2, 3, 4):
        for start in range(0, len(tokens) - size + 1):
            phrases.add(" ".join(tokens[start : start + size]))
    return phrases


def candidate_key(item: dict[str, Any]) -> str:
    return str(item.get("source_id") or f"{item.get('transcript_path')}:{item.get('start_sec')}:{item.get('end_sec')}")


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize_term(term: str) -> str:
    term = re.sub(r"[^A-Za-z0-9' ]+", " ", term)
    return re.sub(r"\s+", " ", term.strip().lower())


def dedupe_strings(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        normalized = normalize_term(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


STOPWORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "just",
    "of",
    "on",
    "or",
    "the",
    "there",
    "this",
    "to",
    "try",
    "trying",
    "was",
    "we",
    "were",
    "with",
}


ORPHAN_FRAGMENTS = {
    "didn",
    "don",
    "doesn",
    "hadn",
    "hasn",
    "haven",
    "isn",
    "ll",
    "re",
    "t",
    "ve",
    "wasn",
    "weren",
}


ANSWER_CUE_TERMS = {
    "answer",
    "called",
    "named",
    "said",
}
