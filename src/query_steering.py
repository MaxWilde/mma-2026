from __future__ import annotations

import json
import re
from typing import Any, Callable

from src.transcript_reasoning_answer import parse_strict_json, resolve_reasoning_model_name, run_local_reasoning_model


ModelFunction = Callable[[str], str]


def suggest_steering_terms(
    question: str,
    retrieved_candidates: list[dict],
    max_terms: int = 12,
    *,
    model_function: ModelFunction | None = None,
) -> dict[str, Any]:
    heuristic = heuristic_steering_terms(question, retrieved_candidates)
    llm = llm_steering_terms(question, retrieved_candidates, model_function=model_function)
    merged = merge_suggestions(heuristic + llm, max_terms=max_terms)
    return {
        "question": question,
        "suggested_terms": merged,
    }


def heuristic_steering_terms(question: str, retrieved_candidates: list[dict] | None = None) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for term in morphology_based_answer_terms(question):
        suggestions.append(
            suggestion(term, "phrase" if " " in term else "keyword", "Morphological answer-bearing variant from the question.", 0.56)
        )
    for phrase in answer_bearing_phrases(retrieved_candidates or [], question):
        suggestions.append(
            suggestion(phrase, "phrase" if " " in phrase else "keyword", "Answer-bearing phrase found in retrieved transcript snippets.", 0.62)
        )
    return suggestions


def llm_steering_terms(
    question: str,
    retrieved_candidates: list[dict],
    *,
    model_function: ModelFunction | None = None,
) -> list[dict[str, Any]]:
    prompt = steering_prompt(question, retrieved_candidates[:8])
    raw_output = ""
    try:
        raw_output = model_function(prompt) if model_function else run_local_reasoning_model(prompt, resolve_reasoning_model_name())
        parsed = parse_strict_json(raw_output)
    except Exception:
        return []
    values = parsed.get("suggested_terms")
    if not isinstance(values, list):
        return []
    out = []
    allowed_terms = allowed_specific_terms(question, retrieved_candidates)
    for item in values:
        if not isinstance(item, dict):
            continue
        term = normalize_term(str(item.get("term") or ""))
        if not term or contains_unseen_specific_entity(term, allowed_terms):
            continue
        if not is_valid_suggestion_term(term, question):
            continue
        term_type = "phrase" if " " in term else "keyword"
        if item.get("type") in {"keyword", "phrase"}:
            term_type = str(item["type"])
        base_confidence = bounded_confidence(item.get("confidence"), default=0.5)
        adjusted_confidence = score_suggestion_confidence(term, question, retrieved_candidates, base_confidence)
        out.append(
            {
                "term": term,
                "type": term_type,
                "reason": str(item.get("reason") or "Suggested by local transcript steering model."),
                "source": "llm",
                "confidence": adjusted_confidence,
            }
        )
    return out


def steering_prompt(question: str, retrieved_candidates: list[dict]) -> str:
    snippets = []
    for rank, item in enumerate(retrieved_candidates, start=1):
        text = re.sub(r"\s+", " ", str(item.get("transcript_snippet") or item.get("text") or "")).strip()
        snippets.append(f"{rank}. {item.get('source_name')} {item.get('timestamp')}: {text[:500]}")
    return (
        "Given the user question and current top retrieved transcript snippets, suggest search terms "
        "that are likely to retrieve answer-bearing transcript evidence that was not strongly represented "
        "in the current top-ranked results. Avoid paraphrasing the question. Avoid generic topic words. "
        "Prefer phrases likely to occur in answers, such as negations, named answers, definitions, quoted "
        "answers, explanations, and short spoken-answer formulations. Use only the question and "
        "retrieved snippets. Do not guess the answer. Do not include specific entities unless they appear "
        "in the question or retrieved snippets. Suggest terms that could surface alternative relevant chunks.\n\n"
        f"Question: {question}\n\n"
        "Retrieved snippets:\n"
        + "\n".join(snippets)
        + "\n\nReturn strict JSON only:\n"
        '{"suggested_terms":[{"term":"alternative phrase","type":"phrase","reason":"...","confidence":0.8}]}'
    )


def merge_suggestions(suggestions: list[dict[str, Any]], *, max_terms: int) -> list[dict[str, Any]]:
    by_term: dict[str, dict[str, Any]] = {}
    for item in suggestions:
        term = normalize_term(str(item.get("term") or ""))
        if not term:
            continue
        if not is_valid_suggestion_term(term, ""):
            continue
        current = by_term.get(term)
        normalized = {
            "term": term,
            "type": item.get("type") if item.get("type") in {"keyword", "phrase"} else ("phrase" if " " in term else "keyword"),
            "reason": str(item.get("reason") or ""),
            "source": str(item.get("source") or "heuristic"),
            "confidence": bounded_confidence(item.get("confidence"), default=0.5),
        }
        if current is None or normalized["confidence"] > current["confidence"]:
            by_term[term] = normalized
    return sorted(by_term.values(), key=lambda value: float(value["confidence"]), reverse=True)[:max_terms]


def suggestion(term: str, term_type: str, reason: str, confidence: float) -> dict[str, Any]:
    return {
        "term": term,
        "type": term_type,
        "reason": reason,
        "source": "heuristic",
        "confidence": confidence,
    }


def allowed_specific_terms(question: str, retrieved_candidates: list[dict]) -> set[str]:
    text = question + " " + " ".join(str(item.get("text") or item.get("transcript_snippet") or "") for item in retrieved_candidates)
    return set(tokenize(text))


def contains_unseen_specific_entity(term: str, allowed_terms: set[str]) -> bool:
    tokens = tokenize(term)
    # Keep generic short steering phrases. Filter only multi-token content that introduces unseen rare-looking words.
    if len(tokens) <= 1:
        return False
    return any(token not in allowed_terms and token not in STOPWORDS for token in tokens)


def answer_bearing_phrases(retrieved_candidates: list[dict], question: str, limit: int = 8) -> list[str]:
    question_vocab = set(content_tokens(question))
    phrases: list[str] = []
    for item in retrieved_candidates[:8]:
        text = str(item.get("text") or item.get("transcript_snippet") or "")
        phrases.extend(pattern_phrases(text))
    return dedupe_strings(
        [
            phrase
            for phrase in phrases
            if is_valid_suggestion_term(phrase, question)
            and strong_answer_pattern(phrase)
            and not mostly_question_overlap(phrase, question_vocab)
        ]
    )[:limit]


def pattern_phrases(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    patterns = [
        r"\bwe\s+(?:do\s+not|don't|dont|have\s+no|haven't|have\s+not)\b(?:\s+\w+){0,4}",
        r"\b(?:don't|dont)\s+have\b(?:\s+\w+){0,4}",
        r"\bnot\s+available\b(?:\s+\w+){0,4}",
        r"\bnothing\b",
        r"\b(?:it's|it\s+is|that\s+is|that's|called|named|answer(?:ed)?)\b(?:\s+\w+){0,4}",
        r"\bsaid\b(?:\s+\w+){0,4}",
        r"\bbecause\b(?:\s+\w+){0,5}",
        r"['\"]([^'\"]{2,60})['\"]",
    ]
    phrases = []
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            if match.groups():
                phrase = match.group(1)
            else:
                phrase = match.group(0)
            phrases.append(clean_phrase(phrase))
    return phrases


def clean_phrase(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9' ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def score_suggestion_confidence(
    term: str,
    question: str,
    retrieved_candidates: list[dict],
    base_confidence: float,
) -> float:
    confidence = base_confidence
    if occurs_in_retrieved(term, retrieved_candidates):
        confidence += 0.12
    if answer_pattern_score(term) > 0:
        confidence += answer_pattern_score(term)
    if mostly_question_overlap(term, set(content_tokens(question))):
        confidence -= 0.35
    return max(0.0, min(1.0, confidence))


def occurs_in_retrieved(term: str, retrieved_candidates: list[dict]) -> bool:
    term = normalize_term(term)
    for item in retrieved_candidates[:8]:
        text = normalize_term(str(item.get("text") or item.get("transcript_snippet") or ""))
        if term and term in text:
            return True
    return False


def answer_pattern_score(term: str) -> float:
    term = normalize_term(term)
    if re.search(r"\b(no|nothing|none|without|don't|dont|not|because|called|named|answer|said|it's|is it)\b", term):
        return 0.16
    return 0.0


def strong_answer_pattern(term: str) -> bool:
    term = normalize_term(term)
    patterns = (
        r"^we\s+have\s+no\b",
        r"^(?:don't|dont)\s+have\b",
        r"^not\s+available\b",
        r"^nothing$",
        r"^(?:it's|it\s+is|that's|that\s+is)\b",
        r"^called\b",
        r"^named\b",
        r"^answer(?:ed)?\b",
        r"^said\b",
        r"^because\b",
    )
    return any(re.search(pattern, term) for pattern in patterns)


def is_valid_suggestion_term(term: str, question: str) -> bool:
    term = normalize_term(term)
    if not term or len(term) > 40:
        return False
    if re.search(r"(.)\1{5,}", term):
        return False
    tokens = content_tokens(term)
    if not tokens:
        return False
    if len(tokens) == 1 and len(tokens[0]) < 4 and not answer_pattern_score(term):
        return False
    if question and not answer_pattern_score(term) and mostly_question_overlap(term, set(content_tokens(question))):
        return False
    return True


def morphology_based_answer_terms(question: str) -> list[str]:
    terms = []
    for token in content_tokens(question):
        if token.startswith("un") and len(token) > 5:
            terms.append(f"not {token[2:]}")
        if token.endswith("less") and len(token) > 6:
            terms.append(f"without {token[:-4]}")
    return dedupe_strings([term for term in terms if is_valid_suggestion_term(term, question="")])


def mostly_question_overlap(term: str, question_vocab: set[str]) -> bool:
    tokens = [token for token in tokenize(term) if token not in STOPWORDS]
    if not tokens:
        return False
    overlap = sum(1 for token in tokens if token in question_vocab)
    return overlap / len(tokens) >= 0.67


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def dedupe_strings(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        normalized = normalize_term(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower())


def bounded_confidence(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
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
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
