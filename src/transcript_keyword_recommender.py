from __future__ import annotations

import json
import re
from typing import Any, Callable

from src.transcript_reasoning_answer import resolve_reasoning_model_name, run_local_reasoning_model


ModelFunction = Callable[[str], str]


def recommend_transcript_keywords(
    question: str,
    transcript_candidates: list[dict],
    top_n: int = 10,
    max_keywords: int = 10,
    *,
    model_function: ModelFunction | None = None,
) -> dict[str, Any]:
    candidates = transcript_candidates[: max(0, top_n)]
    vocabulary = build_keyword_vocabulary(question, candidates)
    prompt = build_prompt(question, candidates, vocabulary)
    debug = {
        "model_invoked": False,
        "raw_model_output": "",
        "parse_success": False,
        "validation_success": False,
        "num_raw_terms": 0,
        "num_valid_terms": 0,
        "fallback_used": False,
        "failure_reason": "",
        "num_candidate_terms": len(vocabulary["candidate_terms"]),
        "num_query_terms": len(vocabulary["query_terms"]),
        "candidate_terms_preview": vocabulary["candidate_terms"][:50],
        "query_terms": vocabulary["query_terms"],
        "transcript_candidates_used": build_debug_candidate_preview(candidates),
    }
    try:
        debug["model_invoked"] = True
        raw_output = model_function(prompt) if model_function else run_local_reasoning_model(prompt, resolve_reasoning_model_name())
        debug["raw_model_output"] = raw_output
        parsed, parse_strategy = parse_keyword_json(raw_output)
        debug["parse_strategy"] = parse_strategy
        debug["parse_success"] = True
        raw_terms = parsed.get("suggested_terms")
        debug["num_raw_terms"] = len(raw_terms) if isinstance(raw_terms, list) else 0
        suggested_terms = normalize_suggestions(
            raw_terms,
            question=question,
            transcript_candidates=candidates,
            vocabulary=vocabulary,
            max_keywords=max_keywords,
        )
        debug["num_valid_terms"] = len(suggested_terms)
        debug["validation_success"] = bool(suggested_terms)
        if len(suggested_terms) >= 3:
            return {
                "source": "qwen_keyword_recommender",
                "top_n": top_n,
                "suggested_terms": suggested_terms,
                "keyword_recommender_debug": debug,
            }
        debug["failure_reason"] = "model_output_had_fewer_than_3_valid_terms"
    except Exception as exc:
        debug.setdefault("parse_strategy", "failed")
        debug["failure_reason"] = f"{type(exc).__name__}: {exc}"
    debug["fallback_used"] = True
    result = fallback_keywords(question, candidates, top_n=top_n, max_keywords=max_keywords, vocabulary=vocabulary)
    result["keyword_recommender_debug"] = debug
    return result


def build_debug_candidate_preview(transcript_candidates: list[dict], limit: int = 20) -> list[dict[str, Any]]:
    preview = []
    for index, item in enumerate(transcript_candidates[:limit], start=1):
        text = clean_snippet(str(item.get("transcript_snippet") or item.get("text") or ""))
        preview.append(
            {
                "candidate_index": index,
                "retrieval_score": safe_float(item.get("score"), 0.0),
                "source_name": item.get("source_name"),
                "timestamp": item.get("timestamp"),
                "transcript_path": item.get("transcript_path"),
                "text_snippet": text[:500],
            }
        )
    return preview


def build_prompt(question: str, transcript_candidates: list[dict], vocabulary: dict[str, list[str]]) -> str:
    candidate_blocks = []
    for index, item in enumerate(transcript_candidates, start=1):
        text = clean_snippet(str(item.get("transcript_snippet") or item.get("text") or ""))
        candidate_blocks.append(
            "\n".join(
                [
                    f"Candidate {index}",
                    f"source: {item.get('source_name', '')}",
                    f"timestamp: {item.get('timestamp', '')}",
                    f"retrieval_score: {safe_float(item.get('score'), 0.0):.6f}",
                    "text:",
                    text[:900],
                ]
            )
        )
    schema = {
        "suggested_terms": [
            {
                "term": "term copied from candidate_terms or query_terms",
                "source": "transcript",
                "reason": "why this would help retrieve better transcript evidence",
                "confidence": 0.8,
            }
        ]
    }
    return (
        "Given the user question and the top retrieved transcript snippets, recommend search steering "
        "keywords or phrases that a user could click to rerun retrieval.\n\n"
        "You must select terms only from the provided candidate_terms and query_terms lists. Do not invent "
        "new terms. When selecting from candidate_terms, prioritize concrete transcript-derived terms over "
        "query-overlap terms.\n\n"
        "Process:\n"
        "1. Select 10-12 useful steering terms from candidate_terms and query_terms.\n"
        "2. Prefer concrete nouns, food/items/objects, noun phrases, and negation phrases like 'no X'.\n"
        "3. Avoid terms from query_terms unless necessary, generic words also present in the question, "
        "verbs like gather/check/trying, and broad terms like ingredients/start.\n"
        "4. Choose terms that would narrow retrieval, not terms that merely match the current query.\n\n"
        "Rules:\n"
        "1. Return 5-10 short search terms or phrases.\n"
        "2. Terms should come from important words in the user question, distinctive words or phrases in "
        "the retrieved transcript snippets, or transcript-style variants that might retrieve similar chunks.\n"
        "3. Prefer distinctive transcript-derived terms over generic query paraphrases when enough "
        "transcript evidence is available. Aim for roughly 70-80% transcript-derived terms and 20-30% "
        "query-derived terms.\n"
        "4. Do not answer the question.\n"
        "5. Do not explain the evidence or choose the best chunk.\n"
        "6. Do not summarize the topic.\n"
        "7. Do not output broad category labels, metadata descriptions, or the whole question.\n"
        "8. Avoid noisy broken fragments.\n"
        "9. Prefer short, clickable, search-like phrases: 1 to 4 words.\n"
        "10. Return JSON only.\n\n"
        "At least 70% of suggestions should be copied exactly from the retrieved transcript snippets. "
        "Only include query/paraphrase terms if they are clearly useful for rerunning retrieval.\n"
        "Do not output a term just because it describes the question. Output it only if it is a useful "
        "search term likely to find a better transcript chunk.\n"
        "Bad style: topic summaries or category labels. Good style: exact words or short phrases found in "
        "the snippets, concrete objects, names, and short answer-like words.\n\n"
        f"Question: {question}\n\n"
        "Transcript candidates:\n\n"
        + "\n\n---\n\n".join(candidate_blocks)
        + "\n\ncandidate_terms copied from transcript snippets:\n"
        + json.dumps(vocabulary["candidate_terms"], ensure_ascii=False)
        + "\n\nquery_terms copied from the user question:\n"
        + json.dumps(vocabulary["query_terms"], ensure_ascii=False)
        + "\n\nReturn strict JSON with this schema:\n"
        + json.dumps(schema)
    )


def parse_keyword_json(raw_output: str) -> tuple[dict[str, Any], str]:
    text = strip_json_fence(raw_output)
    fenced = text != raw_output.strip()
    object_text = extract_outer_json_object(text)
    strategy = "stripped_fence" if fenced else "strict"
    try:
        parsed = json.loads(object_text)
        if isinstance(parsed, dict):
            return parsed, strategy
    except json.JSONDecodeError:
        pass

    repaired = repair_json_object(object_text)
    if repaired != object_text:
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return parsed, "repaired_json"
        except json.JSONDecodeError:
            pass

    recovered_terms = recover_term_objects(text)
    if recovered_terms:
        return {"suggested_terms": recovered_terms}, "regex_term_recovery"
    raise ValueError("keyword JSON parsing failed")


def strip_json_fence(raw_output: str) -> str:
    text = raw_output.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_outer_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object start found")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def repair_json_object(text: str) -> str:
    repaired = re.sub(r",\s*([}\]])", r"\1", text.strip())
    open_braces = repaired.count("{")
    close_braces = repaired.count("}")
    open_brackets = repaired.count("[")
    close_brackets = repaired.count("]")
    if open_brackets > close_brackets:
        repaired += "]" * (open_brackets - close_brackets)
    if open_braces > close_braces:
        repaired += "}" * (open_braces - close_braces)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def recover_term_objects(text: str) -> list[dict[str, Any]]:
    terms = []
    object_pattern = re.compile(r"\{[^{}]*\"term\"\s*:\s*\"[^\"]+\"[^{}]*\}", flags=re.DOTALL)
    for match in object_pattern.finditer(text):
        candidate = re.sub(r",\s*([}\]])", r"\1", match.group(0))
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = recover_term_object_fields(candidate)
        if isinstance(parsed, dict) and parsed.get("term"):
            terms.append(parsed)
    if terms:
        return terms

    # Last-resort recovery for truncated objects that still contain key fields.
    for match in re.finditer(r'"term"\s*:\s*"(?P<term>[^"]+)"(?P<body>[^{}]*)', text, flags=re.DOTALL):
        body = match.group("body")
        parsed = {"term": match.group("term")}
        for key in ("source", "reason", "confidence"):
            value_match = re.search(rf'"{key}"\s*:\s*("([^"]*)"|[0-9.]+)', body)
            if not value_match:
                continue
            value = value_match.group(1)
            if value.startswith('"'):
                parsed[key] = value.strip('"')
            else:
                parsed[key] = value
        terms.append(parsed)
    return terms


def recover_term_object_fields(text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key in ("term", "source", "reason", "confidence"):
        match = re.search(rf'"{key}"\s*:\s*("([^"]*)"|[0-9.]+)', text)
        if not match:
            continue
        value = match.group(1)
        parsed[key] = value.strip('"') if value.startswith('"') else value
    return parsed


def normalize_suggestions(
    values: Any,
    *,
    question: str,
    transcript_candidates: list[dict],
    vocabulary: dict[str, list[str]],
    max_keywords: int,
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    del question, transcript_candidates
    candidate_terms = set(vocabulary["candidate_terms"])
    query_terms = set(vocabulary["query_terms"])
    allowed_terms = candidate_terms | query_terms
    out = []
    seen = set()
    for item in values:
        if not isinstance(item, dict):
            continue
        term = normalize_text(str(item.get("term") or ""))
        if not valid_term(term):
            continue
        if term not in allowed_terms:
            continue
        if term in seen:
            continue
        seen.add(term)
        if term in candidate_terms and term in query_terms:
            source = "both"
        elif term in candidate_terms:
            source = "transcript"
        else:
            source = "query"
        out.append(
            {
                "term": term,
                "type": "phrase" if " " in term else "keyword",
                "source": source,
                "reason": str(item.get("reason") or "Suggested by Qwen keyword recommender."),
                "confidence": bounded_confidence(item.get("confidence"), default=0.5),
            }
        )
        if len(out) >= max_keywords:
            break
    return out


def build_keyword_vocabulary(question: str, transcript_candidates: list[dict], max_candidate_terms: int = 80) -> dict[str, list[str]]:
    # Count terms globally; also track unique-source count for diversity scoring.
    candidate_counts: dict[str, int] = {}
    source_sets: dict[str, set[str]] = {}
    for item in transcript_candidates:
        source = str(item.get("source_name") or item.get("video_id") or "")
        text = str(item.get("transcript_snippet") or item.get("text") or "")
        for term in snippet_terms(text):
            candidate_counts[term] = candidate_counts.get(term, 0) + 1
            source_sets.setdefault(term, set()).add(source)
    # Sort by (unique-source-count, raw-count, term quality) to promote
    # terms that appear across multiple speakers/segments over single-source
    # high-frequency noise.
    candidate_terms = [
        term
        for term, _count in sorted(
            candidate_counts.items(),
            key=lambda kv: (len(source_sets.get(kv[0], set())), kv[1], term_score(kv[0]), len(kv[0])),
            reverse=True,
        )
        if valid_term(term)
    ][:max_candidate_terms]
    query_terms = []
    for token in content_tokens(question):
        if valid_term(token) and token not in query_terms:
            query_terms.append(token)
        if len(query_terms) >= 3:
            break
    return {
        "candidate_terms": candidate_terms,
        "query_terms": query_terms,
    }


def snippet_terms(text: str) -> list[str]:
    tokens = content_tokens(text)
    terms = []
    for size in (2, 1):
        for start in range(0, max(0, len(tokens) - size + 1)):
            term = " ".join(tokens[start : start + size])
            if repeated_stem_phrase(term):
                continue
            if valid_term(term):
                terms.append(term)
    return terms


def term_score(term: str) -> int:
    tokens = tokenize(term)
    score = 0
    if len(tokens) == 2:
        score += 2
    score += sum(1 for token in tokens if len(token) >= 5)
    return score


def fallback_keywords(
    question: str,
    transcript_candidates: list[dict],
    *,
    top_n: int,
    max_keywords: int,
    vocabulary: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    if vocabulary is not None:
        terms = []
        for term in vocabulary["candidate_terms"][: max_keywords]:
            terms.append(
                {
                    "term": term,
                    "type": "phrase" if " " in term else "keyword",
                    "source": "transcript",
                    "reason": "Fallback term from retrieved transcript candidate vocabulary.",
                    "confidence": 0.45,
                }
            )
        remaining = max_keywords - len(terms)
        for term in vocabulary["query_terms"][: max(0, remaining)]:
            if term in {item["term"] for item in terms}:
                continue
            terms.append(
                {
                    "term": term,
                    "type": "phrase" if " " in term else "keyword",
                    "source": "query",
                    "reason": "Fallback term from user query vocabulary.",
                    "confidence": 0.35,
                }
            )
        return {
            "source": "safe_keyword_fallback",
            "top_n": top_n,
            "suggested_terms": terms[: min(max_keywords, 10)],
        }

    suggestions: list[dict[str, Any]] = []
    seen = set()
    for index, item in enumerate(transcript_candidates[:top_n], start=1):
        text = str(item.get("transcript_snippet") or item.get("text") or "")
        del index
        for phrase, count in frequent_terms(short_phrases(text), limit=5):
            if phrase in seen or not valid_term(phrase):
                continue
            seen.add(phrase)
            suggestions.append(
                {
                    "term": phrase,
                    "type": "phrase" if " " in phrase else "keyword",
                    "source": "transcript",
                    "reason": "Frequent short non-noisy term from retrieved transcript text.",
                    "confidence": min(0.6, 0.35 + 0.05 * count),
                }
            )
            if transcript_suggestion_count(suggestions) >= 5:
                break
        if transcript_suggestion_count(suggestions) >= 5:
            break
    for token in content_tokens(question):
        if token in seen or not valid_term(token):
            continue
        seen.add(token)
        suggestions.append(
            {
                "term": token,
                "type": "keyword",
                "source": "query",
                "reason": "Non-stopword token from the query.",
                "confidence": 0.35,
            }
        )
        if query_suggestion_count(suggestions) >= 3:
            break
    return {
        "source": "safe_keyword_fallback",
        "top_n": top_n,
        "suggested_terms": suggestions[: min(max_keywords, 10)],
    }


def short_phrases(text: str) -> list[str]:
    tokens = content_tokens(text)
    phrases = []
    for size in (2, 1):
        for start in range(0, max(0, len(tokens) - size + 1)):
            phrase = " ".join(tokens[start : start + size])
            if repeated_stem_phrase(phrase):
                continue
            if phrase not in phrases:
                phrases.append(phrase)
            if len(phrases) >= 20:
                return phrases
    return phrases


def valid_term(term: str) -> bool:
    if not term or len(term) > 40:
        return False
    if re.search(r"(.)\1{5,}", term):
        return False
    tokens = tokenize(term)
    content = [token for token in tokens if token not in STOPWORDS]
    if not content:
        return False
    if len(tokens) > 4:
        return False
    if tokens[0] in ORPHAN_FRAGMENTS or tokens[-1] in ORPHAN_FRAGMENTS:
        return False
    if tokens[0] in LEADING_NOISE:
        return False
    return True


def frequent_terms(values: list[str], limit: int) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for value in values:
        if valid_term(value):
            counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (item[1], len(item[0])), reverse=True)[:limit]


def transcript_suggestion_count(suggestions: list[dict[str, Any]]) -> int:
    return sum(1 for item in suggestions if item.get("source") == "transcript")


def query_suggestion_count(suggestions: list[dict[str, Any]]) -> int:
    return sum(1 for item in suggestions if item.get("source") == "query")


def repeated_stem_phrase(phrase: str) -> bool:
    tokens = tokenize(phrase)
    return len(tokens) == 2 and (tokens[0].startswith(tokens[1]) or tokens[1].startswith(tokens[0]))


def clean_snippet(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text.strip().lower())


def bounded_confidence(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


STOPWORDS = {
    "a", "about", "actually", "after", "again", "all", "also", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "being",
    "but", "by", "can", "come", "could", "did", "do", "does", "done",
    "each", "even", "for", "from", "get", "getting", "go", "going", "got",
    "had", "has", "have", "having", "he", "her", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "just", "know",
    "let", "like", "look", "m", "make", "me", "more", "my", "need",
    "no", "not", "now", "of", "off", "ok", "okay", "on", "one", "or",
    "our", "out", "really", "s", "said", "saw", "see", "she", "so",
    "some", "that", "the", "their", "them", "then", "there", "these",
    "they", "think", "this", "those", "though", "through", "time", "to",
    "too", "up", "us", "very", "want", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "will", "with", "would",
    "yeah", "yes", "you", "your",
}


LEADING_NOISE = {
    "alright",
    "basically",
    "gonna",
    "gotta",
    "kay",
    "kind",
    "kinda",
    "like",
    "literally",
    "maybe",
    "okay",
    "pretty",
    "right",
    "so",
    "sort",
    "uh",
    "uhh",
    "um",
    "umm",
    "wanna",
    "well",
    "yeah",
    "yes",
}


ORPHAN_FRAGMENTS = {
    "couldn",
    "didn",
    "don",
    "doesn",
    "hadn",
    "hasn",
    "haven",
    "isn",
    "ll",
    "mustn",
    "re",
    "shouldn",
    "t",
    "ve",
    "wasn",
    "weren",
    "won",
    "wouldn",
}
