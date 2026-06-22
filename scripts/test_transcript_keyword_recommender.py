#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transcript_keyword_recommender import recommend_transcript_keywords


def main() -> None:
    test_mocked_qwen_schema_is_stable()
    test_fenced_json_parses()
    test_truncated_json_recovers_term_objects()
    test_terms_outside_provided_vocabulary_are_rejected()
    test_invalid_json_falls_back_safely()
    test_noisy_and_broken_terms_are_filtered()
    test_fallback_terms_are_short_and_non_stopword()
    test_fallback_prefers_transcript_terms_before_query_terms()
    print("transcript keyword recommender smoke tests passed")


def test_mocked_qwen_schema_is_stable() -> None:
    result = recommend_transcript_keywords(
        "Which setup items were unavailable?",
        fake_candidates(),
        model_function=lambda _prompt: json.dumps(
            {
                "suggested_terms": [
                    {"term": "alpha module", "source": "transcript", "reason": "from snippet", "confidence": 0.8},
                    {"term": "bravo cable", "source": "transcript", "reason": "from snippet", "confidence": 0.7},
                    {"term": "calibration token", "source": "transcript", "reason": "from snippet", "confidence": 0.6},
                    {"term": "nothing", "source": "transcript", "reason": "from snippet", "confidence": 0.5},
                    {"term": "unavailable", "source": "query", "reason": "from question", "confidence": 0.4},
                ]
            }
        ),
    )
    assert result["source"] == "qwen_keyword_recommender", result
    debug = result["keyword_recommender_debug"]
    assert debug["model_invoked"] is True, result
    assert debug["parse_success"] is True, result
    assert debug["parse_strategy"] == "strict", result
    assert debug["validation_success"] is True, result
    assert debug["fallback_used"] is False, result
    assert debug["num_raw_terms"] == 5, result
    assert debug["num_valid_terms"] == 5, result
    assert 5 <= len(result["suggested_terms"]) <= 10, result
    for item in result["suggested_terms"]:
        assert set(item) >= {"term", "source", "reason", "confidence"}, item
        assert item["source"] in {"query", "transcript", "both"}, item
        assert 1 <= len(item["term"].split()) <= 4, item
        assert 0.0 <= item["confidence"] <= 1.0, item


def test_fenced_json_parses() -> None:
    raw = """```json
{
  "suggested_terms": [
    {"term": "alpha module", "source": "transcript", "reason": "from snippet", "confidence": 0.8},
    {"term": "bravo cable", "source": "transcript", "reason": "from snippet", "confidence": 0.7},
    {"term": "unavailable", "source": "query", "reason": "from query", "confidence": 0.6}
  ]
}
```"""
    result = recommend_transcript_keywords(
        "Which setup items were unavailable?",
        fake_candidates(),
        model_function=lambda _prompt: raw,
    )
    assert result["source"] == "qwen_keyword_recommender", result
    assert result["keyword_recommender_debug"]["parse_strategy"] == "stripped_fence", result
    terms = {item["term"] for item in result["suggested_terms"]}
    assert {"alpha module", "bravo cable", "unavailable"} <= terms, result


def test_truncated_json_recovers_term_objects() -> None:
    raw = """
{
  "suggested_terms": [
    {"term": "alpha module", "source": "transcript", "reason": "from snippet", "confidence": 0.8},
    {"term": "bravo cable", "source": "transcript", "reason": "from snippet", "confidence": 0.7},
    {"term": "unavailable", "source": "query", "reason": "from query", "confidence": 0.6},
    {"term": "unfinished", "source": "transcript", "reason": "cut off"
"""
    result = recommend_transcript_keywords(
        "Which setup items were unavailable?",
        fake_candidates(),
        model_function=lambda _prompt: raw,
    )
    assert result["source"] == "qwen_keyword_recommender", result
    assert result["keyword_recommender_debug"]["parse_strategy"] == "regex_term_recovery", result
    terms = {item["term"] for item in result["suggested_terms"]}
    assert {"alpha module", "bravo cable", "unavailable"} <= terms, result
    assert result["keyword_recommender_debug"]["num_raw_terms"] >= 2, result


def test_terms_outside_provided_vocabulary_are_rejected() -> None:
    result = recommend_transcript_keywords(
        "Which setup items were unavailable?",
        fake_candidates(),
        model_function=lambda _prompt: json.dumps(
            {
                "suggested_terms": [
                    {"term": "setup unavailable", "source": "query", "reason": "invented topic summary", "confidence": 0.9},
                    {"term": "missing items", "source": "query", "reason": "invented paraphrase", "confidence": 0.8},
                    {"term": "alpha module", "source": "transcript", "reason": "from snippet", "confidence": 0.7},
                    {"term": "bravo cable", "source": "transcript", "reason": "from snippet", "confidence": 0.7},
                    {"term": "calibration token", "source": "transcript", "reason": "from snippet", "confidence": 0.7},
                ]
            }
        ),
    )
    assert result["source"] == "qwen_keyword_recommender", result
    terms = {item["term"] for item in result["suggested_terms"]}
    assert {"alpha module", "bravo cable", "calibration token"} <= terms, result
    assert "setup unavailable" not in terms, result
    assert "missing items" not in terms, result


def test_invalid_json_falls_back_safely() -> None:
    result = recommend_transcript_keywords(
        "Which setup items were unavailable?",
        fake_candidates(),
        model_function=lambda _prompt: "not json",
    )
    assert result["source"] == "safe_keyword_fallback", result
    debug = result["keyword_recommender_debug"]
    assert debug["model_invoked"] is True, result
    assert debug["parse_success"] is False, result
    assert debug["fallback_used"] is True, result
    assert debug["failure_reason"], result
    assert result["suggested_terms"], result
    assert len(result["suggested_terms"]) <= 10, result


def test_noisy_and_broken_terms_are_filtered() -> None:
    result = recommend_transcript_keywords(
        "What was the answer?",
        fake_candidates(),
        model_function=lambda _prompt: json.dumps(
            {
                "suggested_terms": [
                    {"term": "mmmmmmmmmmmm clue", "source": "transcript", "reason": "noise", "confidence": 0.95},
                    {"term": "t know answer", "source": "transcript", "reason": "broken ASR", "confidence": 0.9},
                    {"term": "alpha module", "source": "transcript", "reason": "valid", "confidence": 0.75},
                    {"term": "bravo cable", "source": "transcript", "reason": "valid", "confidence": 0.75},
                    {"term": "calibration token", "source": "transcript", "reason": "valid", "confidence": 0.75},
                    {"term": "so cops", "source": "transcript", "reason": "leading filler", "confidence": 0.7},
                    {"term": "this", "source": "query", "reason": "stopword only", "confidence": 0.6},
                ]
            }
        ),
    )
    terms = {item["term"] for item in result["suggested_terms"]}
    assert "alpha module" in terms, result
    assert "mmmmmmmmmmmm clue" not in terms, result
    assert "t know answer" not in terms, result
    assert "so cops" not in terms, result
    assert "this" not in terms, result


def test_fallback_terms_are_short_and_non_stopword() -> None:
    result = recommend_transcript_keywords(
        "Which setup items were unavailable?",
        [
            candidate("cand1", "so this is about the setup"),
            candidate("cand2", "distinctive transcript clue appears twice distinctive clue"),
            candidate("cand3", "officer office repeated phrase should not dominate"),
        ],
        model_function=lambda _prompt: "not json",
    )
    assert result["source"] == "safe_keyword_fallback", result
    assert len(result["suggested_terms"]) <= 10, result
    terms = {item["term"] for item in result["suggested_terms"]}
    assert "so" not in terms, result
    assert "this" not in terms, result
    assert "so this" not in terms, result
    assert "officer office" not in terms, result
    for item in result["suggested_terms"]:
        assert 1 <= len(item["term"].split()) <= 4, item
        assert item["source"] in {"query", "transcript"}, item


def test_fallback_prefers_transcript_terms_before_query_terms() -> None:
    result = recommend_transcript_keywords(
        "Which setup items were unavailable?",
        fake_candidates(),
        model_function=lambda _prompt: "not json",
    )
    sources = [item["source"] for item in result["suggested_terms"]]
    assert "transcript" in sources, result
    if "query" in sources:
        first_query_index = sources.index("query")
        assert all(source == "transcript" for source in sources[:first_query_index]), result
    else:
        assert all(source == "transcript" for source in sources), result


def fake_candidates() -> list[dict]:
    return [
        candidate("cand1", "They are checking the setup area."),
        candidate("cand2", "We have no alpha module, no bravo cable, no calibration token, nothing."),
        candidate("cand3", "The operator repeats alpha module during the setup check."),
    ]


def candidate(source_id: str, text: str) -> dict:
    return {
        "source_id": source_id,
        "text": text,
        "transcript_snippet": text,
        "source_name": "Test",
        "timestamp": "00:00-00:05",
        "score": 1.0,
    }


if __name__ == "__main__":
    main()
