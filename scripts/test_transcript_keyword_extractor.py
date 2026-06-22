#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transcript_keyword_extractor import extract_keywords_from_transcript_candidates


def main() -> None:
    test_extracts_grounded_terms_from_multiple_candidates()
    test_duplicate_terms_merge_candidate_ids()
    test_noisy_repeated_strings_are_filtered()
    test_question_topic_terms_are_suppressed()
    test_broken_quoted_fragments_are_filtered()
    print("transcript keyword extractor smoke tests passed")


def test_extracts_grounded_terms_from_multiple_candidates() -> None:
    result = extract_keywords_from_transcript_candidates(
        "What ingredients are missing at the start?",
        [
            candidate("cand1", "I'm just trying to gather all the ingredients."),
            candidate("cand2", "We have no celery, we have no bay leaf, nothing."),
            candidate("cand3", "Check the recipe for the list."),
        ],
        top_n=10,
        max_keywords=12,
    )
    terms = {item["term"] for item in result["suggested_terms"]}
    assert {"we have no", "celery", "bay leaf", "nothing"} <= terms, result
    assert "ingredients" not in terms, result
    assert "recipe list" not in terms, result
    assert result["source"] == "multi_candidate_transcript_keywords", result


def test_duplicate_terms_merge_candidate_ids() -> None:
    result = extract_keywords_from_transcript_candidates(
        "What ingredients are missing at the start?",
        [
            candidate("cand1", "We have no celery, nothing."),
            candidate("cand2", "We have no celery, we have no bay leaf, nothing."),
        ],
        top_n=10,
        max_keywords=12,
    )
    terms = {item["term"]: item for item in result["suggested_terms"]}
    assert "we have no" in terms, result
    assert terms["we have no"]["candidate_ids"] == ["cand1", "cand2"], result
    assert terms["we have no"]["candidate_indices"] == [1, 2], result


def test_noisy_repeated_strings_are_filtered() -> None:
    result = extract_keywords_from_transcript_candidates(
        "What supplies were missing?",
        [
            candidate("cand1", "We have no mmmmmmmmmmmmmmmmm tape, nothing."),
            candidate("cand2", "There was no spare tape."),
        ],
        top_n=10,
        max_keywords=12,
    )
    terms = {item["term"] for item in result["suggested_terms"]}
    assert "mmmmmmmmmmmmmmmmm" not in " ".join(terms), result
    assert "spare tape" in terms, result


def test_question_topic_terms_are_suppressed() -> None:
    result = extract_keywords_from_transcript_candidates(
        "What is the answer to the quiz question about The Police rock band?",
        [
            candidate("cand1", "The police band discussion continues and someone said the answer was sting."),
            candidate("cand2", "So where are the cops and rock music topic going?"),
        ],
        top_n=10,
        max_keywords=12,
    )
    terms = {item["term"] for item in result["suggested_terms"]}
    assert "answer" in terms, result
    assert "said" in terms, result
    assert "sting" in terms, result
    assert "police" not in terms, result
    assert "cops" not in terms, result
    assert "rock music" not in terms, result
    assert "band discussion" not in terms, result


def test_broken_quoted_fragments_are_filtered() -> None:
    result = extract_keywords_from_transcript_candidates(
        "What was the answer?",
        [
            candidate("cand1", "\"t know the answer because you didn\" and then someone said correct answer"),
            candidate("cand2", "\"clear answer\" was repeated"),
        ],
        top_n=10,
        max_keywords=12,
    )
    terms = {item["term"] for item in result["suggested_terms"]}
    assert "t know the answer because you didn" not in terms, result
    assert "clear answer" in terms, result


def candidate(source_id: str, text: str) -> dict:
    return {
        "source_id": source_id,
        "text": text,
        "transcript_snippet": text,
        "source_name": "Test",
        "timestamp": "00:00-00:05",
    }


if __name__ == "__main__":
    main()
