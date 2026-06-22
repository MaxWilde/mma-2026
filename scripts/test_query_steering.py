#!/usr/bin/env python
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.query_steering import suggest_steering_terms


def main() -> None:
    test_llm_answer_bearing_phrase_survives()
    test_question_paraphrase_is_penalized_or_rejected()
    test_repeated_character_noise_is_filtered()
    test_morphology_variant_from_question()
    test_retrieval_merging_with_steering()
    test_normal_query_output_includes_steering_without_flag()
    print("query steering smoke tests passed")


def test_llm_answer_bearing_phrase_survives() -> None:
    result = suggest_steering_terms(
        "Which supplies were unavailable during setup?",
        [{"text": "They discuss setup logistics and supply status."}],
        model_function=lambda _prompt: json.dumps(
            {
                "suggested_terms": [
                    {
                        "term": "supply status",
                        "type": "phrase",
                        "reason": "Alternative answer-bearing wording present in the snippets.",
                        "confidence": 0.9,
                    }
                ]
            }
        ),
    )
    terms = {item["term"] for item in result["suggested_terms"]}
    assert "supply status" in terms, result


def test_question_paraphrase_is_penalized_or_rejected() -> None:
    result = suggest_steering_terms(
        "Which supplies were unavailable during setup?",
        [{"text": "They discuss setup logistics and supply status."}],
        model_function=lambda _prompt: json.dumps(
            {
                "suggested_terms": [
                    {
                        "term": "supplies unavailable setup",
                        "type": "phrase",
                        "reason": "Paraphrases the question.",
                        "confidence": 0.95,
                    },
                    {
                        "term": "supply status",
                        "type": "phrase",
                        "reason": "Alternative answer-bearing wording.",
                        "confidence": 0.7,
                    },
                ]
            }
        ),
    )
    terms = [item["term"] for item in result["suggested_terms"]]
    assert "supply status" in terms, result
    if "supplies unavailable setup" in terms:
        assert terms.index("supply status") < terms.index("supplies unavailable setup"), result


def test_repeated_character_noise_is_filtered() -> None:
    result = suggest_steering_terms(
        "Which supplies were unavailable during setup?",
        [{"text": "A repeated vocalization appears in the transcript before useful evidence."}],
        model_function=lambda _prompt: json.dumps(
            {
                "suggested_terms": [
                    {"term": "xxxxxxxxxxxxxxxxxxxx evidence", "type": "phrase", "reason": "noise", "confidence": 0.95},
                    {"term": "useful evidence", "type": "phrase", "reason": "answer-bearing wording.", "confidence": 0.7},
                ]
            }
        ),
    )
    terms = {item["term"] for item in result["suggested_terms"]}
    assert "xxxxxxxxxxxxxxxxxxxx evidence" not in terms, result
    assert "useful evidence" in terms, result


def test_morphology_variant_from_question() -> None:
    result = suggest_steering_terms(
        "Which supplies were unavailable during setup?",
        [{"text": "They discuss setup logistics."}],
        model_function=lambda _prompt: json.dumps({"suggested_terms": []}),
    )
    terms = {item["term"] for item in result["suggested_terms"]}
    assert "not available" in terms, result


def test_retrieval_merging_with_steering() -> None:
    import scripts.query_evidence as query_evidence

    original_retrieve = query_evidence.retrieve_transcript
    try:
        query_evidence.retrieve_transcript = fake_retrieve_transcript
        merged = query_evidence.retrieve_transcript_with_steering(
            "Which supplies were unavailable during setup?",
            "unused",
            10,
            ["supply status", "unavailable items"],
        )
    finally:
        query_evidence.retrieve_transcript = original_retrieve

    keys = {query_evidence.evidence_key(item) for item in merged}
    assert "cand_a" in keys, merged
    assert "cand_b" in keys, merged
    assert len([item for item in merged if query_evidence.evidence_key(item) == "cand_a"]) == 1, merged
    b = next(item for item in merged if query_evidence.evidence_key(item) == "cand_b")
    assert b["retrieval_query_source"] == "steering", b
    assert b["matched_steering_term"] == "supply status", b


def test_normal_query_output_includes_steering_without_flag() -> None:
    import scripts.query_evidence as query_evidence

    original_visual = query_evidence.retrieve_visual
    original_transcript = query_evidence.retrieve_transcript
    original_recommend = query_evidence.recommend_transcript_keywords
    original_route = query_evidence.route_evidence
    argv = sys.argv
    try:
        query_evidence.retrieve_visual = lambda args: ([], 0.0)
        query_evidence.retrieve_transcript = fake_retrieve_transcript
        query_evidence.recommend_transcript_keywords = lambda question, candidates, top_n=10: {
            "source": "qwen_keyword_recommender",
            "top_n": top_n,
            "suggested_terms": [
                {
                    "term": "supply status",
                    "type": "phrase",
                    "reason": "Neutral test suggestion.",
                    "source": "qwen_keyword_recommender",
                    "confidence": 0.8,
                    "derived_from": "transcript",
                    "candidate_indices": [1],
                }
            ],
        }
        query_evidence.route_evidence = lambda question, visual, transcript: {
            **transcript[0],
            "evidence_type": "transcript",
            "router_debug": {},
        }
        sys.argv = ["query_evidence.py", "Which supplies were unavailable during setup?"]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            query_evidence.main()
    finally:
        query_evidence.retrieve_visual = original_visual
        query_evidence.retrieve_transcript = original_transcript
        query_evidence.recommend_transcript_keywords = original_recommend
        query_evidence.route_evidence = original_route
        sys.argv = argv

    text = output.getvalue()
    assert "Recommended steering keywords:" in text, text
    assert "Steering suggestions JSON:" in text, text
    assert "supply status" in text, text


def fake_retrieve_transcript(question: str, index_dir: str, top_k: int) -> list[dict]:
    del index_dir, top_k
    if question == "Which supplies were unavailable during setup?":
        return [
            {
                "source_id": "cand_a",
                "score": 1.0,
                "source_name": "Kitchen",
                "timestamp": "00:00-00:05",
                "text": "They are organizing the setup area.",
            }
        ]
    if "supply status" in question:
        return [
            {
                "source_id": "cand_b",
                "score": 0.95,
                "source_name": "Stevan",
                "timestamp": "00:10-00:15",
                "text": "The supply status is incomplete for the setup.",
            },
            {
                "source_id": "cand_a",
                "score": 0.50,
                "source_name": "Kitchen",
                "timestamp": "00:00-00:05",
                "text": "They are organizing the setup area.",
            },
        ]
    if "unavailable items" in question:
        return [
            {
                "source_id": "cand_b",
                "score": 0.90,
                "source_name": "Stevan",
                "timestamp": "00:10-00:15",
                "text": "The supply status is incomplete for the setup.",
            }
        ]
    return []


if __name__ == "__main__":
    main()
