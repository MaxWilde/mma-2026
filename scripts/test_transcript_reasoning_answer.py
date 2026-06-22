#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transcript_reasoning_answer import reason_over_transcript_candidates


def main() -> None:
    test_reasoning_selects_better_candidate()
    test_invalid_json_falls_back_safely()
    test_reasoning_filters_noisy_steering_terms()
    test_question_paraphrases_are_rejected()
    test_steering_paraphrases_are_ignored()
    test_query_evidence_replacement_metadata()
    test_reasoning_steering_used_without_replacement()
    test_fallback_used_when_no_grounded_reasoning_keywords()
    test_reasoning_ordered_candidates_from_model_output()
    test_query_evidence_uses_reasoning_order_for_keyword_candidates()
    test_query_evidence_falls_back_to_raw_order_when_reasoning_fails()
    test_selected_candidate_without_order_uses_raw_order()
    test_answer_hypothesis_supporting_evidence_priority()
    test_selected_candidate_overridden_to_first_supporting_candidate()
    test_keyword_candidates_fall_back_from_support_to_reasoning_order()
    test_reasoning_indices_are_truncated()
    test_missing_reasoning_order_uses_supporting_indices()
    test_per_candidate_scores_wrong_sense_match_low()
    test_per_candidate_missing_scores_fall_back_safely()
    test_batch_scoring_parse_failure_debug_fields()
    test_batch_scoring_splits_candidates_into_batches_of_five()
    print("transcript reasoning answer smoke tests passed")


def test_reasoning_selects_better_candidate() -> None:
    result = reason_over_transcript_candidates(
        "Which setup supplies are unavailable?",
        fake_candidates(),
        top_n=2,
        model_function=lambda _prompt: json.dumps(
            {
                "selected_candidate_index": 2,
                "answer": "spare batteries and tape",
                "evidence_sentence": "The missing supplies are spare batteries and tape.",
                "confidence": 0.86,
                "reason": "This sentence explicitly states which setup supplies are unavailable.",
                "steering_keywords": [
                    {
                        "term": "spare batteries",
                        "type": "phrase",
                        "reason": "Answer-bearing phrase from the evidence sentence.",
                        "confidence": 0.82,
                    },
                    {
                        "term": "tape",
                        "type": "keyword",
                        "reason": "Answer-bearing word from the evidence sentence.",
                        "confidence": 0.78,
                    }
                ],
                "steering_paraphrases": [
                    {"term": "unavailable supplies", "type": "phrase", "reason": "ignored", "confidence": 0.74}
                ],
            }
        ),
    )
    assert result["selected_candidate_index"] == 2, result
    assert result["answer"] == "spare batteries and tape", result
    assert result["selected_candidate_id"] == "cand2", result
    assert result["reasoning_selected_retrieval_rank"] == 2, result
    assert result["confidence"] == 0.86, result
    assert result["steering_keywords"][0]["term"] == "spare batteries", result
    keyword_terms = {item["term"] for item in result["steering_keywords"]}
    assert {"spare batteries", "tape"} <= keyword_terms, result
    assert result["steering_paraphrases"] == [], result


def test_invalid_json_falls_back_safely() -> None:
    result = reason_over_transcript_candidates(
        "Which setup supplies are unavailable?",
        fake_candidates(),
        top_n=2,
        model_function=lambda _prompt: "not json",
    )
    assert result["selected_candidate_index"] is None, result
    assert result["reasoning_selected_retrieval_rank"] is None, result
    assert result["confidence"] == 0.0, result
    assert "model_or_json_failed" in result["reason"], result
    assert result["steering_keywords"] == [], result
    assert result["steering_paraphrases"] == [], result


def test_reasoning_filters_noisy_steering_terms() -> None:
    result = reason_over_transcript_candidates(
        "Which setup supplies are unavailable?",
        fake_candidates(),
        top_n=2,
        model_function=lambda _prompt: json.dumps(
            {
                "selected_candidate_index": 2,
                "answer": "spare batteries and tape",
                "evidence_sentence": "The missing supplies are spare batteries and tape.",
                "confidence": 0.86,
                "reason": "This sentence explicitly states which setup supplies are unavailable.",
                "steering_keywords": [
                    {"term": "spare batteries", "type": "phrase", "reason": "in evidence", "confidence": 0.8},
                    {"term": "xxxxxxxxxxxxxxxxxxxx", "type": "keyword", "reason": "noise", "confidence": 0.9},
                    {"term": "unseen component", "type": "phrase", "reason": "not in evidence", "confidence": 0.9},
                ],
            }
        ),
    )
    keyword_terms = {item["term"] for item in result["steering_keywords"]}
    assert "spare batteries" in keyword_terms, result
    assert "unseen component" not in keyword_terms, result
    assert "xxxxxxxxxxxxxxxxxxxx" not in keyword_terms, result
    assert result["steering_paraphrases"] == [], result


def test_question_paraphrases_are_rejected() -> None:
    result = reason_over_transcript_candidates(
        "Which setup supplies are unavailable?",
        fake_candidates(),
        top_n=2,
        model_function=lambda _prompt: json.dumps(
            {
                "selected_candidate_index": 2,
                "answer": "spare batteries and tape",
                "evidence_sentence": "The missing supplies are spare batteries and tape.",
                "confidence": 0.86,
                "reason": "This sentence explicitly states which setup supplies are unavailable.",
                "steering_keywords": [
                    {"term": "spare batteries", "type": "phrase", "reason": "in evidence", "confidence": 0.8},
                    {"term": "setup unavailable", "type": "phrase", "reason": "question paraphrase", "confidence": 0.9},
                ],
            }
        ),
    )
    keyword_terms = {item["term"] for item in result["steering_keywords"]}
    assert "spare batteries" in keyword_terms, result
    assert "setup unavailable" not in keyword_terms, result
    assert result["steering_paraphrases"] == [], result


def test_steering_paraphrases_are_ignored() -> None:
    result = reason_over_transcript_candidates(
        "Which setup supplies are unavailable?",
        fake_candidates(),
        top_n=2,
        model_function=lambda _prompt: json.dumps(
            {
                "selected_candidate_index": 2,
                "answer": "spare batteries and tape",
                "evidence_sentence": "The missing supplies are spare batteries and tape.",
                "confidence": 0.86,
                "reason": "This sentence explicitly states which setup supplies are unavailable.",
                "steering_keywords": [
                    {"term": "spare batteries", "type": "phrase", "reason": "in evidence", "confidence": 0.8},
                    {"term": "tape", "type": "keyword", "reason": "in evidence", "confidence": 0.7},
                ],
                "steering_paraphrases": [
                    {"term": "unavailable supplies", "type": "phrase", "reason": "retrieval phrase", "confidence": 0.8},
                    {"term": "missing supplies", "type": "phrase", "reason": "retrieval phrase", "confidence": 0.8},
                    {"term": "don't have", "type": "phrase", "reason": "spoken variant", "confidence": 0.7},
                    {"term": "out of stock", "type": "phrase", "reason": "lexical alternative", "confidence": 0.7},
                    {"term": "equipment", "type": "keyword", "reason": "broad label", "confidence": 0.9},
                    {"term": "objects", "type": "keyword", "reason": "broad label", "confidence": 0.9},
                    {"term": "resources", "type": "keyword", "reason": "broad label", "confidence": 0.9},
                    {"term": "consumables", "type": "keyword", "reason": "broad label", "confidence": 0.9},
                ],
            }
        ),
    )
    keyword_terms = {item["term"] for item in result["steering_keywords"]}
    assert {"spare batteries", "tape"} <= keyword_terms, result
    assert result["steering_paraphrases"] == [], result


def test_query_evidence_replacement_metadata() -> None:
    import scripts.query_evidence as query_evidence

    original_reasoner = query_evidence.reason_over_transcript_candidates
    try:
        query_evidence.reason_over_transcript_candidates = lambda question, candidates, top_n: {
            "selected_candidate_index": 2,
            "selected_candidate_id": "cand2",
            "answer": "spare batteries and tape",
            "evidence_sentence": "The missing supplies are spare batteries and tape.",
            "confidence": 0.86,
            "reason": "This sentence explicitly states which setup supplies are unavailable.",
            "steering_keywords": [
                {
                    "term": "spare batteries",
                    "type": "phrase",
                    "source": "reasoning_lm",
                    "reason": "Answer-bearing phrase from the selected evidence.",
                    "confidence": 0.82,
                }
            ],
            "steering_paraphrases": [
                {"term": "unavailable supplies", "type": "phrase", "source": "reasoning_lm", "reason": "ignored", "confidence": 0.74}
            ],
            "raw_model_output": "{}",
            "model_name": "mock",
            "reasoning_candidates": [],
        }
        candidates = fake_candidates()
        chosen = dict(candidates[0])
        chosen["evidence_type"] = "transcript"
        chosen["router_debug"] = {"combined_transcript_score": 1.0}
        chosen["steering_suggestions"] = {
            "source": "qwen_keyword_recommender",
            "suggested_terms": [
                {
                    "term": "existing recommender term",
                    "type": "phrase",
                    "source": "qwen_keyword_recommender",
                    "reason": "precomputed keyword recommendation",
                    "confidence": 0.7,
                }
            ],
        }
        updated = query_evidence.add_transcript_reasoning_answer(
            "Which setup supplies are unavailable?",
            chosen,
            candidates,
            top_n=2,
        )
    finally:
        query_evidence.reason_over_transcript_candidates = original_reasoner

    assert updated["source_name"] == "Stevan", updated
    assert updated["transcript_reasoning_answer"]["replacement"] is True, updated
    assert updated["transcript_reasoning_answer"]["reasoning_selected_retrieval_rank"] == 2, updated
    assert updated["reasoning_answer"] == "spare batteries and tape", updated
    assert updated["answer_span"]["text"] == "The missing supplies are spare batteries and tape.", updated
    assert updated["steering_suggestions"]["source"] == "qwen_keyword_recommender", updated
    terms = {item["term"] for item in updated["steering_suggestions"]["suggested_terms"]}
    assert "existing recommender term" in terms, updated
    assert "spare batteries" not in terms, updated


def test_reasoning_steering_used_without_replacement() -> None:
    import scripts.query_evidence as query_evidence

    original_reasoner = query_evidence.reason_over_transcript_candidates
    try:
        query_evidence.reason_over_transcript_candidates = lambda question, candidates, top_n: {
            "selected_candidate_index": 2,
            "selected_candidate_id": "cand2",
            "answer": "spare batteries and tape",
            "evidence_sentence": "The missing supplies are spare batteries and tape.",
            "confidence": 0.4,
            "reason": "Valid but below replacement threshold.",
            "steering_keywords": [
                {
                    "term": "spare batteries",
                    "type": "phrase",
                    "source": "reasoning_lm",
                    "reason": "Answer-bearing phrase from the selected evidence.",
                    "confidence": 0.82,
                }
            ],
            "steering_paraphrases": [
                {"term": "unavailable supplies", "type": "phrase", "source": "reasoning_lm", "reason": "ignored", "confidence": 0.74}
            ],
            "raw_model_output": "{}",
            "model_name": "mock",
            "reasoning_candidates": [],
        }
        candidates = fake_candidates()
        chosen = dict(candidates[0])
        chosen["evidence_type"] = "transcript"
        chosen["router_debug"] = {"combined_transcript_score": 1.0}
        chosen["steering_suggestions"] = {
            "source": "query_steering_fallback",
            "suggested_terms": [
                {
                    "term": "fallback term",
                    "type": "phrase",
                    "source": "heuristic",
                    "reason": "fallback",
                    "confidence": 0.5,
                }
            ],
        }
        updated = query_evidence.add_transcript_reasoning_answer(
            "Which setup supplies are unavailable?",
            chosen,
            candidates,
            top_n=2,
        )
    finally:
        query_evidence.reason_over_transcript_candidates = original_reasoner

    assert updated["transcript_reasoning_answer"]["replacement"] is False, updated
    assert updated["transcript_reasoning_answer"]["reasoning_succeeded"] is True, updated
    assert updated["steering_suggestions"]["source"] == "query_steering_fallback", updated
    terms = {item["term"] for item in updated["steering_suggestions"]["suggested_terms"]}
    assert "fallback term" in terms, updated
    assert "spare batteries" not in terms, updated


def test_fallback_used_when_no_grounded_reasoning_keywords() -> None:
    import scripts.query_evidence as query_evidence

    original_reasoner = query_evidence.reason_over_transcript_candidates
    try:
        query_evidence.reason_over_transcript_candidates = lambda question, candidates, top_n: {
            "selected_candidate_index": 2,
            "selected_candidate_id": "cand2",
            "answer": "spare batteries and tape",
            "evidence_sentence": "The missing supplies are spare batteries and tape.",
            "confidence": 0.8,
            "reason": "Valid reasoning, but no grounded steering terms survived.",
            "steering_keywords": [],
            "steering_paraphrases": [],
            "raw_model_output": "{}",
            "model_name": "mock",
            "reasoning_candidates": [],
        }
        candidates = fake_candidates()
        chosen = dict(candidates[0])
        chosen["evidence_type"] = "transcript"
        chosen["router_debug"] = {"combined_transcript_score": 1.0}
        chosen["steering_suggestions"] = {
            "source": "query_steering_fallback",
            "suggested_terms": [
                {
                    "term": "fallback term",
                    "type": "phrase",
                    "source": "heuristic",
                    "reason": "fallback",
                    "confidence": 0.5,
                }
            ],
        }
        updated = query_evidence.add_transcript_reasoning_answer(
            "Which setup supplies are unavailable?",
            chosen,
            candidates,
            top_n=2,
        )
    finally:
        query_evidence.reason_over_transcript_candidates = original_reasoner

    assert updated["transcript_reasoning_answer"]["reasoning_succeeded"] is True, updated
    assert updated["steering_suggestions"]["source"] == "query_steering_fallback", updated
    terms = {item["term"] for item in updated["steering_suggestions"]["suggested_terms"]}
    assert "fallback term" in terms, updated


def test_reasoning_ordered_candidates_from_model_output() -> None:
    candidates = numbered_candidates(30)
    result = reason_over_transcript_candidates(
        "Which candidate is most relevant?",
        candidates,
        top_n=30,
        model_function=lambda _prompt: json.dumps(
            {
                "selected_candidate_index": 23,
                "answer": "candidate 23",
                "evidence_sentence": "Candidate 23 has the relevant evidence.",
                "confidence": 0.91,
                "reason": "Candidate 23 explicitly matches.",
                "steering_keywords": [],
                "reasoning_ordered_candidate_indices": [23, 8, 5, 14],
            }
        ),
    )
    ordered = result["reasoning_ordered_transcript_candidates"]
    assert ordered[0]["candidate_id"] == "cand23", result
    assert ordered[1]["candidate_id"] == "cand8", result
    assert ordered[2]["candidate_id"] == "cand5", result
    assert ordered[0]["raw_retrieval_rank"] == 23, result
    assert ordered[0]["reasoning_rank"] == 1, result
    assert result["reasoning_ordered_candidate_indices"] == [23, 8, 5, 14], result
    assert result["reasoning_order_used"] is True, result


def test_query_evidence_uses_reasoning_order_for_keyword_candidates() -> None:
    import scripts.query_evidence as query_evidence

    raw_candidates = numbered_candidates(30)
    reasoning = {
        "selected_candidate_id": "cand23",
        "reasoning_order_used": True,
        "reasoning_ordered_candidate_indices": [23, 8, 5, 14],
        "reasoning_ordered_transcript_candidates": [
            {
                "candidate_id": "cand23",
                "candidate_index": 23,
                "raw_retrieval_rank": 23,
                "reasoning_rank": 1,
                "reasoning_score": 0.91,
                "reason": "best evidence",
            },
            {
                "candidate_id": "cand8",
                "candidate_index": 8,
                "raw_retrieval_rank": 8,
                "reasoning_rank": 2,
                "reasoning_score": 0.55,
                "reason": "secondary evidence",
            },
            {
                "candidate_id": "cand5",
                "candidate_index": 5,
                "raw_retrieval_rank": 5,
                "reasoning_rank": 3,
                "reasoning_score": 0.50,
                "reason": "third evidence",
            },
            {
                "candidate_id": "cand14",
                "candidate_index": 14,
                "raw_retrieval_rank": 14,
                "reasoning_rank": 4,
                "reasoning_score": 0.45,
                "reason": "fourth evidence",
            },
        ],
    }
    ordered = query_evidence.apply_reasoning_order_to_transcript_results(raw_candidates, reasoning)
    keyword_candidates = ordered[: min(20, len(ordered))]
    assert ordered[0]["source_id"] == "cand23", ordered[:3]
    assert ordered[1]["source_id"] == "cand8", ordered[:3]
    assert ordered[2]["source_id"] == "cand5", ordered[:3]
    assert next(index for index, item in enumerate(ordered, start=1) if item["source_id"] == "cand1") > 4
    assert keyword_candidates[0]["source_id"] == "cand23", keyword_candidates[:3]
    assert "cand23" not in {item["source_id"] for item in raw_candidates[:20]}


def test_query_evidence_falls_back_to_raw_order_when_reasoning_fails() -> None:
    import scripts.query_evidence as query_evidence

    raw_candidates = numbered_candidates(30)
    ordered = query_evidence.apply_reasoning_order_to_transcript_results(
        raw_candidates,
        {"reasoning_ordered_transcript_candidates": [], "reasoning_order_used": False},
    )
    assert ordered[0]["source_id"] == "cand1", ordered[:3]
    assert ordered[0]["raw_retrieval_rank"] == 1, ordered[:3]
    assert ordered[0]["reasoning_order_used"] is False, ordered[:3]


def test_selected_candidate_without_order_uses_raw_order() -> None:
    import scripts.query_evidence as query_evidence

    raw_candidates = numbered_candidates(30)
    reasoning = {
        "selected_candidate_index": 23,
        "selected_candidate_id": "cand23",
        "reasoning_ordered_transcript_candidates": [],
        "reasoning_ordered_candidate_indices": [],
        "reasoning_order_used": False,
    }
    ordered = query_evidence.apply_reasoning_order_to_transcript_results(raw_candidates, reasoning)
    assert ordered[0]["source_id"] == "cand1", ordered[:3]
    assert ordered[0]["reasoning_order_used"] is False, ordered[:3]


def test_answer_hypothesis_supporting_evidence_priority() -> None:
    import scripts.query_evidence as query_evidence

    candidates = supporting_candidates()
    result = reason_over_transcript_candidates(
        "Which items are unavailable?",
        candidates,
        top_n=30,
        model_function=lambda _prompt: json.dumps(
            {
                "answer_hypothesis": "alpha and beta",
                "supporting_candidate_indices": [5, 8],
                "selected_candidate_index": 5,
                "evidence_sentence": "We have no alpha on the shelf.",
                "confidence": 0.88,
                "reason": "Candidates 5 and 8 contain direct item evidence.",
                "steering_keywords": ["no alpha"],
                "reasoning_ordered_candidate_indices": [5, 8, 1, 2],
            }
        ),
    )
    assert result["answer_hypothesis"] == "alpha and beta", result
    assert result["answer"] == "alpha and beta", result
    assert result["selected_candidate_index"] == 5, result
    assert result["supporting_candidate_indices"] == [5, 8], result
    assert result["reasoning_ordered_candidate_indices"][:4] == [5, 8, 1, 2], result

    ordered = query_evidence.apply_reasoning_order_to_transcript_results(candidates, result)
    keyword_candidates, source = query_evidence.select_keyword_candidates(candidates, ordered, result, limit=20)
    assert source == "supporting_evidence", source
    assert keyword_candidates[0]["source_id"] == "cand5", keyword_candidates[:4]
    assert keyword_candidates[1]["source_id"] == "cand8", keyword_candidates[:4]
    assert keyword_candidates[2]["source_id"] == "cand1", keyword_candidates[:4]


def test_selected_candidate_overridden_to_first_supporting_candidate() -> None:
    candidates = supporting_candidates()
    result = reason_over_transcript_candidates(
        "Which items are unavailable?",
        candidates,
        top_n=30,
        model_function=lambda _prompt: json.dumps(
            {
                "answer_hypothesis": "alpha and beta",
                "supporting_candidate_indices": [5, 8],
                "selected_candidate_index": 1,
                "evidence_sentence": "I'm trying to gather all the items.",
                "confidence": 0.82,
                "reason": "Model selected a generic candidate, but support is elsewhere.",
                "steering_keywords": [],
                "reasoning_ordered_candidate_indices": [5, 8, 1, 2],
            }
        ),
    )
    assert result["selected_candidate_index"] == 5, result
    assert result["final_selected_candidate_index"] == 5, result
    assert result["original_selected_candidate_index"] == 1, result
    assert result["selected_candidate_overridden"] is True, result
    assert result["selected_candidate_id"] == "cand5", result
    assert result["reasoning_selected_retrieval_rank"] == 5, result
    assert result["evidence_sentence"] == "We have no alpha on the shelf.", result
    assert "not in supporting_candidate_indices" in result["selected_candidate_override_reason"], result


def test_keyword_candidates_fall_back_from_support_to_reasoning_order() -> None:
    import scripts.query_evidence as query_evidence

    candidates = supporting_candidates()
    reasoning = {
        "supporting_candidate_indices": [],
        "reasoning_order_used": True,
        "reasoning_ordered_transcript_candidates": [
            {
                "candidate_id": "cand8",
                "candidate_index": 8,
                "raw_retrieval_rank": 8,
                "reasoning_rank": 1,
            },
            {
                "candidate_id": "cand5",
                "candidate_index": 5,
                "raw_retrieval_rank": 5,
                "reasoning_rank": 2,
            },
        ],
    }
    ordered = query_evidence.apply_reasoning_order_to_transcript_results(candidates, reasoning)
    keyword_candidates, source = query_evidence.select_keyword_candidates(candidates, ordered, reasoning, limit=20)
    assert source == "reasoning_ordered", source
    assert keyword_candidates[0]["source_id"] == "cand8", keyword_candidates[:3]

    raw_candidates, raw_source = query_evidence.select_keyword_candidates(
        candidates,
        query_evidence.apply_reasoning_order_to_transcript_results(
            candidates,
            {"supporting_candidate_indices": [], "reasoning_ordered_transcript_candidates": [], "reasoning_order_used": False},
        ),
        {"supporting_candidate_indices": [], "reasoning_ordered_transcript_candidates": [], "reasoning_order_used": False},
        limit=20,
    )
    assert raw_source == "raw_retrieval_fallback", raw_source
    assert raw_candidates[0]["source_id"] == "cand1", raw_candidates[:3]


def test_reasoning_indices_are_truncated() -> None:
    candidates = numbered_candidates(50)
    result = reason_over_transcript_candidates(
        "Which candidates provide evidence?",
        candidates,
        top_n=50,
        model_function=lambda _prompt: json.dumps(
            {
                "answer_hypothesis": "several candidates provide evidence",
                "supporting_candidate_indices": list(range(1, 31)),
                "selected_candidate_index": 1,
                "evidence_sentence": "Candidate 1 has transcript evidence.",
                "confidence": 0.8,
                "reason": "Many candidates are relevant.",
                "steering_keywords": ["candidate 1"],
                "reasoning_ordered_candidate_indices": list(range(1, 41)),
            }
        ),
    )
    assert result["supporting_candidate_indices"] == list(range(1, 11)), result
    assert result["reasoning_ordered_candidate_indices"] == list(range(1, 21)), result
    assert result["num_supporting_candidate_indices_raw"] == 30, result
    assert result["num_supporting_candidate_indices_final"] == 10, result
    assert result["num_reasoning_ordered_indices_raw"] == 40, result
    assert result["num_reasoning_ordered_indices_final"] == 20, result


def test_missing_reasoning_order_uses_supporting_indices() -> None:
    candidates = supporting_candidates()
    result = reason_over_transcript_candidates(
        "Which items are unavailable?",
        candidates,
        top_n=30,
        model_function=lambda _prompt: json.dumps(
            {
                "answer_hypothesis": "alpha and beta",
                "supporting_candidate_indices": [5, 8],
                "selected_candidate_index": 5,
                "evidence_sentence": "We have no alpha on the shelf.",
                "confidence": 0.88,
                "reason": "Supporting candidates contain direct evidence.",
                "steering_keywords": ["no alpha"],
            }
        ),
    )
    assert result["supporting_candidate_indices"] == [5, 8], result
    assert result["reasoning_ordered_candidate_indices"] == [5, 8], result
    assert result["num_reasoning_ordered_indices_raw"] == 0, result
    assert result["num_reasoning_ordered_indices_final"] == 2, result


def test_per_candidate_scores_wrong_sense_match_low() -> None:
    candidates = [
        {
            "source_id": "cand1",
            "score": 1.0,
            "source_name": "A",
            "timestamp": "00:00-00:05",
            "transcript_path": "/tmp/a.json",
            "text": "The police officer was at the office.",
            "transcript_snippet": "The police officer was at the office.",
        },
        {
            "source_id": "cand2",
            "score": 0.8,
            "source_name": "B",
            "timestamp": "00:10-00:15",
            "transcript_path": "/tmp/b.json",
            "text": "Generic quiz chatter without the answer.",
            "transcript_snippet": "Generic quiz chatter without the answer.",
        },
        {
            "source_id": "cand3",
            "score": 0.7,
            "source_name": "C",
            "timestamp": "00:20-00:25",
            "transcript_path": "/tmp/c.json",
            "text": "Who was the frontman of The Police? The answer is Sting.",
            "transcript_snippet": "Who was the frontman of The Police? The answer is Sting.",
        },
    ]

    calls = {"scoring": 0, "answer": 0}

    def mock_model(prompt: str) -> str:
        if "answer_hypothesis" in prompt:
            calls["answer"] += 1
            return json.dumps(
                {
                    "answer_hypothesis": "Sting",
                    "selected_candidate_index": 3,
                    "evidence_sentence": "Who was the frontman of The Police? The answer is Sting.",
                    "confidence": 0.92,
                    "reason": "Candidate 3 directly answers the question.",
                    "steering_keywords": ["Sting"],
                }
            )
        if "candidate_scores" in prompt:
            calls["scoring"] += 1
            return json.dumps(
                {
                    "candidate_scores": [
                        {
                            "candidate_index": 1,
                            "evidence_score": 0.05,
                            "answer_likelihood": 0.02,
                            "directness": "none",
                            "supports_answer": False,
                            "possible_answer_terms": [],
                            "reason": "Police is used as a common noun, not the band/title.",
                        },
                        {
                            "candidate_index": 2,
                            "evidence_score": 0.25,
                            "answer_likelihood": 0.2,
                            "directness": "context",
                            "supports_answer": False,
                            "possible_answer_terms": [],
                            "reason": "It has quiz context but no answer evidence.",
                        },
                        {
                            "candidate_index": 3,
                            "evidence_score": 0.95,
                            "answer_likelihood": 0.95,
                            "directness": "direct",
                            "supports_answer": True,
                            "possible_answer_terms": ["Sting"],
                            "reason": "It directly states the answer.",
                        },
                    ]
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt[:300]}")

    result = reason_over_transcript_candidates(
        "Who was the front man of The Police?",
        candidates,
        top_n=3,
        model_function=mock_model,
        reasoning_mode="per_candidate",
    )
    scores = {item["candidate_index"]: item for item in result["candidate_scores"]}
    assert scores[1]["evidence_score"] < 0.2, result
    assert scores[1]["supports_answer"] is False, result
    assert scores[3]["evidence_score"] > 0.9, result
    assert scores[3]["supports_answer"] is True, result
    assert result["selected_candidate_index"] == 3, result
    assert result["supporting_candidate_indices"] == [3], result
    assert calls == {"scoring": 1, "answer": 1}, calls


def test_per_candidate_missing_scores_fall_back_safely() -> None:
    candidates = numbered_candidates(3)
    calls = {"scoring": 0, "answer": 0}

    def mock_model(prompt: str) -> str:
        if "candidate_scores" in prompt:
            calls["scoring"] += 1
            return json.dumps(
                {
                    "candidate_scores": [
                        {
                            "candidate_index": 2,
                            "evidence_score": 0.8,
                            "answer_likelihood": 0.7,
                            "directness": "direct",
                            "supports_answer": True,
                            "possible_answer_terms": ["candidate 2"],
                            "reason": "Only candidate 2 was scored.",
                        }
                    ]
                }
            )
        if "answer_hypothesis" in prompt:
            calls["answer"] += 1
            return json.dumps(
                {
                    "answer_hypothesis": "candidate 2",
                    "selected_candidate_index": 2,
                    "evidence_sentence": "Candidate 2 has transcript evidence.",
                    "confidence": 0.7,
                    "reason": "Candidate 2 was the only direct score.",
                    "steering_keywords": ["candidate 2"],
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt[:300]}")

    result = reason_over_transcript_candidates(
        "Which candidate has evidence?",
        candidates,
        top_n=3,
        model_function=mock_model,
        reasoning_mode="per_candidate",
    )
    scores = {item["candidate_index"]: item for item in result["candidate_scores"]}
    assert scores[1]["evidence_score"] == 0.0, result
    assert "missing from batch candidate_scores" in scores[1]["reason"], result
    assert scores[2]["evidence_score"] == 0.8, result
    assert result["selected_candidate_index"] == 2, result
    assert calls == {"scoring": 1, "answer": 1}, calls


def test_batch_scoring_parse_failure_debug_fields() -> None:
    candidates = numbered_candidates(2)
    bad_json = '{"candidate_scores": [{"candidate_index": 1 "evidence_score": 0.5}]}'

    def mock_model(prompt: str) -> str:
        if "candidate_scores" in prompt:
            return bad_json
        if "answer_hypothesis" in prompt:
            return json.dumps(
                {
                    "answer_hypothesis": "",
                    "selected_candidate_index": 1,
                    "evidence_sentence": "Candidate 1 has transcript evidence.",
                    "confidence": 0.0,
                    "reason": "fallback",
                    "steering_keywords": [],
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt[:300]}")

    result = reason_over_transcript_candidates(
        "Which candidate has evidence?",
        candidates,
        top_n=2,
        model_function=mock_model,
        reasoning_mode="per_candidate",
    )
    assert result["raw_batch_scoring_output"] == bad_json, result
    assert result["raw_batch_scoring_output_length"] == len(bad_json), result
    assert result["batch_scoring_output_length_chars"] == len(bad_json), result
    assert result["batch_scoring_prompt_length_chars"] > 0, result
    assert result["batch_scoring_num_candidates_requested"] == 2, result
    assert result["batch_scoring_num_candidates_completed"] == 0, result
    assert result["batch_scoring_num_batches"] == 1, result
    assert result["batch_scoring_batch_size"] == 5, result
    assert result["batch_scoring_completed_batches"] == 0, result
    assert result["batch_scoring_completed_candidates"] == 0, result
    assert result["batch_scoring_json_parse_failed"] is True, result
    assert "JSONDecodeError" in result["batch_scoring_exception"], result
    assert result["batch_scoring_candidate_count_returned"] == 0, result


def test_batch_scoring_splits_candidates_into_batches_of_five() -> None:
    candidates = numbered_candidates(12)
    calls = {"scoring": 0, "answer": 0}

    def mock_model(prompt: str) -> str:
        if "candidate_scores" in prompt:
            calls["scoring"] += 1
            scores = []
            for candidate_index in range(1, 13):
                marker = f"Candidate {candidate_index}\n"
                if marker not in prompt:
                    continue
                scores.append(
                    {
                        "candidate_index": candidate_index,
                        "evidence_score": 0.7 if candidate_index == 12 else 0.1,
                        "answer_likelihood": 0.7 if candidate_index == 12 else 0.1,
                        "directness": "direct" if candidate_index == 12 else "none",
                        "supports_answer": candidate_index == 12,
                    }
                )
            return json.dumps({"candidate_scores": scores})
        if "answer_hypothesis" in prompt:
            calls["answer"] += 1
            return json.dumps(
                {
                    "answer_hypothesis": "candidate 12",
                    "selected_candidate_index": 12,
                    "evidence_sentence": "Candidate 12 has transcript evidence.",
                    "confidence": 0.7,
                    "reason": "Candidate 12 was the direct score.",
                    "steering_keywords": ["candidate 12"],
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt[:300]}")

    result = reason_over_transcript_candidates(
        "Which candidate has evidence?",
        candidates,
        top_n=12,
        model_function=mock_model,
        reasoning_mode="per_candidate",
    )
    assert calls == {"scoring": 3, "answer": 1}, calls
    assert result["batch_scoring_num_batches"] == 3, result
    assert result["batch_scoring_batch_size"] == 5, result
    assert result["batch_scoring_completed_batches"] == 3, result
    assert result["batch_scoring_completed_candidates"] == 12, result
    assert result["batch_scoring_num_candidates_completed"] == 12, result
    assert result["selected_candidate_index"] == 12, result


def fake_candidates() -> list[dict]:
    return [
        {
            "source_id": "cand1",
            "score": 1.0,
            "source_name": "Kitchen",
            "day": "day1",
            "hour_id": "14",
            "start_sec": 0.0,
            "end_sec": 5.0,
            "timestamp": "00:00-00:05",
            "transcript_path": "/tmp/kitchen.json",
            "text": "They are checking the setup table.",
            "transcript_snippet": "They are checking the setup table.",
            "youtube_timestamp_url": "https://example.com/1",
        },
        {
            "source_id": "cand2",
            "score": 0.75,
            "source_name": "Stevan",
            "day": "day1",
            "hour_id": "14",
            "start_sec": 10.0,
            "end_sec": 15.0,
            "timestamp": "00:10-00:15",
            "transcript_path": "/tmp/stevan.json",
            "text": "The missing supplies are spare batteries and tape.",
            "transcript_snippet": "The missing supplies are spare batteries and tape.",
            "youtube_timestamp_url": "https://example.com/2",
        },
    ]


def numbered_candidates(count: int) -> list[dict]:
    candidates = []
    for index in range(1, count + 1):
        candidates.append(
            {
                "source_id": f"cand{index}",
                "score": 1.0 / index,
                "source_name": f"Source{index}",
                "day": "day1",
                "hour_id": "14",
                "start_sec": float(index),
                "end_sec": float(index + 1),
                "timestamp": f"00:{index:02d}-00:{index + 1:02d}",
                "transcript_path": f"/tmp/source{index}.json",
                "text": f"Candidate {index} has transcript evidence.",
                "transcript_snippet": f"Candidate {index} has transcript evidence.",
                "youtube_timestamp_url": f"https://example.com/{index}",
            }
        )
    return candidates


def supporting_candidates() -> list[dict]:
    candidates = numbered_candidates(30)
    candidates[0]["text"] = "I'm trying to gather all the items."
    candidates[0]["transcript_snippet"] = "I'm trying to gather all the items."
    candidates[4]["text"] = "We have no alpha on the shelf."
    candidates[4]["transcript_snippet"] = "We have no alpha on the shelf."
    candidates[7]["text"] = "Beta is missing from the inventory."
    candidates[7]["transcript_snippet"] = "Beta is missing from the inventory."
    return candidates


if __name__ == "__main__":
    main()
