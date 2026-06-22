#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transcript_answer_reranker import rerank_transcript_answers


def main() -> None:
    question = "What ingredients are missing at the start?"
    candidates = [
        transcript_candidate(
            score=1.0,
            source_name="Kitchen",
            text="I'm just trying to gather all the ingredients.",
            start_sec=10.0,
            end_sec=15.0,
        ),
        transcript_candidate(
            score=0.70,
            source_name="Stevan",
            text="We have no celery, we have no bay leaf, nothing.",
            start_sec=20.0,
            end_sec=25.0,
        ),
    ]
    result = rerank_transcript_answers(
        question,
        candidates,
        top_n=2,
        qa_function=mock_qa,
        relevance_function=mock_relevance,
    )
    best = result["best_transcript"]
    assert best["source_name"] == "Stevan", result
    assert "no celery" in result["best_answer"]["text"], result
    assert result["answer_rerank_candidates"][0]["source_name"] == "Stevan", result
    print("transcript answer reranker smoke test passed")


def mock_qa(question: str, transcript_text: str) -> dict:
    del question
    if "no celery" in transcript_text:
        text = "no celery, we have no bay leaf, nothing"
        return {
            "answer_span_text": text,
            "char_start": transcript_text.index("no celery"),
            "char_end": transcript_text.index("nothing") + len("nothing"),
            "score": 0.95,
            "raw_score": 4.0,
            "method": "mock_qa",
            "qa_context_text": transcript_text,
            "answer_candidates": [{"text": text, "score": 0.95}],
        }
    return {
        "answer_span_text": "all the ingredients",
        "char_start": transcript_text.index("all"),
        "char_end": len(transcript_text) - 1,
        "score": 0.45,
        "raw_score": 0.5,
        "method": "mock_qa",
        "qa_context_text": transcript_text,
        "answer_candidates": [{"text": "all the ingredients", "score": 0.45}],
    }


def mock_relevance(question: str, texts: list[str]) -> list[float]:
    del question
    return [1.0 if "no celery" in text else 0.1 for text in texts]


def transcript_candidate(score: float, source_name: str, text: str, start_sec: float, end_sec: float) -> dict:
    return {
        "score": score,
        "source_name": source_name,
        "day": "day1",
        "hour_id": "13",
        "start_sec": start_sec,
        "end_sec": end_sec,
        "timestamp": f"{int(start_sec)}-{int(end_sec)}",
        "transcript_path": f"/tmp/{source_name}.json",
        "transcript_snippet": text,
        "text": text,
        "youtube_timestamp_url": "https://example.com",
    }


if __name__ == "__main__":
    main()
