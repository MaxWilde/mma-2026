from __future__ import annotations

import re
import math
from typing import Any


VISUAL_CUES = {
    "color",
    "where",
    "visible",
    "wearing",
    "holding",
    "object",
    "on",
    "under",
    "above",
    "beside",
    "near",
    "next",
    "left",
    "right",
}

TRANSCRIPT_CUES = {
    "say",
    "said",
    "mention",
    "mentioned",
    "talk",
    "tell",
    "told",
    "ask",
    "asked",
    "why",
    "think",
    "thought",
    "want",
    "decide",
    "decided",
    "explain",
    "reason",
    "long",
    "minutes",
    "hour",
}


def route_evidence(
    question: str,
    visual_results: list[dict[str, Any]],
    transcript_results: list[dict[str, Any]],
    feedback: dict[str, int] | None = None,
) -> dict[str, Any]:
    visual = visual_results[0] if visual_results else None
    transcript = transcript_results[0] if transcript_results else None
    base_visual_score = route_score(question, visual, "visual")
    base_transcript_score = route_score(question, transcript, "transcript")
    feedback_preference = modality_feedback_preference(feedback)
    feedback_strength = max(
        0.0,
        float((feedback or {}).get("strength", 0.75)),
    )
    visual_score = base_visual_score + feedback_strength * feedback_preference
    transcript_score = base_transcript_score - feedback_strength * feedback_preference
    raw_visual_score = safe_score(visual)
    raw_transcript_score = safe_score(transcript)
    visual_heuristic = heuristic_score(question, "visual")
    transcript_heuristic = heuristic_score(question, "transcript")
    transcript_raw_margin = raw_transcript_score - raw_visual_score

    if (
        feedback_preference <= 0.0
        and visual_heuristic == 0.0
        and transcript_raw_margin >= 0.25
    ):
        chosen = dict(transcript or {})
        chosen["evidence_type"] = "transcript"
        reason = (
            "transcript selected because the query had no visual evidence cues "
            "and raw transcript retrieval score was substantially higher than raw visual retrieval score"
        )
    elif transcript_score > visual_score:
        chosen = dict(transcript or {})
        chosen["evidence_type"] = "transcript"
        if raw_transcript_score > raw_visual_score:
            reason = (
                "transcript selected because combined transcript score exceeded visual score "
                "and raw transcript retrieval score was higher than raw visual retrieval score"
            )
        else:
            reason = "transcript selected because combined transcript score exceeded visual score"
    else:
        chosen = dict(visual or {})
        chosen["evidence_type"] = "visual"
        reason = "visual selected because combined visual score met or exceeded transcript score"

    if feedback_preference > 0:
        reason += "; analyst feedback favored visual evidence"
    elif feedback_preference < 0:
        reason += "; analyst feedback favored transcript evidence"

    router_margin = abs(visual_score - transcript_score)
    router_confidence = bounded_margin_confidence(router_margin)
    router_second_choice = "transcript" if chosen["evidence_type"] == "visual" else "visual"
    chosen["router_confidence"] = router_confidence
    chosen["router_confidence_percent"] = router_confidence * 100.0
    chosen["router_margin"] = router_margin
    chosen["router_second_choice"] = router_second_choice
    chosen["router_confidence_note"] = (
        "Relative route-decision confidence derived from combined score margin; not calibrated probability."
    )
    chosen["router_debug"] = {
        "heuristic_visual_score": visual_heuristic,
        "heuristic_transcript_score": transcript_heuristic,
        "top_visual_score": raw_visual_score,
        "top_transcript_score": raw_transcript_score,
        "raw_transcript_minus_visual": transcript_raw_margin,
        "combined_visual_score": visual_score,
        "combined_transcript_score": transcript_score,
        "base_combined_visual_score": base_visual_score,
        "base_combined_transcript_score": base_transcript_score,
        "feedback_preference": feedback_preference,
        "feedback_strength": feedback_strength,
        "feedback": dict(feedback or {}),
        "router_margin": router_margin,
        "router_confidence": router_confidence,
        "router_confidence_percent": router_confidence * 100.0,
        "router_second_choice": router_second_choice,
        "router_confidence_note": chosen["router_confidence_note"],
        "chosen_route": chosen["evidence_type"],
        "reason": reason,
    }
    return chosen


def modality_feedback_preference(feedback: dict[str, int] | None) -> float:
    """Return a bounded preference in [-1, 1] from analyst feedback."""
    if not feedback:
        return 0.0
    liked_visual = max(0, int(feedback.get("liked_visual", 0)))
    disliked_visual = max(0, int(feedback.get("disliked_visual", 0)))
    liked_transcript = max(0, int(feedback.get("liked_transcript", 0)))
    disliked_transcript = max(0, int(feedback.get("disliked_transcript", 0)))
    total = liked_visual + disliked_visual + liked_transcript + disliked_transcript
    if total <= 0:
        return 0.0
    signal = (
        liked_visual
        + disliked_transcript
        - liked_transcript
        - disliked_visual
    )
    return max(-1.0, min(1.0, signal / total))


def route_score(question: str, result: dict[str, Any] | None, mode: str) -> float:
    if result is None:
        return float("-inf")
    return heuristic_score(question, mode) + normalize_retrieval_score(safe_score(result), mode)


def heuristic_score(question: str, mode: str) -> float:
    tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    if mode == "visual":
        score = 0.0
        score += 1.2 * len(tokens & VISUAL_CUES)
        if re.search(r"\bwhere\s+(?:is|are|was|were)\b", question, flags=re.IGNORECASE):
            score += 2.0
        if re.search(r"\bwhat\s+color\b", question, flags=re.IGNORECASE):
            score += 2.0
        if re.search(r"\bwhat\s+(?:is|are|was|were)\s+(?:on|in|near|beside|under|above)\b", question, flags=re.IGNORECASE):
            score += 2.0
        return score

    score = 0.0
    score += 1.2 * len(tokens & TRANSCRIPT_CUES)
    if re.search(r"\bhow\s+long\b", question, flags=re.IGNORECASE):
        score += 2.5
    if re.search(r"\bwhat\s+did\b", question, flags=re.IGNORECASE):
        score += 2.0
    if re.search(r"\bwhy\b", question, flags=re.IGNORECASE):
        score += 2.0
    return score


def normalize_retrieval_score(score: float, mode: str) -> float:
    return max(0.0, min(score, 1.5))


def safe_score(result: dict[str, Any] | None) -> float:
    if not result:
        return 0.0
    try:
        return float(result.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def bounded_margin_confidence(margin: float) -> float:
    margin = max(0.0, float(margin))
    if not math.isfinite(margin):
        return 1.0
    return margin / (margin + 1.0)
