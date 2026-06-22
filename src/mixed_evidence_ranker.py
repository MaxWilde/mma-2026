from __future__ import annotations

import math
import re
from typing import Any


QUALITY_WEIGHT = 0.75
PRIOR_WEIGHT = 0.25


def build_mixed_evidence_list(
    question: str,
    visual_results: list[dict],
    transcript_results: list[dict],
    router_debug: dict,
    top_k: int = 20,
    *,
    temperature: float = 1.5,
    calibration_mode: str = "max",
    return_debug: bool = False,
) -> list[dict]:
    del question  # Reserved for future query-aware display features.
    validate_calibration_mode(calibration_mode)
    visual_weight, transcript_weight = channel_weights(router_debug, temperature=temperature)
    visual_candidates = diverse_visual_candidates(visual_results)
    transcript_candidates = diverse_transcript_candidates(transcript_results)
    visual_quality = calibrated_quality_scores(visual_candidates, calibration_mode)
    transcript_quality = calibrated_quality_scores(transcript_candidates, calibration_mode)

    mixed: list[dict[str, Any]] = []
    mixed.extend(
        score_candidates(
            visual_candidates,
            visual_quality,
            evidence_type="visual",
            channel_weight=visual_weight,
            calibration_mode=calibration_mode,
        )
    )
    mixed.extend(
        score_candidates(
            transcript_candidates,
            transcript_quality,
            evidence_type="transcript",
            channel_weight=transcript_weight,
            calibration_mode=calibration_mode,
        )
    )
    mixed.sort(key=item_final_score, reverse=True)
    mixed, diversity_debug = suppress_event_duplicates(mixed)
    mixed = mixed[:top_k]
    for rank, item in enumerate(mixed, start=1):
        final_score = float(item["score_components"]["final_score"])
        # Use the raw final_score as confidence instead of normalizing by the
        # top result, which would always make the top result show 100%.
        confidence = min(final_score, 1.0)
        item["rank"] = rank
        item["confidence"] = confidence
        item["confidence_percent"] = confidence * 100.0
        item["mixed_rank_confidence"] = confidence
        item["mixed_rank_confidence_percent"] = confidence * 100.0
        item["confidence_note"] = "Mixed evidence confidence (quality × channel weight); not calibrated probability."
        evidence_confidence = mixed_evidence_confidence(item)
        item["evidence_confidence"] = evidence_confidence
        item["evidence_confidence_percent"] = (
            evidence_confidence * 100.0 if evidence_confidence is not None else None
        )
        item["evidence_confidence_note"] = (
            "Relative mixed evidence confidence combining mixed rank, retrieval confidence, "
            "and router channel weight; not calibrated probability."
        )
    if return_debug:
        return {"items": mixed, "diversity_debug": diversity_debug}
    return mixed


def channel_weights(router_debug: dict, *, temperature: float) -> tuple[float, float]:
    visual_score = safe_float(router_debug.get("combined_visual_score"), 0.0)
    transcript_score = safe_float(router_debug.get("combined_transcript_score"), 0.0)
    temperature = max(0.01, float(temperature))
    visual_logit = visual_score / temperature
    transcript_logit = transcript_score / temperature
    max_logit = max(visual_logit, transcript_logit)
    visual_exp = math.exp(visual_logit - max_logit)
    transcript_exp = math.exp(transcript_logit - max_logit)
    total = visual_exp + transcript_exp
    if total <= 0:
        return 0.5, 0.5
    return visual_exp / total, transcript_exp / total


def score_candidates(
    candidates: list[dict],
    quality_scores: list[dict[str, float | None]],
    *,
    evidence_type: str,
    channel_weight: float,
    calibration_mode: str,
) -> list[dict[str, Any]]:
    scored = []
    for item, quality in zip(candidates, quality_scores):
        calibrated_quality_score = float(quality["calibrated_quality_score"] or 0.0)
        diversity_penalty = float(item.get("_diversity_penalty", 0.0))
        quality_component = QUALITY_WEIGHT * calibrated_quality_score
        prior_component = PRIOR_WEIGHT * channel_weight
        final_score = (quality_component + prior_component) * max(0.0, 1.0 - diversity_penalty)
        updated = dict(item)
        updated["evidence_type"] = evidence_type
        updated["confidence"] = 0.0
        updated["confidence_percent"] = 0.0
        updated["score_components"] = {
            "raw_score": safe_float(item.get("score"), 0.0),
            "calibration_mode": calibration_mode,
            "calibrated_quality_score": calibrated_quality_score,
            "normalized_score": quality.get("normalized_score"),
            "percentile_rank": quality.get("percentile_rank"),
            "router_channel_weight": channel_weight,
            "quality_component": quality_component,
            "prior_component": prior_component,
            "diversity_penalty": diversity_penalty,
            "final_score": final_score,
            "confidence_note": "Relative display confidence within the mixed top-k list, not calibrated probability.",
        }
        updated.pop("_diversity_penalty", None)
        scored.append(updated)
    return scored


def calibrated_quality_scores(candidates: list[dict], calibration_mode: str) -> list[dict[str, float | None]]:
    if calibration_mode == "max":
        return [
            {
                "calibrated_quality_score": score,
                "normalized_score": score,
                "percentile_rank": None,
            }
            for score in normalized_scores(candidates)
        ]
    if calibration_mode == "percentile":
        return [
            {
                "calibrated_quality_score": score,
                "normalized_score": None,
                "percentile_rank": score,
            }
            for score in percentile_quality_scores(candidates)
        ]
    raise ValueError(f"Unsupported calibration mode: {calibration_mode}")


def normalized_scores(candidates: list[dict]) -> list[float]:
    raw_scores = [max(0.0, safe_float(item.get("score"), 0.0)) for item in candidates]
    max_score = max(raw_scores, default=0.0)
    if max_score <= 0:
        return [0.0 for _ in raw_scores]
    return [score / max_score for score in raw_scores]


def percentile_quality_scores(candidates: list[dict]) -> list[float]:
    count = len(candidates)
    if count == 0:
        return []
    if count == 1:
        return [1.0]
    return [1.0 - (rank_index / max(1, count - 1)) for rank_index in range(count)]


def validate_calibration_mode(calibration_mode: str) -> None:
    if calibration_mode not in {"max", "percentile"}:
        raise ValueError(f"Unsupported calibration mode: {calibration_mode}")


def diverse_visual_candidates(candidates: list[dict], *, window_sec: float = 30.0) -> list[dict]:
    selected: list[dict] = []
    for item in sorted(candidates, key=lambda value: safe_float(value.get("score"), 0.0), reverse=True):
        if is_near_duplicate_visual(item, selected, window_sec=window_sec):
            continue
        selected.append(dict(item))
    return selected


def is_near_duplicate_visual(candidate: dict, selected: list[dict], *, window_sec: float) -> bool:
    candidate_time = safe_float(candidate.get("keyframe_time_sec", candidate.get("start_sec")), None)
    if candidate_time is None:
        return False
    candidate_source = (
        candidate.get("day"),
        candidate.get("source_name"),
        candidate.get("video_id", candidate.get("hour_id")),
    )
    for item in selected:
        item_time = safe_float(item.get("keyframe_time_sec", item.get("start_sec")), None)
        item_source = (
            item.get("day"),
            item.get("source_name"),
            item.get("video_id", item.get("hour_id")),
        )
        if item_time is None:
            continue
        if candidate_source == item_source and abs(candidate_time - item_time) <= window_sec:
            return True
    return False


def diverse_transcript_candidates(candidates: list[dict], *, overlap_iou_threshold: float = 0.7) -> list[dict]:
    selected: list[dict] = []
    for item in sorted(candidates, key=lambda value: safe_float(value.get("score"), 0.0), reverse=True):
        if is_overlapping_transcript(item, selected, overlap_iou_threshold=overlap_iou_threshold):
            continue
        selected.append(dict(item))
    return selected


def is_overlapping_transcript(candidate: dict, selected: list[dict], *, overlap_iou_threshold: float) -> bool:
    candidate_start = safe_float(candidate.get("start_sec"), None)
    candidate_end = safe_float(candidate.get("end_sec"), None)
    if candidate_start is None or candidate_end is None:
        return False
    candidate_source = (
        candidate.get("day"),
        candidate.get("source_name"),
        candidate.get("hour_id", candidate.get("video_id")),
    )
    for item in selected:
        item_source = (
            item.get("day"),
            item.get("source_name"),
            item.get("hour_id", item.get("video_id")),
        )
        if candidate_source != item_source:
            continue
        item_start = safe_float(item.get("start_sec"), None)
        item_end = safe_float(item.get("end_sec"), None)
        if item_start is None or item_end is None:
            continue
        if time_iou(candidate_start, candidate_end, item_start, item_end) > overlap_iou_threshold:
            return True
    return False


def suppress_event_duplicates(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for item in items:
        duplicate = find_event_duplicate(item, selected)
        if duplicate is None:
            selected.append(item)
            continue
        reason = duplicate_reason(item, duplicate)
        updated = dict(item)
        components = dict(updated.get("score_components") or {})
        components["duplicate_suppressed_by"] = evidence_identifier(duplicate)
        components["diversity_reason"] = reason
        updated["score_components"] = components
        suppressed.append(suppressed_summary(updated, duplicate, reason))
    return selected, {
        "before_suppression": len(items),
        "after_suppression": len(selected),
        "suppressed_count": len(suppressed),
        "suppressed_type_counts": suppressed_type_counts(suppressed),
        "suppressed_examples": suppressed[:20],
    }


def find_event_duplicate(candidate: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in selected:
        if candidate.get("evidence_type") == item.get("evidence_type"):
            if candidate.get("evidence_type") == "transcript" and is_event_duplicate_transcript(candidate, item):
                return item
            if candidate.get("evidence_type") == "visual" and is_event_duplicate_visual(candidate, item):
                return item
        elif is_conservative_cross_modal_duplicate(candidate, item):
            return item
    return None


def is_event_duplicate_transcript(candidate: dict[str, Any], item: dict[str, Any]) -> bool:
    if same_day_hour(candidate, item):
        candidate_start = safe_float(candidate.get("start_sec"), None)
        candidate_end = safe_float(candidate.get("end_sec"), None)
        item_start = safe_float(item.get("start_sec"), None)
        item_end = safe_float(item.get("end_sec"), None)
        if None not in (candidate_start, candidate_end, item_start, item_end):
            if time_iou(float(candidate_start), float(candidate_end), float(item_start), float(item_end)) > 0.5:
                return True
    return transcript_text_jaccard(candidate, item) > 0.75


def is_event_duplicate_visual(candidate: dict[str, Any], item: dict[str, Any]) -> bool:
    if not same_day_hour(candidate, item):
        return False
    candidate_time = evidence_time(candidate)
    item_time = evidence_time(item)
    if candidate_time is None or item_time is None:
        return False
    delta = abs(candidate_time - item_time)
    if same_source(candidate, item) and delta <= 30.0:
        return True
    return delta <= 10.0


def is_conservative_cross_modal_duplicate(candidate: dict[str, Any], item: dict[str, Any]) -> bool:
    if not same_day_hour(candidate, item):
        return False
    candidate_time = evidence_time(candidate)
    item_time = evidence_time(item)
    if candidate_time is None or item_time is None:
        return False
    return abs(candidate_time - item_time) <= 3.0 and same_source(candidate, item)


def duplicate_reason(candidate: dict[str, Any], item: dict[str, Any]) -> str:
    if candidate.get("evidence_type") == item.get("evidence_type") == "transcript":
        if same_day_hour(candidate, item):
            candidate_start = safe_float(candidate.get("start_sec"), None)
            candidate_end = safe_float(candidate.get("end_sec"), None)
            item_start = safe_float(item.get("start_sec"), None)
            item_end = safe_float(item.get("end_sec"), None)
            if None not in (candidate_start, candidate_end, item_start, item_end):
                iou = time_iou(float(candidate_start), float(candidate_end), float(item_start), float(item_end))
                if iou > 0.5:
                    return f"transcript same day/hour timestamp overlap IoU={iou:.3f}"
        return f"transcript text Jaccard={transcript_text_jaccard(candidate, item):.3f}"
    if candidate.get("evidence_type") == item.get("evidence_type") == "visual":
        delta = time_delta(candidate, item)
        if same_source(candidate, item):
            return f"visual same source within {delta:.1f}s"
        return f"visual same day/hour within {delta:.1f}s across POVs"
    return f"conservative cross-modal same source/event within {time_delta(candidate, item):.1f}s"


def same_day_hour(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        str(a.get("day", "")) == str(b.get("day", ""))
        and str(a.get("hour_id", a.get("video_id", ""))) == str(b.get("hour_id", b.get("video_id", "")))
    )


def same_source(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return str(a.get("source_name", "")) == str(b.get("source_name", ""))


def evidence_time(item: dict[str, Any]) -> float | None:
    return safe_float(item.get("keyframe_time_sec", item.get("start_sec")), None)


def time_delta(a: dict[str, Any], b: dict[str, Any]) -> float:
    a_time = evidence_time(a)
    b_time = evidence_time(b)
    if a_time is None or b_time is None:
        return float("inf")
    return abs(a_time - b_time)


def transcript_text_jaccard(a: dict[str, Any], b: dict[str, Any]) -> float:
    a_tokens = set(normalized_text_tokens(transcript_text(a)))
    b_tokens = set(normalized_text_tokens(transcript_text(b)))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def transcript_text(item: dict[str, Any]) -> str:
    return str(item.get("transcript_snippet") or item.get("text") or item.get("lexical_matched_window") or "")


def normalized_text_tokens(text: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9\s]+", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.split() if normalized else []


def evidence_identifier(item: dict[str, Any]) -> str:
    if item.get("evidence_type") == "visual":
        return str(item.get("keyframe_path") or item.get("source_id") or "")
    return str(item.get("source_id") or f"{item.get('transcript_path')}:{item.get('start_sec')}:{item.get('end_sec')}")


def suppressed_summary(item: dict[str, Any], kept: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "suppressed_type": item.get("evidence_type"),
        "suppressed_id": evidence_identifier(item),
        "suppressed_timestamp": item.get("timestamp"),
        "suppressed_score": item_final_score(item),
        "kept_type": kept.get("evidence_type"),
        "kept_id": evidence_identifier(kept),
        "kept_timestamp": kept.get("timestamp"),
        "kept_score": item_final_score(kept),
        "reason": reason,
    }


def suppressed_type_counts(suppressed: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"visual": 0, "transcript": 0, "cross_modal": 0}
    for item in suppressed:
        suppressed_type = str(item.get("suppressed_type") or "")
        kept_type = str(item.get("kept_type") or "")
        if suppressed_type == kept_type == "visual":
            counts["visual"] += 1
        elif suppressed_type == kept_type == "transcript":
            counts["transcript"] += 1
        else:
            counts["cross_modal"] += 1
    return counts


def item_final_score(item: dict[str, Any]) -> float:
    return float((item.get("score_components") or {}).get("final_score", 0.0))


def time_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if a_end < a_start:
        a_start, a_end = a_end, a_start
    if b_end < b_start:
        b_start, b_end = b_end, b_start
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mixed_evidence_confidence(item: dict[str, Any]) -> float | None:
    components = item.get("score_components") or {}
    if item.get("evidence_type") == "visual":
        retrieval_confidence = item.get("visual_retrieval_confidence")
    else:
        retrieval_confidence = item.get("transcript_retrieval_confidence")
    return geometric_mean(
        [
            item.get("mixed_rank_confidence"),
            retrieval_confidence,
            components.get("router_channel_weight"),
        ]
    )


def geometric_mean(values: list[Any]) -> float | None:
    usable = []
    for value in values:
        parsed = safe_float(value, None)
        if parsed is not None and 0.0 <= parsed <= 1.0:
            usable.append(parsed)
    if not usable:
        return None
    product = 1.0
    for value in usable:
        product *= max(value, 1e-6)
    return product ** (1.0 / len(usable))
