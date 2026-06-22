from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.visual_grounding import GroundingConfig, VisualGroundingResult, ground_visual_evidence
from src.vqa import match_transcript_evidence


EvidenceType = Literal["transcript", "visual", "none"]


@dataclass(frozen=True)
class EvidenceResult:
    evidence_type: EvidenceType
    chunk: dict | None
    transcript_snippet: str | None = None
    visual_caption: str | None = None
    visual_grounding: VisualGroundingResult | None = None


def select_evidence(
    question: str,
    chunks: list[dict],
    embedding_model=None,
    grounding_config: GroundingConfig | None = None,
) -> EvidenceResult:
    if not chunks:
        return EvidenceResult(evidence_type="none", chunk=None)

    transcript_match = match_transcript_evidence(question, chunks, embedding_model=embedding_model)
    if transcript_match.best_sentence and transcript_match.best_chunk and transcript_match.best_source == "transcript":
        return EvidenceResult(
            evidence_type="transcript",
            chunk=transcript_match.best_chunk,
            transcript_snippet=transcript_match.best_sentence,
        )

    visual_chunk = _best_visual_chunk(transcript_match.best_chunk, chunks)
    if visual_chunk is not None:
        return EvidenceResult(
            evidence_type="visual",
            chunk=visual_chunk,
            visual_caption=str(visual_chunk.get("visual_caption", "")).strip() or None,
            visual_grounding=ground_visual_evidence(question, visual_chunk, config=grounding_config),
        )

    if transcript_match.best_sentence and transcript_match.best_chunk:
        return EvidenceResult(
            evidence_type="transcript",
            chunk=transcript_match.best_chunk,
            transcript_snippet=transcript_match.best_sentence,
        )

    return EvidenceResult(evidence_type="none", chunk=chunks[0])


def _best_visual_chunk(extractive_chunk: dict | None, chunks: list[dict]) -> dict | None:
    if extractive_chunk and _has_visual_evidence(extractive_chunk):
        return extractive_chunk
    for chunk in chunks:
        if _has_visual_evidence(chunk):
            return chunk
    return None


def _has_visual_evidence(chunk: dict) -> bool:
    return bool(
        str(chunk.get("visual_caption", "")).strip()
        and (
            chunk.get("keyframe_path")
            or chunk.get("visual_caption_keyframe_path")
            or chunk.get("closest_keyframe_path")
        )
    )
