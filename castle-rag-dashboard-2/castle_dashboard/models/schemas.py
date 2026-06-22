from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TranscriptSegment:
    start: str
    end: str
    speaker: str
    text: str


@dataclass(frozen=True)
class SegmentationMask:
    label: str
    confidence: float
    left: float    # % from left edge
    top: float     # % from top edge
    width: float   # % of image width
    height: float  # % of image height


@dataclass
class RetrievalResult:
    id: str
    rank: int
    title: str
    answer: str
    score: float              # confidence, 0..1 — from the mixed evidence ranker
    timestamp: str           # "HH:MM:SS"
    timestamp_seconds: float
    viewpoint: str           # source_name / POV label
    camera: str              # video_id / hour
    modality: str            # "visual" | "transcript" (== evidence_type)
    grounding: str           # "none" until localized
    keyframe_path: str       # absolute path on disk
    video_path: str
    caption: str
    transcript: tuple[TranscriptSegment, ...]
    mask: SegmentationMask | None
    tags: tuple[str, ...]
    youtube_url: str = ""
    start_sec: float = 0.0
    end_sec: float = 0.0
    source_name: str = ""
    day: str = ""
    video_id: str = ""
    evidence_type: str = ""       # "visual" | "transcript" — raw router/ranker label
    confidence_percent: float = 0.0  # 0..100, relative to the top result in this search
    raw_score: float = 0.0        # underlying retrieval score (cosine sim / rerank score)
    is_routed_choice: bool = False  # True for the single result evidence_router would pick
    faiss_row_id: int | None = None  # positional index in the visual FAISS index; None for transcript


@dataclass(frozen=True)
class ProjectionPoint:
    result_id: str
    x: float
    y: float
    cluster: str
    label: str


@dataclass(frozen=True)
class SearchFilters:
    query: str
    modalities: tuple[str, ...]
    viewpoint: str
    min_score: float
    max_score: float
    top_k: int = 20


@dataclass(frozen=True)
class DashboardStats:
    total_videos: int
    total_keyframes: int
    transcript_chunks: int
    indexed_items: int
    avg_latency_ms: int
    retrieval_quality: float
    grounding_quality: float
