def format_score(value: float) -> str:
    """Raw retrieval score (cosine similarity / cross-encoder logit / RRF score) — unbounded, modality-specific."""
    return f"{value:.3f}"


def format_percent(value: float) -> str:
    """Confidence, 0..1, from the mixed evidence ranker — relative to the top result in a given search."""
    return f"{value * 100:.0f}%"


def format_timestamp(timestamp: str) -> str:
    return timestamp
