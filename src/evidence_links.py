from __future__ import annotations

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


def youtube_timestamp_url(chunk: dict, timestamp_sec: float | None = None) -> str | None:
    youtube_url = str(chunk.get("youtube_url") or "").strip()
    youtube_id = str(chunk.get("youtube_id") or "").strip()
    if not youtube_url and youtube_id:
        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
    if not youtube_url:
        return None

    if timestamp_sec is None:
        timestamp_sec = _timestamp_for_chunk(chunk)
    if timestamp_sec is None:
        return youtube_url

    return _with_timestamp(youtube_url, int(max(0, round(timestamp_sec))))


def _timestamp_for_chunk(chunk: dict) -> float | None:
    for field in ("visual_caption_time_sec", "keyframe_time_sec", "start_sec"):
        value = chunk.get(field)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _with_timestamp(url: str, seconds: int) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if parsed.netloc.endswith("youtu.be"):
        query["t"] = [str(seconds)]
    else:
        query["t"] = [str(seconds)]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))
