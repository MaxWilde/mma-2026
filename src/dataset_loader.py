from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


TRANSCRIPT_SKIP_NAMES = {
    "manifest.json",
    "shots.json",
    "manifest_hyst.json",
    "shots_hyst.json",
}


@dataclass(frozen=True)
class EvidenceItem:
    source_id: str
    area: str
    source_name: str
    day: str
    video_id: str
    start_sec: float
    end_sec: float
    text: str
    video_path: str = ""
    transcript_path: str = ""
    closest_keyframe_path: str | None = None
    youtube_url: str | None = None
    youtube_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoRecord:
    area: str
    source_name: str
    day: str
    video_id: str
    video_path: Path
    transcript_path: Path
    keyframes_dir: Path | None
    fps: float | None
    youtube_url: str | None = None
    youtube_id: str | None = None


@dataclass(frozen=True)
class KeyframeRecord:
    area: str
    source_name: str
    day: str
    video_id: str
    keyframe_path: Path
    keyframe_time_sec: float
    frame_number: int | None = None
    youtube_url: str | None = None
    youtube_id: str | None = None


def find_video_folders(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root).expanduser().resolve()
    return sorted(path.parent for path in root.rglob("video.mp4") if path.is_file())


def find_keyframe_folders(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root).expanduser().resolve()
    if root.name == "keyframes":
        return [root] if root.is_dir() else []
    if (root / "keyframes").is_dir():
        return [root / "keyframes"]
    return sorted(path for path in root.rglob("keyframes") if path.is_dir())


def find_transcript_json(video_dir: str | Path) -> Path:
    candidates: list[Path] = []
    for path in sorted(Path(video_dir).glob("*.json")):
        if path.name in TRANSCRIPT_SKIP_NAMES:
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and isinstance(data.get("chunks"), list):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No transcript JSON with a chunks list found in {video_dir}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"Expected one transcript JSON in {video_dir}, found: {names}")
    return candidates[0]


def discover_videos(dataset_root: str | Path) -> list[VideoRecord]:
    root = Path(dataset_root).expanduser().resolve()
    records: list[VideoRecord] = []

    for video_dir in find_video_folders(root):
        transcript_path = find_transcript_json(video_dir)
        manifest = _load_optional_json(video_dir / "manifest.json")
        fps = _get_fps(video_dir, manifest)

        rel_parts = video_dir.relative_to(root).parts
        area = rel_parts[0] if rel_parts else ""
        day, source_name, video_id = _parse_video_identity(video_dir.name, transcript_path.stem, manifest)
        youtube_url, youtube_id = _extract_youtube_metadata(manifest)

        keyframes_dir = video_dir / "keyframes"
        records.append(
            VideoRecord(
                area=area,
                source_name=source_name,
                day=day,
                video_id=video_id,
                video_path=video_dir / "video.mp4",
                transcript_path=transcript_path,
                keyframes_dir=keyframes_dir if keyframes_dir.is_dir() else None,
                fps=fps,
                youtube_url=youtube_url,
                youtube_id=youtube_id,
            )
        )

    return records


def discover_keyframe_sessions(dataset_root: str | Path) -> list[KeyframeRecord]:
    root = Path(dataset_root).expanduser().resolve()
    records: list[KeyframeRecord] = []

    for keyframes_dir in find_keyframe_folders(root):
        session_dir = keyframes_dir.parent
        manifest = _load_optional_json(session_dir / "manifest.json")
        fps = _get_fps(session_dir, manifest)
        day, source_name, video_id = _parse_keyframe_identity(session_dir, root, manifest)
        youtube_url, youtube_id = _extract_youtube_metadata(manifest)

        try:
            rel_parts = session_dir.relative_to(root).parts
        except ValueError:
            rel_parts = ()
        area = rel_parts[0] if rel_parts else day
        if area in {"", source_name, video_id}:
            area = day

        keyframe_index = _manifest_keyframe_index(manifest)
        for path in sorted(keyframes_dir.glob("*.jpg")):
            frame_number = _parse_frame_number(path)
            keyframe_time_sec = _keyframe_time_sec(path, frame_number, fps, keyframe_index)
            records.append(
                KeyframeRecord(
                    area=area,
                    source_name=source_name,
                    day=day,
                    video_id=video_id,
                    keyframe_path=path,
                    keyframe_time_sec=keyframe_time_sec,
                    frame_number=frame_number,
                    youtube_url=youtube_url,
                    youtube_id=youtube_id,
                )
            )

    return records


def load_evidence_items(dataset_root: str | Path) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for record in discover_videos(dataset_root):
        keyframes = _read_keyframes(record.keyframes_dir, record.fps)
        with record.transcript_path.open("r", encoding="utf-8") as f:
            transcript = json.load(f)

        for idx, chunk in enumerate(transcript.get("chunks", [])):
            item = _chunk_to_evidence(record, chunk, idx, keyframes)
            if item is not None:
                evidence.append(item)
    return evidence


def load_keyframe_records(dataset_root: str | Path) -> list[KeyframeRecord]:
    records: list[KeyframeRecord] = []
    seen_paths: set[str] = set()
    for video in discover_videos(dataset_root):
        for keyframe_time_sec, keyframe_path in _read_keyframes(video.keyframes_dir, video.fps):
            frame_number = _parse_frame_number(keyframe_path)
            records.append(
                KeyframeRecord(
                    area=video.area,
                    source_name=video.source_name,
                    day=video.day,
                    video_id=video.video_id,
                    keyframe_path=keyframe_path,
                    keyframe_time_sec=keyframe_time_sec,
                    frame_number=frame_number,
                    youtube_url=video.youtube_url,
                    youtube_id=video.youtube_id,
                )
            )
            seen_paths.add(str(keyframe_path))
    for record in discover_keyframe_sessions(dataset_root):
        if str(record.keyframe_path) not in seen_paths:
            records.append(record)
            seen_paths.add(str(record.keyframe_path))
    return records


def _chunk_to_evidence(
    record: VideoRecord,
    chunk: dict[str, Any],
    chunk_idx: int,
    keyframes: list[tuple[float, Path]],
) -> EvidenceItem | None:
    timestamp = chunk.get("timestamp")
    text = str(chunk.get("text", "")).strip()
    if not text or not isinstance(timestamp, list) or len(timestamp) != 2:
        return None

    start_sec = _safe_float(timestamp[0])
    end_sec = _safe_float(timestamp[1])
    if start_sec is None or end_sec is None:
        return None
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec

    midpoint = (start_sec + end_sec) / 2.0
    keyframe_path = _closest_keyframe(midpoint, keyframes)
    source_id = f"{record.area}/{record.day}/{record.source_name}/{record.video_id}#{chunk_idx:05d}"

    return EvidenceItem(
        source_id=source_id,
        area=record.area,
        source_name=record.source_name,
        day=record.day,
        video_id=record.video_id,
        start_sec=start_sec,
        end_sec=end_sec,
        text=text,
        video_path=str(record.video_path),
        transcript_path=str(record.transcript_path),
        closest_keyframe_path=str(keyframe_path) if keyframe_path else None,
        youtube_url=record.youtube_url,
        youtube_id=record.youtube_id,
    )


def _load_optional_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _get_fps(video_dir: Path, manifest: dict[str, Any]) -> float | None:
    fps = _safe_float(manifest.get("fps"))
    if fps:
        return fps
    shots = _load_optional_json(video_dir / "shots.json")
    return _safe_float(shots.get("fps"))


def _parse_keyframe_identity(
    session_dir: Path,
    dataset_root: Path,
    manifest: dict[str, Any],
) -> tuple[str, str, str]:
    day = str(manifest.get("day") or "")
    source_name = str(manifest.get("actor") or manifest.get("source_name") or manifest.get("pov") or "")
    video_id = str(manifest.get("video_stem") or manifest.get("video_id") or manifest.get("hour_id") or "")

    try:
        rel_parts = session_dir.relative_to(dataset_root).parts
    except ValueError:
        rel_parts = ()

    parts = rel_parts or session_dir.parts
    if not video_id and len(parts) >= 1:
        video_id = parts[-1]
    if not source_name and len(parts) >= 2:
        source_name = parts[-2]
    if not day and len(parts) >= 3:
        day = parts[-3]

    return day, source_name, video_id


def _parse_video_identity(
    folder_name: str,
    transcript_stem: str,
    manifest: dict[str, Any],
) -> tuple[str, str, str]:
    day = str(manifest.get("day") or "")
    source_name = str(manifest.get("actor") or "")
    video_id = str(manifest.get("video_stem") or "")

    match = re.match(r"^(day\d+)_(.+)_(\d+)$", transcript_stem, flags=re.IGNORECASE)
    if match:
        day = day or match.group(1)
        source_name = source_name or match.group(2)
        video_id = video_id or match.group(3)

    if not (day and source_name and video_id):
        folder_match = re.match(r"^(.+?)_(day\d+)_(\d+)$", folder_name, flags=re.IGNORECASE)
        if folder_match:
            source_name = source_name or folder_match.group(1)
            day = day or folder_match.group(2)
            video_id = video_id or folder_match.group(3)

    return day, source_name, video_id


def _read_keyframes(keyframes_dir: Path | None, fps: float | None) -> list[tuple[float, Path]]:
    if keyframes_dir is None or not fps:
        return []

    keyframes: list[tuple[float, Path]] = []
    for path in sorted(keyframes_dir.glob("*.jpg")):
        frame_number = _parse_frame_number(path)
        if frame_number is None:
            continue
        keyframes.append((frame_number / fps, path))
    return keyframes


def _closest_keyframe(target_sec: float, keyframes: Iterable[tuple[float, Path]]) -> Path | None:
    closest = min(keyframes, key=lambda item: abs(item[0] - target_sec), default=None)
    return closest[1] if closest else None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_frame_number(path: Path) -> int | None:
    match = re.search(r"_frame_(\d+)", path.stem)
    return int(match.group(1)) if match else None


def _keyframe_time_sec(
    path: Path,
    frame_number: int | None,
    fps: float | None,
    keyframe_index: dict[str, float],
) -> float:
    if path.name in keyframe_index:
        return keyframe_index[path.name]
    if frame_number is not None and fps:
        return frame_number / fps
    return 0.0


def _manifest_keyframe_index(manifest: dict[str, Any]) -> dict[str, float]:
    fps = _safe_float(manifest.get("fps"))
    index: dict[str, float] = {}
    keyframes_indexed = manifest.get("keyframes_indexed")
    if not isinstance(keyframes_indexed, dict):
        return index

    for entries in keyframes_indexed.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            filename = str(entry.get("filename") or Path(str(entry.get("path", ""))).name)
            if not filename:
                continue
            time_sec = _safe_float(entry.get("time_sec") or entry.get("timestamp_sec") or entry.get("start_sec"))
            frame = _safe_float(entry.get("frame"))
            if time_sec is None and frame is not None and fps:
                time_sec = frame / fps
            if time_sec is not None:
                index[filename] = time_sec
    return index


def _extract_youtube_metadata(manifest: dict[str, Any]) -> tuple[str | None, str | None]:
    url_keys = ("youtube_url", "youtube", "video_url", "url", "source_url")
    id_keys = ("youtube_id", "youtube_video_id", "yt_id")

    youtube_url = _first_string_value(manifest, url_keys)
    youtube_id = _first_string_value(manifest, id_keys)

    if youtube_url and not youtube_id:
        youtube_id = _youtube_id_from_url(youtube_url)
    if youtube_id and not youtube_url:
        youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"
    return youtube_url, youtube_id


def _first_string_value(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in data.values():
        if isinstance(value, dict):
            found = _first_string_value(value, keys)
            if found:
                return found
    return None


def _youtube_id_from_url(url: str) -> str | None:
    patterns = [
        r"[?&]v=([^&#]+)",
        r"youtu\.be/([^?&#/]+)",
        r"youtube\.com/embed/([^?&#/]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None
