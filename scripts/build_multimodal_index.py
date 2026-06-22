#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_loader import EvidenceItem, load_evidence_items
from src.retriever import DEFAULT_MODEL_NAME, build_faiss_index, embed_texts, load_embedding_model, save_index


CaptionBySource = dict[tuple[str, str, str, str], list[dict[str, Any]]]


@dataclass(frozen=True)
class MatchStats:
    total_chunks: int
    chunks_with_captions: int
    chunks_without_captions: int
    average_caption_distance: float | None
    max_caption_distance: float | None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index over transcript chunks plus Qwen captions.")
    parser.add_argument("--dataset-root", default=str(ROOT))
    parser.add_argument("--captions", default="artifacts/qwen_keyframe_captions.jsonl")
    parser.add_argument("--output-dir", default="artifacts/multimodal_index")
    parser.add_argument(
        "--youtube-map",
        default="artifacts/youtube_video_map.json",
        help="JSON map from day/source/hour_id, e.g. day1/Bjorn/12, to YouTube URL.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-chunks", type=int, default=0, help="Maximum transcript chunks to index. Use 0 for all.")
    parser.add_argument(
        "--max-caption-distance-sec",
        type=float,
        default=60.0,
        help="Only attach the nearest caption if it is within this many seconds of the transcript chunk midpoint.",
    )
    args = parser.parse_args()

    captions_by_source = load_captions(args.captions)
    if not captions_by_source:
        raise SystemExit(f"No caption records found in {args.captions}")
    youtube_map = load_youtube_map(args.youtube_map)

    items = load_evidence_items(args.dataset_root)
    if args.limit_chunks and args.limit_chunks > 0:
        items = items[: args.limit_chunks]
    if items:
        metadata, index_texts, stats = build_multimodal_metadata(
            items,
            captions_by_source,
            youtube_map,
            max_caption_distance_sec=args.max_caption_distance_sec,
        )
    else:
        metadata, index_texts, stats = build_visual_caption_metadata(captions_by_source, youtube_map)
    model = load_embedding_model(args.model_name)
    embeddings = embed_texts(model, index_texts, batch_size=args.batch_size)
    index = build_faiss_index(embeddings)
    save_index(index, metadata, args.output_dir, args.model_name)

    print(f"Total chunks: {stats.total_chunks}")
    print(f"Chunks with captions: {stats.chunks_with_captions}")
    print(f"Chunks without captions: {stats.chunks_without_captions}")
    print(f"Average caption distance: {_format_optional_seconds(stats.average_caption_distance)}")
    print(f"Max caption distance: {_format_optional_seconds(stats.max_caption_distance)}")
    print(f"Saved FAISS index and metadata to {args.output_dir}")


def load_captions(captions_path: str | Path) -> CaptionBySource:
    path = Path(captions_path)
    captions_by_source: CaptionBySource = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc

            caption = str(record.get("caption", "")).strip()
            if not caption:
                continue
            if record.get("is_test_pattern") is True or is_test_pattern_caption(caption):
                continue
            key = (
                str(record.get("area", "")),
                str(record.get("day", "")),
                str(record.get("source_name", "")),
                str(record.get("video_id", "")),
            )
            record["keyframe_time_sec"] = float(record["keyframe_time_sec"])
            captions_by_source[key].append(record)

    for records in captions_by_source.values():
        records.sort(key=lambda item: float(item["keyframe_time_sec"]))
    return dict(captions_by_source)


def load_youtube_map(map_path: str | Path) -> dict[str, str]:
    path = Path(map_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YouTube map must be a JSON object: {path}")
    return {str(key): str(value) for key, value in data.items() if str(value).strip()}


def build_multimodal_metadata(
    items: list[EvidenceItem],
    captions_by_source: CaptionBySource,
    youtube_map: dict[str, str] | None = None,
    max_caption_distance_sec: float = 60.0,
) -> tuple[list[dict[str, Any]], list[str], MatchStats]:
    metadata: list[dict[str, Any]] = []
    index_texts: list[str] = []
    caption_distances: list[float] = []

    for item in items:
        item_metadata = item.to_dict()
        _apply_youtube_map(item_metadata, youtube_map or {})
        caption, caption_distance = nearest_caption(item, captions_by_source)
        visual_caption = ""
        if caption and caption_distance <= max_caption_distance_sec:
            visual_caption = str(caption["caption"]).strip()
            item_metadata["visual_caption"] = visual_caption
            item_metadata["visual_caption_keyframe_path"] = caption.get("keyframe_path")
            item_metadata["visual_caption_time_sec"] = caption.get("keyframe_time_sec")
            item_metadata["visual_caption_distance_sec"] = caption_distance
            item_metadata["visual_caption_frame_number"] = caption.get("frame_number")
            _copy_optional_caption_fields(item_metadata, caption)
            _apply_youtube_map(item_metadata, youtube_map or {})
            caption_distances.append(caption_distance)
        else:
            item_metadata["visual_caption"] = ""
            item_metadata["visual_caption_keyframe_path"] = None
            item_metadata["visual_caption_time_sec"] = None
            item_metadata["visual_caption_distance_sec"] = None
            item_metadata["visual_caption_frame_number"] = None

        combined_text = f"Transcript: {item.text}\nVisual caption: {visual_caption}"
        item_metadata["indexed_text"] = combined_text
        metadata.append(item_metadata)
        index_texts.append(combined_text)

    chunks_with_captions = len(caption_distances)
    stats = MatchStats(
        total_chunks=len(items),
        chunks_with_captions=chunks_with_captions,
        chunks_without_captions=len(items) - chunks_with_captions,
        average_caption_distance=sum(caption_distances) / chunks_with_captions if chunks_with_captions else None,
        max_caption_distance=max(caption_distances) if caption_distances else None,
    )
    return metadata, index_texts, stats


def build_visual_caption_metadata(
    captions_by_source: CaptionBySource,
    youtube_map: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], MatchStats]:
    metadata: list[dict[str, Any]] = []
    index_texts: list[str] = []

    for key, captions in captions_by_source.items():
        area, day, source_name, video_id = key
        for idx, caption in enumerate(captions):
            visual_caption = str(caption.get("caption", "")).strip()
            keyframe_time_sec = float(caption.get("keyframe_time_sec") or 0.0)
            keyframe_path = caption.get("keyframe_path")
            frame_number = caption.get("frame_number")
            item_metadata: dict[str, Any] = {
                "source_id": f"{area}/{day}/{source_name}/{video_id}#keyframe-{idx:06d}",
                "area": area,
                "source_name": source_name,
                "day": day,
                "video_id": video_id,
                "start_sec": keyframe_time_sec,
                "end_sec": keyframe_time_sec,
                "text": "",
                "video_path": "",
                "transcript_path": "",
                "closest_keyframe_path": keyframe_path,
                "keyframe_path": keyframe_path,
                "keyframe_time_sec": keyframe_time_sec,
                "frame_number": frame_number,
                "visual_caption": visual_caption,
                "visual_caption_keyframe_path": keyframe_path,
                "visual_caption_time_sec": keyframe_time_sec,
                "visual_caption_frame_number": frame_number,
                "visual_caption_distance_sec": 0.0,
            }
            _copy_optional_caption_fields(item_metadata, caption)
            _apply_youtube_map(item_metadata, youtube_map or {})
            combined_text = f"Visual caption: {visual_caption}"
            item_metadata["indexed_text"] = combined_text
            metadata.append(item_metadata)
            index_texts.append(combined_text)

    stats = MatchStats(
        total_chunks=len(metadata),
        chunks_with_captions=len(metadata),
        chunks_without_captions=0,
        average_caption_distance=0.0 if metadata else None,
        max_caption_distance=0.0 if metadata else None,
    )
    return metadata, index_texts, stats


def nearest_caption(item: EvidenceItem, captions_by_source: CaptionBySource) -> tuple[dict[str, Any] | None, float]:
    key = (item.area, item.day, item.source_name, item.video_id)
    captions = captions_by_source.get(key, [])
    if not captions:
        return None, float("inf")

    midpoint = (item.start_sec + item.end_sec) / 2.0
    caption = min(captions, key=lambda item_caption: abs(float(item_caption["keyframe_time_sec"]) - midpoint))
    distance = abs(float(caption["keyframe_time_sec"]) - midpoint)
    return caption, distance


def is_test_pattern_caption(caption: str) -> bool:
    normalized = caption.lower()
    return "castle 2024 dataset" in normalized or "test pattern" in normalized


def _format_optional_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


def _copy_optional_caption_fields(metadata: dict[str, Any], caption: dict[str, Any]) -> None:
    for field in ("youtube_url", "youtube_id"):
        if caption.get(field):
            metadata[field] = caption.get(field)
    if metadata.get("youtube_id") and not metadata.get("youtube_url"):
        metadata["youtube_url"] = f"https://www.youtube.com/watch?v={metadata['youtube_id']}"


def _apply_youtube_map(metadata: dict[str, Any], youtube_map: dict[str, str]) -> None:
    if metadata.get("youtube_url") or not youtube_map:
        return
    for key in _youtube_lookup_keys(metadata):
        youtube_url = youtube_map.get(key)
        if youtube_url:
            metadata["youtube_url"] = youtube_url
            return


def _youtube_lookup_keys(metadata: dict[str, Any]) -> list[str]:
    day = str(metadata.get("day", "")).strip()
    source_name = str(metadata.get("source_name", "")).strip()
    video_id = str(metadata.get("video_id", "")).strip()
    if not day or not source_name or not video_id:
        return []

    keys = [f"{day}/{source_name}/{video_id}"]
    if video_id.isdigit():
        normalized_hour = str(int(video_id))
        padded_hour = f"{int(video_id):02d}"
        for hour in (normalized_hour, padded_hour):
            key = f"{day}/{source_name}/{hour}"
            if key not in keys:
                keys.append(key)
    return keys


if __name__ == "__main__":
    main()
