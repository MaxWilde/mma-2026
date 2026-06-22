#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LEGACY_SOURCE_NAMES = {
    "bjorn": "Bjorn",
    "klaus": "Klaus",
    "stevan": "Stevan",
    "kitchen": "Kitchen",
    "living1": "Living1",
    "living2": "Living2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy caption JSONL paths to canonical day1 keyframe paths.")
    parser.add_argument("--input", required=True, help="Legacy caption JSONL file.")
    parser.add_argument("--output", required=True, help="Migrated canonical day1 caption JSONL file.")
    parser.add_argument("--day1-root", default="day1")
    parser.add_argument("--youtube-map", default="artifacts/youtube_video_map.json")
    parser.add_argument("--strict", action="store_true", help="Fail if any record cannot be mapped to an existing day1 keyframe.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    day1_root = Path(args.day1_root)
    youtube_map = load_youtube_map(args.youtube_map)

    stats = {
        "total": 0,
        "migrated": 0,
        "already_canonical": 0,
        "missing_target": 0,
        "unparseable": 0,
        "youtube_injected": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {input_path}") from exc

            migrated, status = migrate_record(record, day1_root, youtube_map)
            stats[status] += 1
            if migrated.get("youtube_url") and not record.get("youtube_url"):
                stats["youtube_injected"] += 1

            if status in {"missing_target", "unparseable"} and args.strict:
                raise SystemExit(
                    f"Failed to migrate line {line_number} ({status}): {record.get('keyframe_path')}"
                )
            if status in {"missing_target", "unparseable"}:
                continue
            dst.write(json.dumps(migrated, ensure_ascii=False) + "\n")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    for key, value in stats.items():
        print(f"{key}: {value}")


def migrate_record(record: dict[str, Any], day1_root: Path, youtube_map: dict[str, str]) -> tuple[dict[str, Any], str]:
    parsed = parse_caption_identity(record)
    if parsed is None:
        return record, "unparseable"

    day, source_name, video_id, filename = parsed
    canonical_path = day1_root / source_name / video_id / "keyframes" / filename
    if not canonical_path.is_file():
        return record, "missing_target"

    migrated = dict(record)
    migrated["area"] = day
    migrated["day"] = day
    migrated["source_name"] = source_name
    migrated["video_id"] = video_id
    migrated["keyframe_path"] = str(canonical_path)

    frame_number = migrated.get("frame_number")
    if frame_number is None:
        frame_number = parse_frame_number(filename)
        if frame_number is not None:
            migrated["frame_number"] = frame_number

    if migrated.get("keyframe_time_sec") is None and frame_number is not None:
        migrated["keyframe_time_sec"] = frame_number / 50.0

    youtube_url = lookup_youtube_url(day, source_name, video_id, youtube_map)
    if youtube_url:
        migrated["youtube_url"] = youtube_url

    original_path = str(record.get("keyframe_path", ""))
    status = "already_canonical" if "/day1/" in original_path or original_path.startswith("day1/") else "migrated"
    return migrated, status


def parse_caption_identity(record: dict[str, Any]) -> tuple[str, str, str, str] | None:
    keyframe_path = str(record.get("keyframe_path") or "")
    if not keyframe_path:
        return None

    path = Path(keyframe_path)
    filename = path.name
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == "day1" and idx + 3 < len(parts):
            source_name = canonical_source_name(parts[idx + 1])
            video_id = canonical_hour(parts[idx + 2])
            if parts[idx + 3] == "keyframes":
                return "day1", source_name, video_id, filename

    legacy = parse_legacy_session(path)
    if legacy is not None:
        source_name, day, video_id = legacy
        return day, source_name, video_id, filename

    day = canonical_day(str(record.get("day") or "day1"))
    source_name = canonical_source_name(str(record.get("source_name") or ""))
    video_id = canonical_hour(str(record.get("video_id") or ""))
    if source_name and video_id:
        return day, source_name, video_id, filename
    return None


def parse_legacy_session(path: Path) -> tuple[str, str, str] | None:
    for part in path.parts:
        match = re.match(r"^([A-Za-z0-9]+)_day(\d+)_(\d+)$", part)
        if not match:
            continue
        source_name = canonical_source_name(match.group(1))
        day = f"day{int(match.group(2))}"
        video_id = canonical_hour(match.group(3))
        return source_name, day, video_id
    return None


def canonical_source_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    return LEGACY_SOURCE_NAMES.get(normalized, value[:1].upper() + value[1:])


def canonical_day(value: str) -> str:
    match = re.search(r"(\d+)", value)
    return f"day{int(match.group(1))}" if match else value


def canonical_hour(value: str) -> str:
    match = re.search(r"\d+", value)
    return f"{int(match.group(0)):02d}" if match else value


def parse_frame_number(filename: str) -> int | None:
    match = re.search(r"_frame_(\d+)", Path(filename).stem)
    return int(match.group(1)) if match else None


def load_youtube_map(path: str | Path) -> dict[str, str]:
    map_path = Path(path)
    if not map_path.exists():
        return {}
    with map_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YouTube map must be a JSON object: {map_path}")
    return {str(key): str(value) for key, value in data.items() if str(value).strip()}


def lookup_youtube_url(day: str, source_name: str, video_id: str, youtube_map: dict[str, str]) -> str | None:
    keys = [f"{day}/{source_name}/{video_id}"]
    if video_id.isdigit():
        keys.append(f"{day}/{source_name}/{int(video_id)}")
    for key in keys:
        if youtube_map.get(key):
            return youtube_map[key]
    return None


if __name__ == "__main__":
    main()
