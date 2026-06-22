#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_loader import KeyframeRecord, load_keyframe_records
from src.vision_qa import DEFAULT_QWEN_MODEL, _get_model_input_device, _load_qwen_model


CAPTION_PROMPT = (
    "Describe the visible scene in this keyframe for multimodal evidence retrieval. "
    "Mention people, objects, food, activities, location cues, and anything being handled. "
    "Be concise and factual. Do not infer beyond what is visible."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline Qwen-VL captions for CASTLE keyframes.")
    parser.add_argument("--dataset-root", default=str(ROOT))
    parser.add_argument("--output", default="artifacts/qwen_keyframe_captions.jsonl")
    parser.add_argument("--limit", type=int, default=5, help="Maximum keyframes to caption. Use 0 for all keyframes.")
    parser.add_argument("--sampling-mode", choices=["first", "uniform"], default="uniform")
    parser.add_argument("--model-name", default=None, help="Qwen model path/name. Defaults to QWEN_VL_MODEL.")
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("QWEN_VL_MAX_NEW_TOKENS", "96")))
    args = parser.parse_args()

    model_id = args.model_name or os.environ.get("QWEN_VL_MODEL")
    if not model_id:
        model_id = DEFAULT_QWEN_MODEL
        print(
            f"QWEN_VL_MODEL is not set; trying default local model id: {model_id}",
            file=sys.stderr,
        )

    keyframes = load_caption_keyframes(args.dataset_root)
    keyframes = select_keyframes(keyframes, args.limit, args.sampling_mode)
    if not keyframes:
        raise SystemExit("No keyframes found to caption.")

    try:
        model, processor, torch, process_vision_info = _load_qwen_model(model_id)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Captioning {len(keyframes)} keyframes with {model_id}")
    print(f"Writing JSONL to {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        for idx, keyframe in enumerate(keyframes, start=1):
            caption = generate_caption(
                keyframe.keyframe_path,
                model=model,
                processor=processor,
                torch=torch,
                process_vision_info=process_vision_info,
                max_new_tokens=args.max_new_tokens,
            )
            record = keyframe_to_json(keyframe)
            record["caption"] = caption
            record["is_test_pattern"] = is_test_pattern_caption(caption)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{idx}/{len(keyframes)}] {keyframe.keyframe_path} -> {caption}")


def load_caption_keyframes(dataset_root: str | Path) -> list[KeyframeRecord]:
    old_records = load_keyframe_records(dataset_root)
    seen_paths = {str(record.keyframe_path) for record in old_records}

    extra_records: list[KeyframeRecord] = []
    for keyframe_path in discover_keyframe_paths(dataset_root):
        path_key = str(keyframe_path)
        if path_key in seen_paths:
            continue
        extra_records.append(keyframe_record_from_path(keyframe_path, dataset_root))
        seen_paths.add(path_key)

    return [*old_records, *extra_records]


def discover_keyframe_paths(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root).expanduser().resolve()
    if root.name == "keyframes":
        return sorted(root.glob("*.jpg"))
    if (root / "keyframes").is_dir():
        return sorted((root / "keyframes").glob("*.jpg"))
    return sorted(root.rglob("keyframes/*.jpg"))


def keyframe_record_from_path(keyframe_path: Path, dataset_root: str | Path) -> KeyframeRecord:
    keyframes_dir = keyframe_path.parent
    session_dir = keyframes_dir.parent
    manifest = load_optional_json(session_dir / "manifest.json")
    shots = load_optional_json(session_dir / "shots.json")
    fps = safe_float(manifest.get("fps")) or safe_float(shots.get("fps")) or 50.0

    day, source_name, video_id = parse_keyframe_identity(session_dir, dataset_root, manifest)
    frame_number = parse_frame_number(keyframe_path)
    keyframe_time_sec = frame_number / fps if frame_number is not None and fps else 0.0

    root = Path(dataset_root).expanduser().resolve()
    try:
        rel_parts = session_dir.relative_to(root).parts
        area = rel_parts[0] if rel_parts else day
    except ValueError:
        area = day
    if area in {"", source_name, video_id}:
        area = day

    return KeyframeRecord(
        area=area,
        source_name=source_name,
        day=day,
        video_id=video_id,
        keyframe_path=keyframe_path,
        keyframe_time_sec=keyframe_time_sec,
    )


def parse_keyframe_identity(session_dir: Path, dataset_root: str | Path, manifest: dict[str, Any]) -> tuple[str, str, str]:
    day = str(manifest.get("day") or "")
    source_name = str(manifest.get("actor") or "")
    video_id = str(manifest.get("video_stem") or "")

    parts = session_dir.parts
    if not video_id:
        video_id = session_dir.name
    if not source_name and len(parts) >= 2:
        source_name = parts[-2]
    if not day and len(parts) >= 3:
        day = parts[-3]

    root = Path(dataset_root).expanduser().resolve()
    try:
        rel_parts = session_dir.relative_to(root).parts
    except ValueError:
        rel_parts = ()
    if not day and len(rel_parts) >= 3:
        day = rel_parts[-3]
    if not source_name and len(rel_parts) >= 2:
        source_name = rel_parts[-2]
    if not video_id and rel_parts:
        video_id = rel_parts[-1]

    return day, source_name, video_id


def parse_frame_number(keyframe_path: Path) -> int | None:
    match = re.search(r"_frame_(\d+)", keyframe_path.stem)
    return int(match.group(1)) if match else None


def load_optional_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def select_keyframes(
    keyframes: list[KeyframeRecord],
    limit: int,
    sampling_mode: str,
) -> list[KeyframeRecord]:
    if not limit or limit <= 0 or limit >= len(keyframes):
        return keyframes
    if sampling_mode == "first":
        return keyframes[:limit]
    if sampling_mode != "uniform":
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")
    if limit == 1:
        return [keyframes[0]]

    last_index = len(keyframes) - 1
    selected_indices = {
        round(position * last_index / (limit - 1))
        for position in range(limit)
    }
    return [keyframes[index] for index in sorted(selected_indices)]


def generate_caption(
    image_path: Path,
    *,
    model: Any,
    processor: Any,
    torch: Any,
    process_vision_info: Any,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.resolve().as_uri()},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_get_model_input_device(model, torch))

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    answers = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return answers[0].strip() if answers else ""


def is_test_pattern_caption(caption: str) -> bool:
    normalized = caption.lower()
    return "castle 2024 dataset" in normalized or "test pattern" in normalized


def keyframe_to_json(keyframe: KeyframeRecord) -> dict[str, Any]:
    return {
        "area": keyframe.area,
        "source_name": keyframe.source_name,
        "day": keyframe.day,
        "video_id": keyframe.video_id,
        "keyframe_path": str(keyframe.keyframe_path),
        "keyframe_time_sec": keyframe.keyframe_time_sec,
        "frame_number": keyframe.frame_number,
        "youtube_url": keyframe.youtube_url,
        "youtube_id": keyframe.youtube_id,
    }


if __name__ == "__main__":
    main()
