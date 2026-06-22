#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_multimodal_index import load_youtube_map  # noqa: E402
from src.clip_retrieval import DEFAULT_SIGLIP_MODEL, embed_images, keyframe_metadata, load_clip_model  # noqa: E402
from src.dataset_loader import load_keyframe_records  # noqa: E402
from src.retriever import build_faiss_index, save_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SigLIP/CLIP FAISS image index over CASTLE keyframes.")
    parser.add_argument("--dataset-root", default="day1")
    parser.add_argument("--output-dir", default="artifacts/siglip_index_day1")
    parser.add_argument("--model-name", default=DEFAULT_SIGLIP_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Maximum keyframes to index. Use 0 for all.")
    parser.add_argument("--youtube-map", default="artifacts/youtube_video_map.json")
    parser.add_argument("--allow-download", action="store_true", help="Allow model download if not cached.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore partial embeddings and rebuild.")
    args = parser.parse_args()

    start = time.perf_counter()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "build_progress.json"
    embeddings_path = output / "embeddings.npy"

    keyframes = load_keyframe_records(args.dataset_root)
    if args.limit and args.limit > 0:
        keyframes = keyframes[: args.limit]
    if not keyframes:
        raise SystemExit(f"No keyframes found under {args.dataset_root}")

    youtube_map = load_youtube_map(args.youtube_map)
    model, processor, torch = load_clip_model(args.model_name, local_files_only=not args.allow_download)

    metadata = [keyframe_metadata(record, youtube_url=lookup_youtube_url(record, youtube_map)) for record in keyframes]
    completed_count = completed_from_progress(
        progress_path=progress_path,
        embeddings_path=embeddings_path,
        args=args,
        total_keyframes=len(keyframes),
    ) if args.resume else 0
    if completed_count:
        print(f"Resuming from {completed_count}/{len(keyframes)} completed keyframes")

    embeddings_array = None
    embedding_dim = None
    for batch_start in range(completed_count, len(keyframes), args.batch_size):
        batch = keyframes[batch_start : batch_start + args.batch_size]
        batch_paths = [record.keyframe_path for record in batch]
        batch_embeddings = embed_images(batch_paths, model=model, processor=processor, torch=torch)
        if embeddings_array is None:
            embedding_dim = int(batch_embeddings.shape[1])
            embeddings_array = open_or_create_embeddings(
                embeddings_path=embeddings_path,
                total_keyframes=len(keyframes),
                embedding_dim=embedding_dim,
                resume=args.resume,
            )
        batch_end = batch_start + len(batch)
        embeddings_array[batch_start:batch_end] = batch_embeddings
        embeddings_array.flush()
        write_progress(
            progress_path=progress_path,
            args=args,
            total_keyframes=len(keyframes),
            completed_count=batch_end,
            embedding_dim=embedding_dim,
            elapsed_sec=time.perf_counter() - start,
        )
        print(f"Embedded {batch_end}/{len(keyframes)} keyframes", flush=True)

    import numpy as np

    if embeddings_array is None:
        embeddings_array = np.load(embeddings_path, mmap_mode="r")
    all_embeddings = np.asarray(embeddings_array[: len(keyframes)], dtype="float32")
    index = build_faiss_index(all_embeddings)
    save_index(index, metadata, args.output_dir, args.model_name)

    elapsed = time.perf_counter() - start
    summary = {
        "dataset_root": args.dataset_root,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
        "keyframes": len(metadata),
        "embeddings_path": str(embeddings_path),
        "elapsed_sec": elapsed,
        "keyframes_per_sec": len(metadata) / elapsed if elapsed > 0 else 0.0,
        "resumed_from_keyframes": completed_count,
        "batch_size": args.batch_size,
    }
    with (output / "build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_progress(
        progress_path=progress_path,
        args=args,
        total_keyframes=len(keyframes),
        completed_count=len(keyframes),
        embedding_dim=int(all_embeddings.shape[1]),
        elapsed_sec=elapsed,
        complete=True,
    )
    print(json.dumps(summary, indent=2))


def lookup_youtube_url(record, youtube_map: dict[str, str]) -> str | None:
    keys = [f"{record.day}/{record.source_name}/{record.video_id}"]
    if str(record.video_id).isdigit():
        keys.append(f"{record.day}/{record.source_name}/{int(record.video_id)}")
    for key in keys:
        if youtube_map.get(key):
            return youtube_map[key]
    return record.youtube_url


def completed_from_progress(
    *,
    progress_path: Path,
    embeddings_path: Path,
    args: argparse.Namespace,
    total_keyframes: int,
) -> int:
    if not progress_path.is_file() or not embeddings_path.is_file():
        return 0
    try:
        with progress_path.open("r", encoding="utf-8") as f:
            progress = json.load(f)
        completed = int(progress.get("completed_count", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0

    if progress.get("dataset_root") != args.dataset_root:
        return 0
    if progress.get("model_name") != args.model_name:
        return 0
    if int(progress.get("total_keyframes", -1)) != total_keyframes:
        return 0
    return max(0, min(completed, total_keyframes))


def open_or_create_embeddings(
    *,
    embeddings_path: Path,
    total_keyframes: int,
    embedding_dim: int,
    resume: bool,
):
    import numpy as np

    shape = (total_keyframes, embedding_dim)
    if resume and embeddings_path.is_file():
        existing = np.load(embeddings_path, mmap_mode="r+")
        if existing.shape == shape:
            return existing
    return np.lib.format.open_memmap(embeddings_path, mode="w+", dtype="float32", shape=shape)


def write_progress(
    *,
    progress_path: Path,
    args: argparse.Namespace,
    total_keyframes: int,
    completed_count: int,
    embedding_dim: int | None,
    elapsed_sec: float,
    complete: bool = False,
) -> None:
    progress: dict[str, Any] = {
        "dataset_root": args.dataset_root,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "total_keyframes": total_keyframes,
        "completed_count": completed_count,
        "embedding_dim": embedding_dim,
        "elapsed_sec": elapsed_sec,
        "keyframes_per_sec": completed_count / elapsed_sec if elapsed_sec > 0 else 0.0,
        "complete": complete,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


if __name__ == "__main__":
    main()
