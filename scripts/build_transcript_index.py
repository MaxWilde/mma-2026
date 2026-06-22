#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_multimodal_index import load_youtube_map  # noqa: E402
from src.retriever import DEFAULT_MODEL_NAME, build_faiss_index, save_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index over CASTLE transcript chunks.")
    parser.add_argument("--transcripts-dir", default="all_transcripts")
    parser.add_argument("--output-dir", default="artifacts/transcript_index_day1")
    parser.add_argument("--youtube-map", default="artifacts/youtube_video_map.json")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--day", default="day1", help="Filter transcripts by day. Use empty string for all days.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true", help="Allow SentenceTransformer download if not cached.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    start = time.perf_counter()
    youtube_map = load_youtube_map(args.youtube_map)
    transcript_files = sorted(Path(args.transcripts_dir).glob("*.json"))
    print(f"Transcript files found: {len(transcript_files)}", flush=True)
    metadata = load_transcript_chunks(args.transcripts_dir, youtube_map, day_filter=args.day or None)
    if args.limit and args.limit > 0:
        metadata = metadata[: args.limit]
    if not metadata:
        raise SystemExit(f"No transcript chunks found in {args.transcripts_dir}")
    print(f"Chunks loaded: {len(metadata)}", flush=True)

    print(f"Model name/path: {args.model_name}", flush=True)
    device = select_device(args.device)
    print(f"Selected device: {device}", flush=True)
    model = load_sentence_transformer(args.model_name, local_files_only=not args.allow_download, device=device)
    print("Model loaded", flush=True)
    print("Embedding started", flush=True)
    embeddings = embed_texts_with_progress(model, [item["text"] for item in metadata], batch_size=args.batch_size)
    print("Embedding complete", flush=True)
    print("Building FAISS index", flush=True)
    index = build_faiss_index(embeddings)
    print("Saving index", flush=True)
    save_index(index, metadata, args.output_dir, args.model_name)

    elapsed = time.perf_counter() - start
    summary = {
        "transcripts_dir": args.transcripts_dir,
        "output_dir": args.output_dir,
        "model_name": args.model_name,
        "day": args.day,
        "chunks": len(metadata),
        "device": device,
        "local_files_only": not args.allow_download,
        "batch_size": args.batch_size,
        "elapsed_sec": elapsed,
        "chunks_per_sec": len(metadata) / elapsed if elapsed > 0 else 0.0,
    }
    output = Path(args.output_dir)
    with (output / "build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def select_device(requested: str) -> str:
    if requested in {"cpu", "cuda"}:
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sentence_transformer(model_name: str, *, local_files_only: bool, device: str):
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    print(f"Loading SentenceTransformer(local_files_only={local_files_only}, device={device})", flush=True)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Install sentence-transformers to build the transcript index.") from exc
    try:
        model = SentenceTransformer(model_name, local_files_only=local_files_only, device=device)
    except Exception as exc:
        raise SystemExit(
            f"Failed to load SentenceTransformer model '{model_name}' with local_files_only={local_files_only}. "
            "Use --allow-download in a compute job if the local cache is incomplete."
        ) from exc
    return model


def embed_texts_with_progress(model: Any, texts: list[str], *, batch_size: int):
    import numpy as np

    batches = []
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, total, batch_size), start=1):
        end = min(start + batch_size, total)
        print(f"Embedding batch {batch_idx}/{total_batches}: chunks {start}-{end - 1}", flush=True)
        batch = model.encode(
            texts[start:end],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        batches.append(np.asarray(batch, dtype="float32"))
    return np.vstack(batches).astype("float32")


def load_transcript_chunks(
    transcripts_dir: str | Path,
    youtube_map: dict[str, str],
    *,
    day_filter: str | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for transcript_path in sorted(Path(transcripts_dir).glob("*.json")):
        parsed = parse_transcript_name(transcript_path.stem)
        if parsed is None:
            continue
        day, source_name, hour_id = parsed
        if day_filter and day != day_filter:
            continue
        with transcript_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = data.get("chunks")
        if not isinstance(chunks, list):
            continue
        youtube_url = lookup_youtube_url(day, source_name, hour_id, youtube_map)
        for idx, chunk in enumerate(chunks):
            text = str(chunk.get("text", "")).strip()
            timestamp = chunk.get("timestamp")
            if not text or not isinstance(timestamp, list) or len(timestamp) != 2:
                continue
            start_sec = safe_float(timestamp[0])
            end_sec = safe_float(timestamp[1])
            if start_sec is None or end_sec is None:
                continue
            if end_sec < start_sec:
                start_sec, end_sec = end_sec, start_sec
            records.append(
                {
                    "source_id": f"{day}/{source_name}/{hour_id}#transcript-{idx:06d}",
                    "area": day,
                    "day": day,
                    "source_name": source_name,
                    "video_id": hour_id,
                    "hour_id": hour_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "text": text,
                    "transcript_path": str(transcript_path),
                    "youtube_url": youtube_url,
                    "video_path": "",
                    "closest_keyframe_path": None,
                }
            )
    return records


def parse_transcript_name(stem: str) -> tuple[str, str, str] | None:
    match = re.match(r"^(day\d+)_(.+)_(\d{1,2})$", stem)
    if not match:
        return None
    day, source_name, hour_id = match.groups()
    return day, source_name, hour_id.zfill(2)


def lookup_youtube_url(day: str, source_name: str, hour_id: str, youtube_map: dict[str, str]) -> str | None:
    keys = [f"{day}/{source_name}/{hour_id}"]
    if hour_id.isdigit():
        keys.append(f"{day}/{source_name}/{int(hour_id)}")
    for key in keys:
        if youtube_map.get(key):
            return youtube_map[key]
    return None


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
