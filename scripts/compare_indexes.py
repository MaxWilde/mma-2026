#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retriever import load_embedding_model, load_index, query_index
from src.vqa import format_timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare transcript-only and multimodal-caption retrieval results.")
    parser.add_argument("question", nargs="?", default="What are they cooking?")
    parser.add_argument("--transcript-index-dir", default="artifacts/transcript_index")
    parser.add_argument("--multimodal-index-dir", default="artifacts/multimodal_index_100")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    transcript_results = search_index(args.transcript_index_dir, args.question, args.top_k)
    multimodal_results = search_index(args.multimodal_index_dir, args.question, args.top_k)

    print(f"Question: {args.question}")
    print(f"Transcript index: {args.transcript_index_dir}")
    print(f"Multimodal index: {args.multimodal_index_dir}")
    print()

    for rank in range(args.top_k):
        print(f"===== Rank {rank + 1} =====")
        print_result("TRANSCRIPT", transcript_results[rank] if rank < len(transcript_results) else None)
        print()
        print_result("MULTIMODAL", multimodal_results[rank] if rank < len(multimodal_results) else None)
        print()


def search_index(index_dir: str, question: str, top_k: int) -> list[dict]:
    index, metadata, model_name = load_index(index_dir)
    model = load_embedding_model(model_name)
    return query_index(question, index, metadata, model, top_k=top_k)


def print_result(label: str, result: dict | None) -> None:
    print(f"[{label}]")
    if result is None:
        print("No result")
        return

    timestamp = format_timestamp(float(result["start_sec"]), float(result["end_sec"]))
    source = f"{result['source_name']} {result['day']} video {result['video_id']}"
    print(f"score: {result['score']:.4f}")
    print(f"source: {source}")
    print(f"timestamp: {timestamp}")
    if result.get("closest_keyframe_path"):
        print(f"keyframe: {result['closest_keyframe_path']}")
    print(f"transcript: {result['text']}")
    visual_caption = result.get("visual_caption")
    if visual_caption:
        distance = result.get("visual_caption_distance_sec")
        if distance is None:
            print(f"visual_caption: {visual_caption}")
        else:
            print(f"visual_caption ({float(distance):.2f}s away): {visual_caption}")
    else:
        print("visual_caption: none")


if __name__ == "__main__":
    main()
