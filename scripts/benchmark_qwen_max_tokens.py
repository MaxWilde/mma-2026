#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_qwen_keyframe_captions import (  # noqa: E402
    generate_caption,
    is_test_pattern_caption,
    keyframe_to_json,
    load_caption_keyframes,
    select_keyframes,
)
from scripts.build_multimodal_index import build_visual_caption_metadata, load_captions  # noqa: E402
from src.retriever import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    build_faiss_index,
    embed_texts,
    load_embedding_model,
    query_index,
)
from src.vision_qa import DEFAULT_QWEN_MODEL, _load_qwen_model  # noqa: E402


DEFAULT_TOKEN_VALUES = (32, 48, 64, 96)
DEFAULT_QUERIES = (
    "What color is the refrigerator?",
    "Where is the kettle?",
    "What is on the stove?",
    "What color is the shirt of the person cooking?",
)


@dataclass(frozen=True)
class BenchmarkResult:
    max_new_tokens: int
    output_path: Path
    image_count: int
    elapsed_sec: float
    captions_per_sec: float
    avg_caption_words: float
    avg_caption_chars: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Qwen caption max_new_tokens settings.")
    parser.add_argument("--dataset-root", default="day1/Stevan/13")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sampling-mode", choices=["first", "uniform"], default="uniform")
    parser.add_argument("--tokens", default="32,48,64,96")
    parser.add_argument("--output-dir", default="artifacts/qwen_token_benchmark")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--queries", default="|".join(DEFAULT_QUERIES), help="Pipe-separated retrieval queries.")
    args = parser.parse_args()

    token_values = parse_int_list(args.tokens)
    queries = [query.strip() for query in args.queries.split("|") if query.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keyframes = select_keyframes(
        load_caption_keyframes(args.dataset_root),
        args.limit,
        args.sampling_mode,
    )
    if not keyframes:
        raise SystemExit(f"No keyframes found for dataset root: {args.dataset_root}")

    model_id = args.model_name or os.environ.get("QWEN_VL_MODEL") or DEFAULT_QWEN_MODEL
    model, processor, torch, process_vision_info = _load_qwen_model(model_id)

    results: list[BenchmarkResult] = []
    for max_new_tokens in token_values:
        output_path = output_dir / f"qwen_captions_{args.limit}_max{max_new_tokens}.jsonl"
        result = run_caption_benchmark(
            keyframes=keyframes,
            output_path=output_path,
            max_new_tokens=max_new_tokens,
            model=model,
            processor=processor,
            torch=torch,
            process_vision_info=process_vision_info,
        )
        results.append(result)

    retrieval_report = run_retrieval_comparison(
        results=results,
        queries=queries,
        embedding_model_name=args.embedding_model,
        top_k=args.top_k,
    )

    summary_path = output_dir / "summary.json"
    summary = {
        "dataset_root": args.dataset_root,
        "limit": len(keyframes),
        "sampling_mode": args.sampling_mode,
        "qwen_model": model_id,
        "results": [
            {
                "max_new_tokens": item.max_new_tokens,
                "output_path": str(item.output_path),
                "image_count": item.image_count,
                "elapsed_sec": item.elapsed_sec,
                "captions_per_sec": item.captions_per_sec,
                "avg_caption_words": item.avg_caption_words,
                "avg_caption_chars": item.avg_caption_chars,
            }
            for item in results
        ],
        "retrieval_report": retrieval_report,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(results, retrieval_report, summary_path)


def run_caption_benchmark(
    *,
    keyframes,
    output_path: Path,
    max_new_tokens: int,
    model: Any,
    processor: Any,
    torch: Any,
    process_vision_info: Any,
) -> BenchmarkResult:
    captions: list[str] = []
    start = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as f:
        for idx, keyframe in enumerate(keyframes, start=1):
            caption = generate_caption(
                keyframe.keyframe_path,
                model=model,
                processor=processor,
                torch=torch,
                process_vision_info=process_vision_info,
                max_new_tokens=max_new_tokens,
            )
            captions.append(caption)
            record = keyframe_to_json(keyframe)
            record["caption"] = caption
            record["is_test_pattern"] = is_test_pattern_caption(caption)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[max={max_new_tokens}] {idx}/{len(keyframes)} {keyframe.keyframe_path} -> {caption}")
    elapsed = time.perf_counter() - start
    words = [len(caption.split()) for caption in captions]
    chars = [len(caption) for caption in captions]
    return BenchmarkResult(
        max_new_tokens=max_new_tokens,
        output_path=output_path,
        image_count=len(captions),
        elapsed_sec=elapsed,
        captions_per_sec=len(captions) / elapsed if elapsed > 0 else 0.0,
        avg_caption_words=mean(words) if words else 0.0,
        avg_caption_chars=mean(chars) if chars else 0.0,
    )


def run_retrieval_comparison(
    *,
    results: list[BenchmarkResult],
    queries: list[str],
    embedding_model_name: str,
    top_k: int,
) -> dict[str, Any]:
    embedding_model = load_embedding_model(embedding_model_name)
    report: dict[str, Any] = {}
    for result in results:
        captions_by_source = load_captions(result.output_path)
        metadata, index_texts, _stats = build_visual_caption_metadata(captions_by_source)
        embeddings = embed_texts(embedding_model, index_texts, batch_size=64)
        index = build_faiss_index(embeddings)
        token_report: dict[str, Any] = {}
        for query in queries:
            hits = query_index(query, index, metadata, embedding_model, top_k=top_k)
            token_report[query] = [
                {
                    "rank": rank,
                    "score": hit.get("score"),
                    "source_name": hit.get("source_name"),
                    "day": hit.get("day"),
                    "video_id": hit.get("video_id"),
                    "timestamp_sec": hit.get("visual_caption_time_sec") or hit.get("keyframe_time_sec"),
                    "keyframe_path": hit.get("visual_caption_keyframe_path") or hit.get("keyframe_path"),
                    "visual_caption": hit.get("visual_caption"),
                }
                for rank, hit in enumerate(hits, start=1)
            ]
        report[str(result.max_new_tokens)] = token_report
    return report


def print_summary(
    results: list[BenchmarkResult],
    retrieval_report: dict[str, Any],
    summary_path: Path,
) -> None:
    print("\nCaption benchmark")
    print("max_new_tokens\timages\tseconds\tcaptions/sec\tavg_words\tavg_chars\toutput")
    for result in results:
        print(
            f"{result.max_new_tokens}\t"
            f"{result.image_count}\t"
            f"{result.elapsed_sec:.2f}\t"
            f"{result.captions_per_sec:.4f}\t"
            f"{result.avg_caption_words:.2f}\t"
            f"{result.avg_caption_chars:.2f}\t"
            f"{result.output_path}"
        )

    print("\nRetrieval comparison")
    for token_value, token_report in retrieval_report.items():
        print(f"\n=== max_new_tokens={token_value} ===")
        for query, hits in token_report.items():
            print(f"Query: {query}")
            for hit in hits:
                caption = str(hit.get("visual_caption") or "")
                short_caption = caption[:180] + ("..." if len(caption) > 180 else "")
                print(
                    f"  {hit['rank']}. score={float(hit['score']):.4f} "
                    f"{hit['source_name']} {hit['video_id']} t={hit['timestamp_sec']} "
                    f"{hit['keyframe_path']} | {short_caption}"
                )
    print(f"\nSummary JSON: {summary_path}")


def parse_int_list(value: str) -> tuple[int, ...]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one token value")
    return tuple(items)


if __name__ == "__main__":
    main()
