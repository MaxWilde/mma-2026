#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence_links import youtube_timestamp_url
from src.evidence_selection import EvidenceResult, select_evidence
from src.retriever import load_embedding_model, load_index, query_index
from src.reranker import rerank_chunks
from src.visual_grounding import GroundingConfig
from src.vqa import format_timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve CASTLE transcript or visual evidence from a FAISS index.")
    parser.add_argument("question")
    parser.add_argument("--index-dir", default="artifacts/transcript_index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-pool-size", type=int, default=20)
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--visual-weight", type=float, default=0.3)
    parser.add_argument(
        "--grounding-threshold",
        default="0.25,0.15,0.10,0.05",
        help="Comma-separated GroundingDINO box/text thresholds to try.",
    )
    parser.add_argument(
        "--grounding-prompts",
        default="",
        help="Comma-separated explicit grounding prompts. If omitted, prompts are derived from question and visual caption.",
    )
    parser.add_argument(
        "--grounding-prompt-strategy",
        default="auto",
        choices=["auto"],
        help="Prompt derivation strategy.",
    )
    parser.add_argument("--debug-grounding", action="store_true", help="Print prompt/threshold grounding attempts.")
    args = parser.parse_args()

    index, metadata, model_name = load_index(args.index_dir)
    model = load_embedding_model(model_name)
    retrieval_k = max(args.top_k, args.rerank_pool_size) if args.rerank else args.top_k
    chunks = query_index(args.question, index, metadata, model, top_k=retrieval_k)

    rerank_method = None
    if args.rerank:
        rerank_result = rerank_chunks(
            args.question,
            chunks,
            cross_encoder_model=args.reranker_model,
            visual_weight=args.visual_weight,
        )
        chunks = rerank_result.chunks[: args.top_k]
        rerank_method = rerank_result.method

    grounding_config = GroundingConfig(
        thresholds=parse_thresholds(args.grounding_threshold),
        prompts=parse_prompts(args.grounding_prompts),
        prompt_strategy=args.grounding_prompt_strategy,
        debug=args.debug_grounding,
    )
    evidence = select_evidence(
        args.question,
        chunks,
        embedding_model=model,
        grounding_config=grounding_config,
    )

    print(f"Question: {args.question}")
    if rerank_method:
        print(f"Rerank method: {rerank_method}")
        print(f"Visual weight: {args.visual_weight:.2f}")
    print_evidence(evidence)

    print("\nRetrieved evidence:")
    for idx, chunk in enumerate(chunks, start=1):
        timestamp = format_timestamp(float(chunk["start_sec"]), float(chunk["end_sec"]))
        score = f"score={chunk['score']:.4f}"
        if "final_score" in chunk:
            score += (
                f" transcript_score={chunk['transcript_score']:.4f}"
                f" visual_score={chunk['visual_score']:.4f}"
                f" final_score={chunk['final_score']:.4f}"
            )
        print(f"\n[{idx}] {score} {chunk['source_name']} {timestamp}")
        print(f"source_id: {chunk['source_id']}")
        video_path = str(chunk.get("video_path") or "")
        if video_path:
            print(f"video: {video_path}")
        youtube_link = youtube_timestamp_url(chunk)
        if youtube_link:
            print(f"youtube: {youtube_link}")
        keyframe = best_keyframe_path(chunk)
        if keyframe:
            print(f"keyframe: {keyframe}")
        if chunk.get("frame_number") is not None:
            print(f"frame: {chunk['frame_number']}")
        if chunk.get("visual_caption"):
            print(f"visual caption: {chunk['visual_caption']}")
        if chunk.get("text"):
            print(chunk["text"])


def print_evidence(evidence: EvidenceResult) -> None:
    print("\nEVIDENCE")
    print(f"Evidence type: {evidence.evidence_type}")
    if evidence.chunk is None:
        print("No retrieved evidence.")
        return

    chunk = evidence.chunk
    timestamp = format_timestamp(float(chunk["start_sec"]), float(chunk["end_sec"]))
    if evidence.evidence_type == "transcript":
        print(f"Matched transcript evidence: {evidence.transcript_snippet or 'n/a'}")
    elif evidence.evidence_type == "visual":
        if evidence.visual_caption:
            print(f"Visual caption evidence: {evidence.visual_caption}")
        grounding = evidence.visual_grounding
        print(f"Keyframe: {grounding.keyframe_path if grounding else best_keyframe_path(chunk) or 'n/a'}")
        print(f"Grounding target: {grounding.grounding_target if grounding else 'n/a'}")
        print(f"Bounding box image: {grounding.output_image_path if grounding else 'n/a'}")
        print(f"Evidence localization method: {grounding.method if grounding else 'none'}")
        if grounding and grounding.box_xyxy:
            box = ", ".join(f"{value:.1f}" for value in grounding.box_xyxy)
            print(f"Bounding box: [{box}]")
        if grounding and grounding.label:
            print(f"Evidence localization note: {grounding.label}")
        if grounding and grounding.confidence is not None:
            print(f"Confidence: {grounding.confidence:.3f}")
        if grounding and grounding.debug_context:
            context = grounding.debug_context
            print("Grounding debug:")
            print(f"- question: {context['question']}")
            print(f"- visual_caption: {context['visual_caption']}")
            print(f"- extracted_grounding_target: {context['grounding_target']}")
            print(f"- prompts_tried: {', '.join(context['prompts'])}")
            print(f"- thresholds_tried: {', '.join(f'{float(item):.3f}' for item in context['thresholds'])}")
            print(f"- selected_detection: {format_selected_detection(context['selected_detection'])}")
        if grounding and grounding.debug:
            print("Grounding attempts:")
            for attempt in grounding.debug:
                print(
                    "- "
                    f"prompt={attempt['prompt']!r} "
                    f"threshold={attempt['threshold']:.3f} "
                    f"detections={attempt['detections']} "
                    f"best_score={format_optional_score(attempt['best_score'])} "
                    f"selected_box={format_optional_box(attempt['selected_box'])}"
                )

    print(f"Timestamp: {timestamp}")
    youtube_link = youtube_timestamp_url(chunk)
    if youtube_link:
        print(f"YouTube: {youtube_link}")


def best_keyframe_path(chunk: dict) -> str | None:
    return (
        chunk.get("keyframe_path")
        or chunk.get("visual_caption_keyframe_path")
        or chunk.get("closest_keyframe_path")
    )


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        thresholds.append(float(item))
    return tuple(thresholds or [0.25])


def parse_prompts(value: str) -> tuple[str, ...] | None:
    prompts = tuple(prompt.strip() for prompt in value.split(",") if prompt.strip())
    return prompts or None


def format_optional_score(value) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def format_optional_box(value) -> str:
    if not value:
        return "n/a"
    return "[" + ", ".join(f"{float(item):.1f}" for item in value) + "]"


def format_selected_detection(value) -> str:
    if not value:
        return "n/a"
    box = format_optional_box(value.get("box_xyxy"))
    return f"label={value.get('label')!r} score={float(value.get('score', 0.0)):.4f} box={box}"


if __name__ == "__main__":
    main()
