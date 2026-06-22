#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transcript_retrieval import retrieve_transcript_evidence_debug  # noqa: E402


EVAL_QUERIES = [
    "How long would the water not boil for?",
    "What did Stevan say about the water boiling?",
    "Why did they think the water would not boil?",
    "What did Tien say about boiling water?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug transcript evidence retrieval stages.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--index-dir", default="artifacts/transcript_index_day1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dense-k", type=int, default=100)
    parser.add_argument("--lexical-k", type=int, default=100)
    parser.add_argument("--rerank-k", type=int, default=50)
    parser.add_argument("--no-cross-encoder", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Run the standard transcript retrieval evaluation queries.")
    parser.add_argument("--output-dir", default="artifacts/transcript_retrieval_eval")
    args = parser.parse_args()

    if args.eval:
        run_eval(args)
        return
    if not args.question:
        parser.error("question is required unless --eval is used")

    debug = retrieve_transcript_evidence_debug(
        args.question,
        args.index_dir,
        top_k=args.top_k,
        dense_k=args.dense_k,
        lexical_k=args.lexical_k,
        rerank_k=args.rerank_k,
        use_cross_encoder=not args.no_cross_encoder,
    )
    print_debug(debug, args.top_k)


def run_eval(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for question in EVAL_QUERIES:
        print(f"Evaluating: {question}", flush=True)
        debug = retrieve_transcript_evidence_debug(
            question,
            args.index_dir,
            top_k=args.top_k,
            dense_k=args.dense_k,
            lexical_k=args.lexical_k,
            rerank_k=args.rerank_k,
            use_cross_encoder=not args.no_cross_encoder,
        )
        records.append(compact_debug(debug, args.top_k))

    report = {
        "index_dir": args.index_dir,
        "dense_k": args.dense_k,
        "lexical_k": args.lexical_k,
        "rerank_k": args.rerank_k,
        "use_cross_encoder": not args.no_cross_encoder,
        "queries": records,
    }
    json_path = output_dir / "transcript_retrieval_eval.json"
    md_path = output_dir / "transcript_retrieval_eval.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(markdown_report(report))
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


def print_debug(debug: dict[str, Any], top_k: int) -> None:
    print(f"Question: {debug['question']}")
    print(f"Index: {debug['index_dir']}")
    print(f"Embedding model: {debug['embedding_model']}")
    print(f"Cross-encoder: {debug['cross_encoder']} ({debug['cross_encoder_status']})")
    print("\nTIMINGS")
    for name, value in debug["timings"].items():
        print(f"{name}: {value:.3f}s")
    print_stage("DENSE TOP 10", debug["dense"], top_k)
    print_stage("LEXICAL TOP 10", debug["lexical"], top_k)
    print_stage("FUSED TOP 10", debug["fused"], top_k)
    print_stage("RERANKED TOP 10", debug["reranked"], top_k)


def print_stage(title: str, items: list[dict[str, Any]], top_k: int) -> None:
    print(f"\n{title}")
    for rank, item in enumerate(items[:top_k], start=1):
        print(format_item(rank, item))


def format_item(rank: int, item: dict[str, Any]) -> str:
    score_bits = [
        f"score={float(item.get('score', 0.0)):.4f}",
        f"dense_rank={item.get('dense_rank', '-')}",
        f"dense={float(item.get('dense_score', 0.0)):.4f}",
        f"lex_rank={item.get('lexical_rank', '-')}",
        f"lex={float(item.get('lexical_score', 0.0)):.4f}",
        f"rrf={float(item.get('rrf_score', 0.0)):.4f}",
    ]
    if item.get("cross_encoder_score") is not None:
        score_bits.append(f"ce={float(item['cross_encoder_score']):.4f}")
    if item.get("minilm_passage_score") is not None:
        score_bits.append(f"passage={float(item['minilm_passage_score']):.4f}")
    if item.get("source_name_boost"):
        score_bits.append(f"source_boost={float(item['source_name_boost']):.4f}")
    source = f"{item.get('source_name')} {item.get('day')} hour {item.get('hour_id', item.get('video_id'))}"
    return (
        f"{rank}. {source} {item.get('timestamp')} | "
        + " | ".join(score_bits)
        + f"\n   {item.get('transcript_snippet', '')}"
    )


def compact_debug(debug: dict[str, Any], top_k: int) -> dict[str, Any]:
    return {
        "question": debug["question"],
        "cross_encoder_status": debug["cross_encoder_status"],
        "timings": debug["timings"],
        "dense_top": [compact_item(item) for item in debug["dense"][:top_k]],
        "lexical_top": [compact_item(item) for item in debug["lexical"][:top_k]],
        "fused_top": [compact_item(item) for item in debug["fused"][:top_k]],
        "reranked_top": [compact_item(item) for item in debug["reranked"][:top_k]],
    }


def compact_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_name": item.get("source_name"),
        "day": item.get("day"),
        "hour_id": item.get("hour_id", item.get("video_id")),
        "timestamp": item.get("timestamp"),
        "start_sec": item.get("start_sec"),
        "end_sec": item.get("end_sec"),
        "youtube": item.get("youtube_timestamp_url"),
        "score": item.get("score"),
        "dense_rank": item.get("dense_rank"),
        "dense_score": item.get("dense_score"),
        "lexical_rank": item.get("lexical_rank"),
        "lexical_score": item.get("lexical_score"),
        "rrf_score": item.get("rrf_score"),
        "cross_encoder_score": item.get("cross_encoder_score"),
        "minilm_passage_score": item.get("minilm_passage_score"),
        "source_name_boost": item.get("source_name_boost"),
        "text_preview": item.get("transcript_snippet"),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Transcript Retrieval Evaluation",
        "",
        f"Index: `{report['index_dir']}`",
        f"Dense candidates: `{report['dense_k']}`",
        f"Lexical candidates: `{report['lexical_k']}`",
        f"Rerank candidates: `{report['rerank_k']}`",
        f"Cross-encoder requested: `{report['use_cross_encoder']}`",
        "",
    ]
    for record in report["queries"]:
        lines.append(f"## {record['question']}")
        lines.append(f"Cross-encoder status: `{record['cross_encoder_status']}`")
        if record["timings"]:
            timing = ", ".join(f"{key}={value:.3f}s" for key, value in record["timings"].items())
            lines.append(f"Timings: {timing}")
        lines.append("")
        for title, key in (
            ("Dense Top 10", "dense_top"),
            ("Lexical Top 10", "lexical_top"),
            ("Fused Top 10", "fused_top"),
            ("Reranked Top 10", "reranked_top"),
        ):
            lines.append(f"### {title}")
            for idx, item in enumerate(record[key], start=1):
                source = f"{item.get('source_name')} {item.get('day')} hour {item.get('hour_id')}"
                lines.append(
                    f"{idx}. `{source} {item.get('timestamp')}` "
                    f"score=`{float(item.get('score') or 0.0):.4f}` "
                    f"rrf=`{float(item.get('rrf_score') or 0.0):.4f}` "
                    f"dense_rank=`{item.get('dense_rank')}` lexical_rank=`{item.get('lexical_rank')}`"
                )
                lines.append(f"   {item.get('text_preview') or ''}")
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
