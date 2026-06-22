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

from src.retriever import load_index  # noqa: E402
from src.transcript_retrieval import retrieve_transcript_evidence_debug  # noqa: E402
from src.vqa import format_timestamp  # noqa: E402


DEFAULT_VARIANTS = ("no celery", "bay leaf", "no bay leaf", "nothing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether a target transcript phrase is recalled.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--target-phrase", required=True)
    parser.add_argument("--transcript-index-dir", default="artifacts/transcript_index_day1")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--output-root", default="artifacts/transcript_recall_debug")
    args = parser.parse_args()

    report = build_report(args)
    output_dir = ROOT / args.output_root / safe_name(args.question)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_console(report))
    print(f"\nWrote: {output_dir / 'report.md'}")
    print(f"Wrote: {output_dir / 'report.json'}")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    _index, metadata, model_name = load_index(args.transcript_index_dir)
    phrase_matches = search_phrases(metadata, args.target_phrase, DEFAULT_VARIANTS)
    debug = retrieve_transcript_evidence_debug(
        args.question,
        args.transcript_index_dir,
        top_k=args.top_k,
        dense_k=args.top_k,
        lexical_k=args.top_k,
        rerank_k=args.top_k,
        use_cross_encoder=True,
        align_playback=False,
    )
    target_keys = {match["source_id"] for match in phrase_matches["target_phrase"]["matches"]}
    stage_reports = {
        stage: stage_rank_report(debug.get(stage, []), target_keys)
        for stage in ("dense", "lexical", "fused", "reranked")
    }
    top_results = [summarize_result(item, rank) for rank, item in enumerate(debug.get("reranked", [])[:20], start=1)]
    return {
        "question": args.question,
        "target_phrase": args.target_phrase,
        "transcript_index_dir": str(args.transcript_index_dir),
        "top_k": args.top_k,
        "metadata_items": len(metadata),
        "embedding_model": model_name,
        "cross_encoder_status": debug.get("cross_encoder_status"),
        "timings": debug.get("timings", {}),
        "phrase_search": phrase_matches,
        "target_in_retrieved_results": stage_reports["reranked"]["found"],
        "target_rank": stage_reports["reranked"].get("rank"),
        "target_retrieval_score": stage_reports["reranked"].get("score"),
        "retrieval_stages": stage_reports,
        "top_20_retrieved_chunks": top_results,
        "available_score_fields": available_score_fields(debug),
        "unavailable_score_fields": unavailable_score_fields(debug),
    }


def search_phrases(metadata: list[dict[str, Any]], target_phrase: str, variants: tuple[str, ...]) -> dict[str, Any]:
    phrases = [target_phrase] + [variant for variant in variants if normalize_text(variant) != normalize_text(target_phrase)]
    results = {}
    for phrase in phrases:
        matches = []
        normalized_phrase = normalize_text(phrase)
        for item in metadata:
            text = str(item.get("text", ""))
            normalized_text = normalize_text(text)
            if normalized_phrase and normalized_phrase in normalized_text:
                matches.append(summarize_match(item, phrase))
        key = "target_phrase" if phrase == target_phrase else phrase
        results[key] = {
            "phrase": phrase,
            "exists": bool(matches),
            "match_count": len(matches),
            "matches": matches[:50],
        }
    return results


def summarize_match(item: dict[str, Any], phrase: str) -> dict[str, Any]:
    text = str(item.get("text", ""))
    start, end = find_phrase_span(text, phrase)
    return {
        "source_id": candidate_key(item),
        "transcript_file": item.get("transcript_path"),
        "source_name": item.get("source_name"),
        "day": item.get("day"),
        "hour_id": item.get("hour_id", item.get("video_id")),
        "start_sec": item.get("start_sec"),
        "end_sec": item.get("end_sec"),
        "timestamp": safe_timestamp(item),
        "surrounding_text": surrounding_text(text, start, end),
    }


def stage_rank_report(results: list[dict[str, Any]], target_keys: set[str]) -> dict[str, Any]:
    if not target_keys:
        return {"found": False, "rank": None, "score": None, "score_fields": []}
    for rank, item in enumerate(results, start=1):
        if candidate_key(item) in target_keys:
            return {
                "found": True,
                "rank": rank,
                "score": item.get("score"),
                "score_fields": score_fields(item),
                "result": summarize_result(item, rank),
            }
    return {"found": False, "rank": None, "score": None, "score_fields": []}


def summarize_result(item: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "source_id": candidate_key(item),
        "source_name": item.get("source_name"),
        "day": item.get("day"),
        "hour_id": item.get("hour_id", item.get("video_id")),
        "timestamp": item.get("timestamp") or safe_timestamp(item),
        "transcript_path": item.get("transcript_path"),
        "score": item.get("score"),
        "dense_rank": item.get("dense_rank"),
        "dense_score": item.get("dense_score"),
        "lexical_rank": item.get("lexical_rank"),
        "lexical_score": item.get("lexical_score"),
        "fused_rank": item.get("fused_rank"),
        "rrf_score": item.get("rrf_score"),
        "rerank_rank": item.get("rerank_rank"),
        "cross_encoder_score": item.get("cross_encoder_score"),
        "minilm_passage_score": item.get("minilm_passage_score"),
        "text_preview": preview_text(str(item.get("text", ""))),
    }


def available_score_fields(debug: dict[str, Any]) -> dict[str, list[str]]:
    return {
        stage: sorted({field for item in debug.get(stage, []) for field in score_fields(item)})
        for stage in ("dense", "lexical", "fused", "reranked")
    }


def unavailable_score_fields(debug: dict[str, Any]) -> dict[str, list[str]]:
    expected = {
        "dense": {"dense_rank", "dense_score", "score"},
        "lexical": {"lexical_rank", "lexical_score", "score"},
        "fused": {"dense_rank", "dense_score", "lexical_rank", "lexical_score", "fused_rank", "rrf_score", "score"},
        "reranked": {"rerank_rank", "cross_encoder_score", "minilm_passage_score", "score"},
    }
    available = available_score_fields(debug)
    return {stage: sorted(expected[stage] - set(available.get(stage, []))) for stage in expected}


def score_fields(item: dict[str, Any]) -> list[str]:
    return [
        field
        for field in (
            "score",
            "dense_rank",
            "dense_score",
            "lexical_rank",
            "lexical_score",
            "fused_rank",
            "rrf_score",
            "rerank_rank",
            "cross_encoder_score",
            "minilm_passage_score",
            "source_name_boost",
        )
        if item.get(field) is not None
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Transcript Recall Debug Report",
        "",
        f"- Question: {report['question']}",
        f"- Target phrase: `{report['target_phrase']}`",
        f"- Transcript index: `{report['transcript_index_dir']}`",
        f"- Top-k: {report['top_k']}",
        f"- Metadata chunks: {report['metadata_items']}",
        f"- Embedding model: `{report['embedding_model']}`",
        f"- Cross-encoder status: `{report['cross_encoder_status']}`",
        "",
        "## Phrase Search",
        "",
    ]
    for key, result in report["phrase_search"].items():
        lines.append(f"### {result['phrase']}")
        lines.append(f"- Exists: {result['exists']}")
        lines.append(f"- Match count: {result['match_count']}")
        for match in result["matches"][:10]:
            lines.extend(
                [
                    f"- `{match['source_name']} {match['day']} hour {match['hour_id']} {match['timestamp']}`",
                    f"  - transcript: `{match['transcript_file']}`",
                    f"  - surrounding: {match['surrounding_text']}",
                ]
            )
        lines.append("")

    lines.extend(["## Retrieval Recall", ""])
    for stage, stage_report in report["retrieval_stages"].items():
        lines.append(
            f"- {stage}: found={stage_report['found']}, "
            f"rank={stage_report.get('rank')}, score={stage_report.get('score')}"
        )
    lines.extend(["", "## Available Score Fields", ""])
    for stage, fields in report["available_score_fields"].items():
        lines.append(f"- {stage}: {', '.join(fields) if fields else 'none'}")
    lines.extend(["", "## Unavailable Expected Score Fields", ""])
    for stage, fields in report["unavailable_score_fields"].items():
        lines.append(f"- {stage}: {', '.join(fields) if fields else 'none'}")

    lines.extend(["", "## Top 20 Retrieved Chunks", ""])
    for item in report["top_20_retrieved_chunks"]:
        lines.extend(
            [
                f"### Rank {item['rank']}: {item['source_name']} {item['day']} hour {item['hour_id']} {item['timestamp']}",
                f"- score: {item['score']}",
                f"- dense: rank={item['dense_rank']} score={item['dense_score']}",
                f"- lexical: rank={item['lexical_rank']} score={item['lexical_score']}",
                f"- fused: rank={item['fused_rank']} rrf={item['rrf_score']}",
                f"- rerank: rank={item['rerank_rank']} cross_encoder={item['cross_encoder_score']} minilm={item['minilm_passage_score']}",
                f"- transcript: `{item['transcript_path']}`",
                f"- text: {item['text_preview']}",
                "",
            ]
        )
    return "\n".join(lines)


def render_console(report: dict[str, Any]) -> str:
    target = report["phrase_search"]["target_phrase"]
    lines = [
        "Transcript recall debug",
        f"Question: {report['question']}",
        f"Target phrase exists: {target['exists']} (matches={target['match_count']})",
        f"Target in retrieved top-{report['top_k']}: {report['target_in_retrieved_results']}",
        f"Target rank: {report['target_rank']}",
        f"Target retrieval score: {report['target_retrieval_score']}",
        "",
        "Stage recall:",
    ]
    for stage, stage_report in report["retrieval_stages"].items():
        lines.append(f"- {stage}: found={stage_report['found']} rank={stage_report.get('rank')} score={stage_report.get('score')}")
    lines.append("")
    lines.append("Top 20 retrieved chunks:")
    for item in report["top_20_retrieved_chunks"]:
        lines.append(
            f"{item['rank']}. {item['source_name']} {item['day']} hour {item['hour_id']} "
            f"{item['timestamp']} score={item['score']} :: {item['text_preview']}"
        )
    return "\n".join(lines)


def candidate_key(item: dict[str, Any]) -> str:
    return str(item.get("source_id") or f"{item.get('transcript_path')}:{item.get('start_sec')}:{item.get('end_sec')}")


def find_phrase_span(text: str, phrase: str) -> tuple[int, int]:
    index = text.lower().find(phrase.lower())
    if index >= 0:
        return index, index + len(phrase)
    normalized_phrase = normalize_text(phrase)
    normalized_text = normalize_text(text)
    normalized_index = normalized_text.find(normalized_phrase)
    if normalized_index < 0:
        return -1, -1
    return -1, -1


def surrounding_text(text: str, start: int, end: int, window: int = 220) -> str:
    if start < 0 or end < 0:
        return preview_text(text, limit=window * 2)
    left = max(0, start - window)
    right = min(len(text), end + window)
    return preview_text(text[left:right], limit=window * 2)


def preview_text(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def normalize_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def safe_timestamp(item: dict[str, Any]) -> str:
    try:
        return format_timestamp(float(item["start_sec"]), float(item["end_sec"]))
    except (KeyError, TypeError, ValueError):
        return ""


def safe_name(value: str, limit: int = 80) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return (safe[:limit].rstrip("_") or "question")


if __name__ == "__main__":
    main()
