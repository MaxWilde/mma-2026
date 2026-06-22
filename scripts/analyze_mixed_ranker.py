#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.batch_query_evidence import parse_question_table, resolve_path  # noqa: E402
from scripts.query_evidence import retrieve_transcript, retrieve_visual  # noqa: E402
from src.evidence_router import route_evidence  # noqa: E402
from src.mixed_evidence_ranker import (  # noqa: E402
    build_mixed_evidence_list,
    channel_weights,
    diverse_transcript_candidates,
    diverse_visual_candidates,
    normalized_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze mixed evidence ranker behavior over ground-truth questions.")
    parser.add_argument("questions_md", type=Path)
    parser.add_argument("--output-dir", default="artifacts/mixed_ranker_analysis")
    parser.add_argument("--visual-index-dir", default="artifacts/siglip_index_day1")
    parser.add_argument("--transcript-index-dir", default="artifacts/transcript_index_day1")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--mixed-top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--mixed-calibration", choices=("max", "percentile"), default="max")
    args = parser.parse_args()

    rows = parse_question_table(resolve_path(args.questions_md))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {row['ID']}: {row['Question']}", flush=True)
        record = analyze_question(row, args)
        records.append(record)
        print(
            f"  route={record['chosen_route']} weights=({record['visual_weight']:.3f}, "
            f"{record['transcript_weight']:.3f}) top10={record['top10_counts']}",
            flush=True,
        )

    summary = build_summary(records, args)
    write_outputs(output_dir, summary, records)
    print(f"Analysis written to: {output_dir}", flush=True)


def analyze_question(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    query_args = argparse.Namespace(
        question=row["Question"],
        visual_index_dir=args.visual_index_dir,
        transcript_index_dir=args.transcript_index_dir,
        text_model_name=None,
        top_k=args.top_k,
        retrieval_top_k=args.top_k,
        visual_query_variants=True,
        synonyms_file=None,
        diversity_window_sec=0.0,
        candidate_multiplier=2,
        allow_download=False,
    )
    visual_results, visual_runtime = retrieve_visual(query_args)
    transcript_results = retrieve_transcript(row["Question"], args.transcript_index_dir, args.top_k)
    chosen = route_evidence(row["Question"], visual_results, transcript_results)
    router_debug = chosen.get("router_debug", {})
    visual_weight, transcript_weight = channel_weights(router_debug, temperature=args.temperature)

    visual_diverse = diverse_visual_candidates(visual_results[: args.top_k])
    transcript_diverse = diverse_transcript_candidates(transcript_results[: args.top_k])
    visual_norm = normalized_scores(visual_diverse)
    transcript_norm = normalized_scores(transcript_diverse)
    visual_final = [score * visual_weight for score in visual_norm]
    transcript_final = [score * transcript_weight for score in transcript_norm]
    mixed = build_mixed_evidence_list(
        row["Question"],
        visual_results[: args.top_k],
        transcript_results[: args.top_k],
        router_debug,
        top_k=args.mixed_top_k,
        temperature=args.temperature,
        calibration_mode=args.mixed_calibration,
    )
    top10_types = [item.get("evidence_type") for item in mixed[: args.mixed_top_k]]
    top10_counts = Counter(str(value) for value in top10_types)
    dominant_type, dominant_count = dominant_modality(top10_counts)

    strongest_visual_raw = max_score(visual_results)
    strongest_transcript_raw = max_score(transcript_results)
    strongest_visual_final = max(visual_final, default=0.0)
    strongest_transcript_final = max(transcript_final, default=0.0)

    raw_visual_gt_transcript = strongest_visual_raw > strongest_transcript_raw
    raw_transcript_gt_visual = strongest_transcript_raw > strongest_visual_raw
    all_visual_top10 = top10_counts.get("visual", 0) == args.mixed_top_k
    all_transcript_top10 = top10_counts.get("transcript", 0) == args.mixed_top_k

    return {
        "id": row["ID"],
        "expected_type": row["Type"].strip().lower(),
        "question": row["Question"],
        "ground_truth": row["Ground Truth Answer / Visual Evidence Target"],
        "chosen_route": chosen.get("evidence_type"),
        "router_debug": router_debug,
        "visual_weight": visual_weight,
        "transcript_weight": transcript_weight,
        "top10_counts": dict(top10_counts),
        "top10_sequence": top10_types,
        "dominant_top10_modality": dominant_type,
        "router_dominance": dominant_count / max(1, args.mixed_top_k),
        "strongest_visual_raw_score": strongest_visual_raw,
        "strongest_transcript_raw_score": strongest_transcript_raw,
        "strongest_visual_final_score": strongest_visual_final,
        "strongest_transcript_final_score": strongest_transcript_final,
        "raw_visual_gt_transcript_but_all_transcript_top10": raw_visual_gt_transcript and all_transcript_top10,
        "raw_transcript_gt_visual_but_all_visual_top10": raw_transcript_gt_visual and all_visual_top10,
        "visual_score_distribution": distribution([float(item.get("score", 0.0)) for item in visual_results[: args.top_k]]),
        "transcript_score_distribution": distribution([float(item.get("score", 0.0)) for item in transcript_results[: args.top_k]]),
        "visual_normalized_distribution": distribution(visual_norm),
        "transcript_normalized_distribution": distribution(transcript_norm),
        "visual_final_distribution": distribution(visual_final),
        "transcript_final_distribution": distribution(transcript_final),
        "mixed_confidence_distribution": distribution([float(item.get("confidence", 0.0)) for item in mixed]),
        "mixed_top10": [
            {
                "rank": item.get("rank"),
                "evidence_type": item.get("evidence_type"),
                "confidence": item.get("confidence"),
                "confidence_percent": item.get("confidence_percent"),
                "raw_score": (item.get("score_components") or {}).get("raw_score"),
                "normalized_score": (item.get("score_components") or {}).get("normalized_score"),
                "final_score": (item.get("score_components") or {}).get("final_score"),
                "source_or_keyframe": item.get("keyframe_path") or item.get("source_name"),
                "timestamp": item.get("timestamp"),
                "youtube": item.get("youtube_timestamp_url"),
            }
            for item in mixed
        ],
        "visual_runtime_sec": visual_runtime,
    }


def build_summary(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    top10_histogram = Counter(composition_key(record["top10_counts"]) for record in records)
    visual_weights = [record["visual_weight"] for record in records]
    transcript_weights = [record["transcript_weight"] for record in records]
    dominance = [record["router_dominance"] for record in records]
    confidence_values = [
        value
        for record in records
        for value in [item.get("confidence") for item in record["mixed_top10"]]
        if value is not None
    ]
    contradictions = {
        "visual_raw_gt_transcript_but_all_transcript_top10": [
            record["id"] for record in records if record["raw_visual_gt_transcript_but_all_transcript_top10"]
        ],
        "transcript_raw_gt_visual_but_all_visual_top10": [
            record["id"] for record in records if record["raw_transcript_gt_visual_but_all_visual_top10"]
        ],
    }

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "questions": len(records),
        "top_k": args.top_k,
        "mixed_top_k": args.mixed_top_k,
        "temperature": args.temperature,
        "mixed_calibration": args.mixed_calibration,
        "top10_modality_histogram": dict(sorted(top10_histogram.items())),
        "visual_weight_distribution": distribution(visual_weights),
        "transcript_weight_distribution": distribution(transcript_weights),
        "router_dominance_distribution": distribution(dominance),
        "mixed_confidence_distribution": distribution(confidence_values),
        "contradiction_counts": {key: len(value) for key, value in contradictions.items()},
        "contradictions": contradictions,
        "most_visual_dominated_query": representative(records, "visual"),
        "most_transcript_dominated_query": representative(records, "transcript"),
        "most_balanced_query": most_balanced(records),
        "per_question": records,
    }


def write_outputs(output_dir: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_markdown(summary, records), encoding="utf-8")


def render_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Mixed Ranker Analysis",
        "",
        "This analysis reuses the existing retrieval, router, and mixed-ranker implementations without modifying them.",
        "",
        "## Summary",
        "",
        f"- Questions: {summary['questions']}",
        f"- Candidate depth per modality: {summary['top_k']}",
        f"- Mixed top-k: {summary['mixed_top_k']}",
        f"- Softmax temperature: {summary['temperature']}",
        f"- Mixed calibration: {summary['mixed_calibration']}",
        "",
        "## Top-10 Modality Histogram",
        "",
    ]
    for key, count in summary["top10_modality_histogram"].items():
        lines.append(f"- `{key}`: {count}")
    lines.extend(
        [
            "",
            "## Router Weight Distributions",
            "",
            f"- Visual weight: `{format_distribution(summary['visual_weight_distribution'])}`",
            f"- Transcript weight: `{format_distribution(summary['transcript_weight_distribution'])}`",
            "",
            "## Router Dominance",
            "",
            "Router dominance = count of the dominant modality in top-10 divided by 10.",
            "",
            f"- Distribution: `{format_distribution(summary['router_dominance_distribution'])}`",
            "",
            "## Mixed Confidence Distribution",
            "",
            f"- Distribution: `{format_distribution(summary['mixed_confidence_distribution'])}`",
            "",
            "## Contradiction Counts",
            "",
            f"- Visual raw score > transcript raw score, but all top-10 transcript: {summary['contradiction_counts']['visual_raw_gt_transcript_but_all_transcript_top10']}",
            f"- Transcript raw score > visual raw score, but all top-10 visual: {summary['contradiction_counts']['transcript_raw_gt_visual_but_all_visual_top10']}",
            "",
            "## Representative Examples",
            "",
        ]
    )
    for label, key in (
        ("Most visual-dominated query", "most_visual_dominated_query"),
        ("Most transcript-dominated query", "most_transcript_dominated_query"),
        ("Most balanced query", "most_balanced_query"),
    ):
        record = summary.get(key)
        lines.extend(render_representative(label, record))

    lines.extend(["", "## Per-Question Overview", ""])
    lines.append("| ID | Expected | Chosen | Weights V/T | Top10 V/T | Dominance | Strongest raw V/T | Question |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for record in records:
        counts = record["top10_counts"]
        lines.append(
            "| {id} | {expected} | {chosen} | {vw:.3f}/{tw:.3f} | {vc}/{tc} | {dom:.2f} | {vs:.3f}/{ts:.3f} | {question} |".format(
                id=record["id"],
                expected=record["expected_type"],
                chosen=record["chosen_route"],
                vw=record["visual_weight"],
                tw=record["transcript_weight"],
                vc=counts.get("visual", 0),
                tc=counts.get("transcript", 0),
                dom=record["router_dominance"],
                vs=record["strongest_visual_raw_score"],
                ts=record["strongest_transcript_raw_score"],
                question=record["question"].replace("|", "\\|"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_representative(label: str, record: dict[str, Any] | None) -> list[str]:
    if not record:
        return [f"### {label}", "", "No record.", ""]
    counts = record["top10_counts"]
    return [
        f"### {label}",
        "",
        f"- ID: `{record['id']}`",
        f"- Question: {record['question']}",
        f"- Chosen route: `{record['chosen_route']}`",
        f"- Visual/transcript weights: `{record['visual_weight']:.3f}` / `{record['transcript_weight']:.3f}`",
        f"- Top-10 visual/transcript counts: `{counts.get('visual', 0)}` / `{counts.get('transcript', 0)}`",
        f"- Router dominance: `{record['router_dominance']:.2f}`",
        f"- Strongest raw visual/transcript scores: `{record['strongest_visual_raw_score']:.4f}` / `{record['strongest_transcript_raw_score']:.4f}`",
        "",
    ]


def representative(records: list[dict[str, Any]], modality: str) -> dict[str, Any] | None:
    return max(
        records,
        key=lambda record: (
            record["top10_counts"].get(modality, 0),
            record["visual_weight"] if modality == "visual" else record["transcript_weight"],
        ),
        default=None,
    )


def most_balanced(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    return min(
        records,
        key=lambda record: (
            abs(record["top10_counts"].get("visual", 0) - record["top10_counts"].get("transcript", 0)),
            abs(record["visual_weight"] - record["transcript_weight"]),
        ),
        default=None,
    )


def dominant_modality(counts: Counter[str]) -> tuple[str, int]:
    visual = counts.get("visual", 0)
    transcript = counts.get("transcript", 0)
    return ("visual", visual) if visual >= transcript else ("transcript", transcript)


def composition_key(counts: dict[str, int]) -> str:
    return f"visual={counts.get('visual', 0)} transcript={counts.get('transcript', 0)}"


def max_score(items: list[dict[str, Any]]) -> float:
    return max((float(item.get("score", 0.0)) for item in items), default=0.0)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"n": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def format_distribution(stats: dict[str, Any]) -> str:
    if not stats or not stats.get("n"):
        return "n=0"
    return "n={n}, min={min:.3f}, mean={mean:.3f}, median={median:.3f}, max={max:.3f}".format(**stats)


if __name__ == "__main__":
    main()
