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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs transcript answer rerank batch outputs.")
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--rerank-dir", required=True, type=Path)
    parser.add_argument("--output", default="artifacts/transcript_rerank_comparison.md")
    args = parser.parse_args()

    baseline_dir = resolve_path(args.baseline_dir)
    rerank_dir = resolve_path(args.rerank_dir)
    baseline = load_batch_results(baseline_dir)
    rerank = load_batch_results(rerank_dir)
    comparison = build_comparison(baseline, rerank, baseline_dir, rerank_dir)

    output_path = resolve_path(Path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(comparison), encoding="utf-8")
    output_path.with_suffix(".json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Wrote {output_path.with_suffix('.json')}")


def load_batch_results(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Missing summary.json: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = {}
    for path in output_dir.glob("*/*/result.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        results[str(item["id"])] = item
    if not results and summary.get("per_question"):
        results = {str(item["id"]): item for item in summary["per_question"]}
    return {"summary": summary, "results": results}


def build_comparison(
    baseline: dict[str, Any],
    rerank: dict[str, Any],
    baseline_dir: Path,
    rerank_dir: Path,
) -> dict[str, Any]:
    baseline_summary = route_metrics(baseline["summary"])
    rerank_summary = route_metrics(rerank["summary"])
    transcript_rows = []
    changed_chunks = []
    changed_spans = []
    improved = []
    worsened = []

    ids = sorted(set(baseline["results"]) & set(rerank["results"]))
    for question_id in ids:
        before = baseline["results"][question_id]
        after = rerank["results"][question_id]
        if normalize_type(before.get("expected_type")) != "transcript":
            continue
        before_hit = answer_hit(before)
        after_hit = answer_hit(after)
        chunk_changed = selected_chunk_key(before) != selected_chunk_key(after)
        span_changed = answer_span_key(before) != answer_span_key(after)
        row = {
            "id": question_id,
            "question": before.get("question"),
            "ground_truth": before.get("ground_truth_answer_or_visual_target"),
            "baseline_route": before.get("predicted_type"),
            "rerank_route": after.get("predicted_type"),
            "baseline_chunk": selected_chunk_key(before),
            "rerank_chunk": selected_chunk_key(after),
            "selected_transcript_chunk_changed": chunk_changed,
            "baseline_answer_span": answer_span_text(before),
            "rerank_answer_span": answer_span_text(after),
            "answer_span_changed": span_changed,
            "baseline_answer_overlap": before_hit,
            "rerank_answer_overlap": after_hit,
            "improved": after_hit["hit"] and not before_hit["hit"],
            "worsened": before_hit["hit"] and not after_hit["hit"],
        }
        transcript_rows.append(row)
        if chunk_changed:
            changed_chunks.append(row)
        if span_changed:
            changed_spans.append(row)
        if row["improved"]:
            improved.append(row)
        if row["worsened"]:
            worsened.append(row)

    return {
        "baseline_dir": str(baseline_dir),
        "rerank_dir": str(rerank_dir),
        "metrics": {
            "baseline": baseline_summary,
            "rerank": rerank_summary,
        },
        "transcript_analysis": {
            "transcript_questions": len(transcript_rows),
            "selected_transcript_chunk_changed": len(changed_chunks),
            "answer_span_changed": len(changed_spans),
            "questions_improved": len(improved),
            "questions_worsened": len(worsened),
            "improved": summarize_rows(improved),
            "worsened": summarize_rows(worsened),
            "changed_chunks": summarize_rows(changed_chunks),
            "changed_spans": summarize_rows(changed_spans),
            "per_question": transcript_rows,
        },
        "accuracy_note": (
            "Route metrics come from the existing benchmark summary. Improved/worsened is a lexical "
            "ground-truth-answer overlap heuristic over the selected answer span/transcript text, not a human judgment."
        ),
    }


def route_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "routing_accuracy": summary.get("routing_accuracy"),
        "transcript_accuracy": summary.get("transcript_accuracy"),
        "visual_accuracy": summary.get("visual_accuracy"),
        "overall_accuracy": summary.get("routing_accuracy"),
        "correct_routes": summary.get("correct_routes"),
        "total_questions": summary.get("total_questions"),
        "transcript_correct": summary.get("transcript_correct"),
        "transcript_total": summary.get("transcript_total"),
        "visual_correct": summary.get("visual_correct"),
        "visual_total": summary.get("visual_total"),
    }


def answer_hit(item: dict[str, Any]) -> dict[str, Any]:
    gt_tokens = content_tokens(str(item.get("ground_truth_answer_or_visual_target") or ""))
    evidence_tokens = content_tokens(answer_span_text(item) + " " + transcript_text_from_result(item))
    if not gt_tokens:
        return {"hit": False, "overlap": 0, "required": 0, "recall": 0.0}
    overlap = len(gt_tokens & evidence_tokens)
    recall = overlap / len(gt_tokens)
    return {
        "hit": recall >= 0.5 or overlap >= min(2, len(gt_tokens)),
        "overlap": overlap,
        "required": len(gt_tokens),
        "recall": recall,
        "matched_tokens": sorted(gt_tokens & evidence_tokens),
        "missing_tokens": sorted(gt_tokens - evidence_tokens),
    }


def content_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in stopwords}


def selected_chunk_key(item: dict[str, Any]) -> str:
    return f"{item.get('parsed_source_or_keyframe')}|{item.get('parsed_timestamp')}|{item.get('parsed_youtube')}"


def answer_span_key(item: dict[str, Any]) -> str:
    span = item.get("answer_span") or {}
    return f"{span.get('text')}|{span.get('char_start')}|{span.get('char_end')}"


def answer_span_text(item: dict[str, Any]) -> str:
    span = item.get("answer_span") or {}
    return str(span.get("text") or span.get("answer_span_text") or "")


def transcript_text_from_result(item: dict[str, Any]) -> str:
    result_path = Path(str(item.get("output_folder") or "")) / "result.txt"
    if not result_path.is_file():
        return ""
    text = result_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'^Transcript:\n"(?P<text>.*?)"', text, flags=re.MULTILINE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group("text")).strip()


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "baseline_chunk": row["baseline_chunk"],
            "rerank_chunk": row["rerank_chunk"],
            "baseline_answer_span": row["baseline_answer_span"],
            "rerank_answer_span": row["rerank_answer_span"],
            "baseline_answer_overlap": row["baseline_answer_overlap"],
            "rerank_answer_overlap": row["rerank_answer_overlap"],
        }
        for row in rows
    ]


def render_markdown(comparison: dict[str, Any]) -> str:
    baseline = comparison["metrics"]["baseline"]
    rerank = comparison["metrics"]["rerank"]
    analysis = comparison["transcript_analysis"]
    lines = [
        "# Transcript Answer Rerank Comparison",
        "",
        f"- Baseline output: `{comparison['baseline_dir']}`",
        f"- Rerank output: `{comparison['rerank_dir']}`",
        "",
        comparison["accuracy_note"],
        "",
        "## Accuracy Summary",
        "",
        "| Metric | Current pipeline | + transcript-answer-rerank-top-n 20 |",
        "|---|---:|---:|",
        f"| Routing accuracy | {format_accuracy(baseline['routing_accuracy'])} | {format_accuracy(rerank['routing_accuracy'])} |",
        f"| Transcript accuracy | {format_accuracy(baseline['transcript_accuracy'])} | {format_accuracy(rerank['transcript_accuracy'])} |",
        f"| Visual accuracy | {format_accuracy(baseline['visual_accuracy'])} | {format_accuracy(rerank['visual_accuracy'])} |",
        f"| Overall accuracy | {format_accuracy(baseline['overall_accuracy'])} | {format_accuracy(rerank['overall_accuracy'])} |",
        "",
        "## Transcript Question Changes",
        "",
        f"- Transcript questions: {analysis['transcript_questions']}",
        f"- Selected transcript chunk changed: {analysis['selected_transcript_chunk_changed']}",
        f"- Answer span changed: {analysis['answer_span_changed']}",
        f"- Questions improved: {analysis['questions_improved']}",
        f"- Questions worsened: {analysis['questions_worsened']}",
        "",
        "## Improved Questions",
        "",
    ]
    lines.extend(render_rows(analysis["improved"]))
    lines.extend(["", "## Worsened Questions", ""])
    lines.extend(render_rows(analysis["worsened"]))
    lines.extend(["", "## Changed Transcript Chunks", ""])
    lines.extend(render_rows(analysis["changed_chunks"]))
    lines.extend(["", "## Changed Answer Spans", ""])
    lines.extend(render_rows(analysis["changed_spans"]))
    lines.append("")
    return "\n".join(lines)


def render_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["None."]
    lines = []
    for row in rows:
        lines.extend(
            [
                f"### {row['id']}",
                "",
                f"- Question: {row['question']}",
                f"- Ground truth: {row['ground_truth']}",
                f"- Baseline chunk: `{row['baseline_chunk']}`",
                f"- Rerank chunk: `{row['rerank_chunk']}`",
                f"- Baseline answer span: `{row['baseline_answer_span']}`",
                f"- Rerank answer span: `{row['rerank_answer_span']}`",
                f"- Baseline overlap: `{row['baseline_answer_overlap']}`",
                f"- Rerank overlap: `{row['rerank_answer_overlap']}`",
                "",
            ]
        )
    return lines


def format_accuracy(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def normalize_type(value: Any) -> str:
    return str(value or "").strip().lower()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


if __name__ == "__main__":
    main()
