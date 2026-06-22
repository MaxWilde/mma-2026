#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


QUESTION_COLUMNS = [
    "ID",
    "Type",
    "Question",
    "Ground Truth Answer / Visual Evidence Target",
    "Video",
    "Start Offset",
    "End Offset",
    "Absolute Time",
    "YouTube Start Link",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run query_evidence.py over the CASTLE Day 1 ground-truth table.")
    parser.add_argument("questions_md", type=Path)
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--visual-index-dir", default="artifacts/siglip_index_day1")
    parser.add_argument("--transcript-index-dir", default="artifacts/transcript_index_day1")
    parser.add_argument("--grounding", default="dino", choices=("none", "dino", "dino-siglip-rerank"))
    parser.add_argument("--mixed-top-k", type=int, default=10)
    parser.add_argument("--debug-diversity", action="store_true")
    parser.add_argument("--transcript-answer-rerank-top-n", type=int, default=0)
    parser.add_argument("--transcript-reasoning-top-n", type=int, default=30)
    parser.add_argument("--reasoning-mode", choices=("global", "per_candidate"), default="per_candidate")
    parser.add_argument("--keyword-extraction-top-n", type=int, default=20)
    parser.add_argument("--suggest-steering-terms", action="store_true")
    parser.add_argument("--steering-terms", default="")
    args = parser.parse_args()

    rows = parse_question_table(resolve_path(args.questions_md))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / args.output_root / f"batch_eval_day1_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_question: list[dict[str, Any]] = []
    confusion: Counter[tuple[str, str]] = Counter()

    print(f"Loaded questions: {len(rows)}")
    print(f"Output directory: {output_dir}")

    for index, row in enumerate(rows, start=1):
        result = run_one_question(row, output_dir, args)
        per_question.append(result)
        expected = normalize_type(result["expected_type"])
        predicted = normalize_type(result["predicted_type"])
        confusion[(expected, predicted)] += 1
        print(
            f"[{index}/{len(rows)}] {result['id']} "
            f"expected={expected} predicted={predicted} correct={result['correct_route']} "
            f"folder={result['output_folder']}",
            flush=True,
        )

    write_summary_files(output_dir, per_question, confusion)
    if args.debug_diversity:
        write_diversity_report(output_dir, per_question)
    summary = build_summary(output_dir, per_question)
    print(json.dumps(summary, indent=2))
    print(f"Batch evaluation complete: {output_dir}")


def run_one_question(row: dict[str, str], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    question_id = safe_name(row["ID"])
    expected_type = normalize_type(row["Type"])
    command = [
        sys.executable,
        "scripts/query_evidence.py",
        row["Question"],
        "--visual-index-dir",
        args.visual_index_dir,
        "--transcript-index-dir",
        args.transcript_index_dir,
        "--grounding",
        args.grounding,
        "--debug-router",
        "--include-transcript-heatmap",
        "--mixed-top-k",
        str(args.mixed_top_k),
    ]
    if args.debug_diversity:
        command.append("--debug-diversity")
    if args.transcript_answer_rerank_top_n > 0:
        command.extend(["--transcript-answer-rerank-top-n", str(args.transcript_answer_rerank_top_n)])
    command.extend(["--transcript-reasoning-top-n", str(args.transcript_reasoning_top_n)])
    command.extend(["--reasoning-mode", args.reasoning_mode])
    command.extend(["--keyword-extraction-top-n", str(args.keyword_extraction_top_n)])
    if args.steering_terms:
        command.extend(["--steering-terms", args.steering_terms])

    stdout = ""
    stderr = ""
    returncode = 0
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    except Exception as exc:
        stderr = f"{type(exc).__name__}: {exc}"
        returncode = 1

    predicted_type = parse_predicted_type(stdout) if returncode == 0 else "ERROR"
    if predicted_type not in {"visual", "transcript"}:
        predicted_type = "ERROR"
    route_folder = predicted_type if predicted_type in {"visual", "transcript"} else "error"
    question_dir = output_dir / route_folder / question_id
    question_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_query_output(stdout)
    copied_files = copy_evidence_files(parsed, question_dir, predicted_type)
    heatmap_path = write_heatmap_if_present(parsed, question_dir, predicted_type)

    result_text = stdout
    if stderr.strip():
        result_text += "\n\nSTDERR:\n" + stderr
    (question_dir / "result.txt").write_text(result_text, encoding="utf-8")

    correct_route = predicted_type == expected_type
    result = {
        "id": row["ID"],
        "question": row["Question"],
        "expected_type": expected_type,
        "predicted_type": predicted_type,
        "correct_route": correct_route,
        "ground_truth_answer_or_visual_target": row["Ground Truth Answer / Visual Evidence Target"],
        "video": row["Video"],
        "start_offset": row["Start Offset"],
        "end_offset": row["End Offset"],
        "absolute_time": row["Absolute Time"],
        "ground_truth_youtube": row["YouTube Start Link"],
        "parsed_source_or_keyframe": parsed.get("source") or parsed.get("keyframe"),
        "parsed_timestamp": parsed.get("timestamp"),
        "parsed_youtube": parsed.get("youtube"),
        "parsed_evidence_confidence": parsed.get("evidence_confidence"),
        "parsed_evidence_confidence_percent": parsed.get("evidence_confidence_percent"),
        "parsed_visual_retrieval_confidence": parsed.get("visual_retrieval_confidence"),
        "parsed_transcript_retrieval_confidence": parsed.get("transcript_retrieval_confidence"),
        "parsed_answer_confidence": parsed.get("answer_confidence"),
        "parsed_grounding_confidence": parsed.get("grounding_confidence"),
        "router_debug": parsed.get("router_debug", {}),
        "answer_span": parsed.get("answer_span"),
        "answer_candidates": parsed.get("answer_candidates"),
        "answer_rerank_candidates": parsed.get("answer_rerank_candidates"),
        "transcript_reasoning_answer": parsed.get("transcript_reasoning_answer"),
        "steering_suggestions": parsed.get("steering_suggestions"),
        "steering_retrieval": parsed.get("steering_retrieval"),
        "mixed_evidence": parsed.get("mixed_evidence"),
        "mixed_diversity_debug": parsed.get("mixed_diversity_debug"),
        "heatmap_path": str(heatmap_path) if heatmap_path else None,
        "copied_evidence_files": copied_files,
        "returncode": returncode,
        "stderr": stderr,
        "output_folder": str(question_dir),
    }

    write_question_markdown(question_dir / "question.md", row, predicted_type, correct_route)
    (question_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def parse_question_table(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, str]] = []
    in_questions = False
    header: list[str] | None = None

    for line in lines:
        stripped = line.strip()
        if stripped == "## Questions":
            in_questions = True
            continue
        if not in_questions or not stripped.startswith("|"):
            continue
        cells = split_markdown_row(stripped)
        if not cells:
            continue
        if cells == QUESTION_COLUMNS:
            header = cells
            continue
        if header is None:
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        if re.match(r"^[VT]\d+", row.get("ID", "")):
            rows.append(row)
    if not rows:
        raise SystemExit(f"No question rows parsed from {path}")
    return rows


def split_markdown_row(line: str) -> list[str]:
    trimmed = line.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|"):
        trimmed = trimmed[:-1]
    return [cell.strip().replace("<br>", "\n") for cell in trimmed.split("|")]


def parse_query_output(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {
        "predicted_type": parse_predicted_type(stdout),
        "source": first_match(stdout, r"^Source:\s*(.+)$"),
        "keyframe": first_match(stdout, r"^Keyframe:\s*(.+)$"),
        "timestamp": first_match(stdout, r"^Timestamp:\s*(.+)$"),
        "youtube": first_match(stdout, r"^YouTube:\s*(.+)$"),
        "bbox_image": first_match(stdout, r"^Bounding box image:\s*(.+)$"),
        "candidate_boxes_image": first_match(stdout, r"^Candidate boxes image:\s*(.+)$"),
        "evidence_confidence": parse_number_or_string(first_match(stdout, r"^Evidence confidence:\s*(.+)$") or ""),
        "evidence_confidence_percent": parse_number_or_string(
            first_match(stdout, r"^Evidence confidence percent:\s*(.+)$") or ""
        ),
        "visual_retrieval_confidence": parse_number_or_string(
            first_match(stdout, r"^Visual retrieval confidence:\s*(.+)$") or ""
        ),
        "visual_retrieval_confidence_percent": parse_number_or_string(
            first_match(stdout, r"^Visual retrieval confidence percent:\s*(.+)$") or ""
        ),
        "transcript_retrieval_confidence": parse_number_or_string(
            first_match(stdout, r"^Transcript retrieval confidence:\s*(.+)$") or ""
        ),
        "transcript_retrieval_confidence_percent": parse_number_or_string(
            first_match(stdout, r"^Transcript retrieval confidence percent:\s*(.+)$") or ""
        ),
        "answer_confidence": parse_number_or_string(first_match(stdout, r"^Answer confidence:\s*(.+)$") or ""),
        "answer_confidence_percent": parse_number_or_string(
            first_match(stdout, r"^Answer confidence percent:\s*(.+)$") or ""
        ),
        "grounding_confidence": parse_number_or_string(first_match(stdout, r"^Grounding confidence:\s*(.+)$") or ""),
        "grounding_confidence_percent": parse_number_or_string(
            first_match(stdout, r"^Grounding confidence percent:\s*(.+)$") or ""
        ),
    }
    parsed["router_debug"] = parse_router_debug(stdout)
    parsed["answer_span"] = parse_json_after_label(stdout, "Answer span JSON:")
    parsed["answer_candidates"] = parse_json_after_label(stdout, "Answer candidates JSON:")
    parsed["answer_rerank_candidates"] = parse_json_after_label(stdout, "Answer rerank candidates JSON:")
    parsed["transcript_reasoning_answer"] = parse_json_after_label(stdout, "Transcript reasoning answer JSON:")
    parsed["steering_suggestions"] = parse_json_after_label(stdout, "Steering suggestions JSON:")
    parsed["steering_retrieval"] = parse_json_after_label(stdout, "Steering retrieval JSON:")
    parsed["transcript_heatmap"] = parse_json_after_label(stdout, "Transcript heatmap JSON:")
    parsed["mixed_evidence"] = parse_json_after_label(stdout, "Mixed evidence JSON:")
    parsed["mixed_diversity_debug"] = parse_json_after_label(stdout, "Mixed diversity debug JSON:")
    return parsed


def parse_predicted_type(stdout: str) -> str:
    match = re.search(r"^Evidence type:\s*(visual|transcript)\s*$", stdout, flags=re.MULTILINE | re.IGNORECASE)
    return match.group(1).lower() if match else "ERROR"


def first_match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    value = match.group(1).strip() if match else None
    return value or None


def parse_router_debug(stdout: str) -> dict[str, Any]:
    keys = {
        "heuristic visual score": "heuristic_visual_score",
        "heuristic transcript score": "heuristic_transcript_score",
        "top visual score": "top_visual_score",
        "top transcript score": "top_transcript_score",
        "combined visual score": "combined_visual_score",
        "combined transcript score": "combined_transcript_score",
        "router margin": "router_margin",
        "router confidence": "router_confidence",
        "router confidence percent": "router_confidence_percent",
        "router second choice": "router_second_choice",
        "chosen route": "chosen_route",
        "reason": "reason",
        "visual retrieval runtime": "visual_retrieval_runtime",
    }
    debug: dict[str, Any] = {}
    for label, key in keys.items():
        value = first_match(stdout, rf"^{re.escape(label)}:\s*(.+)$")
        if value is None:
            continue
        debug[key] = parse_number_or_string(value)
    return debug


def parse_json_after_label(stdout: str, label: str) -> Any | None:
    lines = stdout.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == label and index + 1 < len(lines):
            payload = lines[index + 1].strip()
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return None
    return None


def copy_evidence_files(parsed: dict[str, Any], question_dir: Path, predicted_type: str) -> list[str]:
    copied: list[str] = []
    if predicted_type != "visual":
        return copied
    for key, output_name in (
        ("bbox_image", "evidence_image"),
        ("keyframe", "evidence_image"),
        ("candidate_boxes_image", "candidate_boxes_image"),
    ):
        source_value = parsed.get(key)
        if not source_value:
            continue
        source = resolve_path(Path(source_value))
        if not source.is_file():
            continue
        suffix = source.suffix or ".jpg"
        destination = question_dir / f"{output_name}{suffix}"
        if destination.exists() and output_name == "evidence_image":
            continue
        shutil.copy2(source, destination)
        copied.append(str(destination))
    return copied


def write_heatmap_if_present(parsed: dict[str, Any], question_dir: Path, predicted_type: str) -> Path | None:
    heatmap = parsed.get("transcript_heatmap")
    if predicted_type != "transcript" or heatmap is None:
        return None
    path = question_dir / "heatmap.json"
    path.write_text(json.dumps(heatmap, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_question_markdown(path: Path, row: dict[str, str], predicted_type: str, correct_route: bool) -> None:
    content = [
        f"# {row['ID']}",
        "",
        f"- ID: {row['ID']}",
        f"- Question: {row['Question']}",
        f"- Expected type: {normalize_type(row['Type'])}",
        f"- Predicted type: {predicted_type}",
        f"- Route correct: {str(correct_route).lower()}",
        f"- Ground truth answer / visual target: {row['Ground Truth Answer / Visual Evidence Target']}",
        f"- Ground truth video: {row['Video']}",
        f"- Ground truth time window: {row['Start Offset']}–{row['End Offset']} ({row['Absolute Time']})",
        f"- Ground truth YouTube: {row['YouTube Start Link']}",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def write_summary_files(output_dir: Path, per_question: list[dict[str, Any]], confusion: Counter[tuple[str, str]]) -> None:
    summary_rows = [
        {
            "id": item["id"],
            "expected_type": item["expected_type"],
            "predicted_type": item["predicted_type"],
            "correct_route": item["correct_route"],
            "question": item["question"],
            "ground_truth_answer": item["ground_truth_answer_or_visual_target"],
            "video": item["video"],
            "absolute_time": item["absolute_time"],
            "ground_truth_youtube": item["ground_truth_youtube"],
            "parsed_source_or_keyframe": item["parsed_source_or_keyframe"],
            "parsed_timestamp": item["parsed_timestamp"],
            "parsed_youtube": item["parsed_youtube"],
            "output_folder": item["output_folder"],
        }
        for item in per_question
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = build_summary(output_dir, per_question)
    summary["per_question"] = per_question
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    matrix_rows = []
    for expected in ("visual", "transcript"):
        for predicted in ("visual", "transcript", "error"):
            matrix_rows.append({"expected": expected, "predicted": predicted, "count": confusion[(expected, predicted)]})
    with (output_dir / "routing_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["expected", "predicted", "count"])
        writer.writeheader()
        writer.writerows(matrix_rows)


def write_diversity_report(output_dir: Path, per_question: list[dict[str, Any]]) -> None:
    records = [diversity_record(item) for item in per_question]
    summary = {
        "output_dir": str(output_dir),
        "questions": len(records),
        "average_candidates_before_suppression": average(
            record["before_suppression"] for record in records
        ),
        "average_candidates_after_suppression": average(
            record["after_suppression"] for record in records
        ),
        "average_suppression_count": average(record["suppression_count"] for record in records),
        "transcript_duplicates_removed": sum(record["transcript_duplicates_removed"] for record in records),
        "visual_duplicates_removed": sum(record["visual_duplicates_removed"] for record in records),
        "cross_modal_duplicates_removed": sum(record["cross_modal_duplicates_removed"] for record in records),
        "example_suppressed_transcript_duplicates": collect_suppressed_examples(records, "transcript"),
        "example_suppressed_visual_duplicates": collect_suppressed_examples(records, "visual"),
        "per_question": records,
    }
    (output_dir / "diversity_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "diversity_report.md").write_text(render_diversity_markdown(summary), encoding="utf-8")


def diversity_record(item: dict[str, Any]) -> dict[str, Any]:
    debug = item.get("mixed_diversity_debug") or {}
    counts = debug.get("suppressed_type_counts") or {}
    mixed_counts = Counter(
        str(candidate.get("evidence_type"))
        for candidate in (item.get("mixed_evidence") or [])
        if isinstance(candidate, dict)
    )
    return {
        "id": item["id"],
        "question": item["question"],
        "route": item["predicted_type"],
        "top10_modality_distribution": {
            "visual": mixed_counts.get("visual", 0),
            "transcript": mixed_counts.get("transcript", 0),
        },
        "before_suppression": int(debug.get("before_suppression") or 0),
        "after_suppression": int(debug.get("after_suppression") or 0),
        "suppression_count": int(debug.get("suppressed_count") or 0),
        "transcript_duplicates_removed": int(counts.get("transcript") or 0),
        "visual_duplicates_removed": int(counts.get("visual") or 0),
        "cross_modal_duplicates_removed": int(counts.get("cross_modal") or 0),
        "suppressed_examples": debug.get("suppressed_examples") or [],
    }


def render_diversity_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Mixed Evidence Diversity Report",
        "",
        "This report analyzes post-scoring duplicate suppression for mixed top-k evidence. "
        "It does not change retrieval, routing, timestamps, QA, grounding, confidence, indexes, or ranking formulas.",
        "",
        "## Aggregate Metrics",
        "",
        f"- Questions: {summary['questions']}",
        f"- Average candidates before suppression: {summary['average_candidates_before_suppression']:.2f}",
        f"- Average candidates after suppression: {summary['average_candidates_after_suppression']:.2f}",
        f"- Average suppression count: {summary['average_suppression_count']:.2f}",
        f"- Transcript duplicates removed: {summary['transcript_duplicates_removed']}",
        f"- Visual duplicates removed: {summary['visual_duplicates_removed']}",
        f"- Cross-modal duplicates removed: {summary['cross_modal_duplicates_removed']}",
        "",
        "## Example Suppressed Transcript Duplicates",
        "",
    ]
    lines.extend(render_suppressed_examples(summary["example_suppressed_transcript_duplicates"]))
    lines.extend(["", "## Example Suppressed Visual Duplicates", ""])
    lines.extend(render_suppressed_examples(summary["example_suppressed_visual_duplicates"]))
    lines.extend(
        [
            "",
            "## Per-Question Summary",
            "",
            "| ID | Route | Top10 V/T | Before | After | Suppressed | Transcript dupes | Visual dupes |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in summary["per_question"]:
        top10 = record["top10_modality_distribution"]
        lines.append(
            "| {id} | {route} | {visual}/{transcript} | {before} | {after} | {suppressed} | {td} | {vd} |".format(
                id=record["id"],
                route=record["route"],
                visual=top10["visual"],
                transcript=top10["transcript"],
                before=record["before_suppression"],
                after=record["after_suppression"],
                suppressed=record["suppression_count"],
                td=record["transcript_duplicates_removed"],
                vd=record["visual_duplicates_removed"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_suppressed_examples(examples: list[dict[str, Any]]) -> list[str]:
    if not examples:
        return ["No examples found."]
    lines = []
    for example in examples[:10]:
        lines.extend(
            [
                f"- Question `{example.get('question_id')}`: {example.get('reason')}",
                f"  - suppressed: `{example.get('suppressed_type')}` {example.get('suppressed_timestamp')} {example.get('suppressed_id')}",
                f"  - kept: `{example.get('kept_type')}` {example.get('kept_timestamp')} {example.get('kept_id')}",
            ]
        )
    return lines


def collect_suppressed_examples(records: list[dict[str, Any]], evidence_type: str, limit: int = 10) -> list[dict[str, Any]]:
    examples = []
    for record in records:
        for example in record["suppressed_examples"]:
            if example.get("suppressed_type") == evidence_type and example.get("kept_type") == evidence_type:
                enriched = dict(example)
                enriched["question_id"] = record["id"]
                enriched["question"] = record["question"]
                examples.append(enriched)
                if len(examples) >= limit:
                    return examples
    return examples


def average(values: Any) -> float:
    parsed = [float(value) for value in values]
    return sum(parsed) / len(parsed) if parsed else 0.0


def build_summary(output_dir: Path, per_question: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(per_question)
    visual_items = [item for item in per_question if item["expected_type"] == "visual"]
    transcript_items = [item for item in per_question if item["expected_type"] == "transcript"]
    correct = sum(1 for item in per_question if item["correct_route"])
    visual_correct = sum(1 for item in visual_items if item["correct_route"])
    transcript_correct = sum(1 for item in transcript_items if item["correct_route"])
    return {
        "output_dir": str(output_dir),
        "total_questions": total,
        "visual_questions": len(visual_items),
        "transcript_questions": len(transcript_items),
        "correct_routes": correct,
        "routing_accuracy": safe_ratio(correct, total),
        "visual_correct": visual_correct,
        "visual_total": len(visual_items),
        "visual_accuracy": safe_ratio(visual_correct, len(visual_items)),
        "transcript_correct": transcript_correct,
        "transcript_total": len(transcript_items),
        "transcript_accuracy": safe_ratio(transcript_correct, len(transcript_items)),
    }


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def normalize_type(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "visual":
        return "visual"
    if normalized == "transcript":
        return "transcript"
    if normalized == "error":
        return "error"
    return normalized or "error"


def safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "unknown"


def parse_number_or_string(value: str) -> float | str:
    stripped = value.strip()
    if stripped.endswith("s"):
        stripped = stripped[:-1]
    try:
        return float(stripped)
    except ValueError:
        return value


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
