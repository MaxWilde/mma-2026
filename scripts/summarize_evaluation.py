#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_LOGS = Path.home() / "X_evaluation" / "logs"
DEFAULT_GROUND_TRUTH = Path.home() / "X_evaluation" / "evaluation_ground_truth.json"
DEFAULT_OUTPUT = Path.home() / "X_evaluation" / "generated"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_task(record: dict[str, Any], truth: dict[str, Any] | None) -> dict[str, Any]:
    task = record.get("task") or {}
    summary = record.get("automatic_summary") or {}
    annotation = record.get("expert_annotation") or {}
    counts = summary.get("event_counts") or {}
    return {
        "run_id": record.get("run_id"),
        "session_id": record.get("session_id"),
        "task_id": task.get("task_id"),
        "category": task.get("category"),
        "source": task.get("source"),
        "prompt": task.get("prompt"),
        "ground_truth_available": task.get("ground_truth_available"),
        "reference_answer": (truth or {}).get("answer", ""),
        "reference_anchor": (truth or {}).get("anchor", ""),
        "status": record.get("status"),
        "started_at": record.get("started_at"),
        "ended_at": record.get("ended_at"),
        "duration_seconds": record.get("duration_seconds"),
        "queries_submitted": summary.get("queries_submitted", 0),
        "result_selections": summary.get("result_selections", 0),
        "unique_results_selected": summary.get("unique_results_selected", 0),
        "keyword_clicks": summary.get("keyword_clicks", 0),
        "youtube_links_opened": summary.get("youtube_links_opened", 0),
        "grounding_requests": summary.get("grounding_requests", 0),
        "transcript_highlight_requests": summary.get(
            "transcript_highlight_requests", 0
        ),
        "feedback_actions": summary.get("feedback_actions", 0),
        "rocchio_refinements": summary.get("rocchio_refinements", 0),
        "filter_changes": summary.get("filter_changes", 0),
        "chart_hovers": summary.get("chart_hovers", 0),
        "search_failures": counts.get("search_failed", 0),
        "outcome": annotation.get("outcome"),
        "answer_or_conclusion": annotation.get("answer_or_conclusion"),
        "evidence_verified": annotation.get("evidence_verified"),
        "evidence_correct": annotation.get("evidence_correct"),
        "router_appropriate_1_to_5": annotation.get("router_appropriate_1_to_5"),
        "provenance_clear_1_to_5": annotation.get("provenance_clear_1_to_5"),
        "grounding_correct": annotation.get("grounding_correct"),
        "transcript_highlight_useful_1_to_5": annotation.get(
            "transcript_highlight_useful_1_to_5"
        ),
        "mental_effort_1_to_7": annotation.get("mental_effort_1_to_7"),
        "confidence_in_conclusion_0_to_100": annotation.get(
            "confidence_in_conclusion_0_to_100"
        ),
        "main_problem": annotation.get("main_problem"),
        "notes": annotation.get("notes"),
    }


def numeric(values: list[Any]) -> list[float]:
    output = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            output.append(float(value))
        except (TypeError, ValueError):
            continue
    return output


def study_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    grounded = [row for row in completed if row.get("ground_truth_available")]
    annotated = [row for row in completed if row.get("outcome")]
    successes = [row for row in grounded if row.get("outcome") == "success"]
    evidence_correct = [
        row for row in grounded if row.get("evidence_correct") is True
    ]
    durations = numeric([row.get("duration_seconds") for row in completed])
    queries = numeric([row.get("queries_submitted") for row in completed])
    mental_effort = numeric([row.get("mental_effort_1_to_7") for row in annotated])
    provenance = numeric([row.get("provenance_clear_1_to_5") for row in annotated])
    return {
        "task_logs": len(rows),
        "completed_tasks": len(completed),
        "annotated_tasks": len(annotated),
        "grounded_tasks": len(grounded),
        "grounded_success_rate": (
            len(successes) / len(grounded) if grounded else None
        ),
        "grounded_evidence_correct_rate": (
            len(evidence_correct) / len(grounded) if grounded else None
        ),
        "median_duration_seconds": median(durations) if durations else None,
        "median_queries_submitted": median(queries) if queries else None,
        "median_mental_effort": median(mental_effort) if mental_effort else None,
        "median_provenance_clarity": median(provenance) if provenance else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge CASTLE evaluation task logs into CSV and JSON summaries."
    )
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    truth_payload = load_json(args.ground_truth) if args.ground_truth.exists() else {}
    truth_by_id = {
        item["task_id"]: item for item in truth_payload.get("tasks", [])
    }
    records = [
        load_json(path)
        for path in sorted(args.logs.glob("*.json"))
        if path.is_file()
    ]
    rows = [
        flatten_task(
            record,
            truth_by_id.get((record.get("task") or {}).get("task_id")),
        )
        for record in records
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "task_metrics.csv"
    json_path = args.output_dir / "evaluation_summary.json"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    payload = {"summary": study_summary(rows), "tasks": rows}
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
