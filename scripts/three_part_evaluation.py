#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CONTROL = Path.home() / "X_evaluation" / "runtime" / "evaluation_control.json"
DEFAULT_OUTPUT = Path.home() / "X_evaluation" / "session_logs"

PART_DEFINITIONS = {
    1: {
        "name": "known_answer_known_evidence",
        "label": "Part 1",
        "planned_minutes": 15,
        "knowledge_condition": (
            "The participant knows the ground-truth answer and the corresponding evidence location."
        ),
    },
    2: {
        "name": "known_answer_unknown_evidence",
        "label": "Part 2",
        "planned_minutes": 15,
        "knowledge_condition": (
            "The participant knows the ground-truth answer but does not know where the evidence is."
        ),
    },
    3: {
        "name": "unknown_answer_exploration",
        "label": "Part 3",
        "planned_minutes": 15,
        "knowledge_condition": (
            "The participant does not know the answer and explores the archive without a predefined evidence target."
        ),
    },
}


def now() -> datetime:
    return datetime.now().astimezone()


def iso_now() -> str:
    return now().isoformat(timespec="milliseconds")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def control_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"Evaluation control file does not exist: {path}\n"
            "Enable evaluation mode and start a task in the dashboard first."
        )
    return read_json(path)


def active_log_path(control: dict[str, Any]) -> Path:
    active = control.get("active_task") or {}
    path = active.get("path")
    if not path:
        raise SystemExit(
            "No dashboard evaluation task is active. In the dashboard, enter the "
            "session prompt and press 'Start evaluation task' first."
        )
    return Path(path)


def completed_log_path(control: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    path = control.get("last_completed_task")
    if not path:
        raise SystemExit(
            "No completed dashboard task was found. Stop the task in the dashboard "
            "before running finalize, or provide --log."
        )
    return Path(path)


def append_event(record: dict[str, Any], event: str, data: dict[str, Any]) -> None:
    events = record.setdefault("events", [])
    events.append(
        {
            "sequence": len(events) + 1,
            "timestamp": iso_now(),
            "event": event,
            "data": data,
        }
    )


def current_part(record: dict[str, Any]) -> int | None:
    protocol = record.get("three_part_protocol") or {}
    value = protocol.get("current_part")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def begin(args: argparse.Namespace) -> None:
    control = control_state(args.control)
    path = active_log_path(control)
    record = read_json(path)
    if record.get("three_part_protocol"):
        raise SystemExit("This dashboard task has already been initialized as a three-part session.")

    started_at = iso_now()
    record["schema_version"] = max(2, int(record.get("schema_version", 1)))
    record["three_part_protocol"] = {
        "protocol_version": 1,
        "design": "think_aloud_3x15_minutes",
        "participant_id": args.participant_id,
        "participant_role": args.participant_role,
        "planned_duration_minutes": 45,
        "think_aloud": True,
        "independent_use_without_external_input": True,
        "audio_recording_reference": args.audio_recording or "",
        "session_initialized_at": started_at,
        "current_part": 1,
        "parts": [PART_DEFINITIONS[index] for index in (1, 2, 3)],
    }
    record["part_annotations"] = {
        str(index): part_annotation_template(index) for index in (1, 2, 3)
    }
    record["post_session_annotation"] = post_session_template()
    append_event(
        record,
        "procedure_part_started",
        {"part": 1, **PART_DEFINITIONS[1]},
    )
    atomic_write(path, record)
    print(f"Three-part session initialized for {args.participant_id}.")
    print(f"Part 1 started. Active log: {path}")


def change_part(args: argparse.Namespace) -> None:
    control = control_state(args.control)
    path = active_log_path(control)
    record = read_json(path)
    if not record.get("three_part_protocol"):
        raise SystemExit("Run the begin command before changing parts.")

    previous = current_part(record)
    target = int(args.part)
    if previous == target:
        print(f"Part {target} is already active.")
        return
    if previous is not None:
        append_event(
            record,
            "procedure_part_ended",
            {"part": previous, **PART_DEFINITIONS[previous]},
        )
    record["three_part_protocol"]["current_part"] = target
    append_event(
        record,
        "procedure_part_started",
        {"part": target, **PART_DEFINITIONS[target]},
    )
    atomic_write(path, record)
    print(f"Part {target} started: {PART_DEFINITIONS[target]['name']}")


def add_comment(args: argparse.Namespace) -> None:
    control = control_state(args.control)
    path = active_log_path(control)
    record = read_json(path)
    part = current_part(record)
    if part is None:
        raise SystemExit("Run the begin command before recording comments.")
    append_event(
        record,
        "think_aloud_comment",
        {
            "part": part,
            "text": args.text.strip(),
            "source": "manual_transcription",
        },
    )
    atomic_write(path, record)
    print(f"Comment saved under Part {part}.")


def status(args: argparse.Namespace) -> None:
    control = control_state(args.control)
    active = control.get("active_task")
    if not active:
        print(json.dumps({"active": False, "last_completed_task": control.get("last_completed_task")}, indent=2))
        return
    path = Path(active["path"])
    record = read_json(path)
    protocol = record.get("three_part_protocol") or {}
    print(
        json.dumps(
            {
                "active": True,
                "run_id": record.get("run_id"),
                "participant_id": protocol.get("participant_id"),
                "current_part": protocol.get("current_part"),
                "started_at": record.get("started_at"),
                "log_path": str(path),
            },
            indent=2,
        )
    )


def finalize(args: argparse.Namespace) -> None:
    control = control_state(args.control)
    source_path = completed_log_path(control, args.log)
    record = read_json(source_path)
    protocol = record.get("three_part_protocol")
    if not protocol:
        raise SystemExit(
            "The selected log is not a three-part session. It must be initialized "
            "with the begin command while the dashboard task is active."
        )

    events = record.get("events") or []
    parts = partition_events(events)
    part_records = []
    for index in (1, 2, 3):
        part_events = parts[index]
        part_records.append(
            {
                "part": index,
                **PART_DEFINITIONS[index],
                "started_at": first_event_time(part_events, "procedure_part_started"),
                "ended_at": last_part_time(part_events),
                "duration_seconds": part_duration(part_events),
                "automatic_summary": summarize_events(part_events),
                "events": part_events,
                "participant_annotation": (
                    record.get("part_annotations") or {}
                ).get(str(index), part_annotation_template(index)),
            }
        )

    session = {
        "schema_version": 1,
        "record_type": "three_part_evaluation_session",
        "run_id": record.get("run_id"),
        "session_id": record.get("session_id"),
        "participant": {
            "participant_id": protocol.get("participant_id"),
            "role": protocol.get("participant_role"),
            "familiar_with_castle_dataset": True,
        },
        "experimental_setup": {
            "think_aloud": True,
            "planned_duration_minutes": 45,
            "structure": "3 x 15 minutes",
            "independent_use_without_external_input": True,
            "audio_recording_reference": protocol.get("audio_recording_reference", ""),
        },
        "status": record.get("status"),
        "started_at": record.get("started_at"),
        "ended_at": record.get("ended_at"),
        "duration_seconds": record.get("duration_seconds"),
        "parts": part_records,
        "whole_session_summary": summarize_events(events),
        "final_state": record.get("final_state") or {},
        "post_session_annotation": record.get(
            "post_session_annotation", post_session_template()
        ),
        "source_dashboard_log": str(source_path),
        "all_events": events,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{record.get('run_id', source_path.stem)}-session.json"
    atomic_write(output_path, session)
    print(f"Wrote consolidated three-part session: {output_path}")
    print("Complete participant_annotation inside each part and post_session_annotation.")


def partition_events(events: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    output = {1: [], 2: [], 3: []}
    active_part: int | None = None
    for event in events:
        if event.get("event") == "procedure_part_started":
            try:
                active_part = int((event.get("data") or {}).get("part"))
            except (TypeError, ValueError):
                active_part = None
        if active_part in output:
            output[active_part].append(event)
        if event.get("event") == "procedure_part_ended":
            active_part = None
    return output


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(item.get("event", "")) for item in events)
    queries = [
        (item.get("data") or {}).get("query")
        for item in events
        if item.get("event") == "search_submitted"
    ]
    selected = [
        (item.get("data") or {}).get("result_id")
        for item in events
        if item.get("event") == "result_selected"
    ]
    comments = [
        (item.get("data") or {}).get("text")
        for item in events
        if item.get("event") == "think_aloud_comment"
    ]
    return {
        "event_counts": dict(sorted(counts.items())),
        "queries_submitted": len(queries),
        "query_history": [value for value in queries if value],
        "result_selections": len(selected),
        "unique_results_selected": len({value for value in selected if value}),
        "keyword_clicks": counts.get("keyword_clicked", 0),
        "youtube_links_opened": counts.get("youtube_opened", 0),
        "grounding_requests": counts.get("grounding_requested", 0),
        "transcript_highlight_requests": counts.get("transcript_highlight_requested", 0),
        "feedback_actions": counts.get("feedback_changed", 0),
        "rocchio_refinements": counts.get("rocchio_refinement_completed", 0),
        "filter_changes": counts.get("filters_changed", 0),
        "chart_hovers": counts.get("chart_hovered", 0),
        "think_aloud_comments": [value for value in comments if value],
    }


def first_event_time(events: list[dict[str, Any]], event_name: str) -> str | None:
    for event in events:
        if event.get("event") == event_name:
            return event.get("timestamp")
    return events[0].get("timestamp") if events else None


def last_part_time(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        if event.get("event") in {"procedure_part_ended", "task_stopped"}:
            return event.get("timestamp")
    return events[-1].get("timestamp") if events else None


def part_duration(events: list[dict[str, Any]]) -> float | None:
    if not events:
        return None
    try:
        start = datetime.fromisoformat(str(events[0]["timestamp"]))
        end = datetime.fromisoformat(str(events[-1]["timestamp"]))
    except (KeyError, TypeError, ValueError):
        return None
    return round(max(0.0, (end - start).total_seconds()), 3)


def part_annotation_template(part: int) -> dict[str, Any]:
    return {
        "part": part,
        "prompts_attempted": [],
        "answers_or_conclusions": [],
        "evidence_found": [],
        "evidence_correct": None if part != 3 else "not_applicable_without_reference",
        "effectiveness_notes": {
            "evidence_types_and_router": "",
            "visual_retrieval_quality": "",
            "transcript_retrieval_quality": "",
            "feedback_loop": "",
            "youtube_charts_and_additional_data": "",
        },
        "usability_notes": {
            "search_panel": "",
            "evidence_viewer": "",
            "ranked_results": "",
            "feedback_loop": "",
        },
        "notable_quotes_or_think_aloud_comments": [],
        "example_for_report": {
            "title": "",
            "query_before": "",
            "interaction_or_refinement": "",
            "query_after": "",
            "evidence_before": "",
            "evidence_after": "",
            "figure_or_screenshot_path": "",
            "interpretation": "",
        },
        "main_problem": "",
        "notes": "",
    }


def post_session_template() -> dict[str, Any]:
    return {
        "overall_experience_summary": "",
        "effectiveness": {
            "evidence_types_and_router": "",
            "visual_retrieval_quality": "",
            "transcript_retrieval_quality": "",
            "feedback_loop": "",
            "youtube_charts_and_additional_data": "",
        },
        "usability": {
            "search_panel": "",
            "evidence_viewer": "",
            "ranked_results": "",
            "feedback_loop": "",
        },
        "most_successful_interaction": "",
        "most_important_failure": "",
        "suggested_improvements": [],
        "overall_effectiveness_1_to_5": None,
        "overall_usability_1_to_5": None,
        "mental_effort_1_to_7": None,
        "notes": "",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add three-part think-aloud protocol markers to the existing CASTLE "
            "dashboard evaluation log and consolidate the session afterward."
        )
    )
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    begin_parser = subparsers.add_parser("begin")
    begin_parser.add_argument("--participant-id", required=True)
    begin_parser.add_argument("--participant-role", default="author/evaluator")
    begin_parser.add_argument("--audio-recording")
    begin_parser.set_defaults(func=begin)

    part_parser = subparsers.add_parser("part")
    part_parser.add_argument("part", choices=("1", "2", "3"))
    part_parser.set_defaults(func=change_part)

    comment_parser = subparsers.add_parser("comment")
    comment_parser.add_argument("text")
    comment_parser.set_defaults(func=add_comment)

    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(func=status)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--log", type=Path)
    finalize_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    finalize_parser.set_defaults(func=finalize)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
