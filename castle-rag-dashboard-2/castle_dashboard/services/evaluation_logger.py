from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> datetime:
    return datetime.now().astimezone()


def _iso_now() -> str:
    return _now().isoformat(timespec="milliseconds")


def _safe_slug(value: str, fallback: str = "task") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (slug[:48] or fallback).strip("-")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


class EvaluationLogger:
    """Optional single-evaluator task logger controlled by an external file.

    The external controller enables or disables evaluation mode. Dashboard
    callbacks call ``log_event`` freely; events are ignored unless a task is
    active. Each completed task is a self-contained JSON file with interaction
    events, an automatic summary, and an empty expert annotation template.
    """

    def __init__(self) -> None:
        home = Path.home()
        self.control_path = Path(
            os.getenv(
                "CASTLE_EVALUATION_CONTROL",
                str(home / "X_evaluation" / "runtime" / "evaluation_control.json"),
            )
        )
        self.default_output_dir = Path(
            os.getenv(
                "CASTLE_EVALUATION_OUTPUT_DIR",
                str(home / "X_evaluation" / "logs"),
            )
        )
        self.tasks_path = Path(
            os.getenv(
                "CASTLE_EVALUATION_TASKS",
                str(home / "X_evaluation" / "evaluation_tasks.json"),
            )
        )
        self._lock = threading.RLock()
        self._last_event_signature: tuple[str, str] | None = None
        self._last_event_at: datetime | None = None

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._read_control()

    def is_enabled(self) -> bool:
        return bool(self.state().get("enabled"))

    def is_active(self) -> bool:
        return bool(self.state().get("enabled") and self.state().get("active_task"))

    def start_task(self, prompt: str) -> dict[str, Any]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Enter a task prompt before starting evaluation.")

        with self._lock:
            control = self._read_control()
            if not control.get("enabled"):
                raise RuntimeError("Evaluation mode is not enabled by the external controller.")
            if control.get("active_task"):
                raise RuntimeError("An evaluation task is already active.")

            task_metadata = self._match_task(prompt)
            task_id = str(task_metadata.get("task_id") or "").strip()
            if not task_id:
                task_id = f"custom-{_now().strftime('%Y%m%d-%H%M%S')}"
            run_id = f"{task_id}-{uuid.uuid4().hex[:8]}"
            output_dir = Path(control.get("output_dir") or self.default_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            task_path = output_dir / f"{_safe_slug(run_id)}.json"

            started_at = _iso_now()
            record = {
                "schema_version": 1,
                "run_id": run_id,
                "session_id": control.get("session_id"),
                "task": {
                    "task_id": task_id,
                    "prompt": prompt,
                    "category": task_metadata.get("category", "unregistered"),
                    "source": task_metadata.get("source", "custom"),
                    "ground_truth_available": bool(
                        task_metadata.get("ground_truth_available", False)
                    ),
                },
                "status": "active",
                "started_at": started_at,
                "ended_at": None,
                "duration_seconds": None,
                "events": [
                    {
                        "sequence": 1,
                        "timestamp": started_at,
                        "event": "task_started",
                        "data": {"prompt": prompt},
                    }
                ],
                "automatic_summary": {},
                "final_state": {},
                "expert_annotation": self._expert_annotation_template(),
            }
            self._atomic_write(task_path, record)

            control["active_task"] = {
                "run_id": run_id,
                "task_id": task_id,
                "prompt": prompt,
                "started_at": started_at,
                "path": str(task_path),
            }
            self._write_control(control)
            self._last_event_signature = None
            self._last_event_at = None
            return control["active_task"]

    def stop_task(self, final_state: dict[str, Any] | None = None) -> Path:
        with self._lock:
            control = self._read_control()
            active = control.get("active_task")
            if not active:
                raise RuntimeError("No evaluation task is active.")

            task_path = Path(active["path"])
            record = self._read_json(task_path)
            ended_at = _now()
            started_at = datetime.fromisoformat(record["started_at"])
            duration = max(0.0, (ended_at - started_at).total_seconds())
            events = record.get("events") or []
            events.append(
                {
                    "sequence": len(events) + 1,
                    "timestamp": ended_at.isoformat(timespec="milliseconds"),
                    "event": "task_stopped",
                    "data": _jsonable(final_state or {}),
                }
            )
            record["events"] = events
            record["status"] = "completed"
            record["ended_at"] = ended_at.isoformat(timespec="milliseconds")
            record["duration_seconds"] = round(duration, 3)
            record["final_state"] = _jsonable(final_state or {})
            record["automatic_summary"] = self._summarize(events, duration)
            self._atomic_write(task_path, record)

            control["active_task"] = None
            control["last_completed_task"] = str(task_path)
            self._write_control(control)
            self._last_event_signature = None
            self._last_event_at = None
            return task_path

    def log_event(
        self,
        event: str,
        data: dict[str, Any] | None = None,
        *,
        dedupe_seconds: float = 0.0,
    ) -> None:
        with self._lock:
            control = self._read_control()
            active = control.get("active_task") if control.get("enabled") else None
            if not active:
                return

            payload = _jsonable(data or {})
            signature = (event, json.dumps(payload, sort_keys=True, ensure_ascii=False))
            now = _now()
            if (
                dedupe_seconds > 0
                and signature == self._last_event_signature
                and self._last_event_at is not None
                and (now - self._last_event_at).total_seconds() < dedupe_seconds
            ):
                return

            task_path = Path(active["path"])
            record = self._read_json(task_path)
            events = record.setdefault("events", [])
            events.append(
                {
                    "sequence": len(events) + 1,
                    "timestamp": now.isoformat(timespec="milliseconds"),
                    "event": event,
                    "data": payload,
                }
            )
            self._atomic_write(task_path, record)
            self._last_event_signature = signature
            self._last_event_at = now

    def _read_control(self) -> dict[str, Any]:
        if not self.control_path.exists():
            return {
                "enabled": False,
                "session_id": None,
                "output_dir": str(self.default_output_dir),
                "active_task": None,
            }
        try:
            return self._read_json(self.control_path)
        except (OSError, ValueError, TypeError):
            return {
                "enabled": False,
                "session_id": None,
                "output_dir": str(self.default_output_dir),
                "active_task": None,
                "error": "evaluation control file is unreadable",
            }

    def _write_control(self, control: dict[str, Any]) -> None:
        self.control_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self.control_path, control)

    def _match_task(self, prompt: str) -> dict[str, Any]:
        if not self.tasks_path.exists():
            return {}
        try:
            payload = self._read_json(self.tasks_path)
        except (OSError, ValueError, TypeError):
            return {}
        normalized = " ".join(prompt.lower().split())
        for task in payload.get("tasks", []):
            candidate = " ".join(str(task.get("prompt", "")).lower().split())
            if candidate == normalized:
                return task
        return {}

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)

    @staticmethod
    def _summarize(events: list[dict[str, Any]], duration: float) -> dict[str, Any]:
        counts = Counter(str(item.get("event", "")) for item in events)
        submitted_queries = [
            item.get("data", {}).get("query")
            for item in events
            if item.get("event") == "search_submitted"
        ]
        selected_results = [
            item.get("data", {}).get("result_id")
            for item in events
            if item.get("event") == "result_selected"
        ]
        return {
            "duration_seconds": round(duration, 3),
            "event_counts": dict(sorted(counts.items())),
            "queries_submitted": len(submitted_queries),
            "query_history": [query for query in submitted_queries if query],
            "result_selections": len(selected_results),
            "unique_results_selected": len({item for item in selected_results if item}),
            "keyword_clicks": counts.get("keyword_clicked", 0),
            "youtube_links_opened": counts.get("youtube_opened", 0),
            "grounding_requests": counts.get("grounding_requested", 0),
            "transcript_highlight_requests": counts.get("transcript_highlight_requested", 0),
            "feedback_actions": counts.get("feedback_changed", 0),
            "rocchio_refinements": counts.get("rocchio_refinement_completed", 0),
            "filter_changes": counts.get("filters_changed", 0),
            "chart_hovers": counts.get("chart_hovered", 0),
        }

    @staticmethod
    def _expert_annotation_template() -> dict[str, Any]:
        return {
            "outcome": None,
            "answer_or_conclusion": "",
            "evidence_verified": None,
            "evidence_correct": None,
            "router_appropriate_1_to_5": None,
            "provenance_clear_1_to_5": None,
            "grounding_correct": None,
            "transcript_highlight_useful_1_to_5": None,
            "mental_effort_1_to_7": None,
            "confidence_in_conclusion_0_to_100": None,
            "main_problem": "",
            "notes": "",
        }


evaluation_logger = EvaluationLogger()
