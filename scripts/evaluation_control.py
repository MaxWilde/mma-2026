#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime
from pathlib import Path


DEFAULT_CONTROL = (
    Path.home() / "X_evaluation" / "runtime" / "evaluation_control.json"
)
DEFAULT_OUTPUT = Path.home() / "X_evaluation" / "logs"


def read_control(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_control(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enable or disable optional CASTLE dashboard evaluation controls."
    )
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--session-id")
    args = parser.parse_args()

    control = read_control(args.control)
    active = control.get("active_task")

    if args.action == "status":
        print(json.dumps(control or {"enabled": False}, indent=2))
        return

    if args.action == "stop" and active:
        raise SystemExit(
            "A dashboard evaluation task is active. Stop the task in the dashboard first."
        )

    if args.action == "start":
        session_id = args.session_id or (
            f"expert-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}"
        )
        control = {
            "enabled": True,
            "session_id": session_id,
            "output_dir": str(args.output_dir.resolve()),
            "active_task": active,
            "enabled_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        write_control(args.control, control)
        print(f"Evaluation mode enabled: {session_id}")
        print("The controls will appear in the dashboard within two seconds.")
        return

    control["enabled"] = False
    control["active_task"] = None
    control["disabled_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_control(args.control, control)
    print("Evaluation mode disabled.")


if __name__ == "__main__":
    main()
