#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.transcript_refiner import export_video_clip


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a timestamped evidence clip with audio.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    if args.end_sec <= args.start_sec:
        raise SystemExit("--end-sec must be greater than --start-sec")

    export_video_clip(args.video_path, args.start_sec, args.end_sec, args.output_path)
    print(f"Saved evidence clip to {args.output_path}")


if __name__ == "__main__":
    main()
