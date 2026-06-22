#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_loader import discover_videos, load_evidence_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect legacy transcript/video evidence folders.")
    parser.add_argument("--dataset-root", default="/scratch-shared/group_h/data_goncalo")
    args = parser.parse_args()

    videos = discover_videos(args.dataset_root)
    evidence = load_evidence_items(args.dataset_root)
    by_area = Counter(item.area for item in evidence)

    print(f"Dataset root: {args.dataset_root}")
    print(f"Video folders: {len(videos)}")
    print(f"Transcript chunks: {len(evidence)}")
    print("Chunks by area:")
    for area, count in sorted(by_area.items()):
        print(f"  {area}: {count}")

    print("\nVideos:")
    for video in videos:
        keyframes = "yes" if video.keyframes_dir else "no"
        print(
            f"  {video.area}/{video.source_name} {video.day} {video.video_id} "
            f"fps={video.fps} keyframes={keyframes}"
        )


if __name__ == "__main__":
    main()
