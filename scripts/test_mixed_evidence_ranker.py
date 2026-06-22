#!/usr/bin/env python
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mixed_evidence_ranker import build_mixed_evidence_list, percentile_quality_scores


def main() -> None:
    run_mode_tests("max")
    run_mode_tests("percentile")
    test_percentile_quality_scores()
    test_overlapping_transcript_windows_collapse()
    test_repeated_transcript_text_collapses()
    test_nearby_visual_frames_collapse()
    test_unrelated_evidence_remains()
    print("mixed evidence ranker smoke tests passed")


def run_mode_tests(calibration_mode: str) -> None:
    visual = fake_visual_candidates()
    transcript = fake_transcript_candidates()

    transcript_heavy = build_mixed_evidence_list(
        "test",
        visual,
        transcript,
        {"combined_visual_score": 0.2, "combined_transcript_score": 3.0},
        top_k=10,
        calibration_mode=calibration_mode,
    )
    assert transcript_heavy[0]["evidence_type"] == "transcript", transcript_heavy[:3]
    assert count_types(transcript_heavy)["transcript"] >= count_types(transcript_heavy)["visual"], count_types(transcript_heavy)
    assert all(
        item["score_components"]["calibration_mode"] == calibration_mode
        for item in transcript_heavy
    )

    visual_heavy = build_mixed_evidence_list(
        "test",
        visual,
        transcript,
        {"combined_visual_score": 3.0, "combined_transcript_score": 0.2},
        top_k=10,
        calibration_mode=calibration_mode,
    )
    assert visual_heavy[0]["evidence_type"] == "visual", visual_heavy[:3]
    assert count_types(visual_heavy)["visual"] >= count_types(visual_heavy)["transcript"], count_types(visual_heavy)

    close = build_mixed_evidence_list(
        "test",
        visual,
        transcript,
        {"combined_visual_score": 1.0, "combined_transcript_score": 1.0},
        top_k=10,
        calibration_mode=calibration_mode,
    )
    close_counts = count_types(close)
    assert close_counts["visual"] > 0 and close_counts["transcript"] > 0, close_counts

    lower_weight_stronger = build_mixed_evidence_list(
        "test",
        weak_visual_candidates(),
        strong_transcript_candidates(),
        {"combined_visual_score": 3.0, "combined_transcript_score": 0.2},
        top_k=10,
        calibration_mode=calibration_mode,
    )
    lower_weight_counts = count_types(lower_weight_stronger[:5])
    assert lower_weight_counts["transcript"] > 0, lower_weight_stronger[:5]

    print(f"{calibration_mode} transcript-heavy:", dict(count_types(transcript_heavy)))
    print(f"{calibration_mode} visual-heavy:", dict(count_types(visual_heavy)))
    print(f"{calibration_mode} close:", dict(close_counts))
    print(f"{calibration_mode} lower-weight stronger top5:", dict(lower_weight_counts))


def test_percentile_quality_scores() -> None:
    scores = percentile_quality_scores([{"score": idx} for idx in range(5)])
    assert scores[0] == 1.0, scores
    assert scores[-1] == 0.0, scores
    assert abs(scores[2] - 0.5) < 1e-9, scores


def test_overlapping_transcript_windows_collapse() -> None:
    transcript = [
        transcript_candidate(1.0, "Kitchen", 100.0, 120.0, "The water is starting to boil now."),
        transcript_candidate(0.99, "Kitchen", 108.0, 128.0, "The water is starting to boil now, yes."),
        transcript_candidate(0.90, "Kitchen", 180.0, 190.0, "They discuss a different event."),
    ]
    mixed = build_mixed_evidence_list(
        "test",
        [],
        transcript,
        {"combined_visual_score": 0.0, "combined_transcript_score": 2.0},
        top_k=10,
    )
    timestamps = [item["timestamp"] for item in mixed]
    assert "01:40-02:00" in timestamps, timestamps
    assert "01:48-02:08" not in timestamps, timestamps
    assert "03:00-03:10" in timestamps, timestamps


def test_repeated_transcript_text_collapses() -> None:
    transcript = [
        transcript_candidate(1.0, "Stevan", 10.0, 14.0, "It is boiling."),
        transcript_candidate(0.95, "Kitchen", 300.0, 304.0, "It is boiling!"),
        transcript_candidate(0.90, "Werner", 500.0, 504.0, "A completely different sentence."),
    ]
    mixed = build_mixed_evidence_list(
        "test",
        [],
        transcript,
        {"combined_visual_score": 0.0, "combined_transcript_score": 2.0},
        top_k=10,
    )
    snippets = [item["transcript_snippet"] for item in mixed]
    assert snippets.count("It is boiling.") == 1, snippets
    assert "It is boiling!" not in snippets, snippets
    assert "A completely different sentence." in snippets, snippets


def test_nearby_visual_frames_collapse() -> None:
    visual = [
        visual_candidate(1.0, "Kitchen", 100.0, "frame_a.jpg"),
        visual_candidate(0.98, "Stevan", 108.0, "frame_b.jpg"),
        visual_candidate(0.90, "Kitchen", 200.0, "frame_c.jpg"),
    ]
    mixed = build_mixed_evidence_list(
        "test",
        visual,
        [],
        {"combined_visual_score": 2.0, "combined_transcript_score": 0.0},
        top_k=10,
    )
    paths = [item["keyframe_path"] for item in mixed]
    assert "frame_a.jpg" in paths, paths
    assert "frame_b.jpg" not in paths, paths
    assert "frame_c.jpg" in paths, paths


def test_unrelated_evidence_remains() -> None:
    visual = [
        visual_candidate(1.0, "Kitchen", 100.0, "frame_a.jpg"),
        visual_candidate(0.98, "Kitchen", 180.0, "frame_b.jpg"),
    ]
    transcript = [
        transcript_candidate(1.0, "Kitchen", 130.0, 133.0, "The first sentence."),
        transcript_candidate(0.98, "Kitchen", 230.0, 233.0, "The second unrelated sentence."),
    ]
    mixed = build_mixed_evidence_list(
        "test",
        visual,
        transcript,
        {"combined_visual_score": 1.0, "combined_transcript_score": 1.0},
        top_k=10,
    )
    assert len(mixed) == 4, mixed


def fake_visual_candidates() -> list[dict]:
    return [
        {
            "score": 1.0 - idx * 0.08,
            "keyframe_path": f"frame_{idx}.jpg",
            "day": "day1",
            "source_name": "Kitchen",
            "video_id": "14",
            "keyframe_time_sec": idx * 45.0,
            "timestamp": f"00:{idx:02d}",
            "youtube_timestamp_url": f"https://example.com/v{idx}",
        }
        for idx in range(8)
    ]


def fake_transcript_candidates() -> list[dict]:
    return [
        {
            "score": 0.95 - idx * 0.07,
            "source_name": "Kitchen",
            "day": "day1",
            "hour_id": "14",
            "start_sec": idx * 50.0,
            "end_sec": idx * 50.0 + 10.0,
            "timestamp": f"00:{idx:02d}-00:{idx + 1:02d}",
            "youtube_timestamp_url": f"https://example.com/t{idx}",
            "transcript_snippet": f"transcript snippet {idx}",
        }
        for idx in range(8)
    ]


def visual_candidate(score: float, source_name: str, time_sec: float, keyframe_path: str) -> dict:
    return {
        "score": score,
        "keyframe_path": keyframe_path,
        "day": "day1",
        "source_name": source_name,
        "video_id": "14",
        "keyframe_time_sec": time_sec,
        "timestamp": f"{int(time_sec)}",
        "youtube_timestamp_url": f"https://example.com/v{time_sec}",
    }


def transcript_candidate(score: float, source_name: str, start_sec: float, end_sec: float, text: str) -> dict:
    return {
        "score": score,
        "source_name": source_name,
        "day": "day1",
        "hour_id": "14",
        "start_sec": start_sec,
        "end_sec": end_sec,
        "timestamp": f"{format_test_time(start_sec)}-{format_test_time(end_sec)}",
        "youtube_timestamp_url": f"https://example.com/t{start_sec}",
        "transcript_snippet": text,
    }


def format_test_time(value: float) -> str:
    minutes = int(value // 60)
    seconds = int(value % 60)
    return f"{minutes:02d}:{seconds:02d}"


def weak_visual_candidates() -> list[dict]:
    candidates = fake_visual_candidates()
    for idx, item in enumerate(candidates):
        item["score"] = 1.0 if idx == 0 else 0.30 - idx * 0.02
    return candidates


def strong_transcript_candidates() -> list[dict]:
    candidates = fake_transcript_candidates()
    for idx, item in enumerate(candidates):
        item["score"] = 1.0 - idx * 0.02
    return candidates


def count_types(items: list[dict]) -> Counter:
    return Counter(item["evidence_type"] for item in items)


if __name__ == "__main__":
    main()
