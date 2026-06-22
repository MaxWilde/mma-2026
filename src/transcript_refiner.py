from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PADDING_BEFORE_SEC = 3.0
PADDING_AFTER_SEC = 5.0


@dataclass(frozen=True)
class WhisperSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class RefinedTranscriptEvidence:
    evidence: str
    refined_start_sec: float
    refined_end_sec: float
    full_start_sec: float
    full_end_sec: float
    video_path: str
    matched_text: str
    match_score: float
    segment_start_sec: float
    segment_end_sec: float
    segments: list[WhisperSegment]
    output_clip_path: str | None = None


def refine_transcript_timestamp(
    evidence_sentence: str,
    chunk: dict[str, Any],
    *,
    padding_before_sec: float = PADDING_BEFORE_SEC,
    padding_after_sec: float = PADDING_AFTER_SEC,
    debug: bool = False,
) -> RefinedTranscriptEvidence:
    video_path = str(chunk.get("video_path") or "")
    if not video_path:
        raise ValueError("Cannot refine transcript timestamp: best chunk has no video_path.")
    if chunk.get("start_sec") is None or chunk.get("end_sec") is None:
        raise ValueError("Cannot refine transcript timestamp: best chunk has no start_sec/end_sec.")

    chunk_start = float(chunk["start_sec"])
    chunk_end = float(chunk["end_sec"])
    if chunk_end <= chunk_start:
        raise ValueError(f"Cannot refine transcript timestamp: invalid chunk interval {chunk_start}-{chunk_end}.")

    if debug:
        print("[transcript-refinement] original chunk:", f"{chunk_start:.3f}-{chunk_end:.3f}")
        print("[transcript-refinement] matched evidence:", evidence_sentence)
        print("[transcript-refinement] video:", video_path)

    with tempfile.TemporaryDirectory(prefix="whisper_refine_") as tmpdir:
        audio_path = Path(tmpdir) / "chunk.wav"
        extract_audio_clip(video_path, chunk_start, chunk_end, audio_path)
        if debug:
            print("[transcript-refinement] extracted audio:", audio_path)
        segments = transcribe_audio_segments(audio_path)

    if not segments:
        raise RuntimeError("matching failed: faster-whisper produced no timestamped segments for the selected chunk.")

    if debug:
        print("[transcript-refinement] whisper segments:")
        for idx, segment in enumerate(segments, start=1):
            abs_start = chunk_start + segment.start
            abs_end = chunk_start + segment.end
            print(
                f"  {idx:02d}. local={segment.start:.2f}-{segment.end:.2f} "
                f"absolute={abs_start:.2f}-{abs_end:.2f} text={segment.text}"
            )

    matched_segment, score = best_matching_segment(evidence_sentence, segments)
    if score <= 0:
        raise RuntimeError("matching failed: no Whisper segment had token overlap with the matched evidence.")
    refined_start = max(chunk_start, chunk_start + matched_segment.start - padding_before_sec)
    refined_end = min(chunk_end, chunk_start + matched_segment.end + padding_after_sec)

    if debug:
        print(
            "[transcript-refinement] chosen segment:",
            f"local={matched_segment.start:.2f}-{matched_segment.end:.2f}",
            f"absolute={chunk_start + matched_segment.start:.2f}-{chunk_start + matched_segment.end:.2f}",
            f"score={score:.3f}",
            f"text={matched_segment.text}",
        )

    return RefinedTranscriptEvidence(
        evidence=evidence_sentence,
        refined_start_sec=refined_start,
        refined_end_sec=refined_end,
        full_start_sec=chunk_start,
        full_end_sec=chunk_end,
        video_path=video_path,
        matched_text=matched_segment.text,
        match_score=score,
        segment_start_sec=matched_segment.start,
        segment_end_sec=matched_segment.end,
        segments=segments,
    )


def extract_audio_clip(video_path: str, start_sec: float, end_sec: float, output_path: str | Path) -> None:
    require_ffmpeg()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.01, end_sec - start_sec)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg audio extraction failed with exit code {exc.returncode}: {' '.join(cmd)}") from exc


def export_video_clip(video_path: str, start_sec: float, end_sec: float, output_path: str | Path) -> None:
    require_ffmpeg()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.01, end_sec - start_sec)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg clip export failed with exit code {exc.returncode}: {' '.join(cmd)}") from exc


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg missing: ffmpeg was not found on PATH. Load the Snellius ffmpeg module or install ffmpeg.")


def transcribe_audio_segments(audio_path: str | Path) -> list[WhisperSegment]:
    model_name = os.environ.get("WHISPER_MODEL", "small")
    try:
        return _transcribe_with_faster_whisper(audio_path, model_name)
    except ImportError as exc:
        raise ImportError("faster-whisper missing: install/load faster-whisper to refine transcript timestamps.") from exc
    except Exception as exc:
        raise RuntimeError(f"faster-whisper transcription failed: {exc}") from exc


def _transcribe_with_faster_whisper(audio_path: str | Path, model_name: str) -> list[WhisperSegment]:
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(
            model_name,
            device=os.environ.get("WHISPER_DEVICE", "cpu"),
            compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "int8"),
            local_files_only=os.environ.get("WHISPER_LOCAL_FILES_ONLY", "1").lower() not in {"0", "false", "no"},
        )
    except Exception as exc:
        raise RuntimeError(f"faster-whisper model load failed for model '{model_name}': {exc}") from exc

    try:
        segments, _info = model.transcribe(str(audio_path), beam_size=5, vad_filter=False)
        segments = list(segments)
    except Exception as exc:
        raise RuntimeError(f"faster-whisper transcribe() failed for {audio_path}: {exc}") from exc

    return [
        WhisperSegment(float(segment.start), float(segment.end), str(segment.text).strip())
        for segment in segments
        if str(segment.text).strip()
    ]


def best_matching_segment(evidence_sentence: str, segments: list[WhisperSegment]) -> tuple[WhisperSegment, float]:
    best_segment = segments[0]
    best_score = -1.0
    for index, segment in enumerate(segments):
        candidates = [
            segment,
            _merge_segments(segments[index : index + 2]),
            _merge_segments(segments[index : index + 3]),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            score = token_overlap_score(evidence_sentence, candidate.text)
            if score > best_score:
                best_score = score
                best_segment = candidate
    return best_segment, best_score


def _merge_segments(segments: list[WhisperSegment]) -> WhisperSegment | None:
    if not segments:
        return None
    return WhisperSegment(
        start=segments[0].start,
        end=segments[-1].end,
        text=" ".join(segment.text for segment in segments).strip(),
    )


def token_overlap_score(reference: str, candidate: str) -> float:
    reference_tokens = content_tokens(reference)
    candidate_tokens = content_tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0
    overlap = reference_tokens & candidate_tokens
    precision = len(overlap) / len(candidate_tokens)
    recall = len(overlap) / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def content_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "at",
        "do",
        "does",
        "for",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "there",
        "they",
        "to",
        "what",
        "where",
        "who",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }
