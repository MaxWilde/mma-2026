from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from PIL import Image

from src.vqa import answer_from_chunks, format_timestamp


VisionMode = Literal["mock", "qwen"]
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


@dataclass(frozen=True)
class VisualEvidence:
    chunk_index: int
    source_id: str
    source_name: str
    timestamp: str
    image_path: str
    image_size: tuple[int, int]
    transcript_text: str


@dataclass(frozen=True)
class VisionQAResult:
    transcript_answer: str
    visual_evidence: list[VisualEvidence]
    final_answer: str


def answer_with_visual_evidence(
    question: str,
    retrieved_chunks: list[dict],
    mode: VisionMode = "mock",
    max_images: int = 5,
    qwen_model: str | None = None,
) -> VisionQAResult:
    transcript_answer = answer_from_chunks(question, retrieved_chunks, max_chunks=max_images)
    visual_evidence = collect_visual_evidence(retrieved_chunks, max_images=max_images)

    if mode == "mock":
        final_answer = _mock_vision_answer(question, transcript_answer, visual_evidence)
    elif mode == "qwen":
        final_answer = _qwen_vision_answer(question, retrieved_chunks, visual_evidence, qwen_model)
    else:
        raise ValueError(f"Unsupported vision QA mode: {mode}")

    return VisionQAResult(
        transcript_answer=transcript_answer,
        visual_evidence=visual_evidence,
        final_answer=final_answer,
    )


def collect_visual_evidence(retrieved_chunks: list[dict], max_images: int = 5) -> list[VisualEvidence]:
    evidence: list[VisualEvidence] = []
    seen_paths: set[str] = set()

    for chunk_index, chunk in enumerate(retrieved_chunks, start=1):
        image_path = chunk.get("closest_keyframe_path")
        if not image_path or image_path in seen_paths:
            continue

        path = Path(image_path)
        if not path.exists():
            continue

        try:
            with Image.open(path) as image:
                image_size = image.size
        except OSError:
            continue

        timestamp = format_timestamp(float(chunk["start_sec"]), float(chunk["end_sec"]))
        evidence.append(
            VisualEvidence(
                chunk_index=chunk_index,
                source_id=chunk["source_id"],
                source_name=chunk["source_name"],
                timestamp=timestamp,
                image_path=str(path),
                image_size=image_size,
                transcript_text=chunk["text"],
            )
        )
        seen_paths.add(image_path)

        if len(evidence) >= max_images:
            break

    return evidence


def _mock_vision_answer(
    question: str,
    transcript_answer: str,
    visual_evidence: list[VisualEvidence],
) -> str:
    if not visual_evidence:
        return (
            "No usable keyframe images were found for the retrieved transcript chunks. "
            "Using transcript evidence only.\n\n"
            f"{transcript_answer}"
        )

    lines = [
        "Mock multimodal answer:",
        "A real vision-language model is not enabled yet. The system would analyze these transcript-localized keyframes:",
    ]
    for item in visual_evidence:
        lines.append(
            f"- chunk {item.chunk_index}: {item.source_name} at {item.timestamp}, "
            f"{item.image_path} ({item.image_size[0]}x{item.image_size[1]})"
        )
    lines.extend(["", "Transcript-grounded answer:", transcript_answer])
    return "\n".join(lines)


def _qwen_vision_answer(
    question: str,
    retrieved_chunks: list[dict],
    visual_evidence: list[VisualEvidence],
    model_name: str | None = None,
) -> str:
    if not visual_evidence:
        raise FileNotFoundError(
            "Qwen vision mode requires at least one retrieved keyframe image, but none were found. "
            "Run mock mode or query an index that includes closest_keyframe_path metadata."
        )

    model_id = model_name or os.environ.get("QWEN_VL_MODEL", DEFAULT_QWEN_MODEL)
    model, processor, torch, process_vision_info = _load_qwen_model(model_id)
    top_image = visual_evidence[0]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Path(top_image.image_path).resolve().as_uri()},
                {"type": "text", "text": _build_qwen_prompt(question, retrieved_chunks, top_image)},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_get_model_input_device(model, torch))

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=int(os.environ.get("QWEN_VL_MAX_NEW_TOKENS", "192")))

    generated_ids_trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    answers = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return answers[0].strip() if answers else ""


@lru_cache(maxsize=2)
def _load_qwen_model(model_id: str):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install PyTorch to use Qwen vision mode.") from exc

    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        LocalEntryNotFoundError = OSError

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise ImportError(
            "Install qwen-vl-utils to use Qwen vision mode: pip install qwen-vl-utils"
        ) from exc

    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Install a recent transformers build with Qwen2.5-VL support to use Qwen vision mode."
        ) from exc

    local_only = os.environ.get("QWEN_VL_LOCAL_FILES_ONLY", "1").lower() not in {"0", "false", "no"}
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            local_files_only=local_only,
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            local_files_only=local_only,
            min_pixels=int(os.environ.get("QWEN_VL_MIN_PIXELS", str(256 * 28 * 28))),
            max_pixels=int(os.environ.get("QWEN_VL_MAX_PIXELS", str(1024 * 28 * 28))),
        )
    except (OSError, LocalEntryNotFoundError) as exc:
        raise FileNotFoundError(
            f"Qwen vision model is not available locally: {model_id}. "
            "Download it before running qwen mode, or set QWEN_VL_MODEL to a local checkpoint path. "
            "This code defaults to offline loading so test jobs do not unexpectedly download large models. "
            "To allow HuggingFace downloads explicitly, set QWEN_VL_LOCAL_FILES_ONLY=0."
        ) from exc

    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return model, processor, torch, process_vision_info


def _get_model_input_device(model, torch):
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for device in hf_device_map.values():
            if isinstance(device, str) and device not in {"cpu", "disk"}:
                return torch.device(device)
    return next(model.parameters()).device


def _build_qwen_prompt(question: str, retrieved_chunks: list[dict], top_image: VisualEvidence) -> str:
    lines = [
        "Answer the user's question using both sources of evidence:",
        "1. The retrieved transcript chunks below.",
        "2. The attached keyframe image from the top retrieved visual evidence.",
        "",
        "Do not answer from transcript alone. Inspect the image and reconcile it with the transcript.",
        "If the transcript and image disagree, say so and explain which evidence supports your answer.",
        "Keep the answer concise and grounded in the supplied evidence.",
        "",
        f"Question: {question}",
        "",
        "Attached image metadata:",
        f"- source_id: {top_image.source_id}",
        f"- source_name: {top_image.source_name}",
        f"- timestamp: {top_image.timestamp}",
        "",
        "Retrieved transcript chunks:",
    ]
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        timestamp = format_timestamp(float(chunk["start_sec"]), float(chunk["end_sec"]))
        source = f"{chunk['source_name']} {chunk['day']} {chunk['video_id']}"
        lines.append(f"{idx}. [{source}, {timestamp}] {chunk['text']}")
    return "\n".join(lines)
