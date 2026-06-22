#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"


@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float
    label: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Test GroundingDINO text-conditioned bounding boxes on one image.")
    parser.add_argument("--image", default="day1/Stevan/13/keyframes/shot_0673_frame_116700.jpg")
    parser.add_argument("--text", default="person cooking")
    parser.add_argument("--output", default="artifacts/visual_grounding/dino_test_bbox.jpg")
    parser.add_argument("--model", default=os.environ.get("GROUNDING_DINO_MODEL", DEFAULT_GROUNDING_DINO_MODEL))
    parser.add_argument("--threshold", type=float, default=float(os.environ.get("GROUNDING_DINO_THRESHOLD", "0.25")))
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face download if the GroundingDINO model is not already cached.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    try:
        detections = detect_grounding_dino(
            image_path=image_path,
            text=args.text,
            model_name=args.model,
            threshold=args.threshold,
            local_files_only=not args.allow_download,
        )
    except Exception as exc:
        raise SystemExit(f"GroundingDINO unavailable: {exc}") from exc

    print(f"image: {image_path}")
    print(f"text: {args.text}")
    print(f"model: {args.model}")
    print(f"detections: {len(detections)}")
    for idx, detection in enumerate(detections, start=1):
        box = ", ".join(f"{value:.1f}" for value in detection.box_xyxy)
        print(f"{idx}. score={detection.score:.4f} label={detection.label!r} box=[{box}]")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draw_detections(image_path, detections, output_path)
    print(f"output: {output_path}")


def detect_grounding_dino(
    *,
    image_path: Path,
    text: str,
    model_name: str,
    threshold: float,
    local_files_only: bool,
) -> list[Detection]:
    if local_files_only and not model_available_locally(model_name):
        raise RuntimeError(f"model is not present in local Hugging Face cache: {model_name}")

    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("required dependencies are missing: torch and transformers") from exc

    processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name, local_files_only=local_files_only)
    device = os.environ.get("GROUNDING_DINO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        inputs = processor(images=rgb, text=text, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)

        target_sizes = torch.tensor([rgb.size[::-1]], device=device)
        results = post_process_grounding_dino(
            processor=processor,
            outputs=outputs,
            input_ids=inputs.input_ids,
            threshold=threshold,
            target_sizes=target_sizes,
        )[0]

    boxes = results.get("boxes", [])
    scores = results.get("scores", [])
    labels = results.get("labels", [])
    detections: list[Detection] = []
    for box, score, label in zip(boxes, scores, labels):
        detections.append(
            Detection(
                box_xyxy=tuple(float(value) for value in box.detach().cpu().tolist()),  # type: ignore[arg-type]
                score=float(score.detach().cpu().item()),
                label=str(label),
            )
        )
    return sorted(detections, key=lambda item: item.score, reverse=True)


def post_process_grounding_dino(
    *,
    processor: Any,
    outputs: Any,
    input_ids: Any,
    threshold: float,
    target_sizes: Any,
):
    fn = processor.post_process_grounded_object_detection
    params = inspect.signature(fn).parameters
    kwargs: dict[str, Any] = {
        "outputs": outputs,
        "input_ids": input_ids,
        "text_threshold": threshold,
        "target_sizes": target_sizes,
    }
    if "box_threshold" in params:
        kwargs["box_threshold"] = threshold
    else:
        kwargs["threshold"] = threshold
    return fn(**kwargs)


def model_available_locally(model_name: str) -> bool:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    return bool(try_to_load_from_cache(model_name, "config.json"))


def draw_detections(image_path: Path, detections: list[Detection], output_path: Path) -> None:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        draw = ImageDraw.Draw(rgb)
        font = ImageFont.load_default()
        width = max(3, min(rgb.size) // 160)

        for detection in detections:
            x1, y1, x2, y2 = detection.box_xyxy
            draw.rectangle((x1, y1, x2, y2), outline=(255, 30, 30), width=width)
            label = f"{detection.label} {detection.score:.2f}"
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            label_height = text_bbox[3] - text_bbox[1] + 6
            label_y = max(0, y1 - label_height)
            draw.rectangle(
                (x1, label_y, x1 + (text_bbox[2] - text_bbox[0]) + 8, label_y + label_height),
                fill=(255, 30, 30),
            )
            draw.text((x1 + 4, label_y + 3), label, fill=(255, 255, 255), font=font)

        rgb.save(output_path, quality=92)


if __name__ == "__main__":
    main()
