from __future__ import annotations

import os
import re
import inspect
import time
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha1
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUNDING_DINO_MODEL = "/scratch-shared/group_h/models/grounding-dino-base"
DEFAULT_THRESHOLDS = (0.25, 0.15, 0.10, 0.05)
GroundingMethod = Literal["bbox", "none"]


@dataclass(frozen=True)
class VisualGroundingResult:
    keyframe_path: str | None
    output_image_path: str | None
    method: GroundingMethod
    box_xyxy: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    label: str | None = None
    grounding_target: str | None = None
    candidates_image_path: str | None = None
    frame_score: float | None = None
    crop_siglip_score: float | None = None
    final_score: float | None = None
    frame_rank: int | None = None
    box_rank: int | None = None
    ranked_candidates: list[dict[str, Any]] | None = None
    debug: list[dict[str, Any]] | None = None
    debug_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float
    label: str
    prompt: str | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class GroundingConfig:
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    prompts: tuple[str, ...] | None = None
    prompt_strategy: str = "auto"
    rerank: Literal["none", "siglip-crop"] = "none"
    max_boxes_per_frame: int = 5
    max_prompts: int = 3          # cap forward passes per image (prompts are priority-ranked)
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    debug: bool = False


@dataclass(frozen=True)
class BoxCandidate:
    chunk: dict
    keyframe_path: Path
    frame_rank: int
    box_rank: int
    detection: Detection
    grounding_target: str
    frame_score: float
    crop_siglip_score: float | None = None
    final_score: float | None = None


def ground_visual_evidence(
    question: str,
    chunk: dict,
    config: GroundingConfig | None = None,
) -> VisualGroundingResult:
    config = config or GroundingConfig()
    keyframe_path = _best_keyframe_path(chunk)
    if not keyframe_path:
        return VisualGroundingResult(None, None, "none", label="no keyframe")

    path = Path(keyframe_path)
    if not path.exists():
        return VisualGroundingResult(str(path), None, "none", label="missing keyframe")

    plan = derive_grounding_plan(question, chunk, config)
    prompts = plan["prompts"][:config.max_prompts]
    try:
        detection, attempts = detect_with_grounding_dino(path, prompts, config.thresholds)
    except Exception as exc:
        return VisualGroundingResult(
            str(path),
            None,
            "none",
            label=f"GroundingDINO unavailable: {exc}",
            grounding_target=plan["target"],
            debug=None,
            debug_context=_debug_context(question, chunk, plan, config.thresholds, None) if config.debug else None,
        )

    if detection is None:
        return VisualGroundingResult(
            str(path),
            None,
            "none",
            label=f"no bounding box found for target: {plan['target']}",
            grounding_target=plan["target"],
            debug=attempts if config.debug else None,
            debug_context=_debug_context(question, chunk, plan, config.thresholds, None) if config.debug else None,
        )

    output_path = draw_bounding_box(path, detection)
    display_output_path = output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path
    return VisualGroundingResult(
        keyframe_path=str(path),
        output_image_path=str(display_output_path),
        method="bbox",
        box_xyxy=detection.box_xyxy,
        confidence=detection.score,
        label=detection.label,
        grounding_target=plan["target"],
        debug=attempts if config.debug else None,
        debug_context=_debug_context(question, chunk, plan, config.thresholds, detection) if config.debug else None,
    )


def ground_visual_evidence_with_rerank(
    question: str,
    chunks: list[dict],
    *,
    siglip_model: Any,
    siglip_processor: Any,
    torch: Any,
    config: GroundingConfig | None = None,
    timings: dict[str, float] | None = None,
) -> VisualGroundingResult:
    config = config or GroundingConfig(rerank="siglip-crop")
    if not chunks:
        return VisualGroundingResult(None, None, "none", label="no retrieved keyframes")
    if config.rerank == "none":
        return ground_visual_evidence(question, chunks[0], config=config)

    candidates: list[BoxCandidate] = []
    debug_attempts: list[dict[str, Any]] = []
    first_existing_path: Path | None = None
    first_target: str | None = None
    dino_start = time.perf_counter()
    for frame_rank, chunk in enumerate(chunks, start=1):
        keyframe_path = _best_keyframe_path(chunk)
        if not keyframe_path:
            continue
        path = Path(keyframe_path)
        if not path.exists():
            continue
        first_existing_path = first_existing_path or path
        plan = derive_grounding_plan(question, chunk, config)
        first_target = first_target or plan["target"]
        try:
            detections, attempts = detect_candidates_with_grounding_dino(
                path,
                plan["prompts"][:config.max_prompts],
                config.thresholds,
                max_detections=config.max_boxes_per_frame,
            )
        except Exception as exc:
            return VisualGroundingResult(
                str(path),
                None,
                "none",
                label=f"GroundingDINO unavailable: {exc}",
                grounding_target=plan["target"],
            )
        debug_attempts.extend(
            {**attempt, "frame_rank": frame_rank, "keyframe_path": str(path)}
            for attempt in attempts
        )
        for box_rank, detection in enumerate(detections[: config.max_boxes_per_frame], start=1):
            candidates.append(
                BoxCandidate(
                    chunk=chunk,
                    keyframe_path=path,
                    frame_rank=frame_rank,
                    box_rank=box_rank,
                    detection=detection,
                    grounding_target=plan["target"],
                    frame_score=float(chunk.get("score", 0.0)),
                )
            )
    if timings is not None:
        timings["GroundingDINO"] = time.perf_counter() - dino_start

    if not candidates:
        return VisualGroundingResult(
            str(first_existing_path) if first_existing_path else None,
            None,
            "none",
            label=f"no candidate bounding boxes found for target: {first_target or 'n/a'}",
            grounding_target=first_target,
            debug=debug_attempts if config.debug else None,
        )

    crop_start = time.perf_counter()
    ranked = rerank_candidates_with_siglip(
        question,
        candidates,
        siglip_model=siglip_model,
        siglip_processor=siglip_processor,
        torch=torch,
        config=config,
    )
    if timings is not None:
        timings["crop reranking"] = time.perf_counter() - crop_start
    selected = ranked[0]
    output_path = draw_bounding_box(selected.keyframe_path, selected.detection)
    candidates_path = draw_candidate_boxes(selected.keyframe_path, [item for item in ranked if item.keyframe_path == selected.keyframe_path])
    display_output_path = _display_path(output_path)
    display_candidates_path = _display_path(candidates_path)
    return VisualGroundingResult(
        keyframe_path=str(selected.keyframe_path),
        output_image_path=str(display_output_path),
        method="bbox",
        box_xyxy=selected.detection.box_xyxy,
        confidence=selected.detection.score,
        label=selected.detection.label,
        grounding_target=selected.grounding_target,
        candidates_image_path=str(display_candidates_path),
        frame_score=selected.frame_score,
        crop_siglip_score=selected.crop_siglip_score,
        final_score=selected.final_score,
        frame_rank=selected.frame_rank,
        box_rank=selected.box_rank,
        ranked_candidates=[_candidate_summary(item) for item in ranked],
        debug=debug_attempts if config.debug else None,
    )


def derive_grounding_prompts(question: str, chunk: dict, config: GroundingConfig | None = None) -> tuple[str, ...]:
    return derive_grounding_plan(question, chunk, config)["prompts"]


def derive_grounding_plan(question: str, chunk: dict, config: GroundingConfig | None = None) -> dict[str, Any]:
    config = config or GroundingConfig()
    if config.prompts:
        prompts = _dedupe_prompts(config.prompts)
        return {"target": prompts[0], "prompts": prompts}

    normalized = question.strip().rstrip("?.!")
    caption = str(chunk.get("visual_caption", "")).strip()
    prompts: list[str] = []

    target = extract_grounding_target(normalized, caption)
    if target:
        prompts.append(target)

    question_phrases = _noun_phrases(normalized)
    caption_phrases = _noun_phrases(caption)
    prompts.extend(_caption_confirmed_question_phrases(question_phrases, caption))
    prompts.extend(question_phrases)

    if caption:
        prompts.extend(_caption_phrases_relevant_to_question(normalized, caption_phrases))
        prompts.extend(caption_phrases[:4])

    if _question_mentions_person(normalized):
        prompts.append("person")
    elif not prompts:
        prompts.append("object")

    prompts_tuple = _dedupe_prompts(prompts)
    return {"target": target or prompts_tuple[0], "prompts": prompts_tuple}


def extract_grounding_target(question: str, caption: str) -> str | None:
    location_target = _extract_location_target(question)
    if location_target:
        return _prefer_caption_exact_match(location_target, caption) or location_target

    attribute_target = _extract_attribute_target(question)
    if attribute_target:
        return _prefer_caption_exact_match(attribute_target, caption) or attribute_target

    relation_anchor = _extract_relation_anchor(question)
    if relation_anchor:
        neighbor_target = _caption_neighbor_target(caption, relation_anchor, question)
        if neighbor_target:
            return neighbor_target
        return _prefer_caption_exact_match(relation_anchor, caption) or relation_anchor

    action_target = _caption_action_object_target(question, caption)
    if action_target:
        return action_target

    question_phrases = _noun_phrases(question)
    for phrase in question_phrases:
        if _phrase_in_text(phrase, caption):
            return phrase

    caption_relevant = _caption_phrases_relevant_to_question(question, _noun_phrases(caption))
    if caption_relevant:
        return caption_relevant[0]
    return question_phrases[0] if question_phrases else None


def detect_with_grounding_dino(
    image_path: Path,
    prompts: tuple[str, ...],
    thresholds: tuple[float, ...],
) -> tuple[Detection | None, list[dict[str, Any]]]:
    processor, model, torch = _load_grounding_dino()
    attempts: list[dict[str, Any]] = []

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        device = _model_device(model, torch)
        for prompt in prompts:
            inputs = processor(images=rgb, text=prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            target_sizes = torch.tensor([rgb.size[::-1]], device=device)
            for threshold in thresholds:
                results = post_process_grounding_dino(
                    processor=processor,
                    outputs=outputs,
                    input_ids=inputs.input_ids,
                    threshold=threshold,
                    text_threshold=threshold,
                    target_sizes=target_sizes,
                )[0]
                detection = best_detection_from_results(results, prompt, torch)
                attempts.append(_attempt_summary(prompt, threshold, results, detection))
                if detection is not None:
                    return detection, attempts

    return None, attempts


def detect_candidates_with_grounding_dino(
    image_path: Path,
    prompts: tuple[str, ...],
    thresholds: tuple[float, ...],
    *,
    max_detections: int,
) -> tuple[list[Detection], list[dict[str, Any]]]:
    processor, model, torch = _load_grounding_dino()
    attempts: list[dict[str, Any]] = []
    detections: list[Detection] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        device = _model_device(model, torch)
        for prompt in prompts:
            inputs = processor(images=rgb, text=prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            target_sizes = torch.tensor([rgb.size[::-1]], device=device)
            for threshold in thresholds:
                results = post_process_grounding_dino(
                    processor=processor,
                    outputs=outputs,
                    input_ids=inputs.input_ids,
                    threshold=threshold,
                    text_threshold=threshold,
                    target_sizes=target_sizes,
                )[0]
                batch_detections = detections_from_results(results, prompt, threshold, torch)
                attempts.append(_attempt_summary(prompt, threshold, results, batch_detections[0] if batch_detections else None))
                for detection in batch_detections:
                    box_key = tuple(int(round(value)) for value in detection.box_xyxy)
                    key = (detection.label.lower(), box_key)
                    if key in seen:
                        continue
                    seen.add(key)
                    detections.append(detection)

    return sorted(detections, key=lambda item: item.score, reverse=True)[:max_detections], attempts


def best_detection_from_results(results: dict[str, Any], prompt: str, torch: Any) -> Detection | None:
    detections = detections_from_results(results, prompt, None, torch)
    return detections[0] if detections else None


def detections_from_results(results: dict[str, Any], prompt: str, threshold: float | None, torch: Any) -> list[Detection]:
    scores = results.get("scores")
    boxes = results.get("boxes")
    if scores is None or boxes is None or len(scores) == 0:
        return []

    labels = results.get("labels")
    detections: list[Detection] = []
    for index in range(len(scores)):
        score = float(scores[index].detach().cpu().item())
        box = boxes[index].detach().cpu().tolist()
        label = str(labels[index]) if labels is not None and len(labels) > index else prompt
        detections.append(
            Detection(
                box_xyxy=tuple(float(value) for value in box),  # type: ignore[arg-type]
                score=score,
                label=label,
                prompt=prompt,
                threshold=threshold,
            )
        )
    return sorted(detections, key=lambda item: item.score, reverse=True)


def rerank_candidates_with_siglip(
    question: str,
    candidates: list[BoxCandidate],
    *,
    siglip_model: Any,
    siglip_processor: Any,
    torch: Any,
    config: GroundingConfig,
) -> list[BoxCandidate]:
    from src.clip_retrieval import embed_pil_images, embed_texts_clip

    crops: list[Image.Image] = []
    texts: list[str] = []
    valid_candidates: list[BoxCandidate] = []
    for candidate in candidates:
        crop = crop_detection(candidate.keyframe_path, candidate.detection)
        if crop is None:
            continue
        crops.append(crop)
        texts.append(candidate.grounding_target or question)
        valid_candidates.append(candidate)
    if not valid_candidates:
        return candidates

    crop_embeddings = embed_pil_images(crops, model=siglip_model, processor=siglip_processor, torch=torch)
    text_embeddings = embed_texts_clip(texts, model=siglip_model, processor=siglip_processor, torch=torch)
    crop_scores = np.sum(crop_embeddings * text_embeddings, axis=1)

    reranked: list[BoxCandidate] = []
    for candidate, crop_score in zip(valid_candidates, crop_scores):
        final_score = (
            config.alpha * candidate.frame_score
            + config.beta * candidate.detection.score
            + config.gamma * float(crop_score)
        )
        reranked.append(
            BoxCandidate(
                chunk=candidate.chunk,
                keyframe_path=candidate.keyframe_path,
                frame_rank=candidate.frame_rank,
                box_rank=candidate.box_rank,
                detection=candidate.detection,
                grounding_target=candidate.grounding_target,
                frame_score=candidate.frame_score,
                crop_siglip_score=float(crop_score),
                final_score=float(final_score),
            )
        )
    return sorted(reranked, key=lambda item: item.final_score if item.final_score is not None else float("-inf"), reverse=True)


def _attempt_summary(
    prompt: str,
    threshold: float,
    results: dict[str, Any],
    detection: Detection | None,
) -> dict[str, Any]:
    scores = results.get("scores")
    detection_count = int(len(scores)) if scores is not None else 0
    best_score = None
    if scores is not None and len(scores):
        best_score = float(max(float(score.detach().cpu().item()) for score in scores))
    return {
        "prompt": prompt,
        "threshold": threshold,
        "detections": detection_count,
        "best_score": best_score,
        "selected_box": detection.box_xyxy if detection else None,
    }


def _debug_context(
    question: str,
    chunk: dict,
    plan: dict[str, Any],
    thresholds: tuple[float, ...],
    detection: Detection | None,
) -> dict[str, Any]:
    return {
        "question": question,
        "visual_caption": str(chunk.get("visual_caption", "")).strip(),
        "grounding_target": plan["target"],
        "prompts": list(plan["prompts"]),
        "thresholds": list(thresholds),
        "selected_detection": {
            "label": detection.label,
            "score": detection.score,
            "box_xyxy": detection.box_xyxy,
        }
        if detection
        else None,
    }


@lru_cache(maxsize=1)
def _load_grounding_dino() -> tuple[Any, Any, Any]:
    model_name = os.environ.get("GROUNDING_DINO_MODEL", DEFAULT_GROUNDING_DINO_MODEL)
    local_files_only = os.environ.get("GROUNDING_DINO_LOCAL_FILES_ONLY", "1").lower() not in {"0", "false", "no"}
    if local_files_only and not _model_available_locally(model_name):
        raise RuntimeError(f"model is not present locally: {model_name}")

    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("required dependencies are missing: torch and transformers") from exc

    try:
        processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception as exc:
        raise RuntimeError(f"model load failed: {model_name}") from exc

    device = os.environ.get("GROUNDING_DINO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, torch


def post_process_grounding_dino(
    *,
    processor: Any,
    outputs: Any,
    input_ids: Any,
    threshold: float,
    text_threshold: float,
    target_sizes: Any,
):
    fn = processor.post_process_grounded_object_detection
    params = inspect.signature(fn).parameters
    kwargs: dict[str, Any] = {
        "outputs": outputs,
        "input_ids": input_ids,
        "text_threshold": text_threshold,
        "target_sizes": target_sizes,
    }
    if "box_threshold" in params:
        kwargs["box_threshold"] = threshold
    else:
        kwargs["threshold"] = threshold
    return fn(**kwargs)


def _model_available_locally(model_name: str) -> bool:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    cached_config = try_to_load_from_cache(model_name, "config.json")
    return bool(cached_config)


def draw_bounding_box(image_path: Path, detection: Detection) -> Path:
    output_dir = ROOT / "artifacts" / "visual_grounding"
    output_dir.mkdir(parents=True, exist_ok=True)
    key = f"{image_path}:{detection.label}:{detection.box_xyxy}:{detection.score:.6f}"
    digest = sha1(key.encode("utf-8")).hexdigest()[:12]
    output_path = output_dir / f"{image_path.stem}_{digest}_bbox.jpg"

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        draw = ImageDraw.Draw(rgb)
        x1, y1, x2, y2 = detection.box_xyxy
        width = max(3, min(rgb.size) // 160)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 30, 30), width=width)
        label = f"{detection.label} {detection.score:.2f}"
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        label_height = text_bbox[3] - text_bbox[1] + 6
        label_y = max(0, y1 - label_height)
        draw.rectangle(
            (x1, label_y, x1 + (text_bbox[2] - text_bbox[0]) + 8, label_y + label_height),
            fill=(255, 30, 30),
        )
        draw.text((x1 + 4, label_y + 3), label, fill=(255, 255, 255), font=font)
        rgb.save(output_path, quality=92)

    return output_path


def draw_candidate_boxes(image_path: Path, candidates: list[BoxCandidate]) -> Path:
    output_dir = ROOT / "artifacts" / "visual_grounding"
    output_dir.mkdir(parents=True, exist_ok=True)
    key = f"{image_path}:" + "|".join(
        f"{item.detection.label}:{item.detection.box_xyxy}:{item.final_score}" for item in candidates[:20]
    )
    digest = sha1(key.encode("utf-8")).hexdigest()[:12]
    output_path = output_dir / f"{image_path.stem}_{digest}_candidates.jpg"
    colors = [
        (255, 30, 30),
        (30, 144, 255),
        (34, 180, 80),
        (255, 165, 0),
        (180, 80, 255),
        (255, 80, 180),
    ]

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        draw = ImageDraw.Draw(rgb)
        font = ImageFont.load_default()
        width = max(2, min(rgb.size) // 200)
        for rank, candidate in enumerate(candidates[:20], start=1):
            color = colors[(rank - 1) % len(colors)]
            x1, y1, x2, y2 = candidate.detection.box_xyxy
            draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
            score = candidate.final_score if candidate.final_score is not None else candidate.detection.score
            label = f"{rank}. {candidate.detection.label} {score:.2f}"
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            label_height = text_bbox[3] - text_bbox[1] + 6
            label_y = max(0, y1 - label_height)
            draw.rectangle(
                (x1, label_y, x1 + (text_bbox[2] - text_bbox[0]) + 8, label_y + label_height),
                fill=color,
            )
            draw.text((x1 + 4, label_y + 3), label, fill=(255, 255, 255), font=font)
        rgb.save(output_path, quality=92)

    return output_path


def crop_detection(image_path: Path, detection: Detection) -> Image.Image | None:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        x1, y1, x2, y2 = detection.box_xyxy
        left = max(0, min(width - 1, int(round(x1))))
        top = max(0, min(height - 1, int(round(y1))))
        right = max(left + 1, min(width, int(round(x2))))
        bottom = max(top + 1, min(height, int(round(y2))))
        if right <= left or bottom <= top:
            return None
        return rgb.crop((left, top, right, bottom))


def _display_path(path: Path) -> Path:
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def _candidate_summary(candidate: BoxCandidate) -> dict[str, Any]:
    return {
        "frame_rank": candidate.frame_rank,
        "box_rank": candidate.box_rank,
        "label": candidate.detection.label,
        "prompt": candidate.detection.prompt,
        "threshold": candidate.detection.threshold,
        "dino_score": candidate.detection.score,
        "frame_score": candidate.frame_score,
        "crop_siglip_score": candidate.crop_siglip_score,
        "final_score": candidate.final_score,
        "bbox": candidate.detection.box_xyxy,
        "keyframe_path": str(candidate.keyframe_path),
        "grounding_target": candidate.grounding_target,
    }


def _best_keyframe_path(chunk: dict) -> str | None:
    return (
        chunk.get("keyframe_path")
        or chunk.get("visual_caption_keyframe_path")
        or chunk.get("closest_keyframe_path")
    )


def _model_device(model: Any, torch: Any):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _simple_noun_phrase(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", text.lower())
    tokens = [
        token
        for token in cleaned.split()
        if token not in _stopwords() and len(token) > 2
    ]
    if not tokens:
        return ""
    return " ".join(tokens[:4])


def _extract_location_target(question: str) -> str | None:
    match = re.match(
        r"^\s*where\s+(?:is|are|was|were)\s+(?:a|an|the|any)?\s*(.+?)\s*$",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return _clean_target_phrase(match.group(1))


def _extract_attribute_target(question: str) -> str | None:
    patterns = [
        r"^\s*what\s+color\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(.+?)\s*(?:of|on|in|near|by)\b",
        r"^\s*what\s+color\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, question, flags=re.IGNORECASE)
        if match:
            return _clean_target_phrase(match.group(1))
    return None


def _extract_relation_anchor(question: str) -> str | None:
    match = re.match(
        r"^\s*what\s+(?:is|are|was|were)\s+(?:there\s+)?(?:on|in|inside|near|by|next to|beside|behind|under|above)\s+(?:a|an|the)?\s*(.+?)\s*$",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return _clean_target_phrase(match.group(1))


def _clean_target_phrase(text: str) -> str | None:
    text = re.sub(r"\b(in|on|near|by|at|inside|outside|next to|beside|behind|front of)\b.*$", "", text, flags=re.IGNORECASE)
    cleaned = " ".join(
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _target_stopwords() and len(token) > 1
    )
    return cleaned or None


def _prefer_caption_exact_match(target: str, caption: str) -> str | None:
    if _phrase_in_text(target, caption):
        return target
    target_tokens = target.split()
    caption_lower = caption.lower()
    for token in target_tokens:
        if len(token) > 2 and re.search(rf"\b{re.escape(token)}s?\b", caption_lower):
            return token
    return None


def _caption_confirmed_question_phrases(question_phrases: list[str], caption: str) -> list[str]:
    return [phrase for phrase in question_phrases if _phrase_in_text(phrase, caption)]


def _caption_phrases_relevant_to_question(question: str, caption_phrases: list[str]) -> list[str]:
    question_tokens = set(_tokens(question)) - _target_stopwords()
    scored: list[tuple[int, int, str]] = []
    for phrase in caption_phrases:
        phrase_tokens = set(phrase.split())
        overlap = len(question_tokens & phrase_tokens)
        if overlap:
            scored.append((overlap, -len(phrase_tokens), phrase))
    return [phrase for _overlap, _length, phrase in sorted(scored, reverse=True)]


def _caption_neighbor_target(caption: str, anchor: str, question: str) -> str | None:
    caption_tokens = [token for token in _tokens(caption) if token not in _target_stopwords()]
    anchor_tokens = set(anchor.split())
    question_tokens = set(_tokens(question)) | anchor_tokens
    best: tuple[int, str] | None = None
    for index, token in enumerate(caption_tokens):
        if token not in anchor_tokens:
            continue
        for offset in (1, -1, 2, -2, 3, -3):
            pos = index + offset
            if pos < 0 or pos >= len(caption_tokens):
                continue
            candidate = caption_tokens[pos]
            if candidate in question_tokens or candidate in _target_stopwords():
                continue
            distance = abs(offset)
            if best is None or distance < best[0]:
                best = (distance, candidate)
    return best[1] if best else None


def _caption_action_object_target(question: str, caption: str) -> str | None:
    question_tokens = [token for token in _tokens(question) if token not in _target_stopwords()]
    caption_tokens = [token for token in _tokens(caption) if token not in _target_stopwords()]
    action_tokens = [
        token for token in question_tokens
        if token in caption_tokens and token not in _person_tokens()
    ]
    excluded = set(question_tokens) | _person_tokens()
    for action in action_tokens:
        for index, token in enumerate(caption_tokens):
            if token != action:
                continue
            for candidate in caption_tokens[index + 1 : index + 5]:
                if candidate not in excluded:
                    return candidate
    return None


def _phrase_in_text(phrase: str, text: str) -> bool:
    if not phrase or not text:
        return False
    tokens = phrase.split()
    if not tokens:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(token) + "s?" for token in tokens) + r"\b"
    return bool(re.search(pattern, text.lower()))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _noun_phrases(text: str) -> list[str]:
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", text.lower())
    tokens = [
        token
        for token in cleaned.split()
        if token not in _stopwords() and len(token) > 2
    ]
    phrases: list[str] = []
    for size in (3, 2, 1):
        for start in range(0, max(0, len(tokens) - size + 1)):
            phrases.append(" ".join(tokens[start : start + size]))
    return phrases


def _dedupe_prompts(prompts: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        normalized = re.sub(r"\s+", " ", prompt.strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return tuple(deduped or ["object"])


def _question_mentions_person(question: str) -> bool:
    return bool(set(_tokens(question)) & _person_tokens())


def _person_tokens() -> set[str]:
    return {"person", "people", "man", "woman", "men", "women", "someone"}




def _stopwords() -> set[str]:
    return {
        "are",
        "color",
        "does",
        "for",
        "how",
        "is",
        "the",
        "there",
        "this",
        "what",
        "where",
        "which",
        "who",
        "with",
    }


def _target_stopwords() -> set[str]:
    return _stopwords() | {
        "a",
        "an",
        "any",
        "of",
        "to",
        "left",
        "right",
        "side",
        "image",
        "frame",
        "visible",
        "located",
    }
