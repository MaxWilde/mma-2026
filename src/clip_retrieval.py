from __future__ import annotations

import os
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image


DEFAULT_SIGLIP_MODEL = "google/siglip-base-patch16-224"
DEFAULT_CLIP_MODEL = DEFAULT_SIGLIP_MODEL


def load_clip_model(model_name: str = DEFAULT_CLIP_MODEL, *, local_files_only: bool = True):
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("SigLIP/CLIP retrieval requires torch and transformers.") from exc

    try:
        processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception as exc:
        mode = "local cache only" if local_files_only else "download allowed"
        raise RuntimeError(
            f"Could not load SigLIP/CLIP model '{model_name}' ({mode}). "
            "Cache it first, pass a local model path with --model-name, or rerun with --allow-download."
        ) from exc

    device = os.environ.get("SIGLIP_DEVICE") or os.environ.get("CLIP_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, processor, torch


def load_clip_text_model(model_name: str = DEFAULT_CLIP_MODEL, *, local_files_only: bool = True):
    try:
        import torch
        from transformers import AutoModel, AutoProcessor, Siglip2TextModel, SiglipTextModel
    except ImportError as exc:
        raise RuntimeError("SigLIP/CLIP retrieval requires torch and transformers.") from exc

    try:
        processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        model_type = _local_model_type(model_name)
        if model_type in {"siglip2", "siglip2_text_model"}:
            model = Siglip2TextModel.from_pretrained(model_name, local_files_only=local_files_only)
        elif model_type in {"siglip", "siglip_text_model"}:
            model = SiglipTextModel.from_pretrained(model_name, local_files_only=local_files_only)
        else:
            model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception as exc:
        mode = "local cache only" if local_files_only else "download allowed"
        raise RuntimeError(
            f"Could not load SigLIP/CLIP text model '{model_name}' ({mode}). "
            "Cache it first, pass a local model path with --model-name, or rerun with --allow-download."
        ) from exc

    device = os.environ.get("SIGLIP_DEVICE") or os.environ.get("CLIP_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, processor, torch


def keyframe_metadata(record, youtube_url: str | None = None) -> dict[str, Any]:
    item = asdict(record)
    item["keyframe_path"] = str(record.keyframe_path)
    item["youtube_url"] = youtube_url or record.youtube_url
    item["text"] = ""
    item["visual_caption"] = ""
    item["start_sec"] = record.keyframe_time_sec
    item["end_sec"] = record.keyframe_time_sec
    item["hour_id"] = record.video_id
    item["source_id"] = f"{record.day}/{record.source_name}/{record.video_id}#{record.frame_number or record.keyframe_path.stem}"
    return item


def embed_images(
    image_paths: list[str | Path],
    *,
    model: Any,
    processor: Any,
    torch: Any,
) -> np.ndarray:
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    device = _model_device(model, torch)
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        if hasattr(model, "get_image_features"):
            outputs = model.get_image_features(**inputs)
        else:
            outputs = model(**inputs)
        features = extract_embedding_tensor(outputs, "image", torch=torch)
    return _normalize(features.detach().cpu().float().numpy())


def embed_pil_images(
    images: list[Image.Image],
    *,
    model: Any,
    processor: Any,
    torch: Any,
) -> np.ndarray:
    rgb_images = [image.convert("RGB") for image in images]
    device = _model_device(model, torch)
    inputs = processor(images=rgb_images, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        if hasattr(model, "get_image_features"):
            outputs = model.get_image_features(**inputs)
        else:
            outputs = model(**inputs)
        features = extract_embedding_tensor(outputs, "image", torch=torch)
    return _normalize(features.detach().cpu().float().numpy())


def embed_texts_clip(
    texts: list[str],
    *,
    model: Any,
    processor: Any,
    torch: Any,
) -> np.ndarray:
    device = _model_device(model, torch)
    inputs = processor(text=texts, return_tensors="pt", padding="max_length", truncation=True).to(device)
    with torch.inference_mode():
        if hasattr(model, "get_text_features"):
            outputs = model.get_text_features(**inputs)
        else:
            outputs = model(**inputs)
        features = extract_embedding_tensor(outputs, "text", torch=torch)
    return _normalize(features.detach().cpu().float().numpy())


def embed_texts_clip_profile(
    texts: list[str],
    *,
    model: Any,
    processor: Any,
    torch: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    profile: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_class": model.__class__.__name__,
        "model_device": str(_model_device(model, torch)),
        "text_tower_only": is_text_tower_model(model),
        "has_vision_parameters": has_vision_parameters(model),
    }
    device = _model_device(model, torch)

    start = time.perf_counter()
    inputs = processor(text=texts, return_tensors="pt", padding="max_length", truncation=True)
    profile["padding"] = "max_length"
    profile["tokenizer_time_sec"] = time.perf_counter() - start
    profile["input_devices_before_to"] = _input_devices(inputs)

    start = time.perf_counter()
    inputs = inputs.to(device)
    profile["input_to_device_time_sec"] = time.perf_counter() - start
    profile["input_devices"] = _input_devices(inputs)

    start = time.perf_counter()
    with torch.inference_mode():
        if hasattr(model, "get_text_features"):
            outputs = model.get_text_features(**inputs)
        else:
            outputs = model(**inputs)
        features = extract_embedding_tensor(outputs, "text", torch=torch)
    profile["text_forward_time_sec"] = time.perf_counter() - start
    profile["embedding_tensor_device"] = str(features.device)
    profile["embedding_tensor_shape"] = tuple(int(value) for value in features.shape)

    start = time.perf_counter()
    embeddings = _normalize(features.detach().cpu().float().numpy())
    profile["normalization_time_sec"] = time.perf_counter() - start
    return embeddings, profile


def extract_embedding_tensor(output: Any, kind: Literal["image", "text"], *, torch: Any):
    if _is_tensor(output, torch):
        return _pool_if_needed(output)

    preferred_attrs = [f"{kind}_embeds", "pooler_output", "last_hidden_state"]
    for attr in preferred_attrs:
        value = getattr(output, attr, None)
        if _is_tensor(value, torch):
            return _pool_if_needed(value)

    if isinstance(output, (tuple, list)):
        for item in output:
            if _is_tensor(item, torch):
                return _pool_if_needed(item)
            try:
                return extract_embedding_tensor(item, kind, torch=torch)
            except RuntimeError:
                continue

    if isinstance(output, dict):
        for key in (f"{kind}_embeds", "pooler_output", "last_hidden_state"):
            value = output.get(key)
            if _is_tensor(value, torch):
                return _pool_if_needed(value)
        for value in output.values():
            if _is_tensor(value, torch):
                return _pool_if_needed(value)

    raise RuntimeError(f"Could not extract {kind} embedding tensor from model output of type {type(output).__name__}.")


def _is_tensor(value: Any, torch: Any) -> bool:
    return isinstance(value, torch.Tensor)


def _pool_if_needed(tensor: Any):
    if tensor.ndim == 3:
        return tensor.mean(dim=1)
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    raise RuntimeError(f"Unsupported embedding tensor shape: {tuple(tensor.shape)}")


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float32")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def _model_device(model: Any, torch: Any):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def is_text_tower_model(model: Any) -> bool:
    return model.__class__.__name__ in {"SiglipTextModel", "Siglip2TextModel"}


def has_vision_parameters(model: Any) -> bool:
    try:
        return any(name.startswith("vision") or ".vision" in name for name, _param in model.named_parameters())
    except Exception:
        return False


def _input_devices(inputs: Any) -> dict[str, str]:
    devices: dict[str, str] = {}
    for key, value in dict(inputs).items():
        device = getattr(value, "device", None)
        if device is not None:
            devices[key] = str(device)
    return devices


def _local_model_type(model_name: str) -> str | None:
    config_path = Path(model_name).expanduser() / "config.json"
    if not config_path.is_file():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    model_type = config.get("model_type")
    return str(model_type) if model_type else None
