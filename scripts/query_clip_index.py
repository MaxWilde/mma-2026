#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clip_retrieval import embed_texts_clip_profile, load_clip_model, load_clip_text_model  # noqa: E402
from src.evidence_links import youtube_timestamp_url  # noqa: E402
from src.retriever import require_faiss  # noqa: E402
from src.vqa import format_timestamp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a SigLIP/CLIP keyframe image index.")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--index-dir", default="artifacts/siglip_index_day1")
    parser.add_argument("--text-model-name", default=None, help="Optional text-tower-only SigLIP model path for retrieval.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-download", action="store_true", help="Allow model download if not cached.")
    parser.add_argument("--interactive", action="store_true", help="Load once, then read one question per line from stdin.")
    parser.add_argument("--query-variants", action="store_true", help="Search generic text variants for retrieval diagnostics.")
    parser.add_argument("--merge-variants", action="store_true", help="Print a merged ranking across query variants.")
    parser.add_argument("--synonyms-file", default=None, help="Optional JSON synonym map for query expansion. Disabled by default.")
    parser.add_argument("--no-visual-templates", action="store_true", help="Do not add generic visual prompt templates such as 'a photo of ...'.")
    parser.add_argument("--diversity-window-sec", type=float, default=30.0, help="Suppress repeated frames from the same source/hour within this time window. Use 0 to disable.")
    parser.add_argument("--candidate-multiplier", type=int, default=5, help="Retrieve top_k * multiplier before diversity filtering.")
    parser.add_argument("--grounding", choices=("none", "dino", "dino-siglip-rerank"), default="none")
    parser.add_argument("--grounding-threshold", default="0.25,0.15,0.10,0.05")
    parser.add_argument("--grounding-prompts", default="")
    parser.add_argument("--grounding-rerank", choices=("none", "siglip-crop"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--grounding-max-boxes", type=int, default=5)
    parser.add_argument("--grounding-alpha", type=float, default=1.0, help="Weight for SigLIP frame retrieval score.")
    parser.add_argument("--grounding-beta", type=float, default=1.0, help="Weight for GroundingDINO box confidence.")
    parser.add_argument("--grounding-gamma", type=float, default=1.0, help="Weight for SigLIP crop-text similarity.")
    parser.add_argument("--debug-grounding", action="store_true")
    args = parser.parse_args()
    if not args.interactive and not args.question:
        parser.error("question is required unless --interactive is used")

    if args.grounding_rerank is not None:
        args.grounding = "dino-siglip-rerank" if args.grounding_rerank == "siglip-crop" else "dino"

    timer = StepTimer()
    total_start = time.perf_counter()

    with timer.step("load metadata"):
        metadata, model_name = load_metadata(args.index_dir)
    print(f"Loaded index entries: {len(metadata)}", flush=True)

    with timer.step("load FAISS index"):
        index = load_faiss_index(args.index_dir)

    query_model_name = resolve_query_model_name(model_name, args.text_model_name, args.grounding)
    with timer.step("load SigLIP"):
        if args.grounding == "dino-siglip-rerank":
            model, processor, torch = load_clip_model(model_name, local_files_only=not args.allow_download)
        else:
            model, processor, torch = load_clip_text_model(query_model_name, local_files_only=not args.allow_download)

    print_model_profile(model_name, query_model_name, model, torch, args.grounding)

    if args.interactive:
        print("Interactive mode: enter one question per line. Ctrl-D to exit.", flush=True)
        print_timings(timer, total_start, args.grounding, None)
        for line in sys.stdin:
            question = line.strip()
            if question:
                run_query(question, args, index, metadata, model_name, model, processor, torch)
        return

    assert args.question is not None
    run_query(args.question, args, index, metadata, model_name, model, processor, torch, startup_timer=timer, startup_start=total_start)


def run_query(
    question: str,
    args: argparse.Namespace,
    index: Any,
    metadata: list[dict[str, Any]],
    model_name: str,
    model: Any,
    processor: Any,
    torch: Any,
    *,
    startup_timer: StepTimer | None = None,
    startup_start: float | None = None,
) -> None:
    timer = startup_timer or StepTimer()
    total_start = startup_start or time.perf_counter()

    with timer.step("tokenizer"):
        pass
    timer.timings.pop("tokenizer", None)
    variants = query_variants(
        question,
        load_synonym_map(args.synonyms_file),
        include_visual_templates=not args.no_visual_templates,
    ) if args.query_variants else [question]
    with timer.step("embed question"):
        query_embedding, embed_profile = embed_texts_clip_profile(variants, model=model, processor=processor, torch=torch)
    timer.timings["tokenizer"] = float(embed_profile["tokenizer_time_sec"])
    timer.timings["text model forward"] = float(embed_profile["text_forward_time_sec"])
    timer.timings["normalization"] = float(embed_profile["normalization_time_sec"])

    search_k = min(max(args.top_k, args.top_k * max(1, args.candidate_multiplier)), len(metadata))
    with timer.step("FAISS search"):
        scores, ids = index.search(query_embedding, search_k)

    variant_results = collect_variant_results(scores, ids, metadata, variants, args.top_k, args.diversity_window_sec)
    if args.query_variants:
        results = merge_variant_results(variant_results, args.top_k, args.diversity_window_sec) if args.merge_variants else variant_results[0]["results"]
    else:
        results = variant_results[0]["results"]
    grounding = run_grounding_if_requested(question, args, results, model, processor, torch, timer)

    print(f"\nQuestion: {question}", flush=True)
    print(f"Index: {args.index_dir}", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"Grounding mode: {args.grounding}", flush=True)
    if args.query_variants:
        print("Query variants: " + " | ".join(variants), flush=True)
    print_embedding_profile(embed_profile)
    if args.query_variants:
        print_variant_results(variant_results)
        if args.merge_variants:
            print("\nMERGED VARIANT RANKING", flush=True)
            for rank, result in enumerate(results, start=1):
                print_result(result, rank=rank)
        print_timings(timer, total_start, args.grounding, embed_profile)
        return

    print("\nRETRIEVED KEYFRAMES", flush=True)
    if not results:
        print("No retrieved keyframes.", flush=True)
        print_timings(timer, total_start, args.grounding, embed_profile)
        return
    for rank, result in enumerate(results, start=1):
        print_result(result, rank=rank)

    if grounding:
        selected = selected_result(results, grounding)
        selected_rank = grounding.frame_rank if grounding.frame_rank is not None else 1
        print("\nLOCALIZED EVIDENCE", flush=True)
        print_result(selected, rank=selected_rank)
        print(f"Grounding target: {grounding.grounding_target or 'n/a'}")
        print(f"Evidence localization method: {grounding.method}")
        print(f"Bounding box image: {grounding.output_image_path or 'n/a'}")
        print(f"Candidate boxes image: {grounding.candidates_image_path or 'n/a'}")
        if grounding.box_xyxy:
            print("Bounding box: [" + ", ".join(f"{value:.1f}" for value in grounding.box_xyxy) + "]")
        if grounding.confidence is not None:
            print(f"GroundingDINO confidence: {grounding.confidence:.3f}")
        if grounding.frame_score is not None:
            print(f"SigLIP frame score: {grounding.frame_score:.4f}")
        if grounding.crop_siglip_score is not None:
            print(f"SigLIP crop-text score: {grounding.crop_siglip_score:.4f}")
        if grounding.final_score is not None:
            print(f"Final combined score: {grounding.final_score:.4f}")
        if grounding.frame_rank is not None:
            print(f"Selected frame rank: {grounding.frame_rank}")
        if grounding.box_rank is not None:
            print(f"Selected box rank: {grounding.box_rank}")
        if grounding.label:
            print(f"Evidence localization note: {grounding.label}")
        if args.debug_grounding and grounding.ranked_candidates:
            print_candidate_table(grounding.ranked_candidates)
        if args.debug_grounding and grounding.debug:
            print_grounding_attempts(grounding.debug)

    print_timings(timer, total_start, args.grounding, embed_profile)


def collect_results(scores: Any, ids: Any, metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for score, row_id in zip(scores[0], ids[0]):
        if row_id < 0:
            continue
        item = dict(metadata[int(row_id)])
        item["score"] = float(score)
        results.append(item)
    return results


def collect_variant_results(
    scores: Any,
    ids: Any,
    metadata: list[dict[str, Any]],
    variants: list[str],
    top_k: int,
    diversity_window_sec: float,
) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(variants):
        results = collect_results(scores[variant_index : variant_index + 1], ids[variant_index : variant_index + 1], metadata)
        for result in results:
            result["query_variant"] = variant
        grouped.append({"variant": variant, "results": diversify_results(results, top_k, diversity_window_sec)})
    return grouped


def merge_variant_results(
    variant_results: list[dict[str, Any]],
    top_k: int,
    diversity_window_sec: float,
) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for group in variant_results:
        variant = group["variant"]
        for result in group["results"]:
            key = str(result.get("keyframe_path"))
            existing = by_path.get(key)
            if existing is None or float(result["score"]) > float(existing["score"]):
                merged = dict(result)
                merged["matched_variants"] = [variant]
                by_path[key] = merged
            else:
                existing.setdefault("matched_variants", []).append(variant)
    ranked = sorted(by_path.values(), key=lambda item: float(item["score"]), reverse=True)
    return diversify_results(ranked, top_k, diversity_window_sec)


def diversify_results(results: list[dict[str, Any]], top_k: int, window_sec: float) -> list[dict[str, Any]]:
    if window_sec <= 0:
        return results[:top_k]
    selected: list[dict[str, Any]] = []
    for result in results:
        if not is_near_duplicate(result, selected, window_sec):
            selected.append(result)
        if len(selected) >= top_k:
            break
    return selected


def is_near_duplicate(candidate: dict[str, Any], selected: list[dict[str, Any]], window_sec: float) -> bool:
    candidate_key = source_time_key(candidate)
    if candidate_key is None:
        return False
    candidate_source, candidate_time = candidate_key
    for item in selected:
        item_key = source_time_key(item)
        if item_key is None:
            continue
        item_source, item_time = item_key
        if item_source == candidate_source and abs(candidate_time - item_time) <= window_sec:
            return True
    return False


def source_time_key(item: dict[str, Any]) -> tuple[tuple[str, str, str], float] | None:
    try:
        timestamp = float(item.get("keyframe_time_sec", item.get("start_sec")))
    except (TypeError, ValueError):
        return None
    source = (str(item.get("day", "")), str(item.get("source_name", "")), str(item.get("video_id", "")))
    return source, timestamp


def run_grounding_if_requested(
    question: str,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    model: Any,
    processor: Any,
    torch: Any,
    timer: StepTimer,
):
    if args.grounding == "none" or not results:
        return None

    from src.visual_grounding import GroundingConfig, ground_visual_evidence, ground_visual_evidence_with_rerank

    grounding_config = GroundingConfig(
        thresholds=parse_thresholds(args.grounding_threshold),
        prompts=parse_prompts(args.grounding_prompts),
        rerank="siglip-crop" if args.grounding == "dino-siglip-rerank" else "none",
        max_boxes_per_frame=args.grounding_max_boxes,
        alpha=args.grounding_alpha,
        beta=args.grounding_beta,
        gamma=args.grounding_gamma,
        debug=args.debug_grounding,
    )
    if args.grounding == "dino-siglip-rerank":
        return ground_visual_evidence_with_rerank(
            question,
            results,
            siglip_model=model,
            siglip_processor=processor,
            torch=torch,
            config=grounding_config,
            timings=timer.timings,
        )
    with timer.step("GroundingDINO"):
        return ground_visual_evidence(question, results[0], config=grounding_config)


def query_variants(
    question: str,
    synonym_map: dict[str, list[str]] | None = None,
    *,
    include_visual_templates: bool = True,
) -> list[str]:
    variants: list[str] = [question.strip()]
    noun_phrases = extract_noun_phrases(question)
    variants.extend(noun_phrases[:4])
    if include_visual_templates:
        for phrase in noun_phrases[:4]:
            variants.extend(visual_prompt_templates(phrase))
    synonym_map = synonym_map or {}
    for phrase in noun_phrases[:4]:
        tokens = phrase.split()
        if tokens:
            variants.append(tokens[-1])
        variants.extend(synonym_map.get(phrase, []))
        for token in tokens:
            variants.extend(synonym_map.get(token, []))
    return dedupe_strings(variants)


def visual_prompt_templates(phrase: str) -> list[str]:
    phrase = phrase.strip()
    if not phrase:
        return []
    return [
        f"a photo of {phrase}",
        f"an image of {phrase}",
        f"{phrase} in the scene",
    ]


def extract_noun_phrases(text: str) -> list[str]:
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", text.lower())
    tokens = [token for token in cleaned.split() if token not in query_stopwords() and len(token) > 1]
    phrases: list[str] = []

    color_index = tokens.index("color") if "color" in tokens else -1
    if color_index >= 0 and color_index + 1 < len(tokens):
        phrases.append(tokens[color_index + 1])

    for size in (3, 2, 1):
        for start in range(0, max(0, len(tokens) - size + 1)):
            phrase_tokens = tokens[start : start + size]
            phrase = " ".join(phrase_tokens)
            if phrase:
                phrases.append(phrase)
    return dedupe_strings(phrases)


def load_synonym_map(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise SystemExit(f"Expected synonym map object in {path}")
    synonym_map: dict[str, list[str]] = {}
    for key, values in raw.items():
        normalized_key = re.sub(r"\s+", " ", str(key).strip().lower())
        if isinstance(values, str):
            synonym_map[normalized_key] = [values.lower()]
        elif isinstance(values, list):
            synonym_map[normalized_key] = [str(value).strip().lower() for value in values if str(value).strip()]
        else:
            raise SystemExit(f"Expected string or list for synonym entry {key!r} in {path}")
    return synonym_map


def query_stopwords() -> set[str]:
    return {
        "a",
        "an",
        "are",
        "be",
        "color",
        "does",
        "for",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "the",
        "there",
        "this",
        "to",
        "what",
        "where",
        "which",
        "who",
        "with",
    }


def dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", value.strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


class StepTimer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def step(self, name: str):
        return _TimedStep(self.timings, name)


class _TimedStep:
    def __init__(self, timings: dict[str, float], name: str) -> None:
        self.timings = timings
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.timings[self.name] = time.perf_counter() - self.start


def load_metadata(index_dir: str | Path) -> tuple[list[dict[str, Any]], str]:
    metadata_path = Path(index_dir) / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["items"], data["model_name"]


def load_faiss_index(index_dir: str | Path):
    faiss = require_faiss()
    return faiss.read_index(str(Path(index_dir) / "transcript.faiss"))


def resolve_query_model_name(index_model_name: str, text_model_name: str | None, grounding_mode: str) -> str:
    if grounding_mode == "dino-siglip-rerank":
        return index_model_name
    if text_model_name:
        return text_model_name
    env_text_model = os_environ("SIGLIP_TEXT_MODEL_NAME")
    if env_text_model:
        return env_text_model
    path = Path(index_model_name)
    if path.exists():
        sibling = path.with_name(path.name + "-text")
        if sibling.is_dir():
            return str(sibling)
    return index_model_name


def os_environ(name: str) -> str | None:
    import os

    value = os.environ.get(name)
    return value if value else None


def print_model_profile(index_model_name: str, query_model_name: str, model: Any, torch: Any, grounding_mode: str) -> None:
    model_device = next(model.parameters()).device
    has_vision = False
    try:
        has_vision = any(name.startswith("vision") or ".vision" in name for name, _param in model.named_parameters())
    except Exception:
        pass
    print("\nMODEL DEVICE", flush=True)
    print(f"index model: {index_model_name}", flush=True)
    print(f"query model: {query_model_name}", flush=True)
    print(f"model class: {model.__class__.__name__}", flush=True)
    print(f"model device: {model_device}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"text tower only: {model.__class__.__name__ in {'SiglipTextModel', 'Siglip2TextModel'}}", flush=True)
    print(f"model has vision parameters: {has_vision}", flush=True)
    print(f"grounding mode: {grounding_mode}", flush=True)


def print_embedding_profile(profile: dict[str, Any]) -> None:
    print("\nQUERY EMBEDDING PROFILE", flush=True)
    print(f"CUDA available: {profile.get('cuda_available')}", flush=True)
    print(f"CUDA device: {profile.get('cuda_device')}", flush=True)
    print(f"model class: {profile.get('model_class')}", flush=True)
    print(f"model device: {profile.get('model_device')}", flush=True)
    print(f"input devices: {profile.get('input_devices')}", flush=True)
    print(f"padding: {profile.get('padding')}", flush=True)
    print(f"embedding tensor device: {profile.get('embedding_tensor_device')}", flush=True)
    print(f"embedding tensor shape: {profile.get('embedding_tensor_shape')}", flush=True)
    print(f"text tower only: {profile.get('text_tower_only')}", flush=True)
    print(f"has vision parameters: {profile.get('has_vision_parameters')}", flush=True)


def print_timings(timer: StepTimer, total_start: float, grounding_mode: str, embed_profile: dict[str, Any] | None) -> None:
    total_runtime = time.perf_counter() - total_start
    print("\nTIMINGS", flush=True)
    for name in (
        "load metadata",
        "load FAISS index",
        "load SigLIP",
        "tokenizer",
        "text model forward",
        "normalization",
        "embed question",
        "FAISS search",
        "GroundingDINO",
        "crop reranking",
    ):
        if name in timer.timings:
            print(f"{name}: {timer.timings[name]:.3f}s", flush=True)
        elif name == "GroundingDINO" and grounding_mode == "none":
            print(f"{name}: skipped", flush=True)
        elif name == "crop reranking" and grounding_mode != "dino-siglip-rerank":
            print(f"{name}: skipped", flush=True)
        elif name == "crop reranking" and grounding_mode == "dino-siglip-rerank":
            print(f"{name}: skipped", flush=True)
    if embed_profile:
        print(f"input to device: {float(embed_profile.get('input_to_device_time_sec', 0.0)):.3f}s", flush=True)
    print(f"total runtime: {total_runtime:.3f}s", flush=True)


def print_result(result: dict, rank: int) -> None:
    timestamp = format_timestamp(float(result["start_sec"]), float(result["end_sec"]))
    print(f"\n[{rank}] score={result['score']:.4f} {result['source_name']} {result['day']} {result['video_id']} {timestamp}")
    print(f"keyframe: {result['keyframe_path']}")
    if result.get("frame_number") is not None:
        print(f"frame: {result['frame_number']}")
    youtube_link = youtube_timestamp_url(result)
    if youtube_link:
        print(f"youtube: {youtube_link}")


def print_variant_results(variant_results: list[dict[str, Any]]) -> None:
    print("\nQUERY VARIANT RESULTS", flush=True)
    print("variant | rank | score | keyframe | timestamp | youtube", flush=True)
    for group in variant_results:
        variant = group["variant"]
        for rank, result in enumerate(group["results"], start=1):
            timestamp = format_timestamp(float(result["start_sec"]), float(result["end_sec"]))
            youtube_link = youtube_timestamp_url(result) or ""
            print(
                f"{variant} | "
                f"{rank} | "
                f"{float(result['score']):.4f} | "
                f"{result.get('keyframe_path')} | "
                f"{timestamp} | "
                f"{youtube_link}",
                flush=True,
            )


def selected_result(results: list[dict], grounding) -> dict:
    keyframe_path = getattr(grounding, "keyframe_path", None)
    if keyframe_path:
        for result in results:
            if str(result.get("keyframe_path")) == str(keyframe_path):
                return result
    frame_rank = getattr(grounding, "frame_rank", None)
    if isinstance(frame_rank, int) and 1 <= frame_rank <= len(results):
        return results[frame_rank - 1]
    return results[0]


def parse_thresholds(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    return parsed or (0.25,)


def parse_prompts(value: str) -> tuple[str, ...] | None:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed or None


def print_candidate_table(candidates: list[dict]) -> None:
    print("\nGrounding candidate ranking:")
    print("frame_rank | box_rank | label | dino_score | crop_siglip_score | final_score | bbox")
    for item in candidates:
        bbox = item.get("bbox")
        bbox_text = "[" + ", ".join(f"{float(value):.1f}" for value in bbox) + "]" if bbox else "n/a"
        crop_score = item.get("crop_siglip_score")
        final_score = item.get("final_score")
        crop_text = f"{float(crop_score):.4f}" if crop_score is not None else "n/a"
        final_text = f"{float(final_score):.4f}" if final_score is not None else "n/a"
        print(
            f"{item.get('frame_rank')} | "
            f"{item.get('box_rank')} | "
            f"{item.get('label')} | "
            f"{float(item.get('dino_score', 0.0)):.4f} | "
            f"{crop_text} | "
            f"{final_text} | "
            f"{bbox_text}"
        )


def print_grounding_attempts(attempts: list[dict]) -> None:
    print("\nGrounding attempts:")
    print("frame_rank | prompt | threshold | detections | best_score | selected_box")
    for item in attempts:
        best_score = item.get("best_score")
        selected_box = item.get("selected_box")
        box_text = "[" + ", ".join(f"{float(value):.1f}" for value in selected_box) + "]" if selected_box else "n/a"
        best_text = f"{float(best_score):.4f}" if best_score is not None else "n/a"
        print(
            f"{item.get('frame_rank', 'n/a')} | "
            f"{item.get('prompt')} | "
            f"{float(item.get('threshold', 0.0)):.2f} | "
            f"{item.get('detections')} | "
            f"{best_text} | "
            f"{box_text}"
        )


if __name__ == "__main__":
    main()
