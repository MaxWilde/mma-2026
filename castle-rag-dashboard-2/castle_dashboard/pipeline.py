"""
Retrieval pipeline wrapper around the CASTLE Day 1 evidence-first backend.

This module is a thin, in-process adapter over the backend living at
`src/*` and `scripts/query_evidence.py` in the shared group repo (see
CASTLE_BACKEND_ROOT / app.py sys.path setup). It mirrors the CLI flow in
`scripts/query_evidence.py` — visual retrieval (query variants + FAISS +
diversification), transcript retrieval (dense + lexical + RRF + rerank),
evidence routing (pick one mode), and on-demand enrichment (GroundingDINO,
transcript heatmap + extractive QA span) — but keeps models/indexes resident
in memory across requests instead of re-loading them per CLI invocation.

Index/model paths default to env vars; set them in .env or the environment
before running the app.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

# Compute nodes are offline. Prevent Hugging Face from spending minutes retrying
# network requests before falling back to files that are already available.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ── paths ────────────────────────────────────────────────────────────────────

_BASE = "/gpfs/scratch1/shared/group_h/data_goncalo"
_SHARED_MODELS = Path("/gpfs/scratch1/shared/group_h/models")
_DASHBOARD_ROOT = Path(__file__).resolve().parents[1]


def _complete_model_dir(path: Path) -> bool:
    """Return true only when a local model directory includes actual weights."""
    if not path.is_dir():
        return False
    return any(
        candidate.is_file()
        for pattern in ("*.safetensors", "pytorch_model*.bin", "model*.bin")
        for candidate in path.glob(pattern)
    )


def _complete_index_dir(path: Path) -> bool:
    return (path / "transcript.faiss").is_file() and (path / "metadata.json").is_file()


def _preferred_index(env_name: str, local_name: str, shared_path: str) -> str:
    configured = os.getenv(env_name)
    if configured:
        return configured
    local_path = _DASHBOARD_ROOT / "indexes" / local_name
    return str(local_path if _complete_index_dir(local_path) else Path(shared_path))


def _preferred_model(
    env_name: str,
    local_name: str,
    shared_name: str,
    *,
    fallback_name: str | None = None,
) -> str:
    configured = os.getenv(env_name)
    if configured:
        return configured
    local_path = _DASHBOARD_ROOT / "models" / local_name
    if _complete_model_dir(local_path):
        return str(local_path)
    shared_path = _SHARED_MODELS / shared_name
    if _complete_model_dir(shared_path):
        return str(shared_path)
    return fallback_name or str(shared_path)

VISUAL_INDEX_DIR: str = os.getenv(
    "VISUAL_INDEX_DIR"
) or _preferred_index(
    "VISUAL_INDEX_DIR",
    "siglip_index_day1",
    f"{_BASE}/artifacts/siglip_index_day1",
)
TRANSCRIPT_INDEX_DIR: str = os.getenv(
    "TRANSCRIPT_INDEX_DIR"
) or _preferred_index(
    "TRANSCRIPT_INDEX_DIR",
    "transcript_index_day1",
    f"{_BASE}/artifacts/transcript_index_day1",
)
SIGLIP_TEXT_MODEL_DIR = _preferred_model(
    "SIGLIP_TEXT_MODEL_NAME",
    "siglip2-so400m-patch16-512-text",
    "siglip2-so400m-patch16-512-text",
)
MINILM_MODEL_DIR = _preferred_model(
    "MINILM_MODEL_DIR",
    "all-MiniLM-L6-v2",
    "all-MiniLM-L6-v2",
    fallback_name="sentence-transformers/all-MiniLM-L6-v2",
)
QA_MODEL_DIR = _preferred_model(
    "ANSWER_SPAN_QA_MODEL",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-cased-distilled-squad",
)
os.environ.setdefault("SIGLIP_TEXT_MODEL_NAME", SIGLIP_TEXT_MODEL_DIR)
os.environ.setdefault("MINILM_MODEL_DIR", MINILM_MODEL_DIR)
os.environ.setdefault("ANSWER_SPAN_QA_MODEL", QA_MODEL_DIR)
DIVERSITY_WINDOW_SEC = float(os.getenv("VISUAL_DIVERSITY_WINDOW_SEC", "30"))
CANDIDATE_MULTIPLIER = int(os.getenv("VISUAL_CANDIDATE_MULTIPLIER", "5"))

# Reverse lookup: keyframe_path → FAISS row index (built lazily after index load)
_keyframe_to_row: dict[str, int] = {}

# ── path remapping (old /scratch-shared/ mount → /gpfs/scratch1/shared/) ────

def _fix_path(path: str) -> str:
    if isinstance(path, str) and path.startswith("/scratch-shared/"):
        return path.replace("/scratch-shared/", "/gpfs/scratch1/shared/", 1)
    return path


def _fix_item_paths(item: dict[str, Any]) -> dict[str, Any]:
    for key in ("keyframe_path", "closest_keyframe_path", "video_path", "transcript_path"):
        if item.get(key):
            item[key] = _fix_path(item[key])
    return item


# ── visual resources (index + metadata — fast, no GPU) ──────────────────────

@lru_cache(maxsize=1)
def _load_visual_index() -> tuple[Any, list[dict[str, Any]], str]:
    from src.retriever import load_index  # type: ignore[import]
    index, metadata, model_name = load_index(VISUAL_INDEX_DIR)
    model_name = _fix_path(model_name)
    for item in metadata:
        _fix_item_paths(item)
    return index, metadata, model_name


def _build_keyframe_to_row() -> dict[str, int]:
    global _keyframe_to_row
    if not _keyframe_to_row:
        _, metadata, _ = _load_visual_index()
        _keyframe_to_row = {m["keyframe_path"]: i for i, m in enumerate(metadata)}
    return _keyframe_to_row


def _resolve_query_model_name(index_model_name: str) -> str:
    """Mirrors scripts/query_evidence.py::resolve_query_model_name — prefer a
    text-only SigLIP/SigLIP2 sibling model so query embedding doesn't have to
    load the (unused) vision tower."""
    env_text_model = os.environ.get("SIGLIP_TEXT_MODEL_NAME")
    if env_text_model and _complete_model_dir(Path(env_text_model)):
        return env_text_model
    path = Path(index_model_name)
    if path.exists():
        sibling = path.with_name(path.name + "-text")
        if sibling.is_dir():
            return str(sibling)
    return index_model_name


@lru_cache(maxsize=1)
def _load_siglip_text_model() -> tuple[Any, Any, Any]:
    from src.clip_retrieval import load_clip_text_model  # type: ignore[import]
    _, _, model_name = _load_visual_index()
    query_model_name = _resolve_query_model_name(model_name)
    return load_clip_text_model(query_model_name, local_files_only=True)


@lru_cache(maxsize=1)
def _load_siglip_full_model() -> tuple[Any, Any, Any]:
    """Full image+text SigLIP model — only needed for dino-siglip-rerank grounding."""
    from src.clip_retrieval import load_clip_model  # type: ignore[import]
    _, _, model_name = _load_visual_index()
    return load_clip_model(model_name, local_files_only=True)


# ── public retrieval API ────────────────────────────────────────────────────

def retrieve_visual(question: str, top_k: int = 20, *, use_query_variants: bool = True) -> list[dict[str, Any]]:
    """Visual evidence retrieval — mirrors scripts/query_evidence.py::retrieve_visual.

    Embeds query variants (noun phrases + prompt templates) with the SigLIP
    text tower, searches FAISS, and diversifies results by source/time so
    near-duplicate keyframes from the same shot don't crowd out the list.
    """
    from scripts.query_evidence import (  # type: ignore[import]
        collect_results,
        collect_variant_results,
        merge_variant_results,
        query_variants,
    )
    from src.clip_retrieval import embed_texts_clip_profile  # type: ignore[import]
    from src.evidence_links import youtube_timestamp_url  # type: ignore[import]
    from src.vqa import format_timestamp  # type: ignore[import]

    t0 = time.perf_counter()
    index, metadata, _ = _load_visual_index()
    model, processor, torch = _load_siglip_text_model()

    variants = query_variants(question) if use_query_variants else [question]
    search_k = min(max(top_k, top_k * CANDIDATE_MULTIPLIER), len(metadata))
    embeddings, _profile = embed_texts_clip_profile(variants, model=model, processor=processor, torch=torch)
    scores, ids = index.search(embeddings, search_k)

    if use_query_variants:
        variant_results = collect_variant_results(scores, ids, metadata, variants, top_k, DIVERSITY_WINDOW_SEC)
        results = merge_variant_results(variant_results, top_k, DIVERSITY_WINDOW_SEC)
    else:
        results = collect_results(scores, ids, metadata)[:top_k]

    kp_to_row = _build_keyframe_to_row()
    for result in results:
        result["evidence_type"] = "visual"
        result["timestamp"] = format_timestamp(float(result["start_sec"]), float(result["end_sec"]))
        result["youtube_timestamp_url"] = youtube_timestamp_url(result)
        _fix_item_paths(result)
        kp = result.get("keyframe_path")
        if kp:
            result["faiss_row_id"] = kp_to_row.get(kp)

    elapsed = time.perf_counter() - t0
    print(f"[pipeline] visual retrieve '{question[:60]}' → {len(results)} results in {elapsed:.2f}s", flush=True)
    return results


def retrieve_transcript(question: str, top_k: int = 20) -> list[dict[str, Any]]:
    """Transcript evidence retrieval — dense (MiniLM) + lexical (BM25) candidates,
    fused with RRF, then reranked with MiniLM by default. A local cross-encoder
    such as BGE can still be enabled explicitly through the environment."""
    from src.transcript_retrieval import retrieve_transcript_evidence  # type: ignore[import]

    t0 = time.perf_counter()
    reranker_model = os.environ.get(
        "TRANSCRIPT_RERANKER_MODEL",
        "minilm",
    )
    rerank_k = max(top_k, int(os.environ.get("TRANSCRIPT_RERANK_K", "50")))
    results = retrieve_transcript_evidence(
        question,
        TRANSCRIPT_INDEX_DIR,
        top_k=top_k,
        rerank_k=rerank_k,
        cross_encoder_name=reranker_model,
        align_playback=True,
    )
    for item in results:
        _fix_item_paths(item)
    elapsed = time.perf_counter() - t0
    print(f"[pipeline] transcript retrieve '{question[:60]}' → {len(results)} results in {elapsed:.2f}s", flush=True)
    return results


def route(
    question: str,
    visual_results: list[dict[str, Any]],
    transcript_results: list[dict[str, Any]],
    feedback: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Pick exactly one evidence mode (visual or transcript) for `question`,
    given each channel's top candidate. Returns the chosen item plus a
    `router_debug` dict with the heuristic/raw/combined scores and reason."""
    from src.evidence_router import route_evidence  # type: ignore[import]
    return route_evidence(
        question,
        visual_results,
        transcript_results,
        feedback=feedback,
    )


def get_mixed_evidence(
    question: str,
    visual_results: list[dict[str, Any]],
    transcript_results: list[dict[str, Any]],
    router_debug: dict[str, Any],
    top_k: int = 20,
    *,
    calibration_mode: str = "max",
) -> list[dict[str, Any]]:
    """Rank visual + transcript candidates together onto one 0–100% confidence
    scale, weighted by which channel the router favored for this question.
    This is what drives the dashboard's ranked-results list."""
    from src.mixed_evidence_ranker import build_mixed_evidence_list  # type: ignore[import]
    return build_mixed_evidence_list(
        question,
        visual_results,
        transcript_results,
        router_debug,
        top_k=top_k,
        calibration_mode=calibration_mode,
    )


def recommend_keywords(
    question: str,
    transcript_results: list[dict[str, Any]],
    *,
    top_n: int = 20,
    max_keywords: int = 20,
) -> dict[str, Any]:
    """Return dashboard-ready transcript steering suggestions.

    Keyword recommendation is advisory only: it never changes retrieval unless
    the user explicitly edits the query or selects terms in the UI.
    """
    if not transcript_results:
        return {
            "source": "unavailable_no_transcript_results",
            "suggested_terms": [],
            "keyword_recommender_debug": {
                "failure_reason": "no transcript results available",
            },
        }
    try:
        from src.transcript_keyword_recommender import recommend_transcript_keywords  # type: ignore[import]
        return recommend_transcript_keywords(
            question,
            transcript_results,
            top_n=min(top_n, len(transcript_results)),
            max_keywords=max_keywords,
        )
    except Exception as exc:
        return {
            "source": "unavailable_keyword_recommender_error",
            "suggested_terms": [],
            "keyword_recommender_debug": {
                "failure_reason": f"{type(exc).__name__}: {exc}",
            },
        }


@lru_cache(maxsize=1)
def _load_transcript_index() -> tuple[Any, list[dict[str, Any]], str]:
    """Load transcript FAISS data once for context, statistics and keywords."""
    from src.retriever import load_index  # type: ignore[import]

    index, metadata, model_name = load_index(TRANSCRIPT_INDEX_DIR)
    for item in metadata:
        _fix_item_paths(item)
    return index, metadata, model_name


@lru_cache(maxsize=1)
def _load_minilm_model() -> Any:
    """Load the transcript/keyword MiniLM model once from local offline files."""
    from src.retriever import load_embedding_model  # type: ignore[import]

    _, _, indexed_model_name = _load_transcript_index()
    configured = os.environ.get("MINILM_MODEL_DIR")
    model_name = (
        configured
        if configured and _complete_model_dir(Path(configured))
        else indexed_model_name
    )
    return load_embedding_model(model_name)


def recommend_keywords_semantic(
    question: str,
    transcript_results: list[dict[str, Any]],
    *,
    top_n: int = 20,
    max_keywords: int = 20,
) -> dict[str, Any]:
    """Rank candidate terms by MiniLM cosine similarity to the query.

    Replaces the slow Qwen LLM path with a fast embedding-based approach:
    build the candidate vocabulary from retrieved transcript text, encode
    everything with the already-loaded MiniLM model, and rank by similarity.
    Confidence scores are real cosine similarities, not LLM-generated numbers.
    """
    import numpy as np
    from src.transcript_keyword_recommender import build_keyword_vocabulary  # type: ignore[import]

    if not transcript_results:
        return {"source": "unavailable_no_transcript_results", "suggested_terms": []}

    candidates = transcript_results[:top_n]
    vocabulary = build_keyword_vocabulary(question, candidates)
    # Merge candidate and query terms; candidate terms first (higher priority)
    seen: set[str] = set()
    pool: list[tuple[str, str]] = []  # (term, source)
    for t in vocabulary["candidate_terms"]:
        if t not in seen:
            seen.add(t)
            pool.append((t, "transcript"))
    for t in vocabulary["query_terms"]:
        if t not in seen:
            seen.add(t)
            pool.append((t, "query"))

    if not pool:
        return {"source": "unavailable_no_vocabulary", "suggested_terms": []}

    model = _load_minilm_model()
    terms = [p[0] for p in pool]
    texts = [question] + terms
    embeddings = model.encode(
        texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    query_emb = embeddings[0]
    term_embs = embeddings[1:]
    scores = (term_embs @ query_emb).tolist()

    scored = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)
    suggested_terms = [
        {
            "term": term,
            "type": "phrase" if " " in term else "keyword",
            "source": source,
            "reason": f"MiniLM cosine similarity to query: {score:.3f}",
            "confidence": float(max(0.0, min(1.0, score))),
        }
        for (term, source), score in scored[:max_keywords]
    ]
    return {
        "source": "minilm_semantic_keywords",
        "suggested_terms": suggested_terms,
    }


def recommend_visual_keywords(
    question: str,
    visual_results: list[dict[str, Any]],
    *,
    max_keywords: int = 20,
) -> dict[str, Any]:
    """Return keyword suggestions derived from visual captions for visual-mode searches.

    Reuses the transcript keyword vocabulary + fallback machinery with caption
    text standing in for transcript snippets, so the same quality filters apply.
    """
    caption_candidates = [
        {
            "transcript_snippet": item.get("visual_caption") or item.get("caption") or "",
            "source_name": item.get("source_name", ""),
            "score": item.get("score", 0.0),
        }
        for item in visual_results
        if item.get("visual_caption") or item.get("caption")
    ]
    if not caption_candidates:
        return {
            "source": "unavailable_no_visual_captions",
            "suggested_terms": [],
            "keyword_recommender_debug": {"failure_reason": "no visual captions available"},
        }
    try:
        from src.transcript_keyword_recommender import build_keyword_vocabulary, fallback_keywords  # type: ignore[import]
        vocabulary = build_keyword_vocabulary(question, caption_candidates)
        result = fallback_keywords(
            question,
            caption_candidates,
            top_n=len(caption_candidates),
            max_keywords=max_keywords,
            vocabulary=vocabulary,
        )
        result["source"] = "visual_caption_keywords"
        return result
    except Exception as exc:
        return {
            "source": "visual_caption_keywords_error",
            "suggested_terms": [],
            "keyword_recommender_debug": {"failure_reason": f"{type(exc).__name__}: {exc}"},
        }


def add_transcript_heatmap(question: str, item: dict[str, Any]) -> dict[str, Any]:
    """On-demand transcript enrichment: extractive QA answer span + n-best
    candidates, and a token-level heatmap for highlighting. Mirrors
    scripts/query_evidence.py::add_transcript_heatmap."""
    from scripts.query_evidence import add_transcript_heatmap as _add_transcript_heatmap  # type: ignore[import]
    return _add_transcript_heatmap(question, item)


def warmup_minilm() -> None:
    """Pre-load the MiniLM model used for semantic keyword ranking."""
    try:
        t0 = time.perf_counter()
        _load_minilm_model()
        print(f"[pipeline] MiniLM keyword model pre-loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    except Exception as exc:
        print(f"[pipeline] MiniLM warmup skipped: {exc}", flush=True)


def warmup_grounding_dino() -> None:
    """Pre-load GroundingDINO into GPU memory so the first 'Localize' click
    doesn't pay the disk→GPU transfer cost (~10–60 s on first use)."""
    try:
        from src.visual_grounding import _load_grounding_dino  # type: ignore[import]
        t0 = time.perf_counter()
        _load_grounding_dino()
        print(f"[pipeline] GroundingDINO pre-loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    except Exception as exc:
        print(f"[pipeline] GroundingDINO warmup skipped: {exc}", flush=True)


def run_grounding(query: str, chunk: dict[str, Any]) -> dict[str, Any]:
    """Run GroundingDINO on a keyframe and return a JSON-serialisable result dict."""
    from src.visual_grounding import GroundingConfig, ground_visual_evidence  # type: ignore[import]
    config = GroundingConfig()
    result = ground_visual_evidence(query, chunk, config=config)
    # visual_grounding returns a path relative to the checked-in repository
    # when possible (normally artifacts/visual_grounding/<file>). Preserve that
    # route-relative path instead of incorrectly resolving it under _BASE.
    output_image_path = str(result.output_image_path) if result.output_image_path else None
    return {
        "keyframe_path": result.keyframe_path,
        "output_image_path": output_image_path,
        "method": result.method,
        "box_xyxy": list(result.box_xyxy) if result.box_xyxy else None,
        "confidence": result.confidence,
        "label": result.label,
        "grounding_target": result.grounding_target,
    }


def get_transcript_context(
    source_name: str,
    video_id: str,
    center_sec: float,
    window_sec: float = 120.0,
) -> list[dict[str, Any]]:
    """Return transcript chunks from the same source/hour within ±window_sec."""
    _, metadata, _ = _load_transcript_index()
    chunks = [
        item for item in metadata
        if item.get("source_name") == source_name
        and (item.get("video_id") == video_id or item.get("hour_id") == video_id)
        and abs(float(item.get("start_sec", 0)) - center_sec) <= window_sec
    ]
    return sorted(chunks, key=lambda x: float(x.get("start_sec", 0)))


def get_nearby_keyframes(
    source_name: str,
    video_id: str,
    center_sec: float,
    n: int = 10,
) -> list[dict[str, Any]]:
    """Return n keyframes from the same stream nearest to center_sec."""
    _, metadata, _ = _load_visual_index()
    same_stream = [
        item for item in metadata
        if item.get("source_name") == source_name
        and item.get("video_id") == video_id
    ]
    same_stream.sort(key=lambda x: abs(float(x.get("start_sec", 0)) - center_sec))
    return same_stream[:n]


def get_query_embedding(question: str) -> list[list[float]]:
    """Return a single L2-normalised SigLIP text embedding as a JSON-serialisable
    list-of-lists (shape [[dim]]), suitable for storage in a dcc.Store."""
    from src.clip_retrieval import embed_texts_clip_profile  # type: ignore[import]
    model, processor, torch = _load_siglip_text_model()
    embeddings, _ = embed_texts_clip_profile([question], model=model, processor=processor, torch=torch)
    return embeddings.tolist()


def rocchio_refine(
    query_embedding: list[list[float]],
    liked_row_ids: list[int],
    disliked_row_ids: list[int],
    *,
    alpha: float = 1.0,
    beta: float = 0.15,
    gamma: float = 0.05,
) -> list[list[float]]:
    """Apply the Rocchio relevance-feedback update and return the refined embedding.

    q_new = α·q_orig + β·mean(liked_vectors) − γ·mean(disliked_vectors)

    The result is L2-normalised so it can be used directly with IndexFlatIP.
    alpha/beta/gamma are the classic Rocchio weights (1.0 / 0.75 / 0.25).
    """
    import numpy as np
    index, _, _ = _load_visual_index()
    q = np.array(query_embedding[0], dtype="float32")
    if liked_row_ids:
        pos = np.mean([index.reconstruct(i) for i in liked_row_ids], axis=0).astype("float32")
        q = alpha * q + beta * pos
    else:
        q = alpha * q
    if disliked_row_ids:
        neg = np.mean([index.reconstruct(i) for i in disliked_row_ids], axis=0).astype("float32")
        q = q - gamma * neg
    norm = float(np.linalg.norm(q))
    if norm > 1e-8:
        q = q / norm
    return [q.tolist()]


def retrieve_visual_from_embedding(
    query_embedding: list[list[float]],
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Visual retrieval from a pre-computed embedding vector (used for Rocchio
    refinement). Skips text encoding and query-variant expansion; applies the
    same diversity window and path-fixing as retrieve_visual."""
    import numpy as np
    from src.evidence_links import youtube_timestamp_url  # type: ignore[import]
    from src.vqa import format_timestamp  # type: ignore[import]

    t0 = time.perf_counter()
    index, metadata, _ = _load_visual_index()
    kp_to_row = _build_keyframe_to_row()
    q = np.array(query_embedding, dtype="float32")
    search_k = min(max(top_k, top_k * CANDIDATE_MULTIPLIER), len(metadata))
    scores, ids = index.search(q, search_k)

    results: list[dict[str, Any]] = []
    for score, row_id in zip(scores[0], ids[0]):
        if row_id < 0:
            continue
        item = dict(metadata[int(row_id)])
        item["score"] = float(score)
        item["faiss_row_id"] = int(row_id)
        results.append(item)
    results = results[:top_k]

    for result in results:
        result["evidence_type"] = "visual"
        result["timestamp"] = format_timestamp(float(result["start_sec"]), float(result["end_sec"]))
        result["youtube_timestamp_url"] = youtube_timestamp_url(result)
        _fix_item_paths(result)
        kp = result.get("keyframe_path")
        if kp:
            result["faiss_row_id"] = kp_to_row.get(kp, result["faiss_row_id"])

    elapsed = time.perf_counter() - t0
    print(f"[pipeline] visual refine (Rocchio) → {len(results)} results in {elapsed:.2f}s", flush=True)
    return results


def get_available_viewpoints() -> list[str]:
    """Return sorted list of source_name values from the visual index."""
    _, metadata, _ = _load_visual_index()
    return sorted({item.get("source_name", "") for item in metadata if item.get("source_name")})


def get_stats() -> dict[str, int]:
    _, v_meta, _ = _load_visual_index()
    _, t_meta, _ = _load_transcript_index()
    return {
        "total_keyframes": len(v_meta),
        "transcript_chunks": len(t_meta),
    }
