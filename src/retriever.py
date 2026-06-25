from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset_loader import EvidenceItem


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def require_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError("Install faiss-cpu to build or query the index: pip install faiss-cpu") from exc
    return faiss


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Install sentence-transformers to embed transcript chunks: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name, local_files_only=True)
    print("Loaded embedding model from local cache")
    return model


def embed_texts(model: Any, texts: list[str], batch_size: int = 64) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray):
    faiss = require_faiss()
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index: Any, metadata: list[dict[str, Any]], output_dir: str | Path, model_name: str) -> None:
    faiss = require_faiss()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output / "transcript.faiss"))
    with (output / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"model_name": model_name, "items": metadata}, f, ensure_ascii=False, indent=2)


def load_index(index_dir: str | Path) -> tuple[Any, list[dict[str, Any]], str]:
    faiss = require_faiss()
    index_path = Path(index_dir) / "transcript.faiss"
    metadata_path = Path(index_dir) / "metadata.json"
    index = faiss.read_index(str(index_path))
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return index, data["items"], data["model_name"]


def query_index(
    question: str,
    index: Any,
    metadata: list[dict[str, Any]],
    model: Any,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    query_embedding = embed_texts(model, [question], batch_size=1)
    scores, ids = index.search(query_embedding, min(top_k, len(metadata)))

    results: list[dict[str, Any]] = []
    for score, row_id in zip(scores[0], ids[0]):
        if row_id < 0:
            continue
        item = dict(metadata[int(row_id)])
        item["score"] = float(score)
        results.append(item)
    return results


def evidence_to_metadata(items: list[EvidenceItem]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in items]
